"""
The linear probe on frozen encoder features, used by gaze-probe (Test 3a) and control-probe (Test 3b).

  GazeSeries, HandSeries   fast gaze and palm lookups by VRS timestamp, with a tolerance
  probe_samples            the seeded frame sampler. control-probe replays gaze-probe's rows with it
  Ridge, ridge_cv, r2      ridge regression and its cross-validated strength
  ChannelPCA               shrinks the 1408 channels of each patch, keeping the patch grid
  angular_error_deg        gaze error in degrees
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ego.data import GazeTokenLoader, find_csvs, find_ts_csv, read_vrs_times, video_info


# ---------------------------------------------------------------------------
# Signal lookups
#
# GazeTokenLoader._lookup does (df['ts_us'] - target).abs().idxmin() per call —
# O(rows) each time. Gaze CSVs run 30-60 Hz over ~30 min, so ~100k rows, and the
# probes make tens of thousands of lookups. These reuse the loader's parsing (it
# handles every column-name variant) but look up with searchsorted, and reject a
# sample further than a tolerance away: nearest-neighbour always returns something,
# even for a timestamp past the end of the recording.
# ---------------------------------------------------------------------------

class GazeSeries:
    def __init__(self, gaze_csv, fps, tol_ms=50.0):
        loader = GazeTokenLoader(gaze_csv, fps, standardize=False)  # raw radians/metres
        df = loader.df
        self.ts = df["ts_us"].values.astype(np.int64)
        self.yaw = df["yaw"].values.astype(np.float32)
        self.pitch = df["pitch"].values.astype(np.float32)
        self.depth = df["depth"].values.astype(np.float32)
        self.valid = df["gaze_valid"].values.astype(bool)
        self.tol_us = tol_ms * 1000.0

    def at_ns(self, vrs_ns):
        """Nearest gaze sample to an absolute VRS timestamp, or None."""
        t = int(vrs_ns) // 1000
        i = int(np.searchsorted(self.ts, t))
        best, bestd = -1, None
        for j in (i - 1, i):                      # searchsorted gives the insertion point
            if 0 <= j < len(self.ts):
                d = abs(int(self.ts[j]) - t)
                if bestd is None or d < bestd:
                    best, bestd = j, d
        if best < 0 or bestd > self.tol_us or not self.valid[best]:
            return None
        d = float(self.depth[best])
        if not np.isfinite(d) or d <= 0.05 or d >= 10.0:
            d = 1.0                               # matches GazeTokenLoader.DEPTH_FILL
        v = np.array([self.yaw[best], self.pitch[best], d], dtype=np.float32)
        return None if not np.all(np.isfinite(v)) else v


def point_weights(col, row, sigma=1.0, side=16):
    """
    Weights of the side * side patches, in row-major order as the encoder returns them:
    Gaussian in the distance between each patch centre and (col, row), in patch units, and
    summing to 1.
    """
    c = np.arange(side) + 0.5
    w = np.exp(-((c[None, :] - col) ** 2 + (c[:, None] - row) ** 2) / (2 * sigma ** 2)).ravel()
    return w / w.sum()


def point_features(grid, col, row, sigma=1.0, side=16):
    """
    The features around a point of the patch grid: the patches of grid (..., side * side, D)
    averaged with point_weights.
    """
    return np.tensordot(point_weights(col, row, sigma, side), grid, axes=([0], [-2]))


class HandSeries:
    """
    Palm positions. The hand CSV runs at about 10 Hz against gaze's 30-60 Hz, so the
    tolerance has to be looser than the gaze one, or every lookup would be rejected.
    """
    COLS = ["tx_left_palm_device", "ty_left_palm_device", "tz_left_palm_device",
            "tx_right_palm_device", "ty_right_palm_device", "tz_right_palm_device"]

    def __init__(self, hand_csv, tol_ms=100.0):
        df = pd.read_csv(hand_csv)
        self.ts = df["tracking_timestamp_us"].values.astype(np.int64)
        order = np.argsort(self.ts)
        self.ts = self.ts[order]
        self.xyz = df[self.COLS].values.astype(np.float32)[order]
        self.lvalid = (df["left_tracking_confidence"].values != -1)[order]
        self.rvalid = (df["right_tracking_confidence"].values != -1)[order]
        self.tol_us = tol_ms * 1000.0

    def at_ns(self, vrs_ns):
        """(xyz6, left_valid, right_valid) at the nearest sample, or None if too far."""
        t = int(vrs_ns) // 1000
        i = int(np.searchsorted(self.ts, t))
        best, bestd = -1, None
        for j in (i - 1, i):
            if 0 <= j < len(self.ts):
                d = abs(int(self.ts[j]) - t)
                if bestd is None or d < bestd:
                    best, bestd = j, d
        if best < 0 or bestd > self.tol_us:
            return None
        v = self.xyz[best]
        return (v, bool(self.lvalid[best]), bool(self.rvalid[best])) if np.all(np.isfinite(v)) else None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_indices(n_frames, fps, max_lead_s, windows, window_sec, per_window, rng):
    """
    Frame indices to probe, drawn inside a few short windows rather than scattered
    across the whole recording. load_frames decodes sequentially from min to max
    index, so scattered sampling would decode an entire 30-minute video per
    recording. Windows bound that while keeping frame->timestamp alignment exact
    (no keyframe seeking, which would silently misalign gaze).
    """
    span = int(window_sec * fps)
    usable = n_frames - int(max_lead_s * fps) - 2
    if usable <= span + 1:
        return []
    out = []
    for _ in range(windows):
        start = int(rng.integers(0, usable - span))
        idx = rng.choice(np.arange(start, start + span), size=min(per_window, span), replace=False)
        out.extend(int(i) for i in idx)
    return sorted(set(out))


def probe_samples(video_dir, gaze_dir, participants, recordings, windows, window_sec,
                  per_window, leads, tol_ms, rng, log):
    """
    The probe's frame sampler. For each recording that passes the checks, yields
    (participant, video_path, vrs, keep, g0, g_lead): the kept frame indices, the gaze
    at each kept frame, and the gaze at every lead.

    `recordings` limits each participant to its first recordings, before the checks.
    A frame is kept only if gaze is valid at it and at every lead, so every lead is
    scored on the same frames. `rng` is advanced by one sample_indices() call per
    recording that passes the checks, so the same seed gives the same rows.
    """
    for p in participants:
        vids = sorted((Path(video_dir) / p).glob("*.mp4")) + \
               sorted((Path(video_dir) / p).glob("*.MP4"))
        if recordings:
            vids = vids[:recordings]
        for vp in vids:
            gaze_csv, _ = find_csvs(gaze_dir, p, vp.stem)
            ts_csv = find_ts_csv(str(vp))
            if not gaze_csv or not ts_csv:
                continue
            try:
                n_frames, fps = video_info(str(vp))
                vrs = read_vrs_times(ts_csv)
                gs = GazeSeries(gaze_csv, fps, tol_ms=tol_ms)
            except Exception as e:                          # noqa: BLE001
                log(f"  [skip] {vp.name}: {e}")
                continue
            if not np.isfinite(fps) or fps <= 0 or n_frames <= 0 or len(vrs) < n_frames:
                continue

            idx = sample_indices(n_frames, fps, max(leads), windows, window_sec, per_window, rng)
            keep, g0s, glead = [], [], []
            for i in idx:
                g0 = gs.at_ns(vrs[i])
                if g0 is None:
                    continue
                gl = [gs.at_ns(vrs[i] + int(L * 1e9)) for L in leads]
                if any(g is None for g in gl):
                    continue
                keep.append(i); g0s.append(g0); glead.append(np.stack(gl))
            yield p, vp, vrs, keep, g0s, glead


# ---------------------------------------------------------------------------
# Geometry: (yaw, pitch) -> unit direction, so error is reportable in degrees
# ---------------------------------------------------------------------------

def yawpitch_to_unit(yp):
    yaw, pitch = yp[:, 0], yp[:, 1]
    cp = np.cos(pitch)
    return np.stack([np.sin(yaw) * cp, np.sin(pitch), np.cos(yaw) * cp], axis=1)


def angular_error_deg(pred_yp, true_yp):
    a, b = yawpitch_to_unit(pred_yp), yawpitch_to_unit(true_yp)
    dot = np.clip((a * b).sum(1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# ---------------------------------------------------------------------------
# Ridge regression, solved in whichever form is cheaper
#
# Primal (features D <= samples N):  w = (X'X + aI)^-1 X'Y     -> D x D solve
# Dual   (D > N):                    w = X'(XX' + aI)^-1 Y     -> N x N solve
#
# The headline readout has D = 256 tokens * pca_dim, typically >> N, so the dual
# is the one that runs. Eigendecomposing the Gram matrix ONCE lets every alpha in
# the CV grid be evaluated by a cheap diagonal rescale instead of a fresh solve.
# ---------------------------------------------------------------------------

class Ridge:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, Y):
        self.xm, self.ym = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
        Xc, Yc = X - self.xm, Y - self.ym
        n, d = Xc.shape
        if d <= n:
            A = Xc.T @ Xc + self.alpha * np.eye(d, dtype=np.float64)
            self.W = np.linalg.solve(A, Xc.T @ Yc)
        else:
            K = Xc @ Xc.T + self.alpha * np.eye(n, dtype=np.float64)
            self.W = Xc.T @ np.linalg.solve(K, Yc)
        return self

    def predict(self, X):
        return (X - self.xm) @ self.W + self.ym


def ridge_cv(X, Y, alphas, folds, seed=0):
    """Pick alpha by k-fold CV on the TRAIN split only. Returns (best_alpha, curve)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cuts = np.array_split(order, folds)
    scores = np.zeros(len(alphas))
    for f in range(folds):
        te = cuts[f]
        tr = np.concatenate([cuts[g] for g in range(folds) if g != f])
        Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
        xm, ym = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
        Xc, Yc = Xtr - xm, Ytr - ym
        ntr, d = Xc.shape
        if d > ntr:
            K = Xc @ Xc.T
            s, V = np.linalg.eigh(K)                    # ONE decomposition per fold
            VtY = V.T @ Yc
            Kte = (Xte - xm) @ Xc.T
            for ai, a in enumerate(alphas):
                dual = V @ (VtY / (s[:, None] + a))
                scores[ai] += r2(Kte @ dual + ym, Yte)
        else:
            G = Xc.T @ Xc
            s, V = np.linalg.eigh(G)
            VtXY = V.T @ (Xc.T @ Yc)
            for ai, a in enumerate(alphas):
                W = V @ (VtXY / (s[:, None] + a))
                scores[ai] += r2((Xte - xm) @ W + ym, Yte)
    scores /= folds
    return alphas[int(np.argmax(scores))], scores


def r2(pred, true):
    """Uniform-average R^2 across targets. 0 == predicting the training mean."""
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0, keepdims=True)) ** 2).sum(0)
    return float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-12)))


# ---------------------------------------------------------------------------
# Channel PCA — fit on TRAIN frames only, applied to the patch grid
# ---------------------------------------------------------------------------

class ChannelPCA:
    """(N, tokens, 1408) -> (N, tokens, k). Keeps the spatial grid, shrinks channels."""

    def fit(self, grids, k):
        A = grids.reshape(-1, grids.shape[-1]).astype(np.float64)
        self.mean = A.mean(0, keepdims=True)
        A = A - self.mean
        C = (A.T @ A) / max(len(A) - 1, 1)
        vals, vecs = np.linalg.eigh(C)
        self.comp = vecs[:, ::-1][:, :k].copy()
        self.explained = float(vals[::-1][:k].sum() / max(vals.sum(), 1e-12))
        return self

    def transform(self, grids):
        n, t, _ = grids.shape
        return ((grids.reshape(-1, grids.shape[-1]) - self.mean) @ self.comp) \
            .reshape(n, t, -1).astype(np.float32)
