"""
Gaze-recoverability pretest — how much does the frozen encoder already know about
where the person will look?

THE QUESTION (Ash's Q1, 2026-07-16)
-----------------------------------
The gaze/hand conditioning experiments (EXP-001) and the EK100 probe (EXP-002)
both landed on ~no effect at a ~2 s horizon. Two explanations survive:

  (a) REDUNDANCY — gaze is recoverable from the visual embedding anyway (people
      look at salient objects, which are visible), so a gaze conditioning token
      carries no new information.
  (b) HORIZON — 2 s is short enough that visual continuity alone predicts well,
      leaving conditioning no room to help.

This script distinguishes them, cheaply, before anything expensive is rebuilt.

It measures: can a *small* regressor recover gaze at time t+lead from the frozen
ENCODER embedding at time t? Then it sweeps `lead`.

  marginal information of the gaze token  ~=  1 - recoverability

  * If recoverability stays HIGH as lead grows -> (a). The token is redundant and
    the whole conditioning direction needs rethinking.
  * If recoverability FALLS with lead -> (b). Gaze carries information the encoder
    lacks precisely at longer horizons, which is where the effect was predicted to
    live, and the horizon sweep becomes the priority.

WHY THE ENCODER AND NOT THE PREDICTOR
-------------------------------------
The predictor was *fed* gaze during training, so gaze is trivially recoverable
from it. That would measure nothing. The encoder is frozen and never saw gaze,
so it is the only honest place to ask this question. See
docs/EgoVault/concepts/encoder-vs-predictor.md.

WHY A LINEAR PROBE ("small regressor")
--------------------------------------
Ridge regression - a linear map plus L2 regularisation - has no capacity to
*construct* features. It can only read off directions that already exist linearly
in the representation. That is exactly the standard we want: if a linear map
recovers gaze, the information is not merely present but readily available, which
is what would make a conditioning token redundant. A deep probe would conflate
"the information is there" with "a big enough model can dig it out", which is a
weaker and much less actionable claim.

WHY THE READOUT KEEPS THE PATCH GRID
------------------------------------
Gaze is spatial: yaw/pitch says *where in the scene* the person is looking. The
encoder emits a grid of patch tokens. Mean-pooling over that grid averages away
spatial position and keeps only "what is in the scene", which would systematically
understate recoverability. So the headline readout keeps the grid and reduces the
CHANNEL dimension by PCA instead (1408 -> --pca-dim). The mean-pooled readout is
still computed as an ablation: the gap between them is how much of gaze is
explained by *where* rather than *what*.

Usage
-----
    PY=/mnt/data/home/zj2433/miniconda3/envs/VJEPA2-AC/bin/python
    $PY scripts/gaze_recoverability.py \
        --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --video-dir  data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir   data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --train-participants P01 P02 P03 P04 P05 \
        --test-participants  P06 P07 \
        --out results/gaze_recoverability

Held-out PARTICIPANTS, not held-out clips: otherwise the probe can memorise one
person's gaze habits and the number means nothing.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from ego_common import (
    load_models, encode_independent, load_frames, video_info,
    read_vrs_times, find_csvs, find_ts_csv,
)
from src.datasets.ego_loaders import GazeTokenLoader


# ---------------------------------------------------------------------------
# Fast gaze lookup
#
# GazeTokenLoader._lookup does (df['ts_us'] - target).abs().idxmin() per call —
# O(rows) each time. Gaze CSVs run 30-60 Hz over ~30 min, so ~100k rows, and we
# make tens of thousands of lookups. We reuse the loader's PARSING (it handles
# every column-name variant) but replace the lookup with searchsorted, and add
# the tolerance check the original lacks: nearest-neighbour always returns
# something, even for a timestamp past the end of the recording.
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
# Our headline readout has D = 256 tokens * pca_dim, typically >> N, so the dual
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


def collect(args, encoder, device, participants, split_name, pca, pca_buf, log):
    """Encode sampled frames and pair each with gaze at every lead. Returns arrays."""
    feats_grid, feats_pool, gaze_t, gaze_lead, meta = [], [], [], [], []
    rng = np.random.default_rng(args.seed)
    amp = torch.bfloat16 if (not args.no_amp and device.type == "cuda") else None

    for p in participants:
        vids = sorted((Path(args.video_dir) / p).glob("*.mp4")) + \
               sorted((Path(args.video_dir) / p).glob("*.MP4"))
        if args.recordings:
            vids = vids[:args.recordings]
        for vp in vids:
            gaze_csv, _ = find_csvs(args.gaze_dir, p, vp.stem)
            ts_csv = find_ts_csv(str(vp))
            if not gaze_csv or not ts_csv:
                continue
            try:
                n_frames, fps = video_info(str(vp))
                vrs = read_vrs_times(ts_csv)
                gs = GazeSeries(gaze_csv, fps, tol_ms=args.tol_ms)
            except Exception as e:                          # noqa: BLE001
                log(f"  [skip] {vp.name}: {e}")
                continue
            if not np.isfinite(fps) or fps <= 0 or n_frames <= 0 or len(vrs) < n_frames:
                continue

            idx = sample_indices(n_frames, fps, max(args.leads), args.windows,
                                 args.window_sec, args.per_window, rng)
            keep, y_t, y_lead = [], [], []
            for i in idx:
                g0 = gs.at_ns(vrs[i])
                if g0 is None:
                    continue
                gl = [gs.at_ns(vrs[i] + int(L * 1e9)) for L in args.leads]
                if any(g is None for g in gl):               # paired across leads:
                    continue                                 # one sample set, every lead
                keep.append(i); y_t.append(g0); y_lead.append(np.stack(gl))
            if not keep:
                continue

            frames = load_frames(str(vp), keep, size=args.img_size)      # (K,3,H,W)
            for s in range(0, len(keep), args.batch_size):
                fb = frames[s:s + args.batch_size].unsqueeze(1)          # (b,1,3,H,W)
                h = encode_independent(encoder, fb, device,
                                       normalize_reps=not args.no_normalize_reps,
                                       chunk=args.encode_chunk, amp_dtype=amp)
                g = h.cpu().numpy().astype(np.float32)                   # (b, tokens, D)
                feats_pool.append(g.mean(1))
                if pca[0] is None:
                    pca_buf.append(g)                                    # train-only buffer
                    if sum(b.shape[0] for b in pca_buf) >= args.pca_fit_frames:
                        buf = np.concatenate(pca_buf, 0)
                        pca[0] = ChannelPCA().fit(buf, args.pca_dim)
                        log(f"  [pca] fitted on {len(buf)} frames, "
                            f"{args.pca_dim}/{buf.shape[-1]} channels, "
                            f"{pca[0].explained:.1%} variance")
                        for b in pca_buf:
                            feats_grid.append(pca[0].transform(b).reshape(b.shape[0], -1))
                        pca_buf.clear()
                    else:
                        feats_grid.append(None)                          # placeholder
                else:
                    feats_grid.append(pca[0].transform(g).reshape(g.shape[0], -1))
            gaze_t.append(np.stack(y_t)); gaze_lead.append(np.stack(y_lead))
            meta.extend([(p, vp.stem)] * len(keep))
            log(f"  {split_name} {p}/{vp.stem}: {len(keep)} samples")

    if any(f is None for f in feats_grid):
        raise RuntimeError(
            "PCA was never fitted — not enough training frames buffered. "
            "Lower --pca-fit-frames or raise --windows/--per-window."
        )
    return (np.concatenate(feats_grid, 0), np.concatenate(feats_pool, 0),
            np.concatenate(gaze_t, 0), np.concatenate(gaze_lead, 0), meta)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--train-participants", nargs="+", default=["P01", "P02", "P03", "P04", "P05"])
    ap.add_argument("--test-participants", nargs="+", default=["P06", "P07"])
    ap.add_argument("--leads", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0],
                    help="anticipation lead times in seconds")
    ap.add_argument("--recordings", type=int, default=6, help="max recordings per participant")
    ap.add_argument("--windows", type=int, default=8, help="sampling windows per recording")
    ap.add_argument("--window-sec", type=float, default=8.0)
    ap.add_argument("--per-window", type=int, default=4)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--pca-fit-frames", type=int, default=300)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--encode-chunk", type=int, default=32)
    ap.add_argument("--tol-ms", type=float, default=50.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-normalize-reps", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/gaze_recoverability")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(f"{out}.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"[setup] device={device}  leads={args.leads}")
    log(f"[setup] train={args.train_participants}  test={args.test_participants}")

    t0 = time.time()
    encoder, _, _ = load_models(args.checkpoint, device, context_steps=1)
    log(f"[setup] encoder loaded in {time.time() - t0:.0f}s (predictor built but unused)")

    pca, buf = [None], []
    log("[collect] train split")
    Xg_tr, Xp_tr, g0_tr, gl_tr, _ = collect(args, encoder, device,
                                            args.train_participants, "train", pca, buf, log)
    log("[collect] test split")
    Xg_te, Xp_te, g0_te, gl_te, _ = collect(args, encoder, device,
                                            args.test_participants, "test", pca, buf, log)
    log(f"[collect] train n={len(Xg_tr)}  test n={len(Xg_te)}  "
        f"grid dim={Xg_tr.shape[1]}  pooled dim={Xp_tr.shape[1]}")

    rows = []
    for li, lead in enumerate(args.leads):
        Ytr, Yte = gl_tr[:, li, :], gl_te[:, li, :]

        for name, Xtr, Xte in (("spatial", Xg_tr, Xg_te), ("pooled", Xp_tr, Xp_te)):
            a, _ = ridge_cv(Xtr.astype(np.float64), Ytr.astype(np.float64),
                            args.alphas, args.folds, seed=args.seed)
            pred = Ridge(a).fit(Xtr.astype(np.float64), Ytr.astype(np.float64)) \
                           .predict(Xte.astype(np.float64))
            rows.append(dict(lead_s=lead, readout=name, alpha=a,
                             r2=r2(pred, Yte),
                             r2_yawpitch=r2(pred[:, :2], Yte[:, :2]),
                             ang_err_deg_median=float(np.median(angular_error_deg(pred, Yte))),
                             ang_err_deg_mean=float(np.mean(angular_error_deg(pred, Yte))),
                             depth_mae_m=float(np.mean(np.abs(pred[:, 2] - Yte[:, 2]))),
                             n_test=len(Yte)))

        # Baseline 1 — chance. Predict the TRAIN mean. R^2 near 0 by construction.
        mean_pred = np.repeat(Ytr.mean(0, keepdims=True), len(Yte), axis=0)
        rows.append(dict(lead_s=lead, readout="baseline_mean", alpha=np.nan,
                         r2=r2(mean_pred, Yte), r2_yawpitch=r2(mean_pred[:, :2], Yte[:, :2]),
                         ang_err_deg_median=float(np.median(angular_error_deg(mean_pred, Yte))),
                         ang_err_deg_mean=float(np.mean(angular_error_deg(mean_pred, Yte))),
                         depth_mae_m=float(np.mean(np.abs(mean_pred[:, 2] - Yte[:, 2]))),
                         n_test=len(Yte)))

        # Baseline 2 — persistence. Predict gaze(t+lead) = gaze(t). This uses a
        # PRIVILEGED input the encoder never gets, so it is a reference point for
        # "how predictable is gaze at this lead at all", not a competitor.
        rows.append(dict(lead_s=lead, readout="baseline_persistence", alpha=np.nan,
                         r2=r2(g0_te, Yte), r2_yawpitch=r2(g0_te[:, :2], Yte[:, :2]),
                         ang_err_deg_median=float(np.median(angular_error_deg(g0_te, Yte))),
                         ang_err_deg_mean=float(np.mean(angular_error_deg(g0_te, Yte))),
                         depth_mae_m=float(np.mean(np.abs(g0_te[:, 2] - Yte[:, 2]))),
                         n_test=len(Yte)))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}.csv", index=False)

    log("\n" + "=" * 78)
    log("GAZE RECOVERABILITY  —  R^2 on held-out participants (yaw/pitch only)")
    log("=" * 78)
    piv = df.pivot(index="lead_s", columns="readout", values="r2_yawpitch")
    log(piv.to_string(float_format=lambda v: f"{v:7.3f}"))
    log("\nMedian angular error, degrees")
    log(df.pivot(index="lead_s", columns="readout", values="ang_err_deg_median")
          .to_string(float_format=lambda v: f"{v:7.2f}"))

    sp = df[df.readout == "spatial"].sort_values("lead_s")
    lo, hi = sp.r2_yawpitch.iloc[-1], sp.r2_yawpitch.iloc[0]
    log("\n" + "-" * 78)
    log(f"READING: spatial R^2 goes {hi:.3f} @ {sp.lead_s.iloc[0]:.2f}s "
        f"-> {lo:.3f} @ {sp.lead_s.iloc[-1]:.2f}s")
    if hi - lo < 0.05:
        log("  Recoverability is FLAT across lead. Consistent with REDUNDANCY: the")
        log("  encoder already carries future gaze, so a gaze token adds little at any")
        log("  horizon. This would weaken the behavioral-conditioning direction — see")
        log("  docs/EgoVault/topics/path-2-behavioral-conditioning.md.")
    else:
        log("  Recoverability FALLS with lead. Consistent with the HORIZON reading:")
        log("  marginal information (~1 - R^2) grows with horizon, so the null at ~2s")
        log("  may be a horizon artefact. Prioritise the horizon sweep — see")
        log("  docs/EgoVault/experiments/EXP-002-ek100-probe-null.md.")
    log("  Judge against baseline_mean (chance floor), not against zero.")
    log("-" * 78)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name, style in (("spatial", "-o"), ("pooled", "-s"),
                            ("baseline_persistence", "--^"), ("baseline_mean", ":x")):
            d = df[df.readout == name].sort_values("lead_s")
            ax[0].plot(d.lead_s, d.r2_yawpitch, style, label=name)
            ax[1].plot(d.lead_s, d.ang_err_deg_median, style, label=name)
        ax[0].set_xlabel("anticipation lead (s)"); ax[0].set_ylabel("R² (yaw, pitch)")
        ax[0].set_title("Gaze recoverability from the frozen encoder")
        ax[0].axhline(0, color="k", lw=0.5); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[1].set_xlabel("anticipation lead (s)"); ax[1].set_ylabel("median angular error (°)")
        ax[1].set_title("Angular error"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{out}.png", dpi=150)
        log(f"[out] {out}.png")
    except Exception as e:                                   # noqa: BLE001
        log(f"[warn] plot skipped: {e}")

    with open(f"{out}.json", "w") as f:
        json.dump(dict(config=vars(args), rows=rows,
                       n_train=len(Xg_tr), n_test=len(Xg_te),
                       pca_explained=pca[0].explained), f, indent=2, default=str)
    log(f"[out] {out}.csv  {out}.json  {out}.log")
    logf.close()


if __name__ == "__main__":
    main()
