"""
EXP-003 control probe — is the collapse about GAZE, or about the ENCODER?

THE CONFOUND (adoption plan step 13, found 2026-08-18)
------------------------------------------------------
EXP-003 regressed gaze from frozen encoder features and found skill 0.116 across
held-out RECORDINGS but 0.001 across held-out PARTICIPANTS. That was read as
"gaze does not transfer across people". But in HD-EPIC each participant has their
own kitchen, so the participant split varies people AND scenes together. The same
numbers are equally consistent with "the frozen encoder's features do not transfer
across kitchens" — which would explain the EXP-001 near-null and the EXP-002 null
with no reference to gaze at all.

One probe separates them: regress a DIFFERENT target from the SAME features across
the SAME splits.

  control target TRANSFERS  -> features are fine cross-person; the gaze null is
                               genuinely about gaze. EXP-003 stands, now defended.
  control target COLLAPSES  -> the probe was measuring encoder scene-generalisation.
                               EXP-003's headline needs restating and the project's
                               central problem changes.

THE TARGET
----------
Palm position in the Aria device frame, from the same MPS recordings. It is the
right control for three reasons: it is behavioural like gaze, it is *visible in
the frame* so a transferring encoder should localise it, and it is sampled on the
same clock so it needs no new alignment machinery.

RECOVERING THE ROWS
-------------------
results/gaze_features.npz stores features and gaze but NOT the frame index each row
came from, so a new target cannot be looked up naively. Rather than re-encode
(~25 min), this replays gaze_recoverability.collect()'s sampler: it is seeded, and
its rng is advanced by exactly one sample_indices() call per surviving recording.
Replaying it with the cache's own config reproduces the frame indices.

That replay is then PROVEN rather than assumed: the gaze vectors it recovers must
equal the cached g0_tr / g0_te element-for-element, and the per-recording row counts
must match meta_tr / meta_te. If either check fails the script aborts, because a
silently misaligned row would produce a confident wrong answer — the single most
dangerous failure mode available here.

WHAT IS HELD FIXED
------------------
Same cached features, same PCA, same ridge implementation, same alpha grid, same
folds, same seed, same splits, same leads, same skill metric. Only the target
changes. Hand validity forces some rows to be dropped, so gaze is ALSO re-scored on
the reduced row set: the headline comparison is gaze-vs-hand on identical rows.

Usage
-----
    PY=/mnt/data/home/zj2433/miniconda3/envs/VJEPA2-AC/bin/python
    $PY scripts/control_recoverability.py \
        --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --out results/control_recoverability

No GPU and no video decoding: it reads mp4 headers, CSVs and the feature cache.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from ego_common import video_info, read_vrs_times, find_csvs, find_ts_csv
# Import the ORIGINAL sampler and probe rather than reimplementing them: any
# divergence would silently invalidate the comparison this script exists to make.
from gaze_recoverability import GazeSeries, sample_indices, Ridge, ridge_cv, r2


# ---------------------------------------------------------------------------
# Hand series — searchsorted lookup with a tolerance, mirroring GazeSeries.
#
# The hand CSV runs at ~10 Hz against gaze's ~30-60 Hz, so the tolerance has to be
# looser or every lookup would be rejected. Nearest-neighbour without a tolerance
# would fabricate targets past the end of a recording, which is exactly the bug
# GazeSeries was written to avoid.
# ---------------------------------------------------------------------------

class HandSeries:
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
# Replay of gaze_recoverability.collect()'s sampling, without encoding anything
# ---------------------------------------------------------------------------

def replay(cfg, participants, split_name, log):
    """
    Returns per-row (participant, stem, frame_index, vrs_ns, gaze3), in the exact
    order collect() produced them. The rng is created once per split and advanced
    by one sample_indices() call per recording that survives the same guards, so
    the order of those guards is load-bearing and copied verbatim.
    """
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for p in participants:
        vids = sorted((Path(cfg["video_dir"]) / p).glob("*.mp4")) + \
               sorted((Path(cfg["video_dir"]) / p).glob("*.MP4"))
        if cfg["recordings"]:
            vids = vids[:cfg["recordings"]]
        for vp in vids:
            gaze_csv, _ = find_csvs(cfg["gaze_dir"], p, vp.stem)
            ts_csv = find_ts_csv(str(vp))
            if not gaze_csv or not ts_csv:
                continue
            try:
                n_frames, fps = video_info(str(vp))
                vrs = read_vrs_times(ts_csv)
                gs = GazeSeries(gaze_csv, fps, tol_ms=cfg["tol_ms"])
            except Exception as e:                                # noqa: BLE001
                log(f"  [skip] {vp.name}: {e}")
                continue
            if not np.isfinite(fps) or fps <= 0 or n_frames <= 0 or len(vrs) < n_frames:
                continue

            idx = sample_indices(n_frames, fps, max(cfg["leads"]), cfg["windows"],
                                 cfg["window_sec"], cfg["per_window"], rng)
            kept = 0
            for i in idx:
                g0 = gs.at_ns(vrs[i])
                if g0 is None:
                    continue
                gl = [gs.at_ns(vrs[i] + int(L * 1e9)) for L in cfg["leads"]]
                if any(g is None for g in gl):
                    continue
                rows.append({"participant": p, "stem": vp.stem, "frame": int(i),
                             "vrs_ns": int(vrs[i]), "gaze": g0})
                kept += 1
            if kept:
                log(f"  {split_name} {p}/{vp.stem}: {kept} rows")
    return rows


def verify(rows, g0_cached, meta_cached, split_name, log):
    """Abort unless the replay reproduces the cache exactly."""
    if len(rows) != len(g0_cached):
        raise SystemExit(f"[{split_name}] replay produced {len(rows)} rows, cache has "
                         f"{len(g0_cached)}. The sampler config does not match the cache.")
    g_replay = np.stack([r["gaze"] for r in rows]).astype(np.float32)
    if not np.array_equal(g_replay, g0_cached.astype(np.float32)):
        bad = int((~np.all(g_replay == g0_cached, axis=1)).sum())
        raise SystemExit(f"[{split_name}] replayed gaze differs from the cache on {bad} rows. "
                         "Row alignment is not established; refusing to continue.")
    m_replay = np.array([f"{r['participant']}/{r['stem']}" for r in rows])
    if not np.array_equal(m_replay, meta_cached.astype(str)):
        raise SystemExit(f"[{split_name}] replayed recording labels differ from the cache.")
    log(f"[verify] {split_name}: {len(rows)} rows match the cache exactly "
        f"(gaze values and recording labels)")


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def hand_targets(rows, cfg, log):
    """
    (N, n_leads, 6) palm xyz and (N, n_leads, 2) per-hand validity, aligned to rows.

    Paired across leads exactly as the gaze probe is: a row counts as usable for a
    hand only if that hand is tracked at t AND at every lead. Otherwise the curve
    would be computed on a sample set that shifts with lead.
    """
    leads = cfg["leads"]
    Y = np.full((len(rows), len(leads), 6), np.nan, dtype=np.float32)
    V = np.zeros((len(rows), len(leads), 2), dtype=bool)
    series, missing = {}, set()
    for n, r in enumerate(rows):
        key = (r["participant"], r["stem"])
        if key not in series:
            _, hand_csv = find_csvs(cfg["gaze_dir"], r["participant"], r["stem"])
            series[key] = HandSeries(hand_csv, tol_ms=cfg["hand_tol_ms"]) if hand_csv else None
            if series[key] is None:
                missing.add(key)
        hs = series[key]
        if hs is None:
            continue
        for li, L in enumerate(leads):
            got = hs.at_ns(r["vrs_ns"] + int(L * 1e9))
            if got is None:
                continue
            xyz, lv, rv = got
            Y[n, li] = xyz
            V[n, li] = (lv, rv)
    if missing:
        log(f"[hand] no hand CSV for {len(missing)} recordings: "
            f"{', '.join(sorted(f'{p}/{s}' for p, s in missing))}")
    return Y, V


# ---------------------------------------------------------------------------
# Probe — identical machinery to EXP-003, only the target differs
# ---------------------------------------------------------------------------

def per_col_mse(pred, true):
    """
    Per-column mean squared error, the raw material for both skill definitions.

    Pooled weights columns by their variance, which is what EXP-003 reported for
    yaw/pitch. Per-column is scale-free and is the honest one when a target mixes
    units or ranges; both are printed so a disagreement between them is visible
    rather than buried.
    """
    ss_res = ((true - pred) ** 2).mean(0)
    return ss_res


def run_probe(Xtr, Ytr, Xte, Yte, alphas, folds, seed):
    a, _ = ridge_cv(Xtr.astype(np.float64), Ytr.astype(np.float64), alphas, folds, seed=seed)
    pred = Ridge(a).fit(Xtr.astype(np.float64), Ytr.astype(np.float64)).predict(Xte.astype(np.float64))
    chance = np.repeat(Ytr.mean(0, keepdims=True), len(Yte), axis=0)
    mse_m, mse_c = per_col_mse(pred, Yte), per_col_mse(chance, Yte)
    return {
        "alpha": a,
        "skill": float(1.0 - mse_m.sum() / max(mse_c.sum(), 1e-12)),
        "skill_percol": float(np.mean(1.0 - mse_m / np.maximum(mse_c, 1e-12))),
        "r2": r2(pred, Yte),
        "rmse": float(np.sqrt(mse_m.mean())),
        "rmse_chance": float(np.sqrt(mse_c.mean())),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--cache", default="results/gaze_features.npz")
    ap.add_argument("--source-json", default="results/gaze_recoverability.json",
                    help="the EXP-003 run whose config built the cache")
    ap.add_argument("--hand-tol-ms", type=float, default=100.0,
                    help="hand CSVs run at ~10 Hz, so 50 ms would reject most lookups")
    ap.add_argument("--splits", nargs="+", default=["participant", "recording", "random"])
    ap.add_argument("--readout", default="spatial", choices=["spatial", "pooled", "both"])
    ap.add_argument("--out", default="results/control_recoverability")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(f"{out}.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    src = json.load(open(args.source_json))["config"]
    cfg = {k: src[k] for k in ("video_dir", "gaze_dir", "recordings", "windows", "window_sec",
                               "per_window", "leads", "seed", "tol_ms")}
    cfg["video_dir"], cfg["gaze_dir"] = args.video_dir, args.gaze_dir
    cfg["hand_tol_ms"] = args.hand_tol_ms
    leads = cfg["leads"]
    log(f"[setup] replaying EXP-003 sampler: seed={cfg['seed']} recordings={cfg['recordings']} "
        f"windows={cfg['windows']} window_sec={cfg['window_sec']} per_window={cfg['per_window']}")
    log(f"[setup] leads={leads}  gaze tol={cfg['tol_ms']}ms  hand tol={cfg['hand_tol_ms']}ms")

    z = np.load(args.cache, allow_pickle=True)
    if list(z["leads"]) != list(leads):
        raise SystemExit(f"cache leads {list(z['leads'])} != config leads {leads}")

    log("[replay] train split")
    rows_tr = replay(cfg, src["train_participants"], "train", log)
    log("[replay] test split")
    rows_te = replay(cfg, src["test_participants"], "test", log)
    verify(rows_tr, z["g0_tr"], z["meta_tr"], "train", log)
    verify(rows_te, z["g0_te"], z["meta_te"], "test", log)

    log("[hand] looking up palm positions at every lead")
    Yh_tr, Vh_tr = hand_targets(rows_tr, cfg, log)
    Yh_te, Vh_te = hand_targets(rows_te, cfg, log)

    # Pool the two splits once; the split logic below re-slices this pool exactly
    # as gaze_recoverability.main() does, so "participant" is the original arrays.
    Xg = np.concatenate([z["Xg_tr"], z["Xg_te"]])
    Xp = np.concatenate([z["Xp_tr"], z["Xp_te"]])
    Gl = np.concatenate([z["gl_tr"], z["gl_te"]])
    Yh = np.concatenate([Yh_tr, Yh_te])
    Vh = np.concatenate([Vh_tr, Vh_te])
    M = np.concatenate([z["meta_tr"], z["meta_te"]])
    P = np.array([m.split("/")[0] for m in M])
    n_tr0 = len(z["Xg_tr"])
    log(f"[data] pooled n={len(Xg)}  train={n_tr0}  test={len(Xg)-n_tr0}")

    for hand, col in (("left", 0), ("right", 1)):
        v = Vh[..., col].all(1)
        log(f"[hand] {hand} palm valid at t and every lead: {v.sum()}/{len(v)} rows ({v.mean():.1%})")

    readouts = ([("spatial", Xg), ("pooled", Xp)] if args.readout == "both"
                else [(args.readout, Xg if args.readout == "spatial" else Xp)])
    alphas, folds, seed = src["alphas"], src["folds"], src["seed"]

    rows = []
    for split in args.splits:
        if split == "participant":
            tr_all = np.arange(n_tr0)
            te_all = np.arange(n_tr0, len(Xg))
        else:
            rng = np.random.default_rng(seed)
            if split == "random":
                perm = rng.permutation(len(Xg))
                cut = int(0.75 * len(Xg))
                tr_all, te_all = perm[:cut], perm[cut:]
            else:
                recs = np.unique(M)
                rng.shuffle(recs)
                held = set(recs[int(0.75 * len(recs)):])
                te_all = np.array([i for i, m in enumerate(M) if m in held])
                tr_all = np.array([i for i, m in enumerate(M) if m not in held])

        # target -> (Y array, row mask). Gaze is scored twice: on all rows (the
        # EXP-003 number) and on each hand's rows, so the comparison that decides
        # the confound is made on IDENTICAL samples.
        targets = {
            "gaze_yawpitch": (Gl[:, :, :2], np.ones(len(Xg), bool)),
            "left_palm_xyz": (Yh[:, :, 0:3], Vh[..., 0].all(1)),
            "right_palm_xyz": (Yh[:, :, 3:6], Vh[..., 1].all(1)),
            "gaze_on_left_rows": (Gl[:, :, :2], Vh[..., 0].all(1)),
            "gaze_on_right_rows": (Gl[:, :, :2], Vh[..., 1].all(1)),
        }

        for tname, (Y, keep) in targets.items():
            tr = tr_all[keep[tr_all]]
            te = te_all[keep[te_all]]
            if len(tr) < 50 or len(te) < 20:
                log(f"[skip] {split}/{tname}: n_train={len(tr)} n_test={len(te)} — too few")
                continue
            for rname, X in readouts:
                for li, L in enumerate(leads):
                    res = run_probe(X[tr], Y[tr, li], X[te], Y[te, li], alphas, folds, seed)
                    rows.append(dict(split=split, target=tname, readout=rname, lead_s=L,
                                     n_train=len(tr), n_test=len(te), **res))
            log(f"  [{split}] {tname}: n_train={len(tr)} n_test={len(te)} done")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}.csv", index=False)

    log("\n" + "=" * 92)
    log("CONTROL PROBE — same frozen features, same splits, same ridge. Only the target changes.")
    log("=" * 92)
    for split in args.splits:
        d = df[(df.split == split) & (df.readout == readouts[0][0])]
        if d.empty:
            continue
        log(f"\nSKILL — {split} split")
        log(d.pivot(index="lead_s", columns="target", values="skill")
             .to_string(float_format=lambda v: f"{v:8.3f}"))
        log(f"n_test: " + ", ".join(f"{t}={int(d[d.target==t].n_test.iloc[0])}"
                                    for t in d.target.unique()))

    # ---- the decision ------------------------------------------------------
    def at(split, target, lead=0.0):
        d = df[(df.split == split) & (df.target == target) & (df.lead_s == lead) &
               (df.readout == readouts[0][0])]
        return float(d.skill.iloc[0]) if len(d) else np.nan

    log("\n" + "-" * 92)
    log("READING — skill at lead 0, the number EXP-003's headline rests on")
    tbl = []
    for t in ("gaze_yawpitch", "left_palm_xyz", "right_palm_xyz",
              "gaze_on_left_rows", "gaze_on_right_rows"):
        tbl.append((t, at("recording", t), at("participant", t)))
    for t, rec, par in tbl:
        log(f"  {t:<22} recording={rec:+.3f}   participant={par:+.3f}")

    hands = [v for v in (at("participant", "left_palm_xyz"), at("participant", "right_palm_xyz"))
             if np.isfinite(v)]
    hand_par = max(hands) if hands else np.nan
    hands_rec = [v for v in (at("recording", "left_palm_xyz"), at("recording", "right_palm_xyz"))
                 if np.isfinite(v)]
    hand_rec = max(hands_rec) if hands_rec else np.nan
    log("")
    if not np.isfinite(hand_par):
        log("  Control target could not be scored — check hand CSV coverage above.")
    elif hand_par > 0.05:
        log("  The CONTROL TARGET TRANSFERS across participants while gaze does not.")
        log("  The frozen features carry information that survives the change of person and")
        log("  kitchen, so EXP-003's collapse is not a generic encoder failure. Its headline")
        log("  STANDS and is now defended against the confound.")
    elif hand_rec > 0.05:
        log("  The control target is recoverable WITHIN kitchens but COLLAPSES across them,")
        log("  exactly like gaze. EXP-003 was measuring encoder scene-generalisation, not gaze")
        log("  redundancy. Its headline needs restating and the project's central problem")
        log("  changes: the substrate does not transfer, which explains both nulls without")
        log("  mentioning gaze.")
    else:
        log("  The control target is not recoverable on ANY split, so it says nothing about")
        log("  the encoder. Either palm position is not linearly present in these features or")
        log("  the target is too noisy. Try another control before drawing a conclusion.")
    log("-" * 92)

    # ---- how much of the feature space is person/kitchen identity? ----------
    # A direct reading of the domain gap: if a linear map can name the participant
    # from a held-out RECORDING, person/kitchen identity dominates the features,
    # which is the mechanism the confound proposes.
    recs = np.unique(M)
    rng = np.random.default_rng(seed)
    rng.shuffle(recs)
    held = set(recs[int(0.75 * len(recs)):])
    te = np.array([i for i, m in enumerate(M) if m in held])
    tr = np.array([i for i, m in enumerate(M) if m not in held])
    labels = np.unique(P)
    onehot = (P[:, None] == labels[None, :]).astype(np.float64)
    a, _ = ridge_cv(Xp[tr].astype(np.float64), onehot[tr], alphas, folds, seed=seed)
    pred = Ridge(a).fit(Xp[tr].astype(np.float64), onehot[tr]).predict(Xp[te].astype(np.float64))
    acc = float((labels[pred.argmax(1)] == P[te]).mean())
    log(f"\n[identity] participant recoverable from pooled features on held-out RECORDINGS: "
        f"{acc:.1%} ({len(labels)}-way, chance {1/len(labels):.1%}, n_test={len(te)})")
    log("           High accuracy means the representation is dominated by who/where, which is")
    log("           the mechanism behind the participant/kitchen confound.")

    with open(f"{out}.json", "w") as f:
        json.dump({"config": vars(args), "source_config": src,
                   "identity_accuracy": acc, "rows": rows}, f, indent=2, default=str)
    log(f"\n[out] {out}.csv  {out}.json  {out}.log")
    logf.close()


if __name__ == "__main__":
    main()
