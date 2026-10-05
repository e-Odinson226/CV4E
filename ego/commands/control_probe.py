"""
T5: is the T4 result specific to gaze?

In T4, gaze could be read across recordings of the same people, but not across
participants. In HD-EPIC each participant has their own kitchen, so the participant
split changes the person and the kitchen together. The T4 result could mean that gaze
does not transfer across people. It could also mean that the frozen features do not
transfer across kitchens.

This script runs the same probe on a different target, with the same features and the
same splits:

  * If the control target transfers across participants, the features do transfer,
    and the T4 result is about gaze.
  * If the control target also fails across participants, T4 measured how well the
    features transfer across kitchens.

Target
------
Palm position (x, y, z) in the Aria device frame, from the same MPS recordings. Like
gaze, it is behavioural. It is visible in the frame, so features that transfer should
locate it. It is sampled on the same clock, so it needs no new alignment.

Finding the frames
------------------
The T4 cache (results/gaze_features.npz) stores the features and the gaze, but not the
frame each row came from. The T4 sampler is seeded. The script replays it with the
cache's own settings (--source-json), which gives the frame of each row without
encoding the video again. It then checks that the gaze it finds equals the cached gaze
in every row, and that the number of rows per recording matches. If a check fails, the
script stops, because a misaligned row would give a wrong result without any warning.

What stays the same as T4
-------------------------
The cached features, the PCA, the ridge probe, the regularization grid, the folds, the
seed, the splits, the leads and the skill measure. Only the target changes. Rows
without valid hand data are dropped. Gaze is scored again on the remaining rows, so
gaze and palm are compared on the same rows.

Usage
-----
    python -m ego control-probe \
        --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --out results/control_recoverability

It needs no GPU and decodes no video. It reads mp4 headers, CSV files and the feature
cache.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ego.data import find_csvs
from ego.linprobe import HandSeries, Ridge, probe_samples, r2, ridge_cv
from ego.runlog import Logger


# ---------------------------------------------------------------------------
# T4's rows, rebuilt without encoding anything
# ---------------------------------------------------------------------------

def replay(cfg, participants, split_name, log):
    """
    The rows of the T4 cache, rebuilt with T4's sampler (ego.linprobe.probe_samples) and
    the cache's own settings: per row (participant, stem, frame_index, vrs_ns, gaze3), in
    the order gaze-probe produced them. No video is decoded.
    """
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for p, vp, vrs, keep, g0s, _ in probe_samples(
            cfg["video_dir"], cfg["gaze_dir"], participants, cfg["recordings"], cfg["windows"],
            cfg["window_sec"], cfg["per_window"], cfg["leads"], cfg["tol_ms"], rng, log):
        for i, g0 in zip(keep, g0s):
            rows.append({"participant": p, "stem": vp.stem, "frame": int(i),
                         "vrs_ns": int(vrs[i]), "gaze": g0})
        if keep:
            log(f"  {split_name} {p}/{vp.stem}: {len(keep)} rows")
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
# Probe: the same as in T4 (gaze-probe). Only the target differs.
# ---------------------------------------------------------------------------

def per_col_mse(pred, true):
    """
    Per-column mean squared error, the raw material for both skill definitions.

    Pooled weights columns by their variance. T4 reported pooled skill for yaw and
    pitch. Per-column skill does not depend on scale, so it is the better measure when
    a target mixes units or ranges. Both are printed, so a disagreement between them
    is visible.
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
                    help="the T4 run whose settings built the cache")
    ap.add_argument("--hand-tol-ms", type=float, default=100.0,
                    help="hand CSVs run at ~10 Hz, so 50 ms would reject most lookups")
    ap.add_argument("--splits", nargs="+", default=["participant", "recording", "random"])
    ap.add_argument("--readout", default="spatial", choices=["spatial", "pooled", "both"])
    ap.add_argument("--out", default="results/control_recoverability")
    args = ap.parse_args()

    out = Path(args.out)
    log = Logger(out)

    src = json.load(open(args.source_json))["config"]
    cfg = {k: src[k] for k in ("video_dir", "gaze_dir", "recordings", "windows", "window_sec",
                               "per_window", "leads", "seed", "tol_ms")}
    cfg["video_dir"], cfg["gaze_dir"] = args.video_dir, args.gaze_dir
    cfg["hand_tol_ms"] = args.hand_tol_ms
    leads = cfg["leads"]
    log(f"[setup] replaying the T4 sampler: seed={cfg['seed']} recordings={cfg['recordings']} "
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
    # as gaze-probe does, so "participant" is the original arrays.
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
        # T4 number) and on each hand's rows, so the comparison that decides
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
    log("READING — skill at lead 0, the number the T4 result rests on")
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
        log("  The CONTROL TARGET TRANSFERS across participants, and gaze does not.")
        log("  The frozen features carry information across people and kitchens, so the T4")
        log("  result is not a general failure of the encoder. The T4 result HOLDS.")
    elif hand_rec > 0.05:
        log("  The control target can be read WITHIN kitchens but FAILS across them, like")
        log("  gaze. T4 measured how well the encoder transfers across kitchens, not whether")
        log("  gaze is redundant. This would explain the null results without reference to")
        log("  gaze.")
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
    log.close()


if __name__ == "__main__":
    main()
