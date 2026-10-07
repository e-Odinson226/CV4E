"""
Test 8: does the gaze point tell which object is picked up next?

The plan and the reasons for each choice are in docs/EgoVault/6-next-steps.md and, once run,
docs/EgoVault/4-results.md (Test 8).

For every pick of the 50 most frequent object classes in training (ego/annotations.py), the
frames 0.5, 1, 2 and 4 s before the pick are the moments. A linear probe (ridge regression,
one output per class) names the object from the frozen encoder's features of the moment.
Inputs:

  scene           the 256 patches of the moment frame, PCA-reduced to 64 channels each,
                  plus the patch average of each frame 0.5, 1, 1.5 and 2 s earlier
  angles          scene + yaw, pitch and depth at the moment
  gaze point      scene + the patch features around the gaze point (Gaussian, sigma 1 patch)
  head point      scene + the patch features around a fixed point, the average gaze point of
                  the training moments, where the head points (control)
  gaze history L  scene + gaze point + the gaze-point features of the frames in the last L s,
                  every 0.5 s, averaged (L = 1, 3, 6)
  head history L  scene + head point + the head-point features of the same frames (control)

Each block of features is z-scored and scaled to a total variance of 1. The weight of the
added block against the scene and the ridge penalty are chosen by 3-fold cross-validation
over training recordings, on top-1 accuracy. Every input goes through the same procedure,
so the controls get the same chance as the gaze inputs.

Train on P01-P07, test on P08 and P09. Scores: top-1 and top-5 accuracy with 95% intervals
from a bootstrap over test recordings, and paired differences between inputs.

Stages (--stage)
----------------
  encode  decode and encode each needed frame once; cache per recording in <out>/cache/
          (an existing cache file is kept, so the stage can resume)
  fit     the probes, from the cache
  all     both (default)

Usage
-----
    python -m ego next-object \
        --checkpoint  data/model_checkpoints/vjepa2-ac-vitg.pt \
        --video-dir   data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir    data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --annotations data/epic-kitchen/ek100-hd/HD-EPIC/annotations \
        --out results/next_object
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ego.annotations import FPS, noun_classes, object_class, object_movements
from ego.data import IMAGENET_MEAN, IMAGENET_STD, find_recordings, read_vrs_times
from ego.gaze_geometry import GazeProjector, to_patch
from ego.linprobe import ChannelPCA, GazeSeries, point_weights
from ego.runlog import Logger
from ego.stats import by_recording

STEP = 15                       # frames per 0.5 s
N_BACK = 20                     # frames cached per pick: 0.5 to 10 s before it
HORIZONS = (1, 2, 4, 8)         # moments, in steps of 0.5 s before the pick
PAST = (1, 2, 3, 4)             # scene frames 0.5-2 s before the moment
HISTORY = {1: 2, 3: 6, 6: 12}   # history length in s -> steps of 0.5 s
SEEK_GAP = 300                  # read through gaps up to this many frames instead of seeking
INPUTS = {"scene": [], "angles": ["angles"], "gaze point": ["gp"], "head point": ["hp"]}
INPUTS.update({f"gaze history {L} s": ["gp", f"gh{L}"] for L in HISTORY})
INPUTS.update({f"head history {L} s": ["hp", f"hh{L}"] for L in HISTORY})
DECISION = [("gaze point", "angles"), ("gaze point", "head point"),
            ("gaze history 3 s", "angles"), ("gaze history 3 s", "head history 3 s")]
CONTRASTS = DECISION + [("gaze history 1 s", "head history 1 s"),
                        ("gaze history 6 s", "head history 6 s"),
                        ("angles", "scene"), ("gaze point", "scene"), ("head point", "scene")]


# ---------------------------------------------------------------------------
# Which frames: picks, classes, gaze
# ---------------------------------------------------------------------------

def plan(args):
    """The picks of the top classes, one row each, with the pick frame f and the class."""
    moves = object_movements(args.annotations)
    classes = noun_classes(args.annotations)
    moves["cls"] = [object_class(n, classes) for n in moves.name]
    moves["f"] = (moves.start_s * FPS).round().astype(int)
    train = moves[moves.participant.isin(args.train_participants) & moves.cls.notna()]
    top = train.cls.value_counts().index[:args.classes].tolist()
    picks = moves[moves.cls.isin(top)
                  & moves.participant.isin(args.train_participants + args.test_participants)
                  & (moves.f >= STEP * N_BACK)]
    return picks, top


def recording_plan(rec, picks, args, rng):
    """Needed frames of one recording, with the gaze at each: yaw, pitch, depth, col, row."""
    mine = picks[picks.video == rec.stem]
    if args.max_picks and len(mine) > args.max_picks:
        mine = mine.iloc[np.sort(rng.choice(len(mine), args.max_picks, replace=False))]
    vrs = read_vrs_times(rec.ts_csv)
    frames = sorted({int(f) - STEP * k for f in mine.f for k in range(1, N_BACK + 1)})
    frames = [f for f in frames if f < len(vrs)]
    gs = GazeSeries(rec.gaze_csv, FPS)
    proj = GazeProjector(args.gaze_dir, rec.participant, rec.stem)
    gaze = np.full((len(frames), 5), np.nan, np.float32)
    for i, f in enumerate(frames):
        g = gs.at_ns(vrs[f])
        xy = proj.project(*g) if g is not None else None
        if xy is not None:
            gaze[i] = (*g, *to_patch(xy))
    return {"rec": rec, "picks": mine, "frames": np.array(frames, np.int64), "gaze": gaze}


def moment_frames(picks):
    return {int(f) - STEP * h for f in picks.f for h in HORIZONS}


# ---------------------------------------------------------------------------
# Decoding and encoding
# ---------------------------------------------------------------------------

def read_frames(mp4, frames, size):
    """{frame: RGB uint8 (size, size, 3)} for sorted frame indices, reading forward."""
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    out, pos = {}, None
    for f in frames:
        if pos is None or f < pos or f - pos > SEEK_GAP:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            pos = int(f)
        while pos < f:
            cap.grab()
            pos += 1
        ok, img = cap.read()
        pos += 1
        if not ok:
            break
        out[int(f)] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))
    cap.release()
    return out


def encode(encoder, imgs, device, chunk):
    """(N, 256, 1408) encoder features of RGB uint8 frames, as in Test 3."""
    from ego.model import encode_independent
    x = np.stack(imgs).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    x = torch.from_numpy((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(1)
    amp = torch.bfloat16 if device.type == "cuda" else None
    return encode_independent(encoder, x, device, normalize_reps=True, chunk=chunk, amp_dtype=amp)


def features(h, gaze, head_w):
    """Scene average, gaze-point and head-point features of a batch, on the GPU."""
    gw = np.stack([point_weights(c, r) if np.isfinite(c) else np.full(256, np.nan)
                   for c, r in gaze[:, 3:5]])
    gw = torch.as_tensor(gw, dtype=h.dtype, device=h.device)
    hw = torch.as_tensor(head_w, dtype=h.dtype, device=h.device)
    return (h.mean(1), torch.einsum("bt,btd->bd", gw, h), torch.einsum("t,btd->bd", hw, h))


def encode_stage(args, log):
    from ego.model import load_models
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache = Path(args.out) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    picks, top = plan(args)
    recs = [r for r in find_recordings(args.video_dir, args.gaze_dir,
                                       args.train_participants + args.test_participants)
            if r.gaze_csv and (picks.video == r.stem).any()]
    if args.limit_recordings:
        per = {}
        recs = [r for r in recs if per.setdefault(r.participant, []).append(r) or
                len(per[r.participant]) <= args.limit_recordings]
    log(f"[plan] {len(picks)} picks of {len(top)} classes; {len(recs)} recordings with gaze")
    plans = [recording_plan(r, picks, args, rng) for r in recs]
    log(f"[plan] {sum(len(p['frames']) for p in plans)} frames to encode, "
        f"{sum(len(p['picks']) for p in plans)} picks")

    meta_f = cache / "meta.json"
    if meta_f.exists():
        meta = json.loads(meta_f.read_text())
    else:
        pts = np.concatenate([p["gaze"][np.isin(p["frames"], list(moment_frames(p["picks"])))][:, 3:5]
                              for p in plans if p["rec"].participant in args.train_participants])
        pts = pts[np.isfinite(pts[:, 0])]
        meta = {"head_point": [float(pts[:, 0].mean()), float(pts[:, 1].mean())],
                "classes": top, "pca_dim": args.pca_dim}
    head_w = point_weights(*meta["head_point"])
    log(f"[plan] head point: column {meta['head_point'][0]:.2f}, row {meta['head_point'][1]:.2f}")

    encoder, predictor, _ = load_models(args.checkpoint, device, context_steps=1)
    del predictor
    torch.cuda.empty_cache()

    pca_f = cache / "pca.npz"
    pca = ChannelPCA()
    if pca_f.exists():
        z = np.load(pca_f)
        pca.mean, pca.comp, pca.explained = z["mean"], z["comp"], float(z["explained"])
    else:
        train_plans = [p for p in plans if p["rec"].participant in args.train_participants]
        per_rec = max(1, args.pca_fit_frames // len(train_plans))
        grids = []
        for p in train_plans:
            m = sorted(moment_frames(p["picks"]))
            sel = sorted(rng.choice(m, size=min(per_rec, len(m)), replace=False))
            imgs = read_frames(p["rec"].mp4, sel, args.img_size)
            grids.append(encode(encoder, [imgs[f] for f in sel if f in imgs], device,
                                args.encode_chunk).cpu().numpy())
        pca.fit(np.concatenate(grids), args.pca_dim)
        np.savez(pca_f, mean=pca.mean, comp=pca.comp, explained=pca.explained)
        meta["pca_explained"] = pca.explained
    meta_f.write_text(json.dumps(meta, indent=2))
    log(f"[pca] {args.pca_dim} of 1408 channels, {pca.explained:.1%} of the variance")
    pca_mean = torch.as_tensor(pca.mean, dtype=torch.float32, device=device)
    pca_comp = torch.as_tensor(pca.comp, dtype=torch.float32, device=device)

    todo = [p for p in plans if not (cache / f"{p['rec'].stem}.npz").exists()]
    log(f"[encode] {len(plans) - len(todo)} recordings cached, {len(todo)} to do")
    pool = ThreadPoolExecutor(max_workers=1)
    nxt = pool.submit(read_frames, todo[0]["rec"].mp4, todo[0]["frames"], args.img_size) if todo else None
    for i, p in enumerate(todo):
        imgs = nxt.result()
        if i + 1 < len(todo):
            nxt = pool.submit(read_frames, todo[i + 1]["rec"].mp4, todo[i + 1]["frames"], args.img_size)
        keep = np.array([f in imgs for f in p["frames"]])
        frames, gaze = p["frames"][keep], p["gaze"][keep]
        moments = moment_frames(p["picks"])
        scene, gazef, headf, grid, grid_frames = [], [], [], [], []
        for s in range(0, len(frames), args.batch_size):
            fb = frames[s:s + args.batch_size]
            with torch.no_grad():
                h = encode(encoder, [imgs[f] for f in fb], device, args.encode_chunk)
                sc, gp, hp = features(h, gaze[s:s + args.batch_size], head_w)
                is_m = np.array([f in moments for f in fb])
                if is_m.any():
                    g = (h[torch.as_tensor(is_m, device=device)] - pca_mean) @ pca_comp
                    grid.append(g.half().cpu().numpy())
                    grid_frames.append(fb[is_m])
            scene.append(sc.half().cpu().numpy())
            gazef.append(gp.half().cpu().numpy())
            headf.append(hp.half().cpu().numpy())
        if not grid:
            grid, grid_frames = [np.zeros((0, 256, args.pca_dim), np.float16)], [np.zeros(0, np.int64)]
        np.savez(cache / f"{p['rec'].stem}.npz", frames=frames, gaze=gaze,
                 scene=np.concatenate(scene), gazef=np.concatenate(gazef), headf=np.concatenate(headf),
                 grid=np.concatenate(grid), grid_frames=np.concatenate(grid_frames))
        log(f"[encode] {i + 1}/{len(todo)} {p['rec'].stem}: {len(frames)} frames "
            f"({(~keep).sum()} unreadable), {len(p['picks'])} picks")
    pool.shutdown()


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def samples(args, picks, top, horizon, cache):
    """Feature blocks, labels and recordings of every pick at one moment (steps before it)."""
    cls_idx = {c: i for i, c in enumerate(top)}
    blocks = {k: [] for k in ["grid", "past", "angles", "gp", "hp"] +
              [f"gh{L}" for L in HISTORY] + [f"hh{L}" for L in HISTORY]}
    y, recs, parts = [], [], []
    for stem, z in cache.items():
        row = {int(f): i for i, f in enumerate(z["frames"])}
        grow = {int(f): i for i, f in enumerate(z["grid_frames"])}
        for _, p in picks[picks.video == stem].iterrows():
            m = int(p.f) - STEP * horizon
            back = [m - STEP * j for j in range(1, max(HISTORY.values()) + 1)]
            if m not in grow or m not in row or any(b not in row for b in back):
                continue
            i = row[m]
            if not np.isfinite(z["gaze"][i, 3]):
                continue
            blocks["grid"].append(z["grid"][grow[m]].reshape(-1))
            blocks["past"].append(np.concatenate([z["scene"][row[m - STEP * j]] for j in PAST]))
            blocks["angles"].append(z["gaze"][i, :3])
            blocks["gp"].append(z["gazef"][i])
            blocks["hp"].append(z["headf"][i])
            for L, n in HISTORY.items():
                g = np.stack([z["gazef"][row[b]] for b in back[:n]]).astype(np.float32)
                ok = np.isfinite(g[:, 0])
                blocks[f"gh{L}"].append(g[ok].mean(0) if ok.any() else z["gazef"][i])
                blocks[f"hh{L}"].append(np.stack([z["headf"][row[b]] for b in back[:n]]).astype(np.float32).mean(0))
            y.append(cls_idx[p.cls])
            recs.append(stem)
            parts.append(p.participant)
    return ({k: np.stack(v).astype(np.float32) for k, v in blocks.items()},
            np.array(y), np.array(recs), np.array(parts))


# ---------------------------------------------------------------------------
# Kernel ridge on the GPU
# ---------------------------------------------------------------------------

def block_kernels(blocks, train, device):
    """Linear kernel of each block over all samples, after z-scoring with the training rows
    and scaling the block to a total variance of 1."""
    K = {}
    for k, X in blocks.items():
        X = torch.as_tensor(X, device=device)
        mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-6)
        X = (X - mu) / sd / np.sqrt(X.shape[1])
        K[k] = X @ X.T
        del X
    return K


def centred(K, a, b):
    """Kernel between rows b and a, for features centred on the mean of rows a."""
    Kaa, Kba = K[a][:, a], K[b][:, a]
    ra, m = Kaa.mean(1), Kaa.mean()
    return Kaa - ra[None] - ra[:, None] + m, Kba - ra[None] - Kba.mean(1, keepdim=True) + m


def solve(K, a, b, Y, alphas):
    """Scores of rows b for each alpha (relative to the mean diagonal), fitted on rows a."""
    Kaa, Kba = centred(K, a, b)
    s, V = torch.linalg.eigh(Kaa)
    ym = Y[a].mean(0)
    VtY = V.T @ (Y[a] - ym)
    scale = Kaa.diagonal().mean()
    return [Kba @ (V @ (VtY / (s + c * scale)[:, None])) + ym for c in alphas]


def fit_input(Kb, extra, weights, alphas, Y, train, test, folds):
    """Choose the block weight and alpha by CV on top-1 accuracy; fit on train; score test."""
    best = (-1.0, None, None)
    for w in (weights if extra else [0.0]):
        K = Kb["grid"] + Kb["past"]
        for k in extra:
            K = K + w / len(extra) * Kb[k]
        correct = np.zeros(len(alphas))
        for f in folds:
            a, b = train[~torch.isin(train, f)], f
            for ai, S in enumerate(solve(K, a, b, Y, alphas)):
                correct[ai] += (S.argmax(1) == Y[b].argmax(1)).sum().item()
        acc = correct / len(train)
        ai = int(np.argmax(acc))
        if acc[ai] > best[0]:
            best = (float(acc[ai]), w, alphas[ai])
    cv, w, alpha = best
    K = Kb["grid"] + Kb["past"]
    for k in extra:
        K = K + w / len(extra) * Kb[k]
    S = solve(K, train, test, Y, [alpha])[0]
    return S.cpu().numpy(), {"cv_top1": cv, "weight": w, "alpha": alpha}


def recording_folds(recs, train, n, seed):
    """Split the training rows into n folds of whole recordings, balanced by size."""
    rng = np.random.default_rng(seed)
    names, counts = np.unique(recs[train], return_counts=True)
    order = rng.permutation(len(names))
    load, fold_of = np.zeros(n), {}
    for i in sorted(order, key=lambda i: -counts[i]):
        k = int(np.argmin(load))
        fold_of[names[i]] = k
        load[k] += counts[i]
    return [train[np.array([fold_of[r] == k for r in recs[train]])] for k in range(n)]


def fit_stage(args, log):
    import pandas as pd
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out, cache_dir = Path(args.out), Path(args.out) / "cache"
    meta = json.loads((cache_dir / "meta.json").read_text())
    picks, top = plan(args)
    if top != meta["classes"]:
        raise RuntimeError("the class list differs from the one the cache was built with")
    cache = {f.stem: dict(np.load(f)) for f in sorted(cache_dir.glob("P*.npz"))}
    log(f"[fit] {len(cache)} cached recordings")
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    weights = [0.3, 1.0, 3.0]
    rows, contrast_rows, decision, preds = [], [], {}, {}
    for h in HORIZONS:
        sec = h * STEP / FPS
        blocks, y, recs, parts = samples(args, picks, top, h, cache)
        train = np.flatnonzero(np.isin(parts, args.train_participants))
        test = np.flatnonzero(np.isin(parts, args.test_participants))
        log(f"\n[fit] {sec:g} s before the pick: {len(train)} training and {len(test)} test samples, "
            f"{len(np.unique(recs[test]))} test recordings")
        Y = torch.zeros(len(y), len(top), device=device)
        Y[torch.arange(len(y)), torch.as_tensor(y)] = 1.0
        tr_t, te_t = torch.as_tensor(train, device=device), torch.as_tensor(test, device=device)
        folds = [torch.as_tensor(f, device=device) for f in recording_folds(recs, train, 3, args.seed)]
        Kb = block_kernels(blocks, tr_t, device)
        del blocks
        freq = np.bincount(y[train], minlength=len(top)).argsort()[::-1]
        correct = {"frequency guess": (y[test] == freq[0]).astype(float),
                   "frequency guess top5": np.isin(y[test], freq[:5]).astype(float)}
        res = {"frequency guess": {"top1": by_recording(correct["frequency guess"], recs[test]),
                                   "top5": by_recording(correct["frequency guess top5"], recs[test])}}
        for name, extra in INPUTS.items():
            S, info = fit_input(Kb, extra, weights, alphas, Y, tr_t, te_t, folds)
            rank = (-S).argsort(1)
            c1 = (rank[:, 0] == y[test]).astype(float)
            c5 = (rank[:, :5] == y[test][:, None]).any(1).astype(float)
            correct[name], correct[name + " top5"] = c1, c5
            res[name] = {"top1": by_recording(c1, recs[test]), "top5": by_recording(c5, recs[test]), **info}
            log(f"  {name:18} top-1 {c1.mean():6.1%}  top-5 {c5.mean():6.1%}   "
                f"(cv top-1 {info['cv_top1']:.1%}, weight {info['weight']}, alpha {info['alpha']})")
        for name, r in res.items():
            rows.append({"seconds_before": sec, "input": name, "n_test": len(test),
                         "top1": r["top1"]["mean"], "top1_lo": r["top1"]["ci_lo"], "top1_hi": r["top1"]["ci_hi"],
                         "top5": r["top5"]["mean"], "top5_lo": r["top5"]["ci_lo"], "top5_hi": r["top5"]["ci_hi"],
                         "cv_top1": r.get("cv_top1"), "weight": r.get("weight"), "alpha": r.get("alpha")})
        for a, b in CONTRASTS:
            for m, suffix in (("top1", ""), ("top5", " top5")):
                d = correct[a + suffix] - correct[b + suffix]
                c95 = by_recording(d, recs[test])
                c975 = by_recording(d, recs[test], level=0.975)
                contrast_rows.append({"seconds_before": sec, "a": a, "b": b, "measure": m, "diff": c95["mean"],
                                      "lo95": c95["ci_lo"], "hi95": c95["ci_hi"],
                                      "lo975": c975["ci_lo"], "hi975": c975["ci_hi"]})
                if m == "top1":
                    log(f"  {a} - {b}: {c95['mean']:+.1%} [{c95['ci_lo']:+.1%}, {c95['ci_hi']:+.1%}]"
                        f"  97.5%: [{c975['ci_lo']:+.1%}, {c975['ci_hi']:+.1%}]")
        if h in (1, 2):
            lo = {(r["a"], r["b"]): r["lo975"] for r in contrast_rows
                  if r["seconds_before"] == sec and r["measure"] == "top1"}
            decision[f"{sec:g} s"] = bool(all(lo[c] > 0 for c in DECISION[:2]) or
                                          all(lo[c] > 0 for c in DECISION[2:]))
        preds[f"{sec:g}s_recording"] = recs[test]
        preds[f"{sec:g}s_label"] = y[test]
        for name in INPUTS:
            preds[f"{sec:g}s_{name}"] = correct[name].astype(np.int8)
        del Kb
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out / "next_object.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(out / "next_object_contrasts.csv", index=False)
    np.savez(out / "next_object_predictions.npz", **preds)
    go = any(decision.values())
    summary = {"classes": top, "head_point": meta["head_point"], "pca_explained": meta.get("pca_explained"),
               "decision_by_horizon": decision, "test9_goes_ahead": go}
    (out / "next_object.json").write_text(json.dumps(summary, indent=2))
    log(f"\n[decision] {decision} -> Test 9 {'goes ahead' if go else 'is postponed'}")
    log(f"[out] {out}/next_object.csv  next_object_contrasts.csv  next_object.json  next_object_predictions.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="data/model_checkpoints/vjepa2-ac-vitg.pt")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--train-participants", nargs="+", default=[f"P0{i}" for i in range(1, 8)])
    ap.add_argument("--test-participants", nargs="+", default=["P08", "P09"])
    ap.add_argument("--classes", type=int, default=50)
    ap.add_argument("--stage", choices=["encode", "fit", "all"], default="all")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--pca-fit-frames", type=int, default=400)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--encode-chunk", type=int, default=32)
    ap.add_argument("--max-picks", type=int, default=0, help="picks per recording, 0 = all (for a quick check)")
    ap.add_argument("--limit-recordings", type=int, default=0, help="recordings per participant, 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/next_object")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    log = Logger(Path(args.out) / "next_object")
    if args.stage in ("encode", "all"):
        encode_stage(args, log)
    if args.stage in ("fit", "all"):
        fit_stage(args, log)
    log.close()


if __name__ == "__main__":
    main()
