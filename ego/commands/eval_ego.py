"""
Test 11: score the predictors trained by `python -m ego train-ego` on people not in the
training data.

    python -m ego eval-ego --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --video-dir <videos> --gaze-dir <gaze> --runs checkpoints/test11 --out results/test11

Test set. The held-out sampler of train-ego (fixed_val_clips, seed 12345) on every recording of
P08 and P09 with signals, 24 clips each: about 600 clips from the 25 recordings that Tests 9 and
10 test on. The clips are not the clips of Tests 9 and 10, because a clip here spans the 2.1 s of
context and the farthest horizon.

Measures. At each horizon, the error of the prediction against the frozen encoder's tokens of
the target frame, as L1 and as MSE: over the whole frame (256 patches), and near the gaze point
(the patches within 2 patches of the gaze point of the last observed frame; a clip without that
point is left out of this measure), as in Tests 9 and 10.

Inputs. Every model is scored with two inputs:

    present   the points of the observed frames: what maps, none and shuffled train on
    hidden    no points: the maps add nothing

Models. Every <arm>_s<seed> folder under --runs whose arm train-ego has, at best.pt (named <arm>_s<seed>) and at every
epoch<N>.pt (named <arm>_s<seed>@<N>). Two references without training: repeating the last
observed frame, and the layer-normalized blend of the observed frames of Tests 9 and 10
(0.2 x the last frame + 0.8 x the mean of all 8).

Comparisons. The gain of A over B is the error of B minus the error of A, positive when A
predicts better, clip by clip, with a 95% interval from a bootstrap over the test recordings
(ego.stats.by_recording). Seeds of one arm are averaged clip by clip; the spread between seeds
is the standard deviation of their means, empty with one seed.

Writes <out>/scores.npz (every per-clip error), <out>/eval_ego.csv (the means of every model and
input) and <out>/eval_ego_comparisons.csv.
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ego.commands.train_ego import ARMS, fixed_val_clips
from ego.data import load_frames
from ego.ego_predictor import ego_predictor, signal_strength
from ego.model import encode_independent, load_models, maybe_norm
from ego.stats import by_recording

log = logging.getLogger("eval_ego")

INPUTS = ("present", "hidden")
OWN_INPUT = {"maps": "present", "none": "present", "shuffled": "present"}
MEASURES = ("l1", "mse", "near_l1", "near_mse")
RADIUS = 2.0
BLEND = 0.2
REFERENCES = ("repeat last frame", "blend of past frames")


def load_ego(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    p = ego_predictor(embed_dim=1408, pred_dim=cfg["pred_dim"], depth=cfg["depth"],
                      num_heads=cfg["num_heads"], grid=cfg["img_size"] // 16,
                      sigma=cfg["sigma"], use_hand=not cfg["no_hand"],
                      learn_sigma=not cfg["fixed_sigma"])
    p.load_state_dict(ck["predictor"])
    return p.to(device).eval(), cfg


def find_models(runs):
    """[(name, arm, path)]: best.pt and every epoch<N>.pt of each <arm>_s<seed> folder."""
    out = []
    for d in sorted(Path(r) for r in runs):
        for run in sorted(d.glob("*_s[0-9]*")):
            arm = run.name.rsplit("_s", 1)[0]
            if arm not in ARMS:
                continue
            if (run / "best.pt").exists():
                out.append((run.name, arm, run / "best.pt"))
            for ep in sorted(run.glob("epoch*.pt")):
                out.append((f"{run.name}@{ep.stem[5:]}", f"{arm}@{ep.stem[5:]}", ep))
    return out


def near_mask(gaze_last, valid, grid):
    """(B, grid*grid) bool: the patches within RADIUS of the gaze point, all False without one."""
    c = torch.arange(grid, dtype=torch.float32, device=gaze_last.device) + 0.5
    pc, pr = c.repeat(grid)[None], c.repeat_interleave(grid)[None]       # row-major centres
    col, row = gaze_last[:, 0:1], gaze_last[:, 1:2]
    return ((pc - col) ** 2 + (pr - row) ** 2 <= RADIUS ** 2) & valid[:, None]


def errors(pred, target, near, n_tgt, HW, patches=False):
    """
    {measure: (B, n_tgt)} for one prediction against the target tokens. With patches=True also
    "patch": the MSE of every patch, (B, n_tgt, HW), for the figures.
    """
    d = (pred.float() - target.float()).view(pred.size(0), n_tgt, HW, -1)
    out = {}
    nm = near.float()[:, None, :]                                     # (B, 1, HW)
    has = near.any(1)[:, None]
    for name, e in (("l1", d.abs().mean(-1)), ("mse", d.pow(2).mean(-1))):
        out[name] = e.mean(-1)
        local = (e * nm).sum(-1) / nm.sum(-1).clamp_min(1)
        out["near_" + name] = torch.where(has, local, torch.full_like(local, float("nan")))
        if patches and name == "mse":
            out["patch"] = e
    return {k: v.cpu().numpy() for k, v in out.items()}


def batch_points(clips, device):
    return tuple(torch.from_numpy(np.stack([c[3][k] for c in clips])).to(device) for k in range(4))


def hidden_points(pts):
    g_pt, g_val, h_pt, h_val = pts
    return (torch.full_like(g_pt, -1.0), torch.zeros_like(g_val),
            torch.full_like(h_pt, -1.0), torch.zeros_like(h_val))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="the V-JEPA 2-AC checkpoint; only its encoder is used")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--runs", nargs="+", default=["checkpoints/test11"])
    ap.add_argument("--out", default="results/test11")
    ap.add_argument("--participants", nargs="+", default=["P08", "P09"])
    ap.add_argument("--recordings", type=int, default=1000, help="at most this many recordings")
    ap.add_argument("--clips", type=int, default=24, help="clips per recording")
    ap.add_argument("--batch", type=int, default=1,
                    help="clips per encoder pass. 1 reproduces the check after each epoch of "
                         "train-ego exactly; 8 shifts every error by about 3e-5 (bf16 numerics), "
                         "the same for every model")
    ap.add_argument("--patches", action="store_true",
                    help="also keep the MSE of every patch, for the figures, in <out>/cache/patches.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "eval_ego.log")],
                        force=True)
    device = torch.device(args.device)
    amp = torch.bfloat16

    models = find_models(args.runs)
    preds = {}
    cfg0 = None
    for name, arm, path in models:
        preds[name], cfg = load_ego(path, device)
        cfg0 = cfg0 or cfg
        log.info(f"[model] {name} ({arm}) from {path}  "
                 + "  ".join(f"|{k}|={v:.4f}" for k, v in signal_strength(preds[name]).items())
                 + f"  sigma={preds[name].sigma.item():.3f}")
    if cfg0 is None:
        raise SystemExit(f"no train-ego runs (<arm>_s<seed>/best.pt) under {args.runs}")
    T, stride, slots = cfg0["context_steps"], cfg0["frame_stride"], tuple(cfg0["target_slots"])
    grid, img = cfg0["img_size"] // 16, cfg0["img_size"]
    HW, n_tgt = grid * grid, len(slots)
    horizons = [s * stride / 30.0 for s in slots]
    log.info(f"[horizon] slots {slots} = {', '.join(f'{h:.2f} s' for h in horizons)}")

    now = fixed_val_clips(args.video_dir, args.gaze_dir, args.participants, T, stride, slots,
                          args.recordings, args.clips, img, grid)

    encoder, _, _ = load_models(args.checkpoint, device, context_steps=T)
    scores, patch, points, recording = {}, {}, {}, []
    norms = {name: [] for name, _, _ in models}
    for i in range(0, len(now), args.batch):
        b_now = now[i:i + args.batch]
        obs, tgt = [], []
        for rec, obs_idx, tgt_idx, _ in b_now:
            all_idx = sorted(set(obs_idx + tgt_idx))
            frames = load_frames(rec.mp4, all_idx, size=img)
            pos = {j: k for k, j in enumerate(all_idx)}
            obs.append(frames[[pos[j] for j in obs_idx]])
            tgt.append(frames[[pos[j] for j in tgt_idx]])
        enc_obs = encode_independent(encoder, torch.stack(obs), device, True, chunk=48, amp_dtype=amp)
        enc_tgt = encode_independent(encoder, torch.stack(tgt), device, True, chunk=48, amp_dtype=amp)
        p_now = batch_points(b_now, device)
        inputs = {"present": p_now, "hidden": hidden_points(p_now)}
        near = near_mask(p_now[0][:, T - 1], p_now[1][:, T - 1], grid)
        recording += [c[0].stem for c in b_now]
        for k, part in enumerate(("gaze", "gaze_valid", "hands", "hands_valid")):
            points.setdefault(f"points|present|{part}", []).append(p_now[k][:, T - 1].cpu().numpy())

        def add(key, e):
            for m, v in e.items():
                (patch if m == "patch" else scores).setdefault(key + (m,), []).append(v)

        frames_obs = enc_obs.view(enc_obs.size(0), T, HW, -1)
        last = frames_obs[:, -1]
        blend = F.layer_norm(BLEND * last + (1 - BLEND) * frames_obs.mean(1), (last.size(-1),))
        add(("repeat last frame", "-"), errors(last.repeat(1, n_tgt, 1), enc_tgt, near, n_tgt, HW, args.patches))
        add(("blend of past frames", "-"), errors(blend.repeat(1, n_tgt, 1), enc_tgt, near, n_tgt, HW, args.patches))

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp):
            for name, _, _ in models:
                p = preds[name]
                norms[name].append(p.predictor_embed(enc_obs).float().norm(dim=-1).mean().item())
                for inp in INPUTS:
                    pred = maybe_norm(p(enc_obs, *inputs[inp], target_slots=slots))
                    add((name, inp), errors(pred, enc_tgt, near, n_tgt, HW, args.patches))
        log.info(f"[clips] {min(i + args.batch, len(now))}/{len(now)}")

    scores = {k: np.concatenate(v) for k, v in scores.items()}
    recording = np.array(recording)
    points = {k: np.concatenate(v) for k, v in points.items()}
    np.savez(out / "scores.npz", recording=recording, horizons=np.array(horizons), **points,
             **{"|".join(k): v for k, v in scores.items()})
    if args.patches:
        (out / "cache").mkdir(exist_ok=True)
        np.savez(out / "cache" / "patches.npz", recording=recording, horizons=np.array(horizons),
                 **points, **{"|".join(k[:2]): np.concatenate(v).astype(np.float32) for k, v in patch.items()})
    for name, _, _ in models:
        log.info(f"[token norm] {name}: mean norm of the embedded image tokens {np.mean(norms[name]):.3f}")

    # Means of every model and input.
    rows = []
    keys = sorted({k[:2] for k in scores}, key=lambda k: (k[0] not in REFERENCES, k))
    for key in keys:
        for h, hz in enumerate(horizons):
            rows.append({"model": key[0], "input": key[1], "horizon_s": round(hz, 3),
                         "clips": int(np.isfinite(scores[key + ("l1",)][:, h]).sum()),
                         **{m: float(np.nanmean(scores[key + (m,)][:, h])) for m in MEASURES}})
    with open(out / "eval_ego.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Comparisons: arms pooled over their seeds, each with its own input unless named.
    arms = {}
    for name, arm, _ in models:
        arms.setdefault(arm, []).append(name)

    def side(spec, m, h):
        """
        'arm' (with its own input), 'arm:input', or a reference -> the per-clip errors at
        horizon h, averaged over the arm's seeds, and the mean of each seed.
        """
        if spec in REFERENCES:
            return scores[(spec, "-", m)][:, h], []
        arm, _, inp = spec.partition(":")
        inp = inp or OWN_INPUT[arm.split("@")[0]]
        arrs = [scores[(n, inp, m)][:, h] for n in arms[arm]]
        return np.mean(arrs, axis=0), [float(np.nanmean(a)) for a in arrs]

    wanted = [
        ("maps", "none", "the matched comparison: the value of the maps"),
        ("maps", "maps:hidden", "within maps: real points against hidden"),
        ("none", "maps:hidden", "maps with its points hidden against the matched model"),
        ("maps@3", "none@3", "the matched comparison after 3 epochs"),
        ("shuffled", "none", "the extra input without the information"),
        ("maps", "shuffled", "information against extra input"),
    ] + [(a, "blend of past frames", "against the blend") for a in ("maps", "none", "repeat last frame")]

    comp = []
    for a, b, what in wanted:
        if any(s.split(":")[0] not in arms and s not in REFERENCES for s in (a, b)):
            continue
        for m in MEASURES:
            for h, hz in enumerate(horizons):
                va, sa = side(a, m, h)
                vb, sb = side(b, m, h)
                gain = vb - va
                ok = np.isfinite(gain)
                r = by_recording(gain[ok], recording[ok])
                spread = max([float(np.std(s)) for s in (sa, sb) if len(s) > 1], default=float("nan"))
                comp.append({"A": a, "B": b, "what": what, "measure": m, "horizon_s": round(hz, 3),
                             "gain": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                             "seed_spread": spread, "clips": r["n"], "recordings": r["recordings"]})
    with open(out / "eval_ego_comparisons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0]))
        w.writeheader()
        w.writerows(comp)
    for r in comp:
        if r["measure"] in ("l1", "near_l1"):
            log.info(f"[gain] {r['A']:>22} - {r['B']:<22} {r['measure']:>8} {r['horizon_s']:.2f}s "
                     f"{r['gain']:+.6f} [{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]")
    log.info(f"[done] {len(recording)} clips from {len(set(recording))} recordings -> {out}")


if __name__ == "__main__":
    main()
