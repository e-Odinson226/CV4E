"""
Test 9: compare the gaze forms on people not in the training data.

Every finished training run under --runs (a folder <form>_s<seed> with final.pt; best.pt is
scored) is evaluated on the same test set, plus ego_sd1p0 as "none", seed 0. ego_ft_v2 is
scored as a reference for the reproduction check and is not pooled with the new runs. The plan
is in docs/EgoVault/6-next-steps.md (Test 9).

Test set: the fixed clips (ego/clips.py, seed 12345) of every P08 and P09 recording with
signals, 24 per recording. The frozen encoder's features of these clips are computed once and
cached in <out>/cache/, so a new run costs only its predictor passes.

For each clip and model, the error of the predicted next step (0.27 s ahead):

  mse        over all 256 patches, with the model's signals
  mse_hide   the same with gaze and hand hidden (mask tokens); Delta = mse_hide - mse
  near       over the patches within 2 patches of the gaze point of the last context frame
             (clips where that frame has a gaze point)
  near_hide  the same with the signals hidden

"The model's signals" are the real signals of the context frames; for the future control the
signals one step later; for "none" (trained with signal dropout 1.0) the hidden signals, the
only input it was trained on. A shuffled control is scored with the real signals.

Comparisons, fixed in the plan before the runs (gain of A over B = error of B - error of A,
positive when A predicts better): pe+rope vs angles, rope vs angles, pe vs angles, every model
vs none, rope vs pe, pe+rope vs pe+rope shuffled, future vs none. A form with several seeds is
averaged over its seeds clip by clip; the 95% interval comes from a bootstrap over the 25 test
recordings, and the spread between seeds is reported next to it.

Scores of each model are kept in <out>/scores/<model>.npz and recomputed only when the
checkpoint is newer, so the command can be rerun as runs finish.

Usage
-----
    python -m ego gaze-forms \
        --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --runs checkpoints/test9 --out results/test9
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from ego import signals
from ego.clips import VAL_SEED, fixed_clips
from ego.data import find_recordings, load_frames
from ego.model import encode_independent, load_trained_predictor, maybe_norm
from ego.runlog import Logger
from ego.stats import by_recording

T, STRIDE, CLIPS = 8, 8, 24
RADIUS = 2.0
COMPARISONS = [("pe+rope", "angles"), ("rope", "angles"), ("pe", "angles"),
               ("angles", "none"), ("pe", "none"), ("rope", "none"), ("pe+rope", "none"),
               ("rope", "pe"), ("pe+rope", "pe+rope shuffled"), ("future", "none")]


# ---------------------------------------------------------------------------
# Test set and encoder cache
# ---------------------------------------------------------------------------

def test_clips(args):
    recs = find_recordings(args.video_dir, args.gaze_dir, args.participants, require_signal=True)
    return fixed_clips(recs, T, STRIDE, CLIPS, VAL_SEED, True)


def encoder_cache(args, clips, log):
    """{recording: (ctx (n, T*256, 1408), fut (n, 256, 1408))}, encoded once (fp32, as in training's check)."""
    from ego.model import load_models
    cdir = Path(args.out) / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    by_rec = {}
    for c in clips:
        by_rec.setdefault(c.rec.stem, []).append(c)
    todo = [s for s in by_rec if not (cdir / f"{s}.pt").exists()]
    if todo:
        device = torch.device(args.device)
        encoder, predictor, _ = load_models(args.checkpoint, device, T)
        del predictor
        torch.cuda.empty_cache()
        for s in todo:
            ctx, fut = [], []
            for c in by_rec[s]:
                frames = load_frames(c.rec.mp4, c.ctx_idx + [c.fut_idx])
                with torch.no_grad():
                    ctx.append(encode_independent(encoder, frames[:T].unsqueeze(0), device).cpu())
                    fut.append(encode_independent(encoder, frames[T:].unsqueeze(0), device).cpu())
            torch.save({"ctx": torch.cat(ctx), "fut": torch.cat(fut)}, cdir / f"{s}.pt")
            log(f"[cache] {s}: {len(ctx)} clips")
        del encoder
        torch.cuda.empty_cache()
    return {s: torch.load(cdir / f"{s}.pt") for s in by_rec}, by_rec


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def find_models(args):
    """[(name, form, seed, checkpoint path, config)] of the finished runs and the earlier models."""
    models = []
    for d in sorted(Path(args.runs).glob("*_s[0-9]*")):
        if not (d / "final.pt").exists():
            continue
        cfg = torch.load(d / "best.pt", map_location="cpu", weights_only=False, mmap=True).get("config", {})
        models.append((d.name, form_of(cfg), int(re.search(r"_s(\d+)$", d.name).group(1)), d / "best.pt", cfg))
    for name, path, form in (("none_s0 (ego_sd1p0)", args.sd1p0, "none"), ("angles (ego_ft_v2)", args.ft_v2, "reference")):
        if path and Path(path).exists():
            cfg = torch.load(path, map_location="cpu", weights_only=False, mmap=True).get("config", {})
            models.append((name, form, 0, Path(path), cfg))
    return models


def form_of(cfg):
    if cfg.get("signal_dropout", 0.0) >= 1.0:
        return "none"
    if cfg.get("future_signals"):
        return "future"
    form = cfg.get("gaze_form", "angles")
    return form + " shuffled" if cfg.get("shuffle_signals", "off") != "off" else form


def near_mask(gaze_last):
    """(n, 256) bool: patches within RADIUS of the gaze point; all False without a point."""
    col, row = gaze_last[:, 0:1], gaze_last[:, 1:2]
    c = torch.arange(16, dtype=torch.float32) + 0.5
    pc, pr = c.repeat(16)[None], c.repeat_interleave(16)[None]          # row-major patch centres
    m = (pc - col) ** 2 + (pr - row) ** 2 <= RADIUS ** 2
    return m & (col >= 0)


@torch.no_grad()
def score_model(path, cfg, form, feats, by_rec, device):
    """Per-clip errors of one model on the whole test set."""
    predictor, _ = load_trained_predictor(path, device, T)
    HW = 256
    out = {k: [] for k in ("recording", "mse", "mse_hide", "near", "near_hide")}
    for stem, clips in by_rec.items():
        ctx, fut = feats[stem]["ctx"].to(device), feats[stem]["fut"].to(device)
        key = "sig_next" if form == "future" else "sig"
        sig = tuple(torch.from_numpy(np.stack([getattr(c, key)[k] for c in clips])).to(device) for k in range(5))
        hidden = signals.mask_both(sig)
        given = hidden if form == "none" else sig
        near = near_mask(torch.from_numpy(np.stack([c.sig[0][T - 1, 3:5] for c in clips])))
        has_point = near.any(1)
        errs = {}
        for tag, s in (("", given), ("_hide", hidden)):
            pred = maybe_norm(predictor(ctx, *s)[:, -HW:, :])
            e = ((pred.float() - fut.float()) ** 2).mean(-1).cpu()                 # (n, 256)
            errs["mse" + tag] = e.mean(1)
            nm = near.float()
            errs["near" + tag] = torch.where(has_point, (e * nm).sum(1) / nm.sum(1).clamp_min(1),
                                             torch.full_like(e[:, 0], float("nan")))
        out["recording"] += [stem] * len(clips)
        for k, v in errs.items():
            out[k].append(v.numpy())
    del predictor
    torch.cuda.empty_cache()
    return {"recording": np.array(out["recording"]), **{k: np.concatenate(out[k]) for k in out if k != "recording"}}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def per_form(scores, models, measure):
    """{form: (per-clip error averaged over seeds, [per-seed means])}, the reference model excluded."""
    forms = {}
    for name, form, seed, _, _ in models:
        if form == "reference" or name not in scores:
            continue
        forms.setdefault(form, []).append(scores[name][measure])
    return {f: (np.mean(v, axis=0), [float(np.nanmean(x)) for x in v]) for f, v in forms.items()}


def reproduction(args, log):
    """Training's per-epoch numbers of angles_s0 next to ego_ft_v2's."""
    def vals(path):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        return [(r["epoch"], r["mse_A"], r["mse_B"]) for r in rows if r.get("t") == "val"]
    a, b = Path(args.runs) / "angles_s0" / "metrics.jsonl", Path(args.ft_v2).parent / "metrics.jsonl"
    if not (a.exists() and b.exists()):
        return None
    va, vb = vals(a), vals(b)
    log("\nReproduction: training's check on the 96 P08 clips, angles_s0 against ego_ft_v2")
    log(f"{'epoch':>5} {'hidden s0':>10} {'hidden v2':>10} {'real s0':>9} {'real v2':>9}")
    for (e, ha, ra), (_, hb, rb) in zip(va, vb):
        log(f"{e:>5} {ha:10.4f} {hb:10.4f} {ra:9.4f} {rb:9.4f}")
    return {"angles_s0": va, "ego_ft_v2": vb}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="data/model_checkpoints/vjepa2-ac-vitg.pt")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--participants", nargs="+", default=["P08", "P09"])
    ap.add_argument("--runs", default="checkpoints/test9")
    ap.add_argument("--sd1p0", default="checkpoints/ego_sd1p0/best.pt")
    ap.add_argument("--ft-v2", default="checkpoints/ego_ft_v2/best.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/test9")
    args = ap.parse_args()

    import pandas as pd
    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    log = Logger(out / "gaze_forms")
    device = torch.device(args.device)

    clips = test_clips(args)
    feats, by_rec = encoder_cache(args, clips, log)
    log(f"[test set] {len(clips)} clips in {len(by_rec)} recordings of {args.participants}")
    models = find_models(args)
    log(f"[models] {', '.join(m[0] for m in models)}")

    scores = {}
    for name, form, seed, path, cfg in models:
        f = out / "scores" / f"{re.sub(r'[^A-Za-z0-9_.+-]', '_', name)}.npz"
        if f.exists() and f.stat().st_mtime > Path(path).stat().st_mtime:
            scores[name] = dict(np.load(f))
            continue
        scores[name] = score_model(path, cfg, form, feats, by_rec, device)
        np.savez(f, **scores[name])
        log(f"[scored] {name}")

    rows = []
    for name, form, seed, _, _ in models:
        s = scores[name]
        rows.append({"model": name, "form": form, "seed": seed, "clips": len(s["mse"]),
                     "mse": s["mse"].mean(), "mse_hide": s["mse_hide"].mean(),
                     "delta": (s["mse_hide"] - s["mse"]).mean(),
                     "near": np.nanmean(s["near"]), "near_hide": np.nanmean(s["near_hide"])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "gaze_forms.csv", index=False)
    log("\nEvery model (error of the next step, lower is better):")
    log(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    recs = scores[models[0][0]]["recording"]
    comp_rows = []
    for measure in ("mse", "near"):
        forms = per_form(scores, models, measure)
        log(f"\nForms, {measure}, averaged over seeds:")
        for f, (v, seeds) in sorted(forms.items()):
            log(f"  {f:18} {np.nanmean(v):.4f}   seeds: {', '.join(f'{x:.4f}' for x in seeds)}")
        log(f"\nComparisons, {measure} (gain of A over B = error B - error A; > 0: A better):")
        for a, b in COMPARISONS:
            if a not in forms or b not in forms:
                continue
            d = forms[b][0] - forms[a][0]
            ok = np.isfinite(d)
            r = by_recording(d[ok], recs[ok])
            spread = max(np.std(forms[a][1]), np.std(forms[b][1])) if len(forms[a][1]) > 1 or len(forms[b][1]) > 1 else float("nan")
            comp_rows.append({"measure": measure, "a": a, "b": b, "seeds_a": len(forms[a][1]),
                              "seeds_b": len(forms[b][1]), "gain": r["mean"], "ci_lo": r["ci_lo"],
                              "ci_hi": r["ci_hi"], "seed_spread": spread, "clips": int(ok.sum())})
            log(f"  {a:>18} vs {b:<18} {r['mean']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
                f"   seeds {len(forms[a][1])}/{len(forms[b][1])}, seed spread {spread:.4f}")
    pd.DataFrame(comp_rows).to_csv(out / "gaze_forms_comparisons.csv", index=False)
    rep = reproduction(args, log)
    (out / "gaze_forms.json").write_text(json.dumps({"models": [m[0] for m in models],
                                                     "comparisons": comp_rows, "reproduction": rep},
                                                    indent=2, default=float))
    log(f"\n[out] {out}/gaze_forms.csv  gaze_forms_comparisons.csv  gaze_forms.json  scores/")
    log.close()


if __name__ == "__main__":
    main()
