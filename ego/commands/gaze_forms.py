"""
Test 9: compare the gaze forms on people not in the training data.

Every finished training run under --runs (a folder <form>_s<seed> with final.pt; best.pt is
scored) is evaluated on the same test set, plus ego_sd1p0 as "none", seed 0. ego_ft_v2 is
scored as a reference for the reproduction check and is not pooled with the new runs. The plan
is in docs/6-next-steps.md (Test 9).

Test set: the fixed clips (ego/clips.py, seed 12345) of every P08 and P09 recording with
signals, 24 per recording. The frozen encoder's features of these clips are computed once and
cached in <out>/cache/, so a new run costs only its predictor passes.

For each clip and model, the error of the predicted next step (0.27 s ahead):

  mse        over all 256 patches, with the model's signals
  mse_hide   the same with gaze and hand hidden (mask tokens); Delta = mse_hide - mse
  near       over the patches within 2 patches of the gaze point of the last context frame
             (clips where that frame has a gaze point)
  near_hide  the same with the signals hidden
  l1, l1_hide, near_l1, near_l1_hide
             the same four with the mean absolute difference (L1), the loss of V-JEPA 2-AC's
             training, in place of the mean squared difference

"The model's signals" are the real signals of the context frames; for the future control the
signals one step later; for "none" (trained with signal dropout 1.0) the hidden signals, the
only input it was trained on. A shuffled control is scored with the real signals.

Three references without fine-tuning are scored on the same clips (reference_predictor):
repeating the last context frame, a layer-normalized blend of the context frames, and V-JEPA
2-AC's predictor before fine-tuning. Every form and the other references are compared with the
blend.

Comparisons, fixed in the plan before the runs (gain of A over B = error of B - error of A,
positive when A predicts better): pe+rope vs angles, rope vs angles, pe vs angles, every model
vs none, rope vs pe, pe+rope vs pe+rope shuffled, future vs none. A form with several seeds is
averaged over its seeds clip by clip; the 95% interval comes from a bootstrap over the 25 test
recordings, and the spread between seeds is reported next to it.

Scores of each model are kept in <out>/scores/<model>.npz and recomputed only when the
checkpoint is newer or the file lacks the L1 errors, so the command can be rerun as runs finish.
The references are kept there too and computed once.

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
import torch.nn.functional as F

from ego import signals
from ego.clips import VAL_SEED, fixed_clips
from ego.data import find_recordings, load_frames
from ego.model import (
    build_predictor, encode_independent, load_ac_weights_into_ego, load_trained_predictor,
    maybe_norm, strip_prefix,
)
from ego.runlog import Logger
from ego.stats import by_recording

T, STRIDE, CLIPS = 8, 8, 24
HW = 256
RADIUS = 2.0
MEASURES = ("mse", "near", "l1", "near_l1")
COMPARISONS = [("pe+rope", "angles"), ("rope", "angles"), ("pe", "angles"),
               ("angles", "none"), ("pe", "none"), ("rope", "none"), ("pe+rope", "none"),
               ("rope", "pe"), ("pe+rope", "pe+rope shuffled"), ("future", "none")]
# Test 10 (docs/6-next-steps.md): the loss (L1 against MSE) and full fine-tuning
TEST10_COMPARISONS = [("pe l1", "pe"), ("none l1", "none"), ("pe l1", "none l1"),
                      ("none l1 full", "none l1"), ("pe l1 full", "pe l1"), ("pe l1 full", "none l1 full")]
REFERENCES = ("repeat last frame", "blend of past frames", "before fine-tuning")
BLEND = 0.2         # weight of the last context frame in the blend; the rest is the mean of all 8


# ---------------------------------------------------------------------------
# Test set and encoder cache
# ---------------------------------------------------------------------------

def test_clips(args):
    recs = find_recordings(args.video_dir, args.gaze_dir, args.participants, require_signal=True)
    return fixed_clips(recs, T, STRIDE, CLIPS, VAL_SEED, True)


def encoder_cache(args, clips, log):
    """{recording: (ctx (n, T*256, 1408), fut (n, 256, 1408))}, encoded once (fp32, as in training's check),
    in --cache (default <out>/cache)."""
    from ego.model import load_models
    cdir = Path(getattr(args, "cache", None) or Path(args.out) / "cache")
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
    """[(name, form, seed, checkpoint path, config)] of the finished runs in the --runs folders and
    the earlier models."""
    models = []
    folders = [args.runs] if isinstance(args.runs, str) else args.runs
    for d in sorted(d for f in folders for d in Path(f).glob("*_s[0-9]*")):
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
    """
    The model's form, which pools its seeds: the gaze form (or none, future, "<form> shuffled"),
    then " l1" if it was trained with L1, then " full" if the whole predictor was trained
    (" last<n>" and " +embed" for other choices). Models trained before --loss and
    --unfreeze-embed existed used MSE and the last 6 blocks, and get neither.
    """
    if cfg.get("signal_dropout", 0.0) >= 1.0:
        form = "none"
    elif cfg.get("future_signals"):
        form = "future"
    else:
        form = cfg.get("gaze_form", "angles")
        if cfg.get("shuffle_signals", "off") != "off":
            form += " shuffled"
    if cfg.get("loss", "mse") == "l1":
        form += " l1"
    n, embed = cfg.get("unfreeze_last_n", 6), cfg.get("unfreeze_embed", False)
    if n >= 24 and embed:
        form += " full"
    elif n != 6 or embed:
        form += f" last{n}" + (" +embed" if embed else "")
    return form


def near_mask(gaze_last):
    """(n, 256) bool: patches within RADIUS of the gaze point; all False without a point."""
    col, row = gaze_last[:, 0:1], gaze_last[:, 1:2]
    c = torch.arange(16, dtype=torch.float32) + 0.5
    pc, pr = c.repeat(16)[None], c.repeat_interleave(16)[None]          # row-major patch centres
    m = (pc - col) ** 2 + (pr - row) ** 2 <= RADIUS ** 2
    return m & (col >= 0)


@torch.no_grad()
def score(predict, form, feats, by_rec, device, patches=False):
    """
    Per-clip errors of one predictor on the whole test set. predict(ctx, sig) returns the
    layer-normalized prediction of the next step, (n, 256, D); it is called with the model's
    signals and with the signals hidden. Each error is measured as MSE (mse, near) and as L1
    (l1, near_l1), see the module docstring. With patches=True also the squared error of every
    patch with the model's signals ("patch", (n, 256)), for the figures.
    """
    out = {"recording": []}
    for stem, clips in by_rec.items():
        ctx, fut = feats[stem]["ctx"].to(device), feats[stem]["fut"].to(device)
        key = "sig_next" if form == "future" else "sig"
        sig = tuple(torch.from_numpy(np.stack([getattr(c, key)[k] for c in clips])).to(device) for k in range(5))
        hidden = signals.mask_both(sig)
        given = hidden if form in ("none", "before fine-tuning") else sig
        near = near_mask(torch.from_numpy(np.stack([c.sig[0][T - 1, 3:5] for c in clips])))
        has_point, nm = near.any(1), near.float()
        errs = {}
        for tag, s in (("", given), ("_hide", hidden)):
            d = predict(ctx, s).float() - fut.float()
            for whole, local, e in (("mse", "near", d.pow(2).mean(-1).cpu()),      # (n, 256)
                                    ("l1", "near_l1", d.abs().mean(-1).cpu())):
                errs[whole + tag] = e.mean(1)
                errs[local + tag] = torch.where(has_point, (e * nm).sum(1) / nm.sum(1).clamp_min(1),
                                                torch.full_like(e[:, 0], float("nan")))
                if patches and tag == "" and whole == "mse":
                    errs["patch"] = e
        out["recording"] += [stem] * len(clips)
        for k, v in errs.items():
            out.setdefault(k, []).append(v.numpy())
    return {"recording": np.array(out["recording"]), **{k: np.concatenate(v) for k, v in out.items() if k != "recording"}}


def score_model(path, cfg, form, feats, by_rec, device, patches=False):
    """Per-clip errors of one trained model (score); with patches=True also the error of every
    patch with the model's signals ("patch", (n, 256)), for the figures."""
    predictor, _ = load_trained_predictor(path, device, T)
    s = score(lambda ctx, sig: maybe_norm(predictor(ctx, *sig)[:, -HW:, :]), form, feats, by_rec, device, patches)
    del predictor
    torch.cuda.empty_cache()
    return s


def reference_predictor(name, checkpoint, device):
    """
    (predict, form) of one of the REFERENCES, which use no fine-tuning:

      repeat last frame     the embedding of the last context frame
      blend of past frames  BLEND x the last context frame + (1 - BLEND) x the mean of the 8
                            context frames, layer-normalized like the predictor's output. BLEND
                            was chosen on these test clips (docs/review.md), which
                            favours this reference a little.
      before fine-tuning    V-JEPA 2-AC's predictor with new gaze and hand layers drawn with
                            seed 0, the signals hidden: the model before fine-tuning of Test 2
    """
    if name == "repeat last frame":
        return (lambda ctx, sig: ctx[:, -HW:, :]), "baseline"
    if name == "blend of past frames":
        def blend(ctx, sig):
            frames = ctx.view(ctx.shape[0], T, HW, -1)
            return F.layer_norm(BLEND * frames[:, -1] + (1 - BLEND) * frames.mean(1), (ctx.size(-1),))
        return blend, "baseline"
    torch.manual_seed(0)
    predictor = build_predictor(T)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    load_ac_weights_into_ego(predictor, strip_prefix(ck["predictor"]))
    predictor.to(device).eval()
    return (lambda ctx, sig: maybe_norm(predictor(ctx, *sig)[:, -HW:, :])), "before fine-tuning"


def score_file(out, name):
    return Path(out) / "scores" / f"{re.sub(r'[^A-Za-z0-9_.+-]', '_', name)}.npz"


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
    folders = [args.runs] if isinstance(args.runs, str) else args.runs
    a = next((Path(f) / "angles_s0" / "metrics.jsonl" for f in folders
              if (Path(f) / "angles_s0" / "metrics.jsonl").exists()), None)
    b = Path(args.ft_v2).parent / "metrics.jsonl"
    if a is None or not b.exists():
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
    ap.add_argument("--runs", nargs="+", default=["checkpoints/test9"],
                    help="folders of training runs (<name>_s<seed>), pooled into forms (form_of)")
    ap.add_argument("--sd1p0", default="checkpoints/ego_sd1p0/best.pt")
    ap.add_argument("--ft-v2", default="checkpoints/ego_ft_v2/best.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/test9")
    ap.add_argument("--cache", default=None,
                    help="folder of the encoder features of the test clips (default <out>/cache)")
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
        f = score_file(out, name)
        if f.exists() and f.stat().st_mtime > Path(path).stat().st_mtime:
            cached = dict(np.load(f))
            if "l1" in cached:
                scores[name] = cached
                continue
        scores[name] = score_model(path, cfg, form, feats, by_rec, device)
        np.savez(f, **scores[name])
        log(f"[scored] {name}")

    refs = {}
    for name in REFERENCES:
        f = score_file(out, name)
        if f.exists():
            refs[name] = dict(np.load(f))
            continue
        predict, form = reference_predictor(name, args.checkpoint, device)
        refs[name] = score(predict, form, feats, by_rec, device)
        del predict
        torch.cuda.empty_cache()
        np.savez(f, **refs[name])
        log(f"[scored] {name}")

    def summary(s):
        return {"clips": len(s["mse"]),
                "mse": s["mse"].mean(), "mse_hide": s["mse_hide"].mean(),
                "delta": (s["mse_hide"] - s["mse"]).mean(),
                "near": np.nanmean(s["near"]), "near_hide": np.nanmean(s["near_hide"]),
                "l1": s["l1"].mean(), "l1_hide": s["l1_hide"].mean(),
                "delta_l1": (s["l1_hide"] - s["l1"]).mean(),
                "near_l1": np.nanmean(s["near_l1"]), "near_l1_hide": np.nanmean(s["near_l1_hide"])}

    table = pd.DataFrame([{"model": name, "form": form, "seed": seed, **summary(scores[name])}
                          for name, form, seed, _, _ in models])
    table.to_csv(out / "gaze_forms.csv", index=False)
    log("\nEvery model (error of the next step, lower is better):")
    log(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ref_table = pd.DataFrame([{"model": name, **summary(refs[name])} for name in REFERENCES])
    ref_table.to_csv(out / "gaze_forms_references.csv", index=False)
    log("\nReferences without fine-tuning:")
    log(ref_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    recs = scores[models[0][0]]["recording"]
    comp_rows = []
    for measure in MEASURES:
        forms = per_form(scores, models, measure)
        log(f"\nForms, {measure}, averaged over seeds:")
        for f, (v, seeds) in sorted(forms.items()):
            log(f"  {f:18} {np.nanmean(v):.4f}   seeds: {', '.join(f'{x:.4f}' for x in seeds)}")
        log(f"\nComparisons, {measure} (gain of A over B = error B - error A; > 0: A better):")
        for a, b in COMPARISONS + TEST10_COMPARISONS:
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
        base = "blend of past frames"
        log(f"\nAgainst the {base}, {measure} (gain of A = error of the blend - error A; > 0: A better):")
        against = [(f, v, len(seeds)) for f, (v, seeds) in sorted(forms.items())]
        against += [(n, refs[n][measure], 1) for n in REFERENCES if n != base]
        for a, v, n_seeds in against:
            d = refs[base][measure] - v
            ok = np.isfinite(d)
            r = by_recording(d[ok], recs[ok])
            comp_rows.append({"measure": measure, "a": a, "b": base, "seeds_a": n_seeds, "seeds_b": 1,
                              "gain": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                              "seed_spread": float("nan"), "clips": int(ok.sum())})
            log(f"  {a:>18} {r['mean']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
    pd.DataFrame(comp_rows).to_csv(out / "gaze_forms_comparisons.csv", index=False)
    rep = reproduction(args, log)
    (out / "gaze_forms.json").write_text(json.dumps({"models": [m[0] for m in models],
                                                     "references": ref_table.to_dict("records"),
                                                     "comparisons": comp_rows, "reproduction": rep},
                                                    indent=2, default=float))
    log(f"\n[out] {out}/gaze_forms.csv  gaze_forms_references.csv  gaze_forms_comparisons.csv  "
        f"gaze_forms.json  scores/")
    log.close()


if __name__ == "__main__":
    main()
