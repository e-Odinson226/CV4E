"""
Rung B1 — does the stock AC predictor, with no conditioning at all, already predict
ego video as well as the fine-tuned ego predictor?

WHY THIS RUNS BEFORE ANY RETRAIN (adoption plan step 16, no-signal-baselines.md)
--------------------------------------------------------------------------------
EXP-001 designed a five-rung ablation ladder and ran only the top rung. B1 is the
rung that asks what the fine-tune actually bought. If the stock AC predictor —
pretrained on robot video, never shown a kitchen, conditioning suppressed — already
matches the fine-tuned model, then the whole measured effect was architecture plus
domain adaptation and never behavioural signal, which would explain the EXP-001
near-null and the EXP-002 null at the same time. It costs one eval pass on
checkpoints that already exist; the retrain is a separate, later step, and running
B1 first is what stops that retrain from measuring noise.

THREE WAYS TO SAY "NO GAZE", AND THEY ARE NOT EQUIVALENT
--------------------------------------------------------
  zeros  the literal zero vector in the conditioning slot. NOT neutral for gaze:
         GAZE_MEAN = [0, -0.25, 1.0] is non-zero, so zero in standardised space
         decodes to pitch +0.83 sigma and depth -1.0 sigma — a specific, slightly
         unusual gaze asserted with full confidence on every frame.
  mean   the empirical mean of the standardised signal, measured here rather than
         assumed. The gap between this and zeros is exactly how wrong the
         never-recomputed GAZE_MEAN/GAZE_STD constants are.
  mask   gaze_mask / hand_mask, the learned "no signal available" parameters. On
         the fine-tuned model these were trained by --signal-dropout 0.4; on the
         stock model they are still random, which is itself worth seeing.

All three are reported. If they agree the baseline is solid; if they disagree, the
disagreement is a finding about how brittle the conditioning pathway is.

A NOTE ON WHAT eval_ego_mse.py's CONDITION A ACTUALLY IS
--------------------------------------------------------
null_signals() (eval_ego_mse.py:49) returns zero vectors AND valid=False, and the
predictor routes on the validity flag, so the existing Condition A is the MASK
variant, not the zeros variant that no-signal-baselines.md describes B1 as using.
Every arm here is labelled explicitly so that ambiguity cannot recur.

Every arm is scored on the SAME clips against the SAME target, so per-clip
differences are paired and a Wilcoxon signed-rank test is meaningful — the test
EXP-001 never ran.

Usage
-----
    $PY scripts/eval_rung_b1.py \
        --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --predictor-checkpoint checkpoints/ego_ft_v2/best.pt \
        --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --participants P08 --out results/rung_b1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from ego_common import (
    load_models, encode_independent, maybe_norm, load_frames,
    read_vrs_times, find_csvs, find_ts_csv,
)
from eval_ego_mse import load_finetuned
from probe_sensitivity import discover, enumerate_clips
from src.datasets.ego_loaders import GazeTokenLoader, HandTokenLoader


# ---------------------------------------------------------------------------
# Empirical signal statistics
# ---------------------------------------------------------------------------

def signal_stats(video_dir, gaze_dir, participants, per_participant, standardize, log):
    """
    Mean/std of the STANDARDISED gaze and hand vectors over the training
    participants — i.e. what the projector actually saw.

    Perfect standardisation constants would put this mean at exactly 0, so the
    distance from 0 is a direct readout of how wrong GAZE_MEAN/GAZE_STD are. Only
    valid samples count; invalid ones are zeros by construction and would drag the
    mean toward a value nothing ever produced.
    """
    g_sum = np.zeros(3); g_sq = np.zeros(3); g_n = 0
    h_sum = np.zeros(12); h_sq = np.zeros(12); h_n = 0
    for p in participants:
        seen = 0
        for mp4 in sorted(Path(video_dir, p).glob("*.mp4")):
            g_csv, h_csv = find_csvs(gaze_dir, p, mp4.stem)
            if not (g_csv or h_csv):
                continue
            if g_csv:
                gl = GazeTokenLoader(g_csv, 30.0, standardize=standardize)
                df = gl.df[gl.df["gaze_valid"]]
                v = df[["yaw", "pitch", "depth"]].values.astype(np.float64)
                v[:, 2] = np.clip(np.where(v[:, 2] == 0, 1.0, v[:, 2]), 0.05, 10.0)
                if standardize:
                    from src.datasets.ego_loaders import GAZE_MEAN, GAZE_STD
                    v = (v - GAZE_MEAN) / GAZE_STD
                g_sum += v.sum(0); g_sq += (v ** 2).sum(0); g_n += len(v)
            if h_csv:
                hl = HandTokenLoader(h_csv, 30.0, standardize=standardize)
                df = hl.df[hl.df["left_valid"] | hl.df["right_valid"]]
                cols = ["tx_lw", "ty_lw", "tz_lw", "tx_lp", "ty_lp", "tz_lp",
                        "tx_rw", "ty_rw", "tz_rw", "tx_rp", "ty_rp", "tz_rp"]
                v = df[cols].values.astype(np.float64)
                if standardize:
                    from src.datasets.ego_loaders import HAND_MEAN, HAND_STD
                    v = (v - HAND_MEAN) / HAND_STD
                h_sum += v.sum(0); h_sq += (v ** 2).sum(0); h_n += len(v)
            seen += 1
            if seen >= per_participant:
                break
        log(f"  [stats] {p}: cumulative gaze n={g_n:,}  hand n={h_n:,}")
    g_mean = g_sum / max(g_n, 1)
    h_mean = h_sum / max(h_n, 1)
    return {
        "gaze_mean_std_space": g_mean.tolist(),
        "gaze_std_std_space": np.sqrt(np.maximum(g_sq / max(g_n, 1) - g_mean ** 2, 0)).tolist(),
        "hand_mean_std_space": h_mean.tolist(),
        "hand_std_std_space": np.sqrt(np.maximum(h_sq / max(h_n, 1) - h_mean ** 2, 0)).tolist(),
        "n_gaze": g_n, "n_hand": h_n,
    }


# ---------------------------------------------------------------------------
# Conditioning variants
# ---------------------------------------------------------------------------

def make_variants(sig, g_mean_t, h_mean_t):
    gaze, gv, hand, hl, hr = sig
    ones_g = torch.ones_like(gv)
    return {
        "real":  sig,
        "mask":  (gaze, torch.zeros_like(gv), hand, torch.zeros_like(hl), torch.zeros_like(hr)),
        "zeros": (torch.zeros_like(gaze), ones_g, torch.zeros_like(hand),
                  torch.ones_like(hl), torch.ones_like(hr)),
        "mean":  (g_mean_t.expand_as(gaze).contiguous(), ones_g,
                  h_mean_t.expand_as(hand).contiguous(),
                  torch.ones_like(hl), torch.ones_like(hr)),
    }


@torch.no_grad()
def score(predictor, enc_ctx, enc_fut, sig, HW, normalize_reps):
    pred = maybe_norm(predictor(enc_ctx, *sig)[:, -HW:, :], normalize_reps)
    return float(F.mse_loss(pred.float(), enc_fut.float()))


def paired(df, arm_a, arm_b):
    """Per-clip a - b, with a Wilcoxon signed-rank test and a bootstrap CI on the mean."""
    from scipy.stats import wilcoxon
    a = df[df.arm == arm_a].sort_values("clip").mse.values
    b = df[df.arm == arm_b].sort_values("clip").mse.values
    d = a - b
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(5000)])
    try:
        stat, p = wilcoxon(d)
    except ValueError:
        stat, p = np.nan, np.nan
    return {"contrast": f"{arm_a} - {arm_b}", "n": len(d), "mean_delta": float(d.mean()),
            "median_delta": float(np.median(d)), "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "wilcoxon_p": float(p),
            "frac_positive": float((d > 0).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--predictor-checkpoint", required=True, help="the fine-tuned arm (H)")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--participants", nargs="+", default=["P08"])
    ap.add_argument("--recordings", type=int, default=4)
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--stats-participants", nargs="+",
                    default=["P01", "P02", "P03", "P04", "P05", "P06", "P07"])
    ap.add_argument("--stats-recordings", type=int, default=3)
    ap.add_argument("--context-steps", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-normalize-reps", action="store_true")
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--encode-chunk", type=int, default=16)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/rung_b1")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(f"{out}.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    device = torch.device(args.device)
    normalize_reps = not args.no_normalize_reps
    standardize = not args.no_standardize
    # gaze_proj / hand_proj / the mask tokens are randomly initialised, so any arm
    # that uses the UNTRAINED predictor depends on this draw. Seeding keeps those
    # arms reproducible; checkpoint-loaded arms are deterministic regardless.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    T, stride = args.context_steps, args.frame_stride
    t0 = time.time()

    log("[stats] empirical mean of the standardised signals over the training participants")
    stats = signal_stats(args.video_dir, args.gaze_dir, args.stats_participants,
                         args.stats_recordings, standardize, log)
    log(f"[stats] gaze mean in standardised space = "
        f"{np.round(stats['gaze_mean_std_space'], 4).tolist()}  (0 would mean GAZE_MEAN is right)")
    log(f"[stats] gaze std  in standardised space = "
        f"{np.round(stats['gaze_std_std_space'], 4).tolist()}  (1 would mean GAZE_STD is right)")
    log(f"[stats] hand mean = {np.round(stats['hand_mean_std_space'], 4).tolist()}")
    json.dump(stats, open(f"{out}_signal_stats.json", "w"), indent=2)

    g_mean_t = torch.tensor(stats["gaze_mean_std_space"], dtype=torch.float32,
                            device=device).view(1, 1, 3)
    h_mean_t = torch.tensor(stats["hand_mean_std_space"], dtype=torch.float32,
                            device=device).view(1, 1, 12)

    encoder, predictor, _ = load_models(args.checkpoint, device, T, tubelet=2,
                                        encoder_key="target_encoder")
    stock_sd = {k: v.detach().clone() for k, v in predictor.state_dict().items()}
    predictor.eval()
    HW = predictor.grid_height * predictor.grid_width
    log(f"[setup] models built in {time.time()-t0:.0f}s")

    recs = discover(args.video_dir, args.gaze_dir, args.participants, args.recordings)
    clips = enumerate_clips(recs, T, stride, args.clips, args.seed, standardize, device, log)
    log(f"[clips] {len(clips)} clips from {args.participants}")

    # One encode pass, then both models scored on the cached encodings: the encoder
    # is the whole cost, and re-encoding per model would also let the two arms drift.
    cache = []
    for i, clip in enumerate(clips):
        frames = load_frames(clip["rec"]["mp4"], clip["ctx_idx"] + [clip["fut_idx"]])
        enc_ctx = encode_independent(encoder, frames[:T].unsqueeze(0), device,
                                     normalize_reps, chunk=args.encode_chunk)
        enc_fut = encode_independent(encoder, frames[T].unsqueeze(0).unsqueeze(0), device,
                                     normalize_reps, chunk=args.encode_chunk)
        cache.append((enc_ctx.cpu(), enc_fut.cpu(), clip))
        if (i + 1) % 20 == 0:
            log(f"  [encode {i+1}/{len(clips)}] {time.time()-t0:.0f}s")
    del encoder
    torch.cuda.empty_cache()

    rows = []
    for model_name in ("stock_ac", "finetuned"):
        if model_name == "stock_ac":
            predictor.load_state_dict(stock_sd)
            log("\n[model] stock_ac — AC weights, projectors and mask tokens UNTRAINED")
        else:
            load_finetuned(predictor, args.predictor_checkpoint)
            log(f"\n[model] finetuned — {args.predictor_checkpoint}")
        predictor.eval()
        for i, (enc_ctx, enc_fut, clip) in enumerate(cache):
            ec, ef = enc_ctx.to(device), enc_fut.to(device)
            for vname, sig in make_variants(clip["sig"], g_mean_t, h_mean_t).items():
                rows.append({"clip": i, "participant": clip["rec"]["participant"],
                             "recording": clip["rec"]["stem"], "model": model_name,
                             "variant": vname, "arm": f"{model_name}:{vname}",
                             "mse": score(predictor, ec, ef, sig, HW, normalize_reps)})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}.csv", index=False)

    log("\n" + "=" * 78)
    log("RUNG B1 — mean feature-prediction MSE, every arm on the same clips")
    log("=" * 78)
    piv = df.pivot_table(index="model", columns="variant", values="mse", aggfunc="mean")
    log(piv[["real", "mask", "zeros", "mean"]].to_string(float_format=lambda v: f"{v:9.4f}"))

    contrasts = [
        ("stock_ac:mask", "finetuned:mask"),
        ("stock_ac:zeros", "finetuned:zeros"),
        ("stock_ac:real", "finetuned:real"),
        ("finetuned:mask", "finetuned:real"),
        ("finetuned:zeros", "finetuned:real"),
        ("finetuned:mean", "finetuned:real"),
        ("stock_ac:mask", "stock_ac:real"),
        ("finetuned:zeros", "finetuned:mask"),
    ]
    stat_rows = [paired(df, a, b) for a, b in contrasts]
    sdf = pd.DataFrame(stat_rows)
    sdf.to_csv(f"{out}_contrasts.csv", index=False)
    log("\nPaired contrasts (positive mean_delta = the SECOND arm predicts better)")
    log(sdf.to_string(index=False, float_format=lambda v: f"{v:10.5f}"))

    b1_best = piv.loc["stock_ac", ["mask", "zeros", "mean"]].min()
    h_best = piv.loc["finetuned", ["mask", "zeros", "mean"]].min()
    h_real = piv.loc["finetuned", "real"]
    spread = float(piv.loc["finetuned", ["mask", "zeros", "mean"]].max() -
                   piv.loc["finetuned", ["mask", "zeros", "mean"]].min())
    log("\n" + "-" * 78)
    log(f"[B1 vs H] best no-signal stock arm  = {b1_best:.4f}")
    log(f"[B1 vs H] best no-signal tuned arm  = {h_best:.4f}")
    log(f"[B1 vs H] tuned WITH real signals   = {h_real:.4f}")
    log(f"[baselines] spread across the three no-signal definitions on the tuned model: "
        f"{spread:.4f}")
    log(f"[baselines] compare with the gaze effect itself, {h_best - h_real:+.4f}")
    log("\nREADING:")
    if b1_best <= h_best:
        log("  The STOCK AC predictor with no conditioning matches or beats the fine-tuned one.")
        log("  The fine-tune bought nothing measurable, so the reported effect was architecture")
        log("  plus domain adaptation. This explains the EXP-001 near-null and the EXP-002 null")
        log("  at once, and the retrain would be measuring noise as specified.")
    else:
        log(f"  Fine-tuning improves on the stock predictor by {b1_best - h_best:.4f} MSE with no")
        log("  signals at all. That gain is domain adaptation, not behavioural conditioning —")
        log("  it is the part of the effect the ladder was built to separate out.")
    if spread > abs(h_best - h_real):
        log("  The choice of no-signal baseline moves the number MORE than gaze does. Any Delta")
        log("  quoted without naming its baseline is uninterpretable — which is what happened")
        log("  to the +0.0011.")
    log("-" * 78)

    with open(f"{out}.json", "w") as f:
        json.dump({"config": vars(args), "signal_stats": stats,
                   "means": piv.to_dict(), "contrasts": stat_rows}, f, indent=2, default=str)
    log(f"[out] {out}.csv  {out}_contrasts.csv  {out}.json  {out}.log")
    logf.close()


if __name__ == "__main__":
    main()
