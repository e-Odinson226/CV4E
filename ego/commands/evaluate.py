"""
Evaluate the ego predictor with feature-prediction MSE (no labels).

For each sampled clip BOTH conditions are scored on the SAME frames (paired):

    Condition A (null):  ego predictor, all signals masked   -> mse_A
    Condition B (gaze):  ego predictor, real gaze + hand      -> mse_B
    delta = mse_A - mse_B   (positive  => signals help)

Pairing per clip removes sampling noise from the comparison, so the per-clip deltas can be
tested directly. Alignment, encoding and normalisation are shared with training, so
Condition B is fed gaze the same way it was during fine-tuning.

This command samples its own clips, from any participants. The fixed 96-clip set of the
tests is in ego/clips.py.

Usage:
    # Before fine-tuning, both conditions (needs gaze for B):
    python -m ego evaluate \
        --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --video-dir  data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir   data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --participants P08 P09 --out results/mse_paired.csv

    # Fine-tuned predictor:
    python -m ego evaluate ... \
        --predictor-checkpoint checkpoints/ego_ft_v2/best.pt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ego import signals
from ego.clips import paired_mse
from ego.data import find_recordings, load_frames, open_loaders, read_vrs_times, video_info
from ego.model import load_finetuned, load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",            required=True, help="AC checkpoint (architecture + weights)")
    ap.add_argument("--predictor-checkpoint",  default=None, help="Fine-tuned ego predictor (best.pt)")
    ap.add_argument("--video-dir",             required=True)
    ap.add_argument("--gaze-dir",              default=None, help="Required for Condition B")
    ap.add_argument("--participants",          nargs="+", default=["P08", "P09"])
    ap.add_argument("--context-steps",         type=int, default=8)
    ap.add_argument("--frame-stride",          type=int, default=8)
    ap.add_argument("--clips-per-video",       type=int, default=20)
    ap.add_argument("--no-normalize-reps",     action="store_true")
    ap.add_argument("--no-standardize",        action="store_true")
    ap.add_argument("--out",                   default="results/mse_paired.csv")
    ap.add_argument("--seed",                  type=int, default=0)
    ap.add_argument("--device",                default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed)
    device = torch.device(args.device)
    normalize_reps = not args.no_normalize_reps
    T, stride = args.context_steps, args.frame_stride

    print(f"Device={device}  context_steps={T}  frame_stride={stride}  normalize_reps={normalize_reps}")
    encoder, predictor, (n_t, n_s) = load_models(
        args.checkpoint, device, T, tubelet=2, encoder_key="target_encoder"
    )
    print(f"[ckpt] transferred {n_t}, skipped {n_s}")
    if args.predictor_checkpoint:
        load_finetuned(predictor, args.predictor_checkpoint)
        print(f"[ckpt] loaded fine-tuned predictor from {args.predictor_checkpoint}")
    predictor.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    span = T * stride

    for participant in args.participants:
        recs = find_recordings(args.video_dir, args.gaze_dir, [participant], require_ts=False)
        print(f"\n{participant}: {len(recs)} recordings")

        for rec in recs:
            n_frames, _ = video_info(rec.mp4)
            if rec.ts_csv is None or n_frames < span + 1:
                print(f"  {rec.stem}: skip (no ts csv or too short, {n_frames} frames)")
                continue
            vrs_ts = read_vrs_times(rec.ts_csv)
            n_frames = min(n_frames, len(vrs_ts))

            gl, hl = open_loaders(rec, standardize=not args.no_standardize)
            has_B = gl is not None or hl is not None

            mA, mB = [], []
            for _ in range(args.clips_per_video):
                start = int(np.random.randint(0, n_frames - span))
                ctx_idx = [start + i * stride for i in range(T)]
                fut_idx = start + T * stride
                try:
                    frames = load_frames(rec.mp4, ctx_idx + [fut_idx])
                    real = None
                    if has_B:
                        real = signals.as_batch(
                            signals.read(gl, hl, [int(vrs_ts[j]) for j in ctx_idx], T), device)
                    a, b = paired_mse(encoder, predictor, frames, T, device,
                                      real_sig=real, normalize_reps=normalize_reps)
                    mA.append(a)
                    if b is not None:
                        mB.append(b)
                except Exception as e:
                    print(f"    clip error: {e}")

            if not mA:
                continue
            row = {"participant": participant, "recording": rec.stem,
                   "mse_A": float(np.mean(mA)), "n_clips": len(mA),
                   "mse_B": float(np.mean(mB)) if mB else "",
                   "delta": float(np.mean(mA) - np.mean(mB)) if mB else ""}
            rows.append(row)
            msg = f"  {rec.stem}: MSE_A={row['mse_A']:.4f}"
            if mB:
                msg += f"  MSE_B={row['mse_B']:.4f}  delta={row['delta']:+.4f}"
            print(msg + f"  ({len(mA)} clips)")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["participant", "recording", "mse_A", "mse_B", "delta", "n_clips"])
        w.writeheader()
        w.writerows(rows)

    mean_A = np.mean([r["mse_A"] for r in rows]) if rows else float("nan")
    deltas = [r["delta"] for r in rows if r["delta"] != ""]
    print(f"\nOverall MSE_A: {mean_A:.4f}")
    if deltas:
        mean_B = np.mean([r["mse_B"] for r in rows if r["mse_B"] != ""])
        print(f"Overall MSE_B: {mean_B:.4f}")
        print(f"Mean delta (A-B): {np.mean(deltas):+.4f}  over {len(deltas)} recordings  "
              f"({'signals HELP' if np.mean(deltas) > 0 else 'signals do NOT help'})")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
