"""
Evaluate the egocentric predictor with feature-prediction MSE (no labels).

For each sampled clip BOTH conditions are scored on the SAME frames (paired):

    Condition A (null):  ego predictor, all signals masked   -> mse_A
    Condition B (gaze):  ego predictor, real gaze + hand      -> mse_B
    delta = mse_A - mse_B   (positive  => signals help)

Pairing per clip removes sampling noise from the comparison, so the per-clip
deltas can be tested directly (mean delta + a paired Wilcoxon/t-test).

Alignment, encoding and normalisation are shared with training via ego_common,
so Condition B is fed gaze the SAME way it was during fine-tuning (absolute VRS
timestamps + per-frame-independent encoding + normalize_reps).

Usage:
    # Baseline AC weights, both conditions (needs gaze for B):
    python scripts/eval_ego_mse.py \
        --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --video-dir  data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir   data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --participants P08 P09 --out results/mse_paired.csv

    # Fine-tuned predictor:
    python scripts/eval_ego_mse.py ... \
        --predictor-checkpoint checkpoints/ego_finetune/best.pt
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from ego_common import (
    load_models, encode_independent, maybe_norm,
    load_frames, video_info, read_vrs_times, find_csvs, find_ts_csv, strip_prefix,
)
from src.datasets.ego_loaders import GazeTokenLoader, HandTokenLoader


def null_signals(T, device):
    return (torch.zeros(1, T, 3, device=device),
            torch.zeros(1, T, dtype=torch.bool, device=device),
            torch.zeros(1, T, 12, device=device),
            torch.zeros(1, T, dtype=torch.bool, device=device),
            torch.zeros(1, T, dtype=torch.bool, device=device))


@torch.no_grad()
def run_predictor(predictor, enc_ctx, signals, HW, normalize_reps):
    pred = predictor(enc_ctx, *signals)
    pred = maybe_norm(pred[:, -HW:, :], normalize_reps)   # last step = the future
    return pred


@torch.no_grad()
def eval_clip(encoder, predictor, ctx_frames, fut_frame, device,
              real_signals=None, normalize_reps=True):
    """Returns (mse_A, mse_B|None) for one clip, both scored against the same target."""
    HW = predictor.grid_height * predictor.grid_width
    T = ctx_frames.shape[0]

    enc_ctx = encode_independent(encoder, ctx_frames.unsqueeze(0), device, normalize_reps)
    enc_fut = encode_independent(encoder, fut_frame.unsqueeze(0).unsqueeze(0), device, normalize_reps)

    pred_A = run_predictor(predictor, enc_ctx, null_signals(T, device), HW, normalize_reps)
    mse_A = F.mse_loss(pred_A.float(), enc_fut.float()).item()

    mse_B = None
    if real_signals is not None:
        pred_B = run_predictor(predictor, enc_ctx, real_signals, HW, normalize_reps)
        mse_B = F.mse_loss(pred_B.float(), enc_fut.float()).item()
    return mse_A, mse_B


def load_finetuned(predictor, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["predictor"] if "predictor" in ck else ck
    predictor.load_state_dict(strip_prefix(sd), strict=True)
    # Report which layers this checkpoint fine-tuned (from its saved config)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    n = cfg.get("unfreeze_last_n", 6)
    from src.models.ego_finetune import describe_finetuned_layers
    d = describe_finetuned_layers(predictor, n)
    blk = d["unfrozen_block_ids"]
    print(f"[layers] fine-tuned: {', '.join(d['new_projectors'])} + "
          f"blocks {blk[0]}-{blk[-1]} + {', '.join(d['output_head'])}  "
          f"(epoch={ck.get('epoch','?')}, train_loss={ck.get('loss','?')})")


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
        mp4s = sorted(Path(args.video_dir, participant).glob("*.mp4"))
        print(f"\n{participant}: {len(mp4s)} recordings")

        for mp4 in mp4s:
            stem = mp4.stem
            ts_csv = find_ts_csv(str(mp4))
            n_frames, _ = video_info(str(mp4))
            if ts_csv is None or n_frames < span + 1:
                print(f"  {stem}: skip (no ts csv or too short, {n_frames} frames)")
                continue
            vrs_ts = read_vrs_times(ts_csv)
            n_frames = min(n_frames, len(vrs_ts))

            gl = hl = None
            if args.gaze_dir:
                gaze_csv, hand_csv = find_csvs(args.gaze_dir, participant, stem)
                std = not args.no_standardize
                if gaze_csv:
                    gl = GazeTokenLoader(gaze_csv, 30.0, standardize=std)
                if hand_csv:
                    hl = HandTokenLoader(hand_csv, 30.0, standardize=std)
            has_B = gl is not None or hl is not None

            mA, mB = [], []
            for _ in range(args.clips_per_video):
                start = int(np.random.randint(0, n_frames - span))
                ctx_idx = [start + i * stride for i in range(T)]
                fut_idx = start + T * stride
                try:
                    frames = load_frames(str(mp4), ctx_idx + [fut_idx])
                    ctx_frames, fut_frame = frames[:T], frames[T]

                    real = None
                    if has_B:
                        real = _real_signals(gl, hl, [int(vrs_ts[j]) for j in ctx_idx], T, device)

                    a, b = eval_clip(encoder, predictor, ctx_frames, fut_frame, device,
                                     real_signals=real, normalize_reps=normalize_reps)
                    mA.append(a)
                    if b is not None:
                        mB.append(b)
                except Exception as e:
                    print(f"    clip error: {e}")

            if not mA:
                continue
            row = {"participant": participant, "recording": stem,
                   "mse_A": float(np.mean(mA)), "n_clips": len(mA),
                   "mse_B": float(np.mean(mB)) if mB else "",
                   "delta": float(np.mean(mA) - np.mean(mB)) if mB else ""}
            rows.append(row)
            msg = f"  {stem}: MSE_A={row['mse_A']:.4f}"
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


def _real_signals(gl, hl, ctx_vrs_ns, T, device):
    gaze = np.zeros((T, 3), np.float32); gv = np.zeros(T, bool)
    hand = np.zeros((T, 12), np.float32); hlft = np.zeros(T, bool); hrgt = np.zeros(T, bool)
    if gl is not None:
        for t, ns in enumerate(ctx_vrs_ns):
            gaze[t], gv[t] = gl.get_token_for_vrs_ns(ns)
    if hl is not None:
        for t, ns in enumerate(ctx_vrs_ns):
            hand[t], hlft[t], hrgt[t] = hl.get_token_for_vrs_ns(ns)
    return (torch.from_numpy(gaze).unsqueeze(0).to(device),
            torch.from_numpy(gv).unsqueeze(0).to(device),
            torch.from_numpy(hand).unsqueeze(0).to(device),
            torch.from_numpy(hlft).unsqueeze(0).to(device),
            torch.from_numpy(hrgt).unsqueeze(0).to(device))


if __name__ == "__main__":
    main()
