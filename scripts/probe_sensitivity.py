"""
T6: does the prediction change when gaze changes?

This tests hypothesis H2 in docs/EgoVault/1-introduction.md: the model does not use the
signals. The small Delta of T2 has two possible causes. The model may read gaze, and
gaze may say little about the next step. Or the model may have learned to ignore the
gaze token. This script separates the two without any training. It keeps the video
fixed, changes the gaze input, and measures how far the prediction moves:

    rel = || z_pred(changed) - z_pred(real) ||_F / || z_pred(real) ||_F

  rel close to 0      -> the model ignores gaze.
  rel clearly above 0 -> the model reads gaze.

Reference values
----------------
Three reference values are measured on the same clips:

  identity    The same forward pass twice. It must be exactly 0. Otherwise the network
              is not deterministic, and no other number can be read.
  video_swap  The visual context of another clip, with the same gaze. This shows how
              far apart two predictions are after a large input change.
  both_mask   Gaze and hand replaced by their mask tokens. This is the "signals hidden"
              condition of eval_ego_mse.py, so it links this result to Delta.

The gaze change is swept over several sizes (--degrees, in degrees of yaw). A channel
that does not react at any size differs from one that reacts a little at each size.

Clips
-----
The clips come from the same recordings, seed and sampler as the checks in
finetune_ego.validate(). With the defaults these are the 96 P08 test clips of T2. The
MSE columns then reproduce that run's MSE_A and MSE_B, which checks that the two
pipelines match.

Usage
-----
    PY=/mnt/data/home/zj2433/miniconda3/envs/VJEPA2-AC/bin/python
    $PY scripts/probe_sensitivity.py \
        --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --predictor-checkpoint checkpoints/ego_ft_v2/best.pt \
        --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --participants P08 --out results/sensitivity_ego_ft_v2

Everything runs in fp32 by default. validate() ran in fp32, the identity check must be
exact, and the changes being measured can be small. --amp runs the encoder in bf16.
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
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from ego_common import (
    load_models, encode_independent, maybe_norm,
    load_frames, read_vrs_times, find_csvs, find_ts_csv,
)
from eval_ego_mse import _real_signals, load_finetuned
from ego.ego_loaders import GazeTokenLoader, HandTokenLoader, GAZE_STD


# ---------------------------------------------------------------------------
# Perturbations
#
# A "signals" tuple is (gaze, gaze_valid, hand, hand_left_valid, hand_right_valid)
# exactly as the predictor's forward takes it. Every perturbation returns a NEW
# tuple; none mutate in place, so the baseline stays clean across conditions.
#
# Gaze arrives z-scored by GAZE_MEAN/GAZE_STD, so a shift of d degrees of yaw is
# radians(d) / GAZE_STD[0] in the units the projector actually sees.
# ---------------------------------------------------------------------------

def shift_gaze_deg(sig, deg):
    gaze, gv, hand, hl, hr = sig
    g = gaze.clone()
    g[..., 0] = g[..., 0] + float(np.radians(deg)) / float(GAZE_STD[0])
    return (g, gv, hand, hl, hr)


def swap_gaze(sig, other):
    return (other[0], other[1], sig[2], sig[3], sig[4])


def swap_hand(sig, other):
    return (sig[0], sig[1], other[2], other[3], other[4])


def mask_gaze(sig):
    gaze, gv, hand, hl, hr = sig
    return (gaze, torch.zeros_like(gv), hand, hl, hr)


def mask_hand(sig):
    gaze, gv, hand, hl, hr = sig
    return (gaze, gv, hand, torch.zeros_like(hl), torch.zeros_like(hr))


def zero_gaze(sig):
    gaze, gv, hand, hl, hr = sig
    return (torch.zeros_like(gaze), torch.ones_like(gv), hand, hl, hr)


def shuffle_gaze_time(sig, perm):
    gaze, gv, hand, hl, hr = sig
    return (gaze[:, perm, :], gv[:, perm], hand, hl, hr)


def shuffle_hand_time(sig, perm):
    gaze, gv, hand, hl, hr = sig
    return (gaze, gv, hand[:, perm, :], hl[:, perm], hr[:, perm])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rel_l2(z_ref, z):
    """|| z - z_ref ||_F / || z_ref ||_F — scale-free, so clips are comparable."""
    return float((z - z_ref).float().norm() / z_ref.float().norm())


@torch.no_grad()
def predict(predictor, enc_ctx, sig, HW, normalize_reps):
    """Full predictor output, and the last-step slice the loss is actually taken on."""
    out = predictor(enc_ctx, *sig)
    last = maybe_norm(out[:, -HW:, :], normalize_reps)
    full = maybe_norm(out, normalize_reps)
    return full, last


# ---------------------------------------------------------------------------
# Clip enumeration — mirrors finetune_ego.validate() exactly
# ---------------------------------------------------------------------------

def discover(video_dir, gaze_dir, participants, limit_per_participant):
    recs = []
    for p in participants:
        n = 0
        for mp4 in sorted(Path(video_dir, p).glob("*.mp4")):
            ts = find_ts_csv(str(mp4))
            if ts is None:
                continue
            g, h = find_csvs(gaze_dir, p, mp4.stem)
            if not (g or h):
                continue
            recs.append({"participant": p, "stem": mp4.stem, "mp4": str(mp4),
                         "ts_csv": ts, "gaze_csv": g, "hand_csv": h})
            n += 1
            if limit_per_participant and n >= limit_per_participant:
                break
    return recs


def enumerate_clips(recs, T, stride, n_clips, seed, standardize, device, log):
    """
    Frame indices + signals for every clip, with NO video decoding. Signals come
    from the CSVs, so the whole clip list is built before the GPU is touched —
    which is what lets a clip's swap partner be another clip's real signals.
    """
    rng = np.random.RandomState(seed)
    span = T * stride
    clips = []
    for rec in recs:
        vrs = read_vrs_times(rec["ts_csv"])
        n = len(vrs)
        if n < span + 1:
            continue
        gl = GazeTokenLoader(rec["gaze_csv"], 30.0, standardize=standardize) if rec["gaze_csv"] else None
        hl = HandTokenLoader(rec["hand_csv"], 30.0, standardize=standardize) if rec["hand_csv"] else None
        if gl is None and hl is None:
            continue
        for _ in range(n_clips):
            start = int(rng.randint(0, n - span))
            ctx_idx = [start + i * stride for i in range(T)]
            fut_idx = start + T * stride
            sig = _real_signals(gl, hl, [int(vrs[j]) for j in ctx_idx], T, device)
            clips.append({"rec": rec, "ctx_idx": ctx_idx, "fut_idx": fut_idx, "sig": sig})
        log(f"  {rec['participant']}/{rec['stem']}: {n_clips} clips")
    return clips


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="AC checkpoint (architecture + encoder)")
    ap.add_argument("--predictor-checkpoint", default=None,
                    help="Fine-tuned ego predictor; omit to probe the untrained-projector model")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--participants", nargs="+", default=["P08"])
    ap.add_argument("--recordings", type=int, default=4, help="max recordings per participant")
    ap.add_argument("--clips", type=int, default=24, help="clips per recording")
    ap.add_argument("--context-steps", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=8)
    ap.add_argument("--degrees", nargs="+", type=float, default=[1.0, 2.0, 5.0, 10.0, 20.0, 45.0])
    ap.add_argument("--seed", type=int, default=12345, help="12345 = finetune_ego.validate()'s seed")
    ap.add_argument("--no-normalize-reps", action="store_true")
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 encoder (faster, breaks bit-exactness)")
    ap.add_argument("--encode-chunk", type=int, default=16)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/sensitivity_probe")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(f"{out}.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    device = torch.device(args.device)
    normalize_reps = not args.no_normalize_reps
    amp_dtype = torch.bfloat16 if args.amp else None
    # gaze_proj / hand_proj / the mask tokens are randomly initialised, so any arm
    # that uses the UNTRAINED predictor depends on this draw. Seeding keeps those
    # arms reproducible; checkpoint-loaded arms are deterministic regardless.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    T, stride = args.context_steps, args.frame_stride

    log(f"[setup] device={device}  T={T}  stride={stride}  normalize_reps={normalize_reps}  amp={amp_dtype}")
    t0 = time.time()
    encoder, predictor, (n_t, n_s) = load_models(args.checkpoint, device, T, tubelet=2,
                                                 encoder_key="target_encoder")
    log(f"[ckpt] AC weights: transferred {n_t}, skipped {n_s}  ({time.time()-t0:.0f}s)")
    if args.predictor_checkpoint:
        load_finetuned(predictor, args.predictor_checkpoint)
        log(f"[ckpt] fine-tuned predictor <- {args.predictor_checkpoint}")
    else:
        log("[ckpt] NO fine-tuned predictor — probing AC weights with untrained projectors")
    predictor.eval()
    HW = predictor.grid_height * predictor.grid_width

    recs = discover(args.video_dir, args.gaze_dir, args.participants, args.recordings)
    log(f"[clips] {len(recs)} recordings from {args.participants}")
    clips = enumerate_clips(recs, T, stride, args.clips, args.seed,
                            not args.no_standardize, device, log)
    N = len(clips)
    log(f"[clips] {N} clips total")
    if N < 2:
        raise SystemExit("need at least 2 clips (swap conditions pair them)")

    perm = np.random.RandomState(args.seed).permutation(T)   # fixed time permutation
    log(f"[setup] time permutation for shuffle conditions: {perm.tolist()}")

    rows = []
    prev_enc = None
    for i, clip in enumerate(clips):
        rec = clip["rec"]
        try:
            frames = load_frames(rec["mp4"], clip["ctx_idx"] + [clip["fut_idx"]])
        except Exception as e:                                   # noqa: BLE001
            log(f"  [skip] clip {i}: {e}")
            continue
        ctx_frames, fut_frame = frames[:T], frames[T]

        enc_ctx = encode_independent(encoder, ctx_frames.unsqueeze(0), device,
                                     normalize_reps, chunk=args.encode_chunk, amp_dtype=amp_dtype)
        enc_fut = encode_independent(encoder, fut_frame.unsqueeze(0).unsqueeze(0), device,
                                     normalize_reps, chunk=args.encode_chunk, amp_dtype=amp_dtype)

        sig = clip["sig"]
        other = clips[(i + 1) % N]["sig"]                        # swap partner

        conditions = {
            "identity":          (enc_ctx, sig),
            "gaze_swap":         (enc_ctx, swap_gaze(sig, other)),
            "gaze_mask":         (enc_ctx, mask_gaze(sig)),
            "gaze_zero":         (enc_ctx, zero_gaze(sig)),
            "gaze_shuffle_time": (enc_ctx, shuffle_gaze_time(sig, perm)),
            "hand_swap":         (enc_ctx, swap_hand(sig, other)),
            "hand_mask":         (enc_ctx, mask_hand(sig)),
            "hand_shuffle_time": (enc_ctx, shuffle_hand_time(sig, perm)),
            "both_mask":         (enc_ctx, mask_hand(mask_gaze(sig))),
        }
        for d in args.degrees:
            conditions[f"gaze_yaw_{d:g}deg"] = (enc_ctx, shift_gaze_deg(sig, d))
        if prev_enc is not None:
            conditions["video_swap"] = (prev_enc, sig)

        full_ref, last_ref = predict(predictor, enc_ctx, sig, HW, normalize_reps)
        mse_ref = float(F.mse_loss(last_ref.float(), enc_fut.float()))
        gaze_valid_frac = float(sig[1].float().mean())
        hand_valid_frac = float((sig[3] | sig[4]).float().mean())

        base = {"clip": i, "participant": rec["participant"], "recording": rec["stem"],
                "start_frame": clip["ctx_idx"][0],
                "gaze_valid_frac": gaze_valid_frac, "hand_valid_frac": hand_valid_frac,
                "mse_real": mse_ref}
        rows.append(dict(base, condition="real", rel_last=0.0, rel_full=0.0,
                         mse=mse_ref, d_mse=0.0))

        for name, (enc_c, sig_c) in conditions.items():
            full_c, last_c = predict(predictor, enc_c, sig_c, HW, normalize_reps)
            mse_c = float(F.mse_loss(last_c.float(), enc_fut.float()))
            rows.append(dict(base, condition=name,
                             rel_last=rel_l2(last_ref, last_c),
                             rel_full=rel_l2(full_ref, full_c),
                             mse=mse_c, d_mse=mse_c - mse_ref))

        prev_enc = enc_ctx
        if (i + 1) % 10 == 0:
            log(f"  [{i+1}/{N}] {(time.time()-t0):.0f}s elapsed")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}.csv", index=False)

    # ---- summary -----------------------------------------------------------
    agg = df.groupby("condition").agg(
        n=("rel_last", "size"),
        rel_last_mean=("rel_last", "mean"), rel_last_med=("rel_last", "median"),
        rel_last_max=("rel_last", "max"),
        rel_full_mean=("rel_full", "mean"),
        mse_mean=("mse", "mean"), d_mse_mean=("d_mse", "mean"),
    )
    vs = float(agg.loc["video_swap", "rel_last_mean"]) if "video_swap" in agg.index else np.nan
    agg["frac_of_video_swap"] = agg["rel_last_mean"] / vs

    order = ["real", "identity"] + [f"gaze_yaw_{d:g}deg" for d in args.degrees] + \
            ["gaze_shuffle_time", "gaze_swap", "gaze_zero", "gaze_mask",
             "hand_shuffle_time", "hand_swap", "hand_mask", "both_mask", "video_swap"]
    agg = agg.reindex([c for c in order if c in agg.index])

    log("\n" + "=" * 96)
    log("T6 SENSITIVITY — relative movement of z_pred when the input is perturbed, video held fixed")
    log("=" * 96)
    log(agg.to_string(float_format=lambda v: f"{v:11.6f}"))

    # ---- the three gates ---------------------------------------------------
    ident = float(agg.loc["identity", "rel_last_max"]) if "identity" in agg.index else np.nan
    log("\n" + "-" * 96)
    log(f"[gate 1] identity control, max over clips = {ident:.3e}  "
        f"({'OK — network is deterministic' if ident == 0.0 else 'NON-ZERO: nothing below is readable'})")

    g_mask = float(agg.loc["gaze_mask", "rel_last_mean"])
    g_swap = float(agg.loc["gaze_swap", "rel_last_mean"])
    log(f"[gate 2] gaze_swap moves z_pred by {g_swap:.3e} of its norm "
        f"({100*g_swap/vs:.3f}% of a whole-video swap)")
    log(f"[gate 3] gaze_mask (Condition A's gaze half) moves it by {g_mask:.3e}")

    log("\nREADING:")
    if ident != 0.0:
        log("  Identity control is non-zero. Fix determinism before reading anything else.")
    elif g_swap < 1e-4:
        log("  The model IGNORES gaze. Another clip's gaze changes the prediction by less")
        log("  than 0.01% of its norm. This supports H2 (the model does not use the signals),")
        log("  and changes to how gaze enters the model would have a real target.")
    elif g_swap / vs < 0.01:
        log("  The model READS gaze, with a small effect: a completely different gaze moves")
        log("  the prediction less than 1% as far as a different video does. H2 is not")
        log("  supported. Weigh H1 and H3.")
    else:
        log("  The model clearly READS gaze. H2 is rejected: the prediction depends on the")
        log("  gaze input, so Coord-PE, zero initialization and the gate would address a")
        log("  problem that does not exist. Look at H1 (the information does not help at")
        log("  this horizon) or H3 (the 3-epoch model is undertrained).")

    # end-to-end check against the training log
    if "both_mask" in agg.index:
        mse_A = float(agg.loc["both_mask", "mse_mean"])
        mse_B = float(agg.loc["real", "mse_mean"])
        log("\n[end-to-end] Condition A (both_mask) vs B (real), same clips as validate():")
        log(f"             MSE_A={mse_A:.4f}  MSE_B={mse_B:.4f}  Delta(A-B)={mse_A-mse_B:+.4f}")
        log("             Compare with the run's own validation log to confirm the harness matches.")
    log("-" * 96)

    with open(f"{out}.json", "w") as f:
        json.dump({"config": vars(args), "n_clips": N,
                   "summary": agg.reset_index().to_dict(orient="records")},
                  f, indent=2, default=str)
    log(f"[out] {out}.csv  {out}.json  {out}.log")
    logf.close()


if __name__ == "__main__":
    main()
