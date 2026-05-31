"""
Shared building blocks for ego-predictor fine-tuning and evaluation.

Both finetune_ego.py and eval_ego_mse.py import from here so the two pipelines
cannot drift apart (an earlier bug: training aligned gaze by VRS timestamp while
eval aligned by frame/fps, silently sabotaging Condition B).

Design decisions, matched to the original V-JEPA 2-AC droid training
(configs/train/vitg16/droid-256px-8f.yaml + app/vjepa_droid/train.py):

  * Encoder weights:    target_encoder (EMA), used for BOTH context and target.
  * Per-frame encoding: each frame is encoded INDEPENDENTLY as a duplicated
                        2-frame tubelet (forward_target), NOT as one joint clip.
                        The pretrained predictor consumes per-frame latents.
  * Frame striding:     the world model steps at ~4 fps (8 frames spaced apart),
                        so context frames are sampled at a stride, not adjacent.
  * normalize_reps:     reps are LayerNorm'd before the loss (config has it True).
  * Token layout:       [gaze, hand, visual...] per step, cond_tokens=2.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from src.models.vision_transformer import vit_giant_xformers
from src.models.ego_predictor import vit_ego_predictor
from src.models.ego_finetune import load_ac_weights_into_ego

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ---------------------------------------------------------------------------
# Checkpoint / model loading
# ---------------------------------------------------------------------------

def strip_prefix(sd: dict) -> dict:
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def load_models(checkpoint, device, context_steps, tubelet=2, encoder_key="target_encoder"):
    """
    Build the ViT-g encoder + ego predictor from a V-JEPA 2-AC checkpoint.

    encoder_key="target_encoder" matches the droid config, which uses the EMA
    target encoder for both the context and the prediction target. Falls back to
    "encoder" if the requested key is absent.

    The predictor's num_frames is context_steps * tubelet so its causal attention
    mask has exactly context_steps temporal slots.
    """
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    enc_sd = ck.get(encoder_key) or ck.get("encoder")

    encoder = vit_giant_xformers(
        patch_size=16, img_size=(256, 256),
        num_frames=tubelet, tubelet_size=tubelet,       # we feed 2-frame tubelets
        use_sdpa=True, use_SiLU=False, wide_SiLU=True,
        uniform_power=False, use_rope=True,
    )
    encoder.load_state_dict(strip_prefix(enc_sd), strict=False)
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    predictor = vit_ego_predictor(
        img_size=(256, 256), patch_size=16,
        num_frames=context_steps * tubelet, tubelet_size=tubelet,
        embed_dim=encoder.embed_dim, predictor_embed_dim=1024,
        depth=24, num_heads=16,
        use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True,
    )
    transferred, skipped = load_ac_weights_into_ego(predictor, strip_prefix(ck["predictor"]))
    predictor.to(device)
    return encoder, predictor, (len(transferred), len(skipped))


# ---------------------------------------------------------------------------
# Per-frame-independent encoding  (mirrors app/vjepa_droid/train.py forward_target)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_independent(encoder, frames, device, normalize_reps=True, chunk=16, amp_dtype=None):
    """
    frames: (B, T, 3, H, W) — T independent steps.
    Each frame is duplicated into a 2-frame tubelet and encoded ALONE, so the
    output matches the per-frame latents the pretrained predictor was trained on.

    The B*T frames are pushed through the ViT-g encoder in sub-batches of `chunk`
    so peak memory is bounded (B*T can be 64+ otherwise, which OOMs a busy GPU).

    amp_dtype (e.g. torch.bfloat16) runs the encoder under autocast — the encoder
    is the dominant cost, so this is the main throughput lever. Output is cast back
    to fp32 so the downstream LayerNorm / predictor / loss stay numerically stable.

    Returns (B, T*HW, D), optionally LayerNorm'd (normalize_reps).
    """
    B, T = frames.shape[:2]
    c = frames.to(device, non_blocking=True).flatten(0, 1)          # (B*T, 3, H, W)
    c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)                        # (B*T, 3, 2, H, W)
    N = c.shape[0]
    use_amp = amp_dtype is not None and device.type == "cuda"
    outs = []
    for i in range(0, N, chunk):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outs.append(encoder(c[i:i + chunk]).float())           # sub-batched, back to fp32
    h = torch.cat(outs, dim=0)                                     # (B*T, HW, D)
    D = h.size(-1)
    h = h.view(B, T, -1, D).flatten(1, 2)                          # (B, T*HW, D)
    if normalize_reps:
        h = F.layer_norm(h, (D,))
    return h


def maybe_norm(x, normalize_reps=True):
    """LayerNorm predictor output before the loss, matching training."""
    return F.layer_norm(x, (x.size(-1),)) if normalize_reps else x


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------

def load_frames(mp4_path, indices, size=256):
    """Returns (N, 3, H, W) float32, ImageNet-normalised, in the order of `indices`."""
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    lo, hi = min(indices), max(indices)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)          # seek so we don't decode from 0
    target, buf, i = set(indices), {}, lo
    while cap.isOpened() and len(buf) < len(target) and i <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if i in target:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
            buf[i] = frame
        i += 1
    cap.release()
    arr = np.stack([buf.get(j, np.zeros((size, size, 3), np.uint8)) for j in indices])
    arr = arr.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr)


def video_info(mp4_path):
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps


# ---------------------------------------------------------------------------
# VRS timestamp alignment
# ---------------------------------------------------------------------------

def read_vrs_times(ts_csv):
    """Return the per-frame absolute VRS device timestamps (ns) as an int64 array."""
    import pandas as pd
    df = pd.read_csv(ts_csv)
    return df["vrs_device_time_ns"].values.astype(np.int64)


# Aria MPS standard output filenames (inside mps_<rec>_vrs.zip).
# Older drafts referred to eye_gaze.csv / hand_tracking_results.csv — kept as aliases.
_GAZE_NAMES = ("general_eye_gaze.csv", "eye_gaze.csv")
_HAND_NAMES = ("wrist_and_palm_poses.csv", "hand_tracking_results.csv")


def find_csvs(gaze_dir, participant, stem):
    """Locate the gaze / hand MPS CSVs for a recording (extracted, size > 0 only)."""
    base = Path(gaze_dir) / participant / "GAZE_HAND"
    gaze_csv = hand_csv = None
    if not base.exists():
        return gaze_csv, hand_csv
    for root, _, files in os.walk(base):
        if stem not in str(root):
            continue
        for f in files:
            fp = Path(root) / f
            if f in _GAZE_NAMES and fp.stat().st_size > 0:
                gaze_csv = str(fp)
            elif f in _HAND_NAMES and fp.stat().st_size > 0:
                hand_csv = str(fp)
    return gaze_csv, hand_csv


def find_ts_csv(video_path):
    """The frame->VRS timestamp CSV that sits next to a recording's mp4."""
    p = Path(video_path)
    ts = p.parent / f"{p.stem}_mp4_to_vrs_time_ns.csv"
    return str(ts) if ts.exists() else None
