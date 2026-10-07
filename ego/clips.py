"""
The fixed evaluation clips, and how one clip is scored.

Training's check after each epoch, the sensitivity and attention tests (Tests 4a and 4b)
and the comparisons of Test 2 all use the clips from fixed_clips(): the same recordings, the
same seed (12345) and the same sampler. With the defaults these are the 96 P08 clips.
Changing this sampler makes those results incomparable.

A clip is T context frames, `stride` frames apart, and the future frame `stride` frames
after the last one. The model predicts the embedding of the future frame. Each clip carries
the signals of its context frames (sig) and of the frames one step later (sig_next, for the
"future" control of Test 9), both with the gaze point (ego/signals.py).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ego import signals
from ego.data import Recording, open_loaders, read_vrs_times
from ego.gaze_geometry import projector_for
from ego.model import encode_independent, maybe_norm

VAL_SEED = 12345


@dataclass
class Clip:
    rec: Recording
    ctx_idx: list
    fut_idx: int
    sig: tuple          # numpy signals from signals.read(), no batch dimension
    sig_next: tuple = None   # the same, one step later: frames ctx_idx[1:] + [fut_idx]


def fixed_clips(recs, T, stride, n_clips, seed=VAL_SEED, standardize=True, log=None):
    """
    `n_clips` clips from each recording, with their signals read from the CSVs. No video
    is decoded, so the whole list exists before the GPU is used, and a clip's signals can
    be given to another clip.
    """
    rng = np.random.RandomState(seed)
    span = T * stride
    clips = []
    for rec in recs:
        vrs = read_vrs_times(rec.ts_csv)
        n = len(vrs)
        if n < span + 1:
            continue
        gl, hl = open_loaders(rec, standardize)
        if gl is None and hl is None:
            continue
        proj = projector_for(rec)
        for _ in range(n_clips):
            start = int(rng.randint(0, n - span))
            ctx_idx = [start + i * stride for i in range(T)]
            fut_idx = start + T * stride
            sig = signals.read(gl, hl, [int(vrs[j]) for j in ctx_idx], T, proj)
            sig_next = signals.read(gl, hl, [int(vrs[j]) for j in ctx_idx[1:] + [fut_idx]], T, proj)
            clips.append(Clip(rec, ctx_idx, fut_idx, sig, sig_next))
        if log:
            log(f"  {rec.participant}/{rec.stem}: {n_clips} clips")
    return clips


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_clip(encoder, frames, T, device, normalize_reps=True, chunk=16, amp_dtype=None):
    """frames: (T+1, 3, H, W), the context then the future frame. Returns (enc_ctx, enc_fut)."""
    enc_ctx = encode_independent(encoder, frames[:T].unsqueeze(0), device, normalize_reps,
                                 chunk=chunk, amp_dtype=amp_dtype)
    enc_fut = encode_independent(encoder, frames[T].unsqueeze(0).unsqueeze(0), device,
                                 normalize_reps, chunk=chunk, amp_dtype=amp_dtype)
    return enc_ctx, enc_fut


@torch.no_grad()
def predict_last(predictor, enc_ctx, sig, normalize_reps=True):
    """The predictor's output for the last step: the predicted embedding of the future frame."""
    HW = predictor.grid_height * predictor.grid_width
    return maybe_norm(predictor(enc_ctx, *sig)[:, -HW:, :], normalize_reps)


def mse(pred, target):
    return F.mse_loss(pred.float(), target.float()).item()


@torch.no_grad()
def paired_mse(encoder, predictor, frames, T, device, real_sig=None, normalize_reps=True):
    """
    (MSE with the signals hidden, MSE with the real signals) for one clip, both against
    the same target. The second is None when real_sig is None.
    """
    enc_ctx, enc_fut = encode_clip(encoder, frames, T, device, normalize_reps)
    mse_a = mse(predict_last(predictor, enc_ctx, signals.null(T, device), normalize_reps), enc_fut)
    mse_b = None
    if real_sig is not None:
        mse_b = mse(predict_last(predictor, enc_ctx, real_sig, normalize_reps), enc_fut)
    return mse_a, mse_b
