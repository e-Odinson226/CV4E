"""
The gaze and hand inputs of a clip, and their variants.

A signal tuple is (gaze, gaze_valid, hand, hand_left_valid, hand_right_valid), shaped as
the predictor's forward takes it: (B, T, 5), (B, T), (B, T, 12), (B, T), (B, T). A signal
marked invalid is replaced by its learned mask token inside the predictor.

The gaze vector is yaw, pitch and depth (scaled, see below), then the gaze point in the
image as column and row on the 16 x 16 patch grid (ego/gaze_geometry.py). The point is
NO_POINT (-1, -1) when the gaze is invalid or no calibration exists. The angles model uses
only the first three values; the gaze forms of Test 9 use the point. A 3-value gaze vector
(as from null()) is read as having no point. shift_gaze_deg moves the yaw only, not the
point.

Every variant returns a new tuple and leaves its input unchanged.

Gaze arrives scaled by GAZE_MEAN and GAZE_STD, so a shift of d degrees of yaw is
radians(d) / GAZE_STD[0] in the units the projection layer sees.
"""

import numpy as np
import torch

from ego.data import GAZE_MEAN, GAZE_STD, HAND_MEAN, HAND_STD
from ego.gaze_geometry import GRID, to_patch

NO_POINT = -1.0


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read(gl, hl, vrs_ns, T, proj=None):
    """
    Gaze and hand at T frame timestamps (absolute VRS, ns), as numpy arrays with no batch
    dimension. A missing loader leaves its signal at zero and marked invalid. With a
    GazeProjector `proj`, the gaze point is filled in; without one it is NO_POINT.
    """
    gaze = np.zeros((T, 5), np.float32); gv = np.zeros(T, bool)
    gaze[:, 3:] = NO_POINT
    hand = np.zeros((T, 12), np.float32); hlft = np.zeros(T, bool); hrgt = np.zeros(T, bool)
    if gl is not None:
        for t, ns in enumerate(vrs_ns):
            gaze[t, :3], gv[t] = gl.get_token_for_vrs_ns(ns)
            if gv[t] and proj is not None:
                raw = gaze[t, :3] * GAZE_STD + GAZE_MEAN if gl.standardize else gaze[t, :3]
                xy = proj.project(*raw)
                if xy is not None:
                    gaze[t, 3:] = to_patch(xy)
    if hl is not None:
        for t, ns in enumerate(vrs_ns):
            hand[t], hlft[t], hrgt[t] = hl.get_token_for_vrs_ns(ns)
    return gaze, gv, hand, hlft, hrgt


def points(sig, proj, standardize=True, grid=GRID):
    """
    A signal tuple -> the four inputs of the Test 11 predictor (ego/ego_predictor.py), as
    numpy arrays with no batch dimension:

        gaze_pt  (T, 2)     the gaze point, column and row in patch units
        gaze_val (T,)  bool
        hand_pt  (T, 2, 2)  the left then the right palm, the same units
        hand_val (T, 2) bool

    The gaze point is already in the signal tuple, filled in by read(). The palms are not:
    the hand file gives them in the device frame, so each one is projected into the image
    here, with the same camera model as the gaze (GazeProjector.project_device_point). A
    point that is missing, outside the frame or without a calibration is NO_POINT and marked
    invalid, and the predictor's map for it is zero everywhere.
    """
    gaze, gv, hand, hlft, hrgt = sig
    T = gaze.shape[0]

    gaze_pt = np.full((T, 2), NO_POINT, np.float32)
    gaze_val = np.zeros(T, bool)
    if gaze.shape[-1] >= 5:
        gaze_pt[:] = gaze[:, 3:5]
        gaze_val[:] = np.asarray(gv, bool) & (gaze[:, 3] >= 0) & (gaze[:, 4] >= 0)

    hand_pt = np.full((T, 2, 2), NO_POINT, np.float32)
    hand_val = np.zeros((T, 2), bool)
    if proj is not None:
        valid = (np.asarray(hlft, bool), np.asarray(hrgt, bool))
        for h, cols in enumerate(((3, 6), (9, 12))):      # left palm, right palm
            for t in range(T):
                if not valid[h][t]:
                    continue
                xyz = hand[t, cols[0]:cols[1]]
                if standardize:
                    xyz = xyz * HAND_STD + HAND_MEAN
                xy = proj.project_device_point(xyz)
                if xy is not None:
                    hand_pt[t, h] = to_patch(xy, grid=grid)
                    hand_val[t, h] = True
    return gaze_pt, gaze_val, hand_pt, hand_val


def null_points(T):
    """No gaze point and no palm point: the maps of the matched model add nothing."""
    return (np.full((T, 2), NO_POINT, np.float32), np.zeros(T, bool),
            np.full((T, 2, 2), NO_POINT, np.float32), np.zeros((T, 2), bool))


def as_batch(sig, device):
    """Numpy signals from read() -> tensors with a batch dimension of 1, on `device`."""
    return tuple(torch.from_numpy(x).unsqueeze(0).to(device) for x in sig)


def null(T, device):
    """No signal at all: zeros, every frame marked invalid."""
    return (torch.zeros(1, T, 3, device=device),
            torch.zeros(1, T, dtype=torch.bool, device=device),
            torch.zeros(1, T, 12, device=device),
            torch.zeros(1, T, dtype=torch.bool, device=device),
            torch.zeros(1, T, dtype=torch.bool, device=device))


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def mask_gaze(sig):
    gaze, gv, hand, hl, hr = sig
    return (gaze, torch.zeros_like(gv), hand, hl, hr)


def mask_hand(sig):
    gaze, gv, hand, hl, hr = sig
    return (gaze, gv, hand, torch.zeros_like(hl), torch.zeros_like(hr))


def mask_both(sig):
    return mask_hand(mask_gaze(sig))


def zero_gaze(sig):
    gaze, gv, hand, hl, hr = sig
    return (torch.zeros_like(gaze), torch.ones_like(gv), hand, hl, hr)


def shift_gaze_deg(sig, deg):
    gaze, gv, hand, hl, hr = sig
    g = gaze.clone()
    g[..., 0] = g[..., 0] + float(np.radians(deg)) / float(GAZE_STD[0])
    return (g, gv, hand, hl, hr)


def swap_gaze(sig, other):
    return (other[0], other[1], sig[2], sig[3], sig[4])


def swap_hand(sig, other):
    return (sig[0], sig[1], other[2], other[3], other[4])


def shuffle_gaze_time(sig, perm):
    gaze, gv, hand, hl, hr = sig
    return (gaze[:, perm, :], gv[:, perm], hand, hl, hr)


def shuffle_hand_time(sig, perm):
    gaze, gv, hand, hl, hr = sig
    return (gaze, gv, hand[:, perm, :], hl[:, perm], hr[:, perm])


def no_signal_variants(sig, gaze_mean, hand_mean):
    """
    The real signals and three ways to give the model no signal:

      mask   gaze and hand marked invalid, so the predictor uses its mask tokens
      zeros  the zero vector, marked valid. After scaling, zero is the gaze GAZE_MEAN
      mean   the average scaled signal (gaze_mean (1,1,3), hand_mean (1,1,12)), marked valid
    """
    gaze, gv, hand, hl, hr = sig
    ones_g = torch.ones_like(gv)
    return {
        "real":  sig,
        "mask":  (gaze, torch.zeros_like(gv), hand, torch.zeros_like(hl), torch.zeros_like(hr)),
        "zeros": (torch.zeros_like(gaze), ones_g, torch.zeros_like(hand),
                  torch.ones_like(hl), torch.ones_like(hr)),
        "mean":  (gaze_mean.expand_as(gaze).contiguous(), ones_g,
                  hand_mean.expand_as(hand).contiguous(),
                  torch.ones_like(hl), torch.ones_like(hr)),
    }
