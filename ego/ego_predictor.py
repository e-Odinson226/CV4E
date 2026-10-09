"""
The predictor of Test 11: trained from the start, with gaze and hand as maps over the
image tokens.

Why a new predictor. Tests 1-4, 8, 9 and 10 give gaze to the V-JEPA 2-AC predictor in the
slot its pretraining built for the robot's action (ego/predictor.py). That token is rotated
by the frame index alone, so it carries no row and no column and cannot point at the image.
Test 9 placed it at the gaze point instead (ego/gaze_attention.py) and it did not help: in a
RoPE head the attention between two tokens depends on their content, not on their distance,
so a token placed at the gaze point is not a token the frozen heads attend to. The horizon
was inherited in the same way, because 0.27 s is the step of 8 frames that matches the 4
frames per second of the pretraining.

This predictor drops both inheritances.

  * The signals are not tokens. The token of patch p at observed step tau becomes

        x[tau, p] + alpha_g * k(p, g[tau]) * e_g
                  + alpha_h * (k(p, l[tau]) * e_l + k(p, r[tau]) * e_r)

    with k(p, q) = exp(-||p - q||^2 / (2 sigma^2)) over the patch grid. So the signals act
    on the attention between image tokens instead of competing for it as a token of their
    own. They add no tokens and move no positions.

  * An untrained model is exactly the matched model without signals, because the vectors e
    start at zero. tests/test_ego_predictor.py asserts this.

    Which factor starts at zero matters. The gradient of alpha * k * e with respect to e is
    alpha * k, and with respect to alpha is k . e. So starting alpha at 0 and e at random,
    as first written, leaves e with no gradient at all on the first step: only alpha can
    move, and e begins to learn one step later. Starting e at 0 and alpha at 1 keeps the
    property that matters, a model that begins as the matched model, and gives e a gradient
    immediately, because the kernel k is not zero wherever a point exists. alpha is then a
    readable gain rather than a gate; its own gradient is zero while e is zero, so it starts
    to move one step after e does. The norm of e is what says how much the model uses a
    signal, and the training log prints it.

  * The frames to predict are learned mask tokens at their own time slots, given to RoPE
    through explicit token ids. Attention spans every token and is not causal over frames,
    so several horizons are predicted in one forward pass: pass target_slots=(4, 8) for
    0.5 s and 1 s at a step of 0.125 s.

A point is (column, row) in continuous patch coordinates, as ego.gaze_geometry.to_patch
returns them, and NO_POINT (-1, -1) when it is missing. A missing signal adds nothing: its
kernel is zero everywhere, so the token is unchanged.
"""

import math

import torch
import torch.nn as nn

from src.models.utils.modules import Block
from src.utils.tensors import trunc_normal_

NO_POINT = -1.0


class EgoPredictor(nn.Module):
    """
    Observed encoder tokens and the signals of the observed frames in, the tokens of the
    frames to predict out.

    Parameters
    ----------
    embed_dim : the encoder's width, the width in and out
    pred_dim, depth, num_heads : the predictor itself. The defaults give about 21 million
        parameters, the size of V-JEPA 2.1's predictor. The V-JEPA 2-AC predictor holds
        305.2 million, too many to train from the start on this data.
    grid : patches per side, 16 for the V-JEPA 2 encoder at 256 pixels
    sigma : the width of the kernel in patches. 1 patch is about twice the error of the gaze
        projection, and a picked object covers about 3 of 256 patches.
    use_hand : whether the hand maps exist at all. With no hand point in the data, leave it
        on and pass hand_valid all False; the maps then add nothing.
    """

    def __init__(self, embed_dim, pred_dim=384, depth=12, num_heads=6, grid=16,
                 sigma=1.0, mlp_ratio=4.0, qkv_bias=True, use_hand=True,
                 learn_sigma=True, init_std=0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.pred_dim = pred_dim
        self.grid = grid
        self.use_hand = use_hand
        self.init_std = init_std

        self.predictor_embed = nn.Linear(embed_dim, pred_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))

        # The signal maps. e starts at 0, so the model starts as the matched model and e
        # still receives a gradient on the first step (see the module docstring).
        self.e_gaze = nn.Parameter(torch.zeros(pred_dim))
        self.alpha_gaze = nn.Parameter(torch.ones(1))
        if use_hand:
            self.e_left = nn.Parameter(torch.zeros(pred_dim))
            self.e_right = nn.Parameter(torch.zeros(pred_dim))
            self.alpha_hand = nn.Parameter(torch.ones(1))
        log_sigma = torch.tensor([math.log(sigma)])
        self.log_sigma = nn.Parameter(log_sigma) if learn_sigma else None
        self.register_buffer("_log_sigma_fixed", log_sigma, persistent=False)

        self.blocks = nn.ModuleList([
            Block(dim=pred_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  use_rope=True, is_causal=False, grid_size=grid, use_sdpa=True)
            for _ in range(depth)
        ])
        self.predictor_norm = nn.LayerNorm(pred_dim)
        self.predictor_proj = nn.Linear(pred_dim, embed_dim, bias=True)

        # Patch centres in patch units, in the (row, column) order of the token layout.
        rows = torch.arange(grid).float() + 0.5
        cols = torch.arange(grid).float() + 0.5
        centres = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
        self.register_buffer("centres", centres.reshape(grid * grid, 2), persistent=False)

        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=init_std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # The maps
    # ------------------------------------------------------------------

    @property
    def sigma(self):
        p = self.log_sigma if self.log_sigma is not None else self._log_sigma_fixed
        return p.exp()

    def kernel(self, points, valid):
        """
        points : (B, T, 2) as (column, row) in patch units; valid : (B, T) bool.
        Returns (B, T, grid*grid): exp(-||p - q||^2 / 2 sigma^2) at every patch, and zero
        at a step whose point is missing or outside the frame.
        """
        col, row = points[..., 0], points[..., 1]
        q = torch.stack([row, col], dim=-1)                      # match the (row, col) layout
        d2 = (self.centres.view(1, 1, -1, 2) - q.unsqueeze(2)).pow(2).sum(-1)
        w = torch.exp(-d2 / (2.0 * self.sigma.to(d2.dtype) ** 2))
        inside = valid & (col >= 0) & (row >= 0) & (col <= self.grid) & (row <= self.grid)
        return w * inside.unsqueeze(-1).to(w.dtype)

    def add_signal_maps(self, x, gaze_pt, gaze_valid, hand_pt=None, hand_valid=None):
        """
        x : (B, T, grid*grid, pred_dim), the embedded tokens of the observed frames.
        gaze_pt : (B, T, 2); hand_pt : (B, T, 2, 2) as (left, right); hand_valid : (B, T, 2).
        """
        x = x + self.alpha_gaze * self.kernel(gaze_pt, gaze_valid).unsqueeze(-1) * self.e_gaze
        if self.use_hand and hand_pt is not None:
            left = self.kernel(hand_pt[..., 0, :], hand_valid[..., 0]).unsqueeze(-1)
            right = self.kernel(hand_pt[..., 1, :], hand_valid[..., 1]).unsqueeze(-1)
            x = x + self.alpha_hand * (left * self.e_left + right * self.e_right)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, ctx, gaze_pt, gaze_valid, hand_pt=None, hand_valid=None,
                obs_slots=None, target_slots=(4, 8)):
        """
        ctx : (B, T_obs * grid*grid, embed_dim), the frozen encoder's tokens of the observed
              frames, in (frame, row, column) order.
        obs_slots : the time slot of each observed frame. The default is 0 .. T_obs-1.
        target_slots : the time slots to predict. With a step of 0.125 s, (4, 8) is 0.5 s and
              1 s after the last observed frame when the observed frames are 0 .. 7.

        Returns (B, len(target_slots) * grid*grid, embed_dim): the predicted tokens of each
        target slot, in the order of target_slots.
        """
        HW = self.grid * self.grid
        B = ctx.size(0)
        T_obs = ctx.size(1) // HW
        if obs_slots is None:
            obs_slots = list(range(T_obs))
        assert len(obs_slots) == T_obs, f"{len(obs_slots)} slots for {T_obs} observed frames"

        x = self.predictor_embed(ctx).view(B, T_obs, HW, self.pred_dim)
        x = self.add_signal_maps(x, gaze_pt, gaze_valid, hand_pt, hand_valid)

        n_tgt = len(target_slots)
        tgt = self.mask_token.expand(B, n_tgt * HW, self.pred_dim)
        x = torch.cat([x.flatten(1, 2), tgt], dim=1)

        slots = torch.tensor(list(obs_slots) + list(target_slots), device=ctx.device)
        within = torch.arange(HW, device=ctx.device)
        ids = (slots.view(-1, 1) * HW + within.view(1, -1)).reshape(1, -1).expand(B, -1)

        for blk in self.blocks:
            x = blk(x, mask=ids, T=T_obs + n_tgt, H_patches=self.grid, W_patches=self.grid)

        x = x[:, T_obs * HW:, :]
        return self.predictor_proj(self.predictor_norm(x))


def ego_predictor(embed_dim=1408, **kwargs):
    """The predictor of Test 11, randomly initialised, on the CPU."""
    return EgoPredictor(embed_dim=embed_dim, **kwargs)


def signal_strength(model):
    """
    How far each signal has moved from "no signal", as alpha * ||e||. Zero for a model that
    does not use a signal at all, which is where every run starts.
    """
    out = {"gaze": (model.alpha_gaze.abs() * model.e_gaze.norm()).item()}
    if model.use_hand:
        out["left"] = (model.alpha_hand.abs() * model.e_left.norm()).item()
        out["right"] = (model.alpha_hand.abs() * model.e_right.norm()).item()
    return out


def param_counts(model):
    """(trainable, total) parameter counts, and the signal parameters on their own."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = ("e_gaze", "alpha_gaze", "e_left", "e_right", "alpha_hand", "log_sigma")
    signal = sum(p.numel() for n, p in model.named_parameters() if n in names)
    return trainable, total, signal
