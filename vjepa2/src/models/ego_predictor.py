import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_


class VisionTransformerPredictorEgo(nn.Module):
    """
    Egocentric predictor: conditions on gaze + hand tracking instead of robot actions.

    The token layout per frame is [gaze_token, hand_token, visual_tokens...], identical
    to the AC predictor's [action_token, state_token, visual_tokens...] (cond_tokens=2).
    This means all transformer block weights from a pretrained AC checkpoint transfer
    directly — only the two input projectors are newly initialised.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        gaze_dim=3,
        hand_dim=12,
        **kwargs,
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal

        # Projects encoder tokens into predictor hidden space — same role as in AC predictor
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Ego signal projectors — replace action_encoder + state_encoder
        self.gaze_proj = nn.Linear(gaze_dim, predictor_embed_dim, bias=True)
        self.hand_proj = nn.Linear(hand_dim, predictor_embed_dim, bias=True)

        # Learned mask tokens used when a signal is invalid for a frame
        self.gaze_mask = nn.Parameter(torch.randn(predictor_embed_dim) * 0.02)
        self.hand_mask = nn.Parameter(torch.randn(predictor_embed_dim) * 0.02)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.use_activation_checkpointing = use_activation_checkpointing
        self.uniform_power = uniform_power
        self.use_rope = use_rope

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # cond_tokens=2 mirrors AC predictor (no extrinsics) so the mask is compatible
        attn_mask = None
        if self.is_frame_causal:
            grid_depth = num_frames // tubelet_size
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth, self.grid_height, self.grid_width, add_tokens=2
            )
        self.attn_mask = attn_mask

    # ------------------------------------------------------------------
    # Weight init (identical to AC predictor)
    # ------------------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # ------------------------------------------------------------------
    # Signal encoders
    # ------------------------------------------------------------------

    def _encode_gaze(
        self,
        gaze_vecs: torch.Tensor,  # (B, T, 3)
        gaze_valid: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:  # (B, T, D)
        B, T, _ = gaze_vecs.shape
        tokens = self.gaze_proj(gaze_vecs.reshape(B * T, -1)).reshape(B, T, -1)
        mask = self.gaze_mask.view(1, 1, -1).expand(B, T, -1)
        return torch.where(gaze_valid.unsqueeze(-1), tokens, mask)

    def _encode_hand(
        self,
        hand_vecs: torch.Tensor,  # (B, T, 12)
        hand_left_valid: torch.Tensor,  # (B, T) bool
        hand_right_valid: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:  # (B, T, D)
        B, T, _ = hand_vecs.shape
        tokens = self.hand_proj(hand_vecs.reshape(B * T, -1)).reshape(B, T, -1)
        both_invalid = ~(hand_left_valid | hand_right_valid)
        mask = self.hand_mask.view(1, 1, -1).expand(B, T, -1)
        return torch.where(both_invalid.unsqueeze(-1), mask, tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,  # (B, T*H*W, embed_dim)  encoder context tokens
        gaze_vecs: torch.Tensor,  # (B, T, 3)
        gaze_valid: torch.Tensor,  # (B, T) bool
        hand_vecs: torch.Tensor,  # (B, T, 12)
        hand_left_valid: torch.Tensor,  # (B, T) bool
        hand_right_valid: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:  # (B, T*H*W, embed_dim)
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)

        gaze_tokens = self._encode_gaze(gaze_vecs, gaze_valid)  # (B, T, D)
        hand_tokens = self._encode_hand(hand_vecs, hand_left_valid, hand_right_valid)  # (B, T, D)

        # Interleave: [gaze, hand, visual...] per frame — same layout as AC [action, state, visual...]
        x = x.view(B, T, self.grid_height * self.grid_width, D)
        x = torch.cat([gaze_tokens.unsqueeze(2), hand_tokens.unsqueeze(2), x], dim=2)
        x = x.flatten(1, 2)  # (B, T*(H*W+2), D)

        cond_tokens = 2
        attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                )

        # Strip conditioning tokens, keep visual tokens only
        x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)
        x = x[:, :, cond_tokens:, :].flatten(1, 2)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x


def vit_ego_predictor(**kwargs):
    return VisionTransformerPredictorEgo(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


# from src.models.ego_predictor import vit_ego_predictor
# from src.models.ego_finetune import (
#     load_ac_weights_into_ego,
#     freeze_for_ego_finetune,
#     get_ego_finetune_param_groups,
#     trainable_parameter_summary,
# )

# # 1. Build predictor
# predictor = vit_ego_predictor(embed_dim=1024, predictor_embed_dim=1024, depth=24, ...)

# # 2. Transfer pretrained AC weights (blocks, norm, proj — everything except action encoders)
# transferred, skipped = load_ac_weights_into_ego(predictor, ac_ckpt['predictor'])

# # 3. Freeze all but new projectors + last 6 blocks + output head
# freeze_for_ego_finetune(predictor, unfreeze_last_n_blocks=6)
# print(trainable_parameter_summary(predictor))   # sanity check

# # 4. Optimizer with two LRs
# param_groups = get_ego_finetune_param_groups(predictor, lr_proj=1e-3, lr_blocks=1e-4)
# optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

# # 5. Forward pass
# out = predictor(encoder_tokens, gaze_vecs, gaze_valid, hand_vecs, left_valid, right_valid)
# What gets frozen vs. trained
# Layer	Status	Reason
# gaze_proj, hand_proj, *_mask	Train @ 1e-3	Randomly init, must learn fast
# Last 6 transformer blocks	Train @ 1e-4	Pretrained, adapt to ego signals
# predictor_norm, predictor_proj	Train @ 1e-4	Output head needs calibration
# predictor_embed	Frozen	Maps encoder tokens, no ego-specific change
# First 18 transformer blocks	Frozen	Low-level feature processing stays fixed
