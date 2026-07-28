"""
Fine-tuning utilities for VisionTransformerPredictorEgo.

Typical workflow
----------------
1.  Build the ego predictor and load a pretrained AC checkpoint:

        predictor = vit_ego_predictor(...)
        transferred, skipped = load_ac_weights_into_ego(predictor, ac_ckpt['predictor'])

2.  Freeze everything except the layers that need to adapt:

        freeze_for_ego_finetune(predictor, unfreeze_last_n_blocks=6)

3.  Build an optimizer with separate LRs for new vs. pretrained params:

        param_groups = get_ego_finetune_param_groups(predictor, lr_proj=1e-3, lr_blocks=1e-4)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
"""

import torch


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------

def freeze_for_ego_finetune(predictor, unfreeze_last_n_blocks: int = 6) -> None:
    """
    Freeze the entire predictor, then selectively unfreeze:

    Always trained (new, randomly initialised):
        gaze_proj, hand_proj, gaze_mask, hand_mask

    Trained at lower LR (pretrained, need to adapt to ego signals):
        last `unfreeze_last_n_blocks` transformer blocks
        predictor_norm, predictor_proj

    Frozen (pretrained, kept fixed):
        predictor_embed (encoder→predictor projection)
        first (depth - unfreeze_last_n_blocks) transformer blocks
    """
    for p in predictor.parameters():
        p.requires_grad_(False)

    # New ego projectors — always train
    for name in ('gaze_proj', 'hand_proj'):
        for p in getattr(predictor, name).parameters():
            p.requires_grad_(True)
    predictor.gaze_mask.requires_grad_(True)
    predictor.hand_mask.requires_grad_(True)

    # Last N transformer blocks
    total = len(predictor.predictor_blocks)
    n = min(unfreeze_last_n_blocks, total)
    for blk in predictor.predictor_blocks[total - n:]:
        for p in blk.parameters():
            p.requires_grad_(True)

    # Output norm + projection
    for p in predictor.predictor_norm.parameters():
        p.requires_grad_(True)
    for p in predictor.predictor_proj.parameters():
        p.requires_grad_(True)


def unfreeze_all(predictor) -> None:
    """Unfreeze every parameter — use for full fine-tuning after the warm-up phase."""
    for p in predictor.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Optimizer param groups
# ---------------------------------------------------------------------------

_NEW_PARAM_KEYS = {'gaze_proj', 'hand_proj', 'gaze_mask', 'hand_mask'}


def get_ego_finetune_param_groups(
    predictor,
    lr_proj: float = 1e-3,
    lr_blocks: float = 1e-4,
) -> list:
    """
    Returns two optimizer param groups:

    - new params  (gaze_proj, hand_proj, mask tokens) → lr_proj   (higher: random init)
    - adapted params (unfrozen blocks, output norm/proj) → lr_blocks (lower: pretrained)

    Only parameters with requires_grad=True are included.

    Example
    -------
        groups = get_ego_finetune_param_groups(predictor, lr_proj=1e-3, lr_blocks=1e-4)
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    """
    new_params, pretrained_params = [], []

    for name, p in predictor.named_parameters():
        if not p.requires_grad:
            continue
        # Match by the top-level attribute name (first segment before '.')
        top = name.split('.')[0]
        if top in _NEW_PARAM_KEYS:
            new_params.append(p)
        else:
            pretrained_params.append(p)

    return [
        {'params': new_params,       'lr': lr_proj},
        {'params': pretrained_params, 'lr': lr_blocks},
    ]


# ---------------------------------------------------------------------------
# AC checkpoint transfer
# ---------------------------------------------------------------------------

_AC_SKIP_KEYS = {
    'action_encoder.weight', 'action_encoder.bias',
    'state_encoder.weight',  'state_encoder.bias',
    'extrinsics_encoder.weight', 'extrinsics_encoder.bias',
}


def load_ac_weights_into_ego(ego_predictor, ac_state_dict: dict) -> tuple[list, list]:
    """
    Transfer compatible weights from a pretrained AC predictor state-dict.

    Transferred (shape-compatible):
        predictor_embed          — same Linear(embed_dim → pred_dim)
        all predictor_blocks     — transformer weights; cond_tokens=2 is identical
        predictor_norm           — same LayerNorm
        predictor_proj           — same Linear(pred_dim → embed_dim)

    Skipped (incompatible — replaced by ego projectors):
        action_encoder, state_encoder, extrinsics_encoder

    Returns
    -------
    transferred : list of key names that were copied
    skipped     : list of key names that were not copied
    """
    own_state = ego_predictor.state_dict()
    transferred, skipped = [], []

    for k, v in ac_state_dict.items():
        if k in _AC_SKIP_KEYS:
            skipped.append(k)
            continue
        if k in own_state and own_state[k].shape == v.shape:
            own_state[k].copy_(v)
            transferred.append(k)
        else:
            skipped.append(k)

    ego_predictor.load_state_dict(own_state)
    return transferred, skipped


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe_finetuned_layers(predictor, unfreeze_last_n: int = 6) -> dict:
    """
    Human-readable description of which layers ego fine-tuning trains vs freezes.
    Used for logging in both training and eval so the two always agree.
    """
    total = len(predictor.predictor_blocks)
    n = min(unfreeze_last_n, total)
    return {
        "new_projectors": ["gaze_proj", "hand_proj", "gaze_mask", "hand_mask"],
        "unfrozen_block_ids": list(range(total - n, total)),
        "output_head": ["predictor_norm", "predictor_proj"],
        "frozen": [f"predictor_blocks[0:{total - n}]", "predictor_embed"],
        "n_blocks_total": total,
    }


def trainable_parameter_names(predictor) -> list:
    """List of (name, numel) for every parameter with requires_grad=True."""
    return [(name, p.numel()) for name, p in predictor.named_parameters() if p.requires_grad]


def log_finetuned_layers(log, predictor, unfreeze_last_n: int = 6) -> None:
    """Emit a compact, explicit description of trainable vs frozen layers."""
    d = describe_finetuned_layers(predictor, unfreeze_last_n)
    blk = d["unfrozen_block_ids"]
    blk_str = f"{blk[0]}-{blk[-1]}" if blk else "(none)"
    s = trainable_parameter_summary(predictor)
    log.info(f"[layers] TRAIN  new: {', '.join(d['new_projectors'])}")
    log.info(f"[layers] TRAIN  blocks {blk_str} (last {len(blk)} of {d['n_blocks_total']})  "
             f"+ {', '.join(d['output_head'])}")
    log.info(f"[layers] FROZEN {', '.join(d['frozen'])}")
    log.info(f"[layers] {len(trainable_parameter_names(predictor))} trainable tensors  "
             f"{s['trainable']:,}/{s['total']:,} params ({s['trainable_pct']:.1f}%)")


def trainable_parameter_summary(predictor) -> dict:
    """
    Returns a dict with total / trainable parameter counts and a per-group breakdown.
    Useful for a quick sanity-check after freezing.
    """
    total, trainable = 0, 0
    groups = {}

    for name, p in predictor.named_parameters():
        n = p.numel()
        total += n
        top = name.split('.')[0]
        if p.requires_grad:
            trainable += n
            groups[top] = groups.get(top, 0) + n

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_pct': round(100.0 * trainable / total, 2) if total else 0.0,
        'by_group': groups,
    }
