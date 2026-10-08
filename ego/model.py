"""
Building, loading and running the model: the frozen ViT-g encoder and the ego predictor.

The setup matches the original V-JEPA 2-AC droid training
(configs/train/vitg16/droid-256px-8f.yaml and app/vjepa_droid/train.py in vjepa2/):

  * The EMA target encoder encodes both the context and the target.
  * Each frame is encoded on its own, as a 2-frame tubelet of the same image
    (encode_independent). The pretrained predictor was trained on such per-frame latents.
  * Embeddings are layer-normalized before the loss (normalize_reps).

Fine-tuning trains the new layers (gaze_proj, hand_proj, the mask tokens) at one learning
rate, and the last blocks and the output layers at a lower one:

    encoder, predictor, _ = load_models(checkpoint, device, context_steps=8)
    freeze_for_ego_finetune(predictor, unfreeze_last_n_blocks=6)
    groups = get_ego_finetune_param_groups(predictor, lr_proj=1e-3, lr_blocks=1e-4)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
"""

import torch
import torch.nn.functional as F

from src.models.vision_transformer import vit_giant_xformers

from ego.predictor import vit_ego_predictor


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def strip_prefix(sd: dict) -> dict:
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


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


def load_models(checkpoint, device, context_steps, tubelet=2, encoder_key="target_encoder",
                gaze_form="angles"):
    """
    Build the ViT-g encoder + ego predictor from a V-JEPA 2-AC checkpoint.

    encoder_key="target_encoder" matches the droid config, which uses the EMA
    target encoder for both the context and the prediction target. Falls back to
    "encoder" if the requested key is absent.

    The predictor's num_frames is context_steps * tubelet so its causal attention
    mask has exactly context_steps temporal slots. gaze_form sets how the predictor takes
    gaze (ego/predictor.py).
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

    predictor = build_predictor(context_steps, tubelet, encoder.embed_dim, gaze_form)
    transferred, skipped = load_ac_weights_into_ego(predictor, strip_prefix(ck["predictor"]))
    predictor.to(device)
    return encoder, predictor, (len(transferred), len(skipped))


def load_trained_predictor(ckpt_path, device, context_steps=8):
    """
    A predictor written by `python -m ego train`, built with the gaze form saved in its
    config, loaded, frozen and in eval mode. Returns (predictor, config). Checkpoints from
    before the gaze forms have no gaze_form in their config; they are "angles".
    """
    cfg = dict(torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True).get("config", {}))
    cfg.setdefault("gaze_form", "angles")
    predictor = build_predictor(context_steps, gaze_form=cfg["gaze_form"])
    load_finetuned(predictor, ckpt_path)
    predictor.to(device).eval()
    for p in predictor.parameters():
        p.requires_grad_(False)
    return predictor, cfg


def build_predictor(context_steps, tubelet=2, embed_dim=1408, gaze_form="angles"):
    """The ego predictor with V-JEPA 2-AC's shape, randomly initialised, on the CPU."""
    return vit_ego_predictor(
        img_size=(256, 256), patch_size=16,
        num_frames=context_steps * tubelet, tubelet_size=tubelet,
        embed_dim=embed_dim, predictor_embed_dim=1024,
        depth=24, num_heads=16,
        use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True, gaze_form=gaze_form,
    )


def load_finetuned(predictor, ckpt_path):
    """Load a checkpoint written by `python -m ego train` and print which layers it trained."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["predictor"] if "predictor" in ck else ck
    predictor.load_state_dict(strip_prefix(sd), strict=True)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    n = cfg.get("unfreeze_last_n", 6)
    d = describe_finetuned_layers(predictor, n, cfg.get("unfreeze_embed", False))
    blk = d["unfrozen_block_ids"]
    print(f"[layers] fine-tuned: {', '.join(d['new_projectors'])} + "
          f"blocks {blk[0]}-{blk[-1]} + {', '.join(d['output_head'])}  "
          f"(epoch={ck.get('epoch','?')}, train_loss={ck.get('loss','?')})")


# ---------------------------------------------------------------------------
# What trains
# ---------------------------------------------------------------------------

def freeze_for_ego_finetune(predictor, unfreeze_last_n_blocks: int = 6, unfreeze_embed: bool = False) -> None:
    """
    Freeze the entire predictor, then selectively unfreeze:

    Always trained (new, randomly initialised):
        gaze_proj, hand_proj, gaze_mask, hand_mask

    Trained at lower LR (pretrained, need to adapt to ego signals):
        last `unfreeze_last_n_blocks` transformer blocks
        predictor_norm, predictor_proj
        predictor_embed, if unfreeze_embed

    Frozen (pretrained, kept fixed):
        predictor_embed (encoder→predictor projection), unless unfreeze_embed
        first (depth - unfreeze_last_n_blocks) transformer blocks

    With unfreeze_last_n_blocks = depth (24) and unfreeze_embed, the whole predictor trains:
    a full fine-tune.
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

    if unfreeze_embed:
        for p in predictor.predictor_embed.parameters():
            p.requires_grad_(True)


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


def describe_finetuned_layers(predictor, unfreeze_last_n: int = 6, unfreeze_embed: bool = False) -> dict:
    """
    Human-readable description of which layers ego fine-tuning trains vs freezes.
    Used for logging in both training and eval so the two always agree.
    """
    total = len(predictor.predictor_blocks)
    n = min(unfreeze_last_n, total)
    frozen = [f"predictor_blocks[0:{total - n}]"] if n < total else []
    return {
        "new_projectors": ["gaze_proj", "hand_proj", "gaze_mask", "hand_mask"],
        "unfrozen_block_ids": list(range(total - n, total)),
        "output_head": ["predictor_norm", "predictor_proj"] + (["predictor_embed"] if unfreeze_embed else []),
        "frozen": frozen + ([] if unfreeze_embed else ["predictor_embed"]),
        "n_blocks_total": total,
    }


def trainable_parameter_names(predictor) -> list:
    """List of (name, numel) for every parameter with requires_grad=True."""
    return [(name, p.numel()) for name, p in predictor.named_parameters() if p.requires_grad]


def trainable_parameter_summary(predictor) -> dict:
    """Total and trainable parameter counts, with a per-group breakdown."""
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


def log_finetuned_layers(log, predictor, unfreeze_last_n: int = 6, unfreeze_embed: bool = False) -> None:
    """Emit a compact, explicit description of trainable vs frozen layers."""
    d = describe_finetuned_layers(predictor, unfreeze_last_n, unfreeze_embed)
    blk = d["unfrozen_block_ids"]
    blk_str = f"{blk[0]}-{blk[-1]}" if blk else "(none)"
    s = trainable_parameter_summary(predictor)
    log.info(f"[layers] TRAIN  new: {', '.join(d['new_projectors'])}")
    log.info(f"[layers] TRAIN  blocks {blk_str} (last {len(blk)} of {d['n_blocks_total']})  "
             f"+ {', '.join(d['output_head'])}")
    log.info(f"[layers] FROZEN {', '.join(d['frozen']) or '(nothing)'}")
    log.info(f"[layers] {len(trainable_parameter_names(predictor))} trainable tensors  "
             f"{s['trainable']:,}/{s['total']:,} params ({s['trainable_pct']:.1f}%)")


# ---------------------------------------------------------------------------
# Encoding (mirrors app/vjepa_droid/train.py forward_target)
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
