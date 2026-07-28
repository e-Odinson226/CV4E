import torch
import torch.nn.functional as F


def predictor_forward_with_behavioral(predictor, visual_tokens,
                                       gaze_tokens, hand_tokens,
                                       device, B, T, N):

    # Project visual tokens from encoder dim to predictor dim (1408 → 1024)
    x = predictor.predictor_embed(visual_tokens)   # (B, T*N, 1024)
    D = x.shape[-1]

    # Reshape to (B, T, N, D)
    x = x.view(B, T, N, D)

    # Gaze → action slot, Hand → state slot
    a = gaze_tokens.unsqueeze(2)   # (B, T, 1, 1024)
    s = hand_tokens.unsqueeze(2)   # (B, T, 1, 1024)

    # Interleave [action, state, patches] per timestep → (B, T*(N+2), 1024)
    x = torch.cat([a, s, x], dim=2).flatten(1, 2)

    # Block-causal attention mask
    seq_len   = x.shape[1]
    attn_mask = predictor.attn_mask[:seq_len, :seq_len].to(device)

    # Forward through frozen transformer blocks
    for blk in predictor.predictor_blocks:
        x = blk(
            x,
            mask=None,
            attn_mask=attn_mask,
            T=T,
            H=predictor.grid_height,
            W=predictor.grid_width,
            action_tokens=2,
        )

    # Remove action+state tokens, keep only visual tokens
    x = x.view(B, T, 2 + N, D)
    x = x[:, :, 2:, :].flatten(1, 2)   # (B, T*N, 1024)

    # Project back to encoder dimension (1024 → 1408)
    x = predictor.predictor_norm(x)
    x = predictor.predictor_proj(x)     # (B, T*N, 1408)

    return x


def optimization_step(encoder, predictor,
                       gaze_projector, hand_projector,
                       optimizer, batch, device):

    frames       = batch['frames'].to(device)        # (B, T, C, H, W)
    gaze_vecs    = batch['gaze_vecs'].to(device)     # (B, T, 3)
    gaze_valids  = batch['gaze_valids'].to(device)   # (B, T) bool
    hand_vecs    = batch['hand_vecs'].to(device)     # (B, T, 12)
    left_valids  = batch['left_valids'].to(device)   # (B, T) bool
    right_valids = batch['right_valids'].to(device)  # (B, T) bool

    B, T, C, H, W = frames.shape

    # Step 1 — encoder: video frames → visual tokens (frozen)
    with torch.no_grad():
        frames_flat   = frames.view(B * T, C, H, W)
        visual_flat   = encoder(frames_flat)               # (B*T, N, 1408)
        N             = visual_flat.shape[1]
        visual_tokens = visual_flat.view(B, T * N, 1408)   # (B, T*N, 1408)

    # Step 2 — gaze projector: gaze → tokens (trainable)
    gaze_flat   = gaze_vecs.view(B * T, 3)
    valid_flat  = gaze_valids.view(B * T)
    gaze_tokens = gaze_projector(gaze_flat, valid_flat).view(B, T, 1024)

    # Step 3 — hand projector: hand → tokens (trainable)
    hand_flat    = hand_vecs.view(B * T, 12)
    l_flat       = left_valids.view(B * T)
    r_flat       = right_valids.view(B * T)
    hand_tokens  = hand_projector(hand_flat, l_flat, r_flat).view(B, T, 1024)

    # Step 4 — predictor: imagine future conditioned on gaze + hand (frozen)
    with torch.no_grad():
        predicted = predictor_forward_with_behavioral(
            predictor, visual_tokens, gaze_tokens, hand_tokens, device, B, T, N
        )                                                   # (B, T*N, 1408)

    # Step 5 — target: actual next frames from encoder (frozen)
    with torch.no_grad():
        target_frames = frames[:, 1:].contiguous().view(B * (T - 1), C, H, W)
        target_flat   = encoder(target_frames)
        actual        = target_flat.view(B, (T - 1) * N, 1408)

    # Step 6 — L1 loss: predicted future vs actual future
    pred_for_loss = predicted[:, :actual.shape[1], :]
    loss = F.l1_loss(pred_for_loss, actual)

    # Step 7 — backprop: only projector weights update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()