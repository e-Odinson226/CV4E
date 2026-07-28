import torch
import torch.nn as nn


class GazeZProjector(nn.Module):
    PREDICTOR_DIM = 1024

    def __init__(self):
        super().__init__()
        # Projects [yaw, pitch, depth] → 1024-d token — only trained layer
        self.gaze_proj = nn.Linear(3, self.PREDICTOR_DIM)
        # Learned mask token — used when gaze is invalid
        # Starts random, trained to represent "no gaze signal"
        self.gaze_mask = nn.Parameter(torch.randn(self.PREDICTOR_DIM) * 0.02)

    def forward(self, gaze_vec: torch.Tensor, gaze_valid: torch.Tensor):
        B = gaze_vec.shape[0]

        # All vectors go through linear projection
        gaze_token = self.gaze_proj(gaze_vec)

        # Replace invalid frames with learned mask token
        # Valid   → use real projection
        # Invalid → use gaze_mask (trained to represent "no gaze")
        gaze_token = torch.where(
            gaze_valid.unsqueeze(1),                          # (B, 1) bool
            gaze_token,                                        # (B, 1024)
            self.gaze_mask.unsqueeze(0).expand(B, -1)         # (B, 1024)
        )

        return gaze_token


# ── How to use in training loop ────────────────────────────────────────────
#
# projector   = GazeZProjector().to(device)
# optimizer   = torch.optim.Adam(projector.parameters(), lr=1e-4)
#
# # GazeTokenLoader returns numpy — convert once per batch
# gaze_vecs   = torch.from_numpy(gaze_vecs_numpy).to(device)    # (B, 3)
# gaze_valids = torch.from_numpy(gaze_valids_numpy).to(device)  # (B,) bool
#
# gaze_tokens = projector(gaze_vecs, gaze_valids)               # (B, 1024)