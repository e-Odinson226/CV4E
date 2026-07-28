import torch
import torch.nn as nn


class HandZProjector(nn.Module):
    PREDICTOR_DIM = 1024

    def __init__(self):
        super().__init__()
        self.hand_proj = nn.Linear(12, self.PREDICTOR_DIM)
        self.hand_mask = nn.Parameter(torch.randn(self.PREDICTOR_DIM) * 0.02)

    def forward(self, hand_vec: torch.Tensor,
                left_valid: torch.Tensor,
                right_valid: torch.Tensor):

        B = hand_vec.shape[0]

        # Project all vectors
        hand_token = self.hand_proj(hand_vec)

        # Both hands invalid (-1) → use mask token
        # At least one valid    → use projection
        both_invalid = ~(left_valid | right_valid)

        hand_token = torch.where(
            both_invalid.unsqueeze(1),
            self.hand_mask.unsqueeze(0).expand(B, -1),
            hand_token
        )

        return hand_token