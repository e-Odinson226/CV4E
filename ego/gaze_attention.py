"""
The gaze token placed at the gaze point in the predictor's attention (Test 9, the forms
"rope" and "pe+rope").

V-JEPA 2-AC's attention (vjepa2/src/models/utils/modules.py, ACRoPEAttention) rotates the
channels of each head in groups: 20 by the frame index, 20 by the row and 20 by the column
of an image patch; the last 4 are not rotated. The conditioning tokens (here gaze and hand)
are rotated by the frame index only, so in the other channels they sit at row 0, column 0:
the top-left patch.

GazeRoPEAttention changes one thing. The gaze token, the first conditioning token, is also
rotated by a row and a column: the gaze point of its frame, set in `gaze_pos` before each
forward. Everything else is the upstream code. With every position at 0 the rotation is the
identity, and the output equals that of ACRoPEAttention exactly (tests/test_gaze_forms.py).
"""

import torch
import torch.nn.functional as F

from src.models.utils.modules import ACRoPEAttention, rotate_queries_or_keys


class GazeRoPEAttention(ACRoPEAttention):
    gaze_pos = None     # (row, column), each (B, T) in patches: the gaze token's position

    def _rotate_signal(self, t, T, i):
        """q or k (B, heads, T, head_dim) of conditioning token i, rotated."""
        d, h, w = self.d_dim, self.h_dim, self.w_dim
        td = rotate_queries_or_keys(t[..., :d], pos=torch.arange(T, device=t.device))
        if i != 0 or self.gaze_pos is None:
            return torch.cat([td, t[..., d:]], dim=-1)
        row, col = self.gaze_pos
        th = rotate_queries_or_keys(t[..., d:d + h], pos=row.unsqueeze(1))
        tw = rotate_queries_or_keys(t[..., d + h:d + h + w], pos=col.unsqueeze(1))
        return torch.cat([td, th, tw, t[..., d + h + w:]], dim=-1)

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        # The upstream forward, with the conditioning tokens rotated by _rotate_signal.
        B, N, C = x.size()

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)
        else:
            mask = torch.arange(int(T * H * W), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)

        h_mask *= self.grid_size / H
        w_mask *= self.grid_size / W

        if action_tokens > 0:
            x = x.view(B, -1, action_tokens + H * W, C)
            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                a = x[:, :, i : i + 1, :].flatten(1, 2)
                qkv = self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                action_q += [self._rotate_signal(q, T, i).view(B, self.num_heads, T, 1, -1)]
                action_k += [self._rotate_signal(k, T, i).view(B, self.num_heads, T, 1, -1)]
                action_v += [v.view(B, self.num_heads, T, 1, -1)]
            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        if s < self.head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0:
            def merge_(tx, ta):
                tx = tx.view(B, self.num_heads, T, H * W, -1)
                ta = ta.view(B, self.num_heads, T, action_tokens, -1)
                return torch.cat([ta, tx], dim=3).flatten(2, 3)
            q = merge_(q, action_q)
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
