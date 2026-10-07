"""Self-test for the gaze forms of Test 9: ego/predictor.py and ego/gaze_attention.py (no GPU, no data)."""
import sys
import torch

from ego.gaze_attention import GazeRoPEAttention
from ego.predictor import PE_FREQS, vit_ego_predictor
from src.models.utils.modules import ACRoPEAttention, rotate_queries_or_keys

ok = True
def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    ok &= bool(cond)

T, G, D = 2, 4, 32                      # 2 steps, a 4 x 4 patch grid, encoder width 32


def tiny(form, seed=0):
    torch.manual_seed(seed)
    p = vit_ego_predictor(img_size=(G * 16, G * 16), patch_size=16, num_frames=2 * T, tubelet_size=2,
                          embed_dim=D, predictor_embed_dim=64, depth=2, num_heads=2,
                          use_silu=False, wide_silu=True, uniform_power=False, use_rope=True,
                          gaze_form=form)
    return p.eval()


def inputs(col, row, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T * G * G, D, generator=g)
    gaze = torch.randn(1, T, 5, generator=g)
    gaze[..., 3], gaze[..., 4] = col, row
    hand = torch.randn(1, T, 12, generator=g)
    yes = torch.ones(1, T, dtype=torch.bool)
    return x, gaze, yes, hand, yes, yes


with torch.no_grad():
    angles, rope = tiny("angles"), tiny("rope")
    check("rope builds the same weights as angles",
          all(torch.equal(a, b) for a, b in zip(angles.state_dict().values(), rope.state_dict().values())))
    check("rope uses GazeRoPEAttention in every block",
          all(type(b.attn) is GazeRoPEAttention for b in rope.predictor_blocks))
    check("angles keeps the upstream attention",
          all(type(b.attn) is ACRoPEAttention for b in angles.predictor_blocks))

    # --- 1. rope at position 0 is exactly the angles model --------------------
    inp = inputs(0.5, 0.5)              # the centre of the top-left patch -> position (0, 0)
    check("rope with every point at the top-left patch == angles, exactly",
          torch.equal(rope(*inp), angles(*inp)))
    inp_none = inputs(-1.0, -1.0)       # NO_POINT -> position (0, 0)
    check("rope without a point == angles, exactly", torch.equal(rope(*inp_none), angles(*inp_none)))
    inp_far = inputs(3.5, 2.5)
    check("rope with the point elsewhere differs from angles",
          not torch.allclose(rope(*inp_far), angles(*inp_far)))

    # --- 2. angles ignores the point; 3 and 5 values give the same output ------
    x, gaze, gv, hand, hl, hr = inputs(3.5, 2.5)
    check("angles: 5-value and 3-value gaze vectors give the same output",
          torch.equal(angles(x, gaze, gv, hand, hl, hr), angles(x, gaze[..., :3], gv, hand, hl, hr)))

    # --- 3. the position convention: row -> height channels, column -> width -----
    rope._set_gaze_positions(*inputs(3.5, 2.5)[1:3])
    row, col = rope.predictor_blocks[0].attn.gaze_pos
    check("a point at column 3.5, row 2.5 gets position (row 2, column 3)",
          torch.equal(row, torch.full((1, T), 2.0)) and torch.equal(col, torch.full((1, T), 3.0)))
    attn = rope.predictor_blocks[0].attn
    k = torch.randn(1, 2, T, attn.head_dim)
    attn.gaze_pos = (torch.full((1, T), 2.0), torch.full((1, T), 3.0))
    got = attn._rotate_signal(k, T, 0)
    d, h, w = attn.d_dim, attn.h_dim, attn.w_dim
    frame = torch.arange(T, dtype=torch.float32)
    want = torch.cat([rotate_queries_or_keys(k[..., :d], pos=frame),
                      rotate_queries_or_keys(k[..., d:d + h], pos=torch.full((T,), 2.0)),
                      rotate_queries_or_keys(k[..., d + h:d + h + w], pos=torch.full((T,), 3.0)),
                      k[..., d + h + w:]], dim=-1)
    check("the gaze key is rotated like the patch at row 2, column 3", torch.allclose(got, want, atol=1e-6))
    check("the hand token keeps position 0", torch.equal(attn._rotate_signal(k, T, 1)[..., d:], k[..., d:]))
    swapped = torch.cat([want[..., :d], want[..., d + h:d + h + w], want[..., d:d + h], want[..., d + h + w:]], -1)
    check("row and column are not swapped", not torch.allclose(got, swapped, atol=1e-4))

    # --- 4. Coord-PE features ------------------------------------------------------
    pe = tiny("pe")
    check("pe: gaze_proj takes 4 * PE_FREQS + 2 inputs", pe.gaze_proj.in_features == 4 * PE_FREQS + 2)
    g = torch.zeros(1, 1, 5); g[..., 2] = 0.7; g[..., 3] = 8.0; g[..., 4] = 8.0     # the frame centre
    f = pe._gaze_input(g)[0, 0]
    K = PE_FREQS
    c = 1 / (2 * K) ** 0.5
    check("pe at the centre: sines 0, cosines 1, scaled by 1/sqrt(2K)",
          torch.allclose(f[:K], torch.zeros(K), atol=1e-6) and torch.allclose(f[K:2 * K], torch.full((K,), c)))
    check("pe: the waves have a joint length of 1", abs(f[:4 * K].norm() - 1.0) < 1e-5)
    check("pe keeps the depth and sets the flag", abs(f[-2] - 0.7) < 1e-6 and f[-1] == 1.0)
    g2 = g.clone(); g2[..., 3:] = -1.0
    f2 = pe._gaze_input(g2)[0, 0]
    check("pe without a point: flag 0", f2[-1] == 0.0)
    g3 = g.clone(); g3[..., 3] = 9.0                                                  # one patch right
    f3 = pe._gaze_input(g3)[0, 0]
    check("pe: the finest wave repeats once per patch, the next one every two",
          abs(f3[2 * K - 1] - c) < 1e-4 and abs(f3[2 * K - 2] + c) < 1e-4)
    pr = tiny("pe+rope")
    check("pe+rope: pe input and rope attention",
          pr.gaze_proj.in_features == 4 * K + 2 and type(pr.predictor_blocks[0].attn) is GazeRoPEAttention)

sys.exit(0 if ok else 1)
