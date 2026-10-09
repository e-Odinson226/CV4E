"""
The Test 11 predictor (ego/ego_predictor.py): the maps, the scales and the target slots.

Two checks matter most.

An untrained predictor must equal the matched model exactly, whatever the points are. That is
the property the rope form of Test 9 lacked: it moved the gaze token in every frame from the
first step of training, away from the position the frozen blocks relied on. Here training
starts at the no-signal model and the signals can only grow from it.

The vectors e must also receive a gradient on that first step. They are the zero factor in
alpha * k * e, so the gradient reaches them (it is alpha * k) while alpha's own gradient
(k . e) is still zero. Putting the zero on alpha instead would freeze e for a step, which is
why the parametrisation is this way round.
"""

import numpy as np
import torch

from ego import signals
from ego.ego_predictor import ego_predictor, param_counts

GRID = 16
HW = GRID * GRID
D = 64        # a small encoder width keeps the test on the CPU


def build(**kw):
    torch.manual_seed(0)
    return ego_predictor(embed_dim=D, pred_dim=48, depth=2, num_heads=4, grid=GRID, **kw).eval()


def inputs(B=2, T=4, valid=True):
    torch.manual_seed(1)
    ctx = torch.randn(B, T * HW, D)
    gaze_pt = torch.rand(B, T, 2) * GRID
    hand_pt = torch.rand(B, T, 2, 2) * GRID
    gv = torch.full((B, T), valid, dtype=torch.bool)
    hv = torch.full((B, T, 2), valid, dtype=torch.bool)
    return ctx, gaze_pt, gv, hand_pt, hv


def test_shapes():
    m = build()
    ctx, g, gv, h, hv = inputs(B=2, T=4)
    for slots in [(1,), (2, 4), (1, 2, 3)]:
        out = m(ctx, g, gv, h, hv, target_slots=slots)
        assert out.shape == (2, len(slots) * HW, D), (out.shape, slots)
    print("ok  one output frame per target slot")


def test_kernel():
    m = build()
    B, T = 1, 1
    # a point at the centre of patch (row 3, column 5) must peak on that patch
    pt = torch.tensor([[[5.5, 3.5]]])          # (column, row)
    k = m.kernel(pt, torch.ones(B, T, dtype=torch.bool))
    assert k.shape == (B, T, HW)
    assert int(k[0, 0].argmax()) == 3 * GRID + 5, int(k[0, 0].argmax())
    assert abs(k[0, 0].max().item() - 1.0) < 1e-5, k[0, 0].max().item()

    # a missing point adds nothing anywhere
    miss = m.kernel(torch.full((B, T, 2), signals.NO_POINT), torch.zeros(B, T, dtype=torch.bool))
    assert miss.abs().max().item() == 0.0
    # a point marked valid but outside the frame is also ignored
    outside = m.kernel(torch.tensor([[[-1.0, -1.0]]]), torch.ones(B, T, dtype=torch.bool))
    assert outside.abs().max().item() == 0.0
    print("ok  the map peaks on the patch of the point, and is zero without one")


def test_untrained_is_the_matched_model():
    m = build()
    ctx, g, gv, h, hv = inputs()
    assert m.e_gaze.abs().max().item() == 0.0 and m.e_left.abs().max().item() == 0.0
    with torch.no_grad():
        with_points = m(ctx, g, gv, h, hv, target_slots=(2, 4))
        no_points = m(ctx, torch.full_like(g, signals.NO_POINT), torch.zeros_like(gv),
                      torch.full_like(h, signals.NO_POINT), torch.zeros_like(hv),
                      target_slots=(2, 4))
    assert torch.equal(with_points, no_points), (with_points - no_points).abs().max().item()

    # and once e moves off zero, the points change the output
    with torch.no_grad():
        m.e_gaze.normal_(0.0, 0.02)
        moved = m(ctx, g, gv, h, hv, target_slots=(2, 4))
    assert not torch.equal(moved, no_points)
    print("ok  an untrained predictor is exactly the matched model; a trained one is not")


def test_vectors_get_a_gradient_on_the_first_step():
    m = build()
    ctx, g, gv, h, hv = inputs()
    m(ctx, g, gv, h, hv, target_slots=(2,)).square().mean().backward()
    for name in ("e_gaze", "e_left", "e_right"):
        p = getattr(m, name)
        assert p.grad is not None and p.grad.abs().sum().item() > 0, f"{name} has no gradient"
    # alpha and sigma multiply e, which is still zero, so they move one step later
    for name in ("alpha_gaze", "alpha_hand", "log_sigma"):
        p = getattr(m, name)
        assert p.grad is None or p.grad.abs().sum().item() == 0, f"{name} moved too early"

    # after one step of e, the gain and the width start to move too
    with torch.no_grad():
        for name in ("e_gaze", "e_left", "e_right"):
            getattr(m, name).sub_(getattr(m, name).grad * 1.0)
    m.zero_grad()
    m(ctx, g, gv, h, hv, target_slots=(2,)).square().mean().backward()
    for name in ("alpha_gaze", "alpha_hand", "log_sigma"):
        p = getattr(m, name)
        assert p.grad is not None and p.grad.abs().sum().item() > 0, f"{name} still frozen"
    print("ok  e learns from the first step; the gain and sigma follow one step later")


def test_no_hand():
    m = build(use_hand=False)
    ctx, g, gv, h, hv = inputs()
    out = m(ctx, g, gv, None, None, target_slots=(2,))
    assert out.shape == (2, HW, D)
    assert not hasattr(m, "alpha_hand") and not hasattr(m, "e_left")
    print("ok  gaze-only predictor builds and runs")


def test_target_slots_change_the_prediction():
    """A target slot is a RoPE position, so predicting 2 slots ahead is not the same job as 4."""
    m = build()
    ctx, g, gv, h, hv = inputs()
    with torch.no_grad():
        a = m(ctx, g, gv, h, hv, target_slots=(2,))
        b = m(ctx, g, gv, h, hv, target_slots=(4,))
    assert not torch.allclose(a, b), "the horizon does not reach the predictor"
    print("ok  the target slot reaches the attention as a position")


def test_matched_model_arm():
    """The `none` arm of Test 11: every signal parameter held where it starts."""
    from ego.commands.train_ego import freeze_signals
    m = build()
    freeze_signals(m)
    ctx, g, gv, h, hv = inputs()
    out = m(ctx, g, gv, h, hv, target_slots=(2,))
    out.square().mean().backward()
    for name in ("e_gaze", "alpha_gaze", "e_left", "e_right", "alpha_hand"):
        p = getattr(m, name)
        assert not p.requires_grad, f"{name} still trains in the matched model"
        assert p.grad is None, f"{name} received a gradient in the matched model"
    assert m.e_gaze.abs().max().item() == 0.0
    print("ok  the matched model cannot learn to use the signals at all")


def test_size():
    m = ego_predictor(embed_dim=1408)
    _, total, sig_n = param_counts(m)
    assert 15e6 < total < 30e6, f"{total / 1e6:.1f}M is not the intended size"
    assert sig_n < 2000, sig_n
    print(f"ok  {total / 1e6:.1f}M parameters, {sig_n} of them the signal parameters")


def test_points_without_a_projector():
    """signals.points with no calibration: the gaze point survives, the palms are missing."""
    T = 3
    gaze = np.zeros((T, 5), np.float32)
    gaze[:, 3:] = [4.0, 7.0]
    sig = (gaze, np.ones(T, bool), np.zeros((T, 12), np.float32),
           np.ones(T, bool), np.ones(T, bool))
    g_pt, g_val, h_pt, h_val = signals.points(sig, None)
    assert g_val.all() and np.allclose(g_pt[:, 0], 4.0) and np.allclose(g_pt[:, 1], 7.0)
    assert not h_val.any() and np.allclose(h_pt, signals.NO_POINT)

    g_pt, g_val, h_pt, h_val = signals.null_points(T)
    assert not g_val.any() and not h_val.any()
    print("ok  signals.points passes the gaze point through and marks absent palms invalid")


if __name__ == "__main__":
    test_shapes()
    test_kernel()
    test_untrained_is_the_matched_model()
    test_vectors_get_a_gradient_on_the_first_step()
    test_no_hand()
    test_target_slots_change_the_prediction()
    test_matched_model_arm()
    test_size()
    test_points_without_a_projector()
    print("ego predictor: all checks pass")
