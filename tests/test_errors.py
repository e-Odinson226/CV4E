"""Self-test for the two error measures, MSE and L1: the training loss of `train`
(ego/commands/train.py), the per-clip errors of ego/clips.py, and the scoring and the
references of `gaze-forms` (ego/commands/gaze_forms.py). No GPU, no data."""
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ego.clips import l1, mse
from ego.commands.gaze_forms import BLEND, HW, T, near_mask, reference_predictor, score
from ego.commands.train import LOSSES

ok = True
def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    ok &= bool(cond)

D = 8
g = torch.Generator().manual_seed(0)
p, t = torch.randn(2, 5, D, generator=g), torch.randn(2, 5, D, generator=g)

# V-JEPA 2-AC's loss (app/vjepa_droid/train.py): torch.mean(torch.abs(z - h) ** loss_exp) / loss_exp
check("--loss l1 is V-JEPA 2-AC's loss with loss_exp 1.0",
      torch.allclose(LOSSES["l1"](p, t), torch.mean(torch.abs(p - t) ** 1.0) / 1.0))
check("--loss mse is the mean squared difference", torch.allclose(LOSSES["mse"](p, t), ((p - t) ** 2).mean()))
check("clips.l1 and clips.mse", abs(l1(p, t) - (p - t).abs().mean().item()) < 1e-6
      and abs(mse(p, t) - ((p - t) ** 2).mean().item()) < 1e-6)


def clip(col, row):
    gaze = np.zeros((T, 5), np.float32)
    gaze[:, 3], gaze[:, 4] = col, row
    sig = (gaze, np.ones(T, bool), np.zeros((T, 12), np.float32), np.ones(T, bool), np.ones(T, bool))
    return SimpleNamespace(sig=sig, sig_next=sig)


def ln(x):
    return F.layer_norm(x, (x.size(-1),))


# two recordings, three clips; the last clip has no gaze point
by_rec = {"a": [clip(3.5, 4.5), clip(10.0, 12.0)], "b": [clip(-1.0, -1.0)]}
feats = {s: {"ctx": ln(torch.randn(len(c), T * HW, D, generator=g)), "fut": ln(torch.randn(len(c), HW, D, generator=g))}
         for s, c in by_rec.items()}
blend, form = reference_predictor("blend of past frames", None, torch.device("cpu"))
s = score(blend, form, feats, by_rec, torch.device("cpu"))

check("score: every measure, one value per clip",
      all(s[k].shape == (3,) for k in ("mse", "mse_hide", "near", "near_hide", "l1", "l1_hide", "near_l1", "near_l1_hide")))
check("score: recordings in order", list(s["recording"]) == ["a", "a", "b"])

frames = feats["a"]["ctx"][1].view(T, HW, D)
pred = ln(BLEND * frames[-1] + (1 - BLEND) * frames.mean(0))
e2 = ((pred - feats["a"]["fut"][1]) ** 2).mean(-1)
e1 = (pred - feats["a"]["fut"][1]).abs().mean(-1)
check("score: the blend's MSE and L1 by hand", abs(s["mse"][1] - e2.mean().item()) < 1e-6
      and abs(s["l1"][1] - e1.mean().item()) < 1e-6)
near = near_mask(torch.tensor([[10.0, 12.0]]))[0]
check("score: near uses only the patches within 2 patches of the gaze point",
      abs(s["near"][1] - e2[near].mean().item()) < 1e-6 and abs(s["near_l1"][1] - e1[near].mean().item()) < 1e-6,
      f"({int(near.sum())} patches)")
check("score: no gaze point gives no near value", np.isnan(s["near"][2]) and np.isnan(s["near_l1"][2]))
check("score: a reference that ignores the signals is the same with them hidden",
      np.array_equal(s["mse"], s["mse_hide"]) and np.array_equal(s["l1"], s["l1_hide"]))

last, _ = reference_predictor("repeat last frame", None, torch.device("cpu"))
r = score(last, "baseline", feats, by_rec, torch.device("cpu"))
check("repeat last frame: the error of the last context frame",
      abs(r["mse"][0] - ((feats["a"]["ctx"][0, -HW:] - feats["a"]["fut"][0]) ** 2).mean().item()) < 1e-6)
check("MSE of layer-normalized vectors is 2 x (1 - correlation)",
      abs(r["mse"][0] - 2 * (1 - (ln(feats["a"]["ctx"][0, -HW:]) * feats["a"]["fut"][0]).mean().item())) < 1e-4)

sys.exit(0 if ok else 1)
