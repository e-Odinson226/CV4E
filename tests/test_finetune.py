"""Self-test for what fine-tuning trains (ego/model.py: freeze_for_ego_finetune, the optimizer
groups) and for how `gaze-forms` names the models it pools (form_of) and recovers their base
form (base_form). No GPU, no data."""
import sys

import torch

from ego.commands.gaze_forms import base_form, form_of
from ego.model import describe_finetuned_layers, freeze_for_ego_finetune, get_ego_finetune_param_groups
from ego.predictor import vit_ego_predictor

ok = True
def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    ok &= bool(cond)


def tiny(depth=4):
    torch.manual_seed(0)
    return vit_ego_predictor(img_size=(64, 64), patch_size=16, num_frames=4, tubelet_size=2,
                             embed_dim=32, predictor_embed_dim=64, depth=depth, num_heads=2,
                             use_silu=False, wide_silu=True, uniform_power=False, use_rope=True)


def trains(p, prefix):
    return {q.requires_grad for n, q in p.named_parameters() if n.startswith(prefix)}


p = tiny()
freeze_for_ego_finetune(p, unfreeze_last_n_blocks=2)
check("last 2 of 4 blocks: blocks 0-1 frozen, 2-3 train",
      trains(p, "predictor_blocks.0.") == {False} and trains(p, "predictor_blocks.1.") == {False}
      and trains(p, "predictor_blocks.2.") == {True} and trains(p, "predictor_blocks.3.") == {True})
check("by default predictor_embed stays frozen", trains(p, "predictor_embed") == {False})
check("the new layers and the output layers train",
      all(trains(p, k) == {True} for k in ("gaze_proj", "hand_proj", "gaze_mask", "hand_mask",
                                           "predictor_norm", "predictor_proj")))

f = tiny()
freeze_for_ego_finetune(f, unfreeze_last_n_blocks=4, unfreeze_embed=True)
check("full fine-tune: every parameter of the predictor trains", all(q.requires_grad for q in f.parameters()))
g = get_ego_finetune_param_groups(f)
n_new = sum(q.numel() for q in g[0]["params"])
n_new_ref = sum(q.numel() for n, q in f.named_parameters() if n.split(".")[0] in ("gaze_proj", "hand_proj", "gaze_mask", "hand_mask"))
check("full fine-tune: the new layers keep their own learning rate, all else gets lr_blocks",
      n_new == n_new_ref and n_new + sum(q.numel() for q in g[1]["params"]) == sum(q.numel() for q in f.parameters()))
d = describe_finetuned_layers(f, 4, True)
check("full fine-tune: the description lists nothing frozen", d["frozen"] == [] and "predictor_embed" in d["output_head"])

base = {"signal_dropout": 0.4, "gaze_form": "pe", "shuffle_signals": "off", "unfreeze_last_n": 6}
check("form_of: a Test 9 run keeps its name", form_of(base) == "pe"
      and form_of({**base, "signal_dropout": 1.0}) == "none"
      and form_of({**base, "gaze_form": "pe+rope", "shuffle_signals": "batch"}) == "pe+rope shuffled"
      and form_of({**base, "future_signals": True}) == "future")
check("form_of: L1 and full fine-tuning are named", form_of({**base, "loss": "l1"}) == "pe l1"
      and form_of({**base, "loss": "mse"}) == "pe"
      and form_of({**base, "loss": "l1", "unfreeze_last_n": 24, "unfreeze_embed": True}) == "pe l1 full"
      and form_of({**base, "signal_dropout": 1.0, "loss": "l1", "unfreeze_last_n": 24, "unfreeze_embed": True}) == "none l1 full")
check("form_of: other choices are not pooled with the default",
      form_of({**base, "loss": "l1", "unfreeze_last_n": 12}) == "pe l1 last12"
      and form_of({**base, "unfreeze_last_n": 24}) == "pe last24")

# base_form must recover "none" and "future" through every suffix form_of can add, so that
# score() (ego/commands/gaze_forms.py) keeps feeding a "none" model its hidden-signal input and
# a "future" model its next-step signals, whatever loss or fine-tune choice produced the run.
# This guards the bug of 2026-10-08: "none l1" and "none l1 full" fell through to the real-signal
# branch because the check compared against the literal string "none".
check("base_form: none and future survive every suffix",
      all(base_form(f) == "none" for f in
          (form_of({**base, "signal_dropout": 1.0}),
           form_of({**base, "signal_dropout": 1.0, "loss": "l1"}),
           form_of({**base, "signal_dropout": 1.0, "loss": "l1", "unfreeze_last_n": 24, "unfreeze_embed": True}),
           form_of({**base, "signal_dropout": 1.0, "unfreeze_last_n": 12})))
      and all(base_form(f) == "future" for f in
          (form_of({**base, "future_signals": True}),
           form_of({**base, "future_signals": True, "loss": "l1"}),
           form_of({**base, "future_signals": True, "loss": "l1", "unfreeze_last_n": 24, "unfreeze_embed": True}))))
check("base_form: a signal-bearing form keeps its gaze form through every suffix",
      base_form(form_of({**base, "loss": "l1", "unfreeze_last_n": 24, "unfreeze_embed": True})) == "pe"
      and base_form(form_of({**base, "gaze_form": "pe+rope", "shuffle_signals": "batch", "loss": "l1"})) == "pe+rope shuffled"
      and base_form("before fine-tuning") == "before fine-tuning")

sys.exit(0 if ok else 1)
