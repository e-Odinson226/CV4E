"""
T8: did training shrink the gaze layer?

A new module in a pretrained network adds noise at first. Training can reduce that
noise by driving the module's weights toward zero, and the pathway may then never
recover. GazeQwen describes this (docs/EgoVault/papers/literature/gazeqwen.md). If it
happened here, the norm of gaze_proj.weight shrank relative to its value at the start.

Weight decay
------------
Training uses AdamW with weight decay 0.01 on all trained parameters, so every
trained weight shrinks, whatever the data. The shrinkage of gaze_proj alone says
nothing. Each norm is therefore reported in three ways:

  ratio_to_init   The norm divided by the norm at the start.
  vs reference    The same ratio for parameters trained with the same decay and the
                  same schedule: blocks 18-23 and the output layers. Suppression means
                  that gaze_proj shrinks more than these.
  frozen control  predictor_embed and blocks 0-17 were never trained, so their ratio
                  must be exactly 1.000. Any other value means the checkpoint is not
                  what it claims to be, and no other number can be read.

Limits
------
The finished run saved checkpoints only at epoch 3 (--save-every 3 with --epochs 3).
So the script compares the start with epoch 3. It cannot show the path in between. A
trace over training needs a run that logs the norms.

The starting values are rebuilt by running the same construction with the same seed.
The norm of a 3072-element random draw varies by only about 1.3%, so the starting norm
hardly depends on the seed.

Usage
-----
    python -m ego weight-norms \
        --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --predictor-checkpoints checkpoints/ego_ft_v2/best.pt checkpoints/ego_sd1p0/best.pt \
        --out results/weight_norms

It runs on the CPU by default, so it can run next to a GPU job.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ego.model import load_models, strip_prefix
from ego.runlog import Logger


# Groups reported. Each is (label, predicate over parameter name).
GROUPS = [
    ("gaze_proj.weight",  lambda n: n == "gaze_proj.weight"),
    ("gaze_proj.bias",    lambda n: n == "gaze_proj.bias"),
    ("gaze_mask",         lambda n: n == "gaze_mask"),
    ("hand_proj.weight",  lambda n: n == "hand_proj.weight"),
    ("hand_proj.bias",    lambda n: n == "hand_proj.bias"),
    ("hand_mask",         lambda n: n == "hand_mask"),
    ("blocks_18-23",      lambda n: n.startswith("predictor_blocks.") and
                                    int(n.split(".")[1]) >= 18),
    ("predictor_norm",    lambda n: n.startswith("predictor_norm.")),
    ("predictor_proj",    lambda n: n.startswith("predictor_proj.")),
    ("blocks_0-17 [frozen]", lambda n: n.startswith("predictor_blocks.") and
                                       int(n.split(".")[1]) < 18),
    ("predictor_embed [frozen]", lambda n: n.startswith("predictor_embed.")),
]


def group_norm(sd, pred):
    """Frobenius norm over a group, as sqrt of the summed squared norms."""
    tot = 0.0
    n_par = 0
    for k, v in sd.items():
        if pred(k):
            tot += float(v.float().pow(2).sum())
            n_par += v.numel()
    return float(np.sqrt(tot)), n_par


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="AC checkpoint — the init reference")
    ap.add_argument("--predictor-checkpoints", nargs="+", required=True)
    ap.add_argument("--context-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="seed the runs used")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/weight_norms")
    args = ap.parse_args()

    out = Path(args.out)
    log = Logger(out)

    # Rebuild the exact construction path training takes, so the
    # projectors get the initialisation the runs actually started from.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _, predictor, _ = load_models(args.checkpoint, torch.device(args.device),
                                  args.context_steps, tubelet=2, encoder_key="target_encoder")
    init_sd = {k: v.detach().cpu() for k, v in predictor.state_dict().items()}
    log(f"[init] rebuilt with seed={args.seed}")

    rows = []
    base = {}
    for label, pred in GROUPS:
        nrm, npar = group_norm(init_sd, pred)
        base[label] = nrm
        rows.append({"checkpoint": "init", "group": label, "norm": nrm,
                     "n_params": npar, "ratio_to_init": 1.0})
        log(f"  init  {label:<28} ||W||={nrm:12.4f}  ({npar:,} params)")

    for ckpt_path in args.predictor_checkpoints:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = strip_prefix(ck["predictor"] if "predictor" in ck else ck)
        cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
        tag = Path(ckpt_path).parent.name + "/" + Path(ckpt_path).name
        log(f"\n[ckpt] {tag}  epoch={ck.get('epoch','?')}  train_loss={ck.get('loss','?')}  "
            f"wd={cfg.get('weight_decay','?')} lr_proj={cfg.get('lr_proj','?')} "
            f"epochs={cfg.get('epochs','?')} clips/rec={cfg.get('clips_per_recording','?')}")
        for label, pred in GROUPS:
            nrm, npar = group_norm(sd, pred)
            # Biases initialise to exactly 0 (predictor._init_weights), so a
            # ratio against init is a division by zero dressed up as a huge number.
            # Those groups are reported as absolute norms only.
            ratio = nrm / base[label] if base[label] > 1e-8 else float("nan")
            rows.append({"checkpoint": tag, "group": label, "norm": nrm,
                         "n_params": npar, "ratio_to_init": ratio,
                         "epoch": ck.get("epoch"), "loss": ck.get("loss")})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}.csv", index=False)

    log("\n" + "=" * 88)
    log("T8 — parameter norm relative to initialisation (1.000 = unchanged)")
    log("=" * 88)
    piv = df.pivot(index="group", columns="checkpoint", values="ratio_to_init") \
            .reindex([g for g, _ in GROUPS])
    log(piv.to_string(float_format=lambda v: f"{v:9.4f}", na_rep="    n/a"))
    log("\n(n/a = the group initialises to exactly 0, so a ratio is undefined; absolutes below)")
    absn = df.pivot(index="group", columns="checkpoint", values="norm") \
             .reindex([g for g, _ in GROUPS])
    log("\nAbsolute Frobenius norms")
    log(absn.to_string(float_format=lambda v: f"{v:12.5f}"))

    log("\n" + "-" * 88)
    for ck in [c for c in piv.columns if c != "init"]:
        frozen = [piv.loc[g, ck] for g in ("blocks_0-17 [frozen]", "predictor_embed [frozen]")]
        ok = all(abs(f - 1.0) < 1e-6 for f in frozen)
        log(f"[control] {ck}: frozen groups at {frozen[0]:.6f}, {frozen[1]:.6f} "
            f"({'OK' if ok else 'NOT 1.000 — the checkpoint trained parameters it claims it did not'})")
        g = piv.loc["gaze_proj.weight", ck]
        h = piv.loc["hand_proj.weight", ck]
        ref = piv.loc["blocks_18-23", ck]
        head = piv.loc["predictor_proj", ck]
        log(f"          gaze_proj {g:.4f}  hand_proj {h:.4f}  vs decayed reference "
            f"blocks_18-23 {ref:.4f}, predictor_proj {head:.4f}")
        if g < 0.9 * ref:
            log("          gaze_proj shrank FASTER than parameters under the same decay — "
                "consistent with active suppression.")
        elif g > 1.05:
            log("          gaze_proj GREW against its initialisation. Training was building the "
                "pathway up, not suppressing it — the suppression story is not what happened.")
        else:
            log("          gaze_proj tracks the decayed reference. No evidence of suppression "
                "beyond what weight decay alone produces.")
    log("-" * 88)
    log("\nCaveat: --save-every was >= --epochs on every completed run, so these are endpoints,")
    log("not a trace. A trace over training needs a run that logs the norms at each step.")

    with open(f"{out}.json", "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2, default=str)
    log(f"[out] {out}.csv  {out}.json  {out}.log")
    log.close()


if __name__ == "__main__":
    main()
