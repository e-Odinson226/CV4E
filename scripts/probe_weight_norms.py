"""
A3 projector weight-norm trace — was the gaze pathway suppressed during training?

THE FAILURE MODE BEING TESTED (adoption plan step 15, measuring-signal-use.md A3)
---------------------------------------------------------------------------------
[[gazeqwen]] describes it: a fresh module injects structured noise into a pretrained
network, and the fastest way for training to reduce the damage is to suppress the new
channel — drive its weights toward zero — after which the pathway may never come back.
If that happened here, ||gaze_proj.weight|| shrank relative to its initialisation.

THE CONFOUND, AND HOW IT IS HANDLED
-----------------------------------
finetune_ego.py runs AdamW at --weight-decay 1e-2 over ALL trainable parameters, so
every trainable weight shrinks whether or not the data asks for it. An absolute
shrinkage of gaze_proj therefore means nothing on its own.

So every norm is reported three ways:

  ratio_to_init   — raw shrinkage, uninterpretable alone
  vs reference    — the same ratio for parameters that were trained under the SAME
                    decay and the SAME schedule (the unfrozen blocks 18-23 and the
                    output head). Suppression is gaze_proj shrinking MORE than these.
  frozen control  — predictor_embed and blocks 0-17 were never trained, so their
                    ratio must be exactly 1.000. Anything else means the checkpoint
                    is not what it claims to be, and nothing else here is readable.

WHAT THIS CAN AND CANNOT SHOW
-----------------------------
Only three checkpoints of the completed run exist and all three are at epoch 3
(--save-every 3 with --epochs 3), so there is no intra-training trace. This gives
init -> epoch 3 endpoints, plus whatever the abandoned runs happen to add. A real
per-step trace is A4 and needs the rerun to log it.

The initialisation reference is rebuilt by running the same construction path with
the same seed. That reference is robust even if the RNG stream has drifted: the
Frobenius norm of a 3072-element i.i.d. draw is concentrated to about 1.3%, so the
init norm is effectively deterministic regardless of seed.

Usage
-----
    $PY scripts/probe_weight_norms.py \
        --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --predictor-checkpoints checkpoints/ego_ft_v2/best.pt checkpoints/ego_ft_v2/final.pt \
        --out results/weight_norms

CPU-only by default so it can run beside a GPU job.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from ego_common import load_models, strip_prefix


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

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(f"{out}.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    # Rebuild the exact construction path finetune_ego.main() takes, so the
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
            # Biases initialise to exactly 0 (ego_predictor._init_weights), so a
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
    log("A3 — parameter norm relative to initialisation (1.000 = unchanged)")
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
    log("not a trace. When it dies mid-pathway matters (A4) and needs per-step logging in the rerun.")

    with open(f"{out}.json", "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2, default=str)
    log(f"[out] {out}.csv  {out}.json  {out}.log")
    logf.close()


if __name__ == "__main__":
    main()
