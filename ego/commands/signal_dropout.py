"""
Test 2: does a model trained without signals do as well?

Compares, clip by clip, the model trained with signals (ego_ft_v2) and the matched model
trained without them (ego_sd1p0, --signal-dropout 1.0). It reads the per-clip errors that
stock-vs-tuned wrote for each model on the same 96 clips, so it needs no GPU.

Each contrast is a - b per clip. A positive mean means b predicts better.

  sd1.0 masked - ft_v2 real    the value of the information in the signals. This is the
                               paper's central number (+0.0003, p = 0.40)
  sd1.0 masked - ft_v2 masked  the two models with the signals hidden
  ft_v2 masked - ft_v2 real    the Delta of ego_ft_v2 on these clips
  sd1.0 real   - sd1.0 masked  real signals fed to the model that never saw them

The confidence intervals use 10000 bootstrap resamples (seed 0).

Usage
-----
    python -m ego stock-vs-tuned ... --predictor-checkpoint checkpoints/ego_ft_v2/best.pt \
        --out results/rung_b1
    python -m ego stock-vs-tuned ... --predictor-checkpoint checkpoints/ego_sd1p0/best.pt \
        --out results/rung_b1_sd1p0
    python -m ego signal-dropout
"""

import argparse
from pathlib import Path

import pandas as pd

from ego.stats import paired


def per_clip(df, arm):
    return df[df.arm == arm].sort_values("clip").mse.values


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-signals", default="results/rung_b1.csv",
                    help="stock-vs-tuned output for ego_ft_v2")
    ap.add_argument("--without-signals", default="results/rung_b1_sd1p0.csv",
                    help="stock-vs-tuned output for ego_sd1p0")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="results/signal_dropout_contrasts.csv")
    args = ap.parse_args()

    ft = pd.read_csv(args.with_signals)
    sd = pd.read_csv(args.without_signals)
    for name, df in (("with", ft), ("without", sd)):
        n = len(per_clip(df, "finetuned:mask"))
        if n != len(per_clip(df, "finetuned:real")) or n == 0:
            raise SystemExit(f"--{name}-signals: the mask and real arms differ in clip count")
    if len(per_clip(ft, "finetuned:mask")) != len(per_clip(sd, "finetuned:mask")):
        raise SystemExit("the two runs have different numbers of clips; they must use the same clips")

    contrasts = [
        ("sd1.0 masked - ft_v2 masked", per_clip(sd, "finetuned:mask"), per_clip(ft, "finetuned:mask")),
        ("sd1.0 masked - ft_v2 real",   per_clip(sd, "finetuned:mask"), per_clip(ft, "finetuned:real")),
        ("ft_v2 masked - ft_v2 real",   per_clip(ft, "finetuned:mask"), per_clip(ft, "finetuned:real")),
        ("sd1.0 real   - sd1.0 masked", per_clip(sd, "finetuned:real"), per_clip(sd, "finetuned:mask")),
    ]
    rows = [{"contrast": name, **paired(a, b, n_boot=args.n_boot)} for name, a, b in contrasts]
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:10.5f}"))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
