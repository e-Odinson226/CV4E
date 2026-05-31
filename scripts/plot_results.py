"""
Plot fine-tuning results into presentation-ready figures (PNG).

Reads metrics.jsonl (preferred) or train.log from a run dir and produces a
3-panel figure: (1) train loss, (2) held-out MSE_A vs MSE_B, (3) Δ = A−B with a
zero baseline. Pass multiple --dir to overlay the Δ curves for comparison.

    python scripts/plot_results.py --dir checkpoints/ego_ft_v2
    python scripts/plot_results.py --dir checkpoints/ego_ft_v2 checkpoints/ego_finetune
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAL_RE = re.compile(r"\[val \w*(\d+)\].*?MSE_A\(masked\)=([\d.]+).*?MSE_B\(real\)=([\d.]+).*?Delta\(A-B\)=([-+\d.]+)")
STEP_RE = re.compile(r"epoch (\d+)\s+step\s+(\d+)/(\d+).*?loss=([\d.]+)")


def load(run_dir):
    """Return (steps, vals). steps: list of (frac_epoch, loss); vals: list of dict(epoch,A,B,delta)."""
    d = Path(run_dir)
    steps, vals = [], []
    jl = d / "metrics.jsonl"
    if jl.exists():
        for line in jl.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("t") == "step":
                fe = (r["epoch"] - 1) + r["step"] / r["total_steps"]
                steps.append((fe, r.get("ema", r["loss"])))
            elif r.get("t") == "val":
                vals.append({"epoch": r["epoch"], "A": r["mse_A"], "B": r["mse_B"], "delta": r["delta"]})
    if not steps and not vals:  # fall back to train.log
        for line in (d / "train.log").read_text(errors="ignore").splitlines():
            m = STEP_RE.search(line)
            if m:
                fe = (int(m.group(1)) - 1) + int(m.group(2)) / int(m.group(3))
                steps.append((fe, float(m.group(4))))
            m = VAL_RE.search(line)
            if m:
                vals.append({"epoch": int(m.group(1)), "A": float(m.group(2)),
                             "B": float(m.group(3)), "delta": float(m.group(4))})
    return steps, vals


def plot_run(run_dir, out_path):
    steps, vals = load(run_dir)
    name = Path(run_dir).name
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(f"Gaze/Hand conditioning — {name}", fontsize=13, fontweight="bold")

    # 1) train loss
    ax = axes[0]
    if steps:
        xs, ys = zip(*steps)
        ax.plot(xs, ys, color="#1f77b4")
    ax.set_title("Training loss"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (feature pred.)")
    ax.grid(alpha=0.3)

    # 2) MSE_A vs MSE_B
    ax = axes[1]
    if vals:
        e = [v["epoch"] for v in vals]
        ax.plot(e, [v["A"] for v in vals], "o-", label="MSE_A (signals masked)", color="#d62728")
        ax.plot(e, [v["B"] for v in vals], "s-", label="MSE_B (real gaze+hand)", color="#2ca02c")
        ax.legend(fontsize=9)
    ax.set_title("Held-out prediction error (P08)"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)

    # 3) Delta
    ax = axes[2]
    if vals:
        e = [v["epoch"] for v in vals]; d = [v["delta"] for v in vals]
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.plot(e, d, "o-", color="#9467bd")
        ax.fill_between(e, d, 0, where=[x > 0 for x in d], color="green", alpha=0.15, interpolate=True)
        ax.fill_between(e, d, 0, where=[x <= 0 for x in d], color="red", alpha=0.15, interpolate=True)
        for x, y in zip(e, d):
            ax.annotate(f"{y:+.4f}", (x, y), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8)
    ax.set_title("Δ = MSE_A − MSE_B   (>0 ⇒ signals help)"); ax.set_xlabel("epoch"); ax.set_ylabel("Δ")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_compare(run_dirs, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="gray", lw=1, ls="--")
    for rd in run_dirs:
        _, vals = load(rd)
        if vals:
            ax.plot([v["epoch"] for v in vals], [v["delta"] for v in vals], "o-", label=Path(rd).name)
    ax.set_title("Δ = MSE_A − MSE_B across runs"); ax.set_xlabel("epoch"); ax.set_ylabel("Δ (>0 ⇒ signals help)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for rd in args.dir:
        out = Path(args.out_dir) / f"{Path(rd).name}_results.png"
        print("wrote", plot_run(rd, out))
    if len(args.dir) > 1:
        out = Path(args.out_dir) / "delta_compare.png"
        print("wrote", plot_compare(args.dir, out))


if __name__ == "__main__":
    main()
