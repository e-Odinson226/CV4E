"""
Summarize a training run into a JSON + Markdown report.

Parses train.log: config, per-epoch train loss + wall time, held-out
MSE_A / MSE_B / Delta, throughput, and total time.

    python -m ego summarize --dir checkpoints/ego_ft_v2
    python -m ego summarize --dir checkpoints/ego_ft_v2 --out results/ego_ft_v2
"""

import argparse
import json
from pathlib import Path

from ego.runlog import parse_train_log


def build(dir_path):
    r = parse_train_log(Path(dir_path) / "train.log")
    epochs, vals = r["epochs"], r["vals"]
    its = [s["it_s"] for s in r["steps"] if s["it_s"] is not None]

    # merge per-epoch train + val by index (val has epoch0 baseline first)
    val_by_epoch = {}
    for v in vals:
        k = v["tag"].replace("epoch", "")
        val_by_epoch[int(k) if k.isdigit() else k] = v

    rows = []
    base = val_by_epoch.get(0)
    if base:
        rows.append({"epoch": 0, "train_loss": None, "time_s": None, **{k: base[k] for k in ("mse_A", "mse_B", "delta")}})
    for e in epochs:
        v = val_by_epoch.get(e["epoch"], {})
        rows.append({"epoch": e["epoch"], "train_loss": e["loss"], "time_s": e["time"],
                     "mse_A": v.get("mse_A"), "mse_B": v.get("mse_B"), "delta": v.get("delta")})

    total_time = sum(e["time"] for e in epochs)
    summary = {
        "dir": str(dir_path),
        "config": r["config"],
        "trainable_layers": r["layers"],
        "epochs_completed": len(epochs),
        "epochs_planned": epochs[-1]["total"] if epochs else None,
        "avg_epoch_time_s": round(total_time / len(epochs), 1) if epochs else None,
        "total_train_time_s": total_time,
        "mean_it_s": round(sum(its) / len(its), 3) if its else None,
        "best_train_loss": min((e["loss"] for e in epochs), default=None),
        "delta_first": rows[0]["delta"] if rows else None,
        "delta_last": next((r["delta"] for r in reversed(rows) if r["delta"] is not None), None),
        "rows": rows,
    }
    return summary


def to_md(s):
    L = [f"# Run summary: {s['dir']}", ""]
    L.append(f"- epochs: **{s['epochs_completed']}/{s['epochs_planned']}**  "
             f"| avg epoch: **{s['avg_epoch_time_s']}s**  | total: **{s['total_train_time_s']}s**  "
             f"| throughput: **{s['mean_it_s']} it/s**")
    L.append(f"- best train loss: **{s['best_train_loss']}**")
    if s["delta_first"] is not None and s["delta_last"] is not None:
        L.append(f"- held-out Δ(A−B): **{s['delta_first']:+.4f} → {s['delta_last']:+.4f}**")
    L += ["", "## Trainable layers", ""]
    L += [f"- {x}" for x in s["trainable_layers"]]
    L += ["", "## Per-epoch", "", "| epoch | train_loss | time(s) | MSE_A | MSE_B | Δ(A−B) |",
          "|---|---|---|---|---|---|"]
    for r in s["rows"]:
        def f(x, p=".4f"):
            return "—" if x is None else (f"{x:{p}}" if isinstance(x, float) else str(x))
        L.append(f"| {r['epoch']} | {f(r['train_loss'])} | {f(r['time_s'],'.0f')} | "
                 f"{f(r['mse_A'])} | {f(r['mse_B'])} | {f(r['delta'],'+.4f')} |")
    L += ["", "## Config", ""]
    L += [f"- `{k}` = {v}" for k, v in s["config"].items()]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None, help="output path prefix (default results/<dirname>)")
    args = ap.parse_args()

    s = build(args.dir)
    out = args.out or f"results/{Path(args.dir).name}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out + "_summary.json").write_text(json.dumps(s, indent=2))
    Path(out + "_summary.md").write_text(to_md(s))
    print(to_md(s))
    print(f"\nSaved -> {out}_summary.json  and  {out}_summary.md")


if __name__ == "__main__":
    main()
