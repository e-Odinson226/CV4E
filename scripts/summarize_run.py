"""
Summarize a finetune_ego.py run into a JSON + Markdown report.

Parses train.log (and metrics.jsonl if present): config, per-epoch train loss +
wall time, held-out MSE_A / MSE_B / Delta, throughput, and total time.

    python scripts/summarize_run.py --dir checkpoints/ego_ft_quick
    python scripts/summarize_run.py --dir checkpoints/ego_ft_quick --out results/ego_ft_quick
"""

import argparse
import json
import re
from pathlib import Path

STEP_RE = re.compile(r"epoch (\d+)\s+step\s+(\d+)/(\d+).*?loss=([\d.]+).*?(?:it/s=([\d.]+))?.*?ETA=(\d+)s")
EPOCH_RE = re.compile(r"\[epoch\s+(\d+)/(\d+)\]\s+loss=([\d.]+)(\*?).*?time=(\d+)s\s+ETA=(\d+)s")
VAL_RE = re.compile(r"\[val (\w+)\]\s+MSE_A\(masked\)=([\d.]+)\s+MSE_B\(real\)=([\d.]+)\s+Delta\(A-B\)=([-+\d.]+)")
CFG_RE = re.compile(r"INFO\s+  (\w[\w ]*?)\s{2,}(.+)$")
LAYER_RE = re.compile(r"\[layers\] (.+)$")


def parse(log_path):
    cfg, epochs, vals, layers, steps_its = {}, [], [], [], []
    in_cfg = False
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Ego predictor fine-tuning" in line:
            in_cfg = True; continue
        if in_cfg:
            if "===" in line and cfg:
                in_cfg = False
            else:
                m = CFG_RE.search(line)
                if m:
                    cfg[m.group(1).strip()] = m.group(2).strip()
        m = LAYER_RE.search(line)
        if m: layers.append(m.group(1).strip())
        m = EPOCH_RE.search(line)
        if m:
            epochs.append({"epoch": int(m.group(1)), "total": int(m.group(2)),
                           "train_loss": float(m.group(3)), "best": m.group(4) == "*",
                           "time_s": int(m.group(5))})
        m = VAL_RE.search(line)
        if m:
            vals.append({"tag": m.group(1), "mse_A": float(m.group(2)),
                         "mse_B": float(m.group(3)), "delta": float(m.group(4))})
        m = STEP_RE.search(line)
        if m and m.group(5):
            steps_its.append(float(m.group(5)))
    return cfg, epochs, vals, layers, steps_its


def build(dir_path):
    log_path = Path(dir_path) / "train.log"
    cfg, epochs, vals, layers, its = parse(log_path)

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
        rows.append({"epoch": e["epoch"], "train_loss": e["train_loss"], "time_s": e["time_s"],
                     "mse_A": v.get("mse_A"), "mse_B": v.get("mse_B"), "delta": v.get("delta")})

    total_time = sum(e["time_s"] for e in epochs)
    summary = {
        "dir": str(dir_path),
        "config": cfg,
        "trainable_layers": layers,
        "epochs_completed": len(epochs),
        "epochs_planned": epochs[-1]["total"] if epochs else None,
        "avg_epoch_time_s": round(total_time / len(epochs), 1) if epochs else None,
        "total_train_time_s": total_time,
        "mean_it_s": round(sum(its) / len(its), 3) if its else None,
        "best_train_loss": min((e["train_loss"] for e in epochs), default=None),
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
