"""
Live terminal dashboard for a running `python -m ego train` job.

Reads the run's train.log and renders a
live-updating view: training-loss sparkline, the per-epoch held-out
MSE_A / MSE_B / Delta table (the metric that matters), throughput, and GPU stats.

Works on an ALREADY-RUNNING job (it only reads the log), so no restart needed.

    python -m ego watch --dir checkpoints/ego_ft_v2
    python -m ego watch --dir checkpoints/ego_ft_v2 --once   # one snapshot, no live loop
"""

import argparse
import subprocess
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ego.runlog import parse_train_log

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(vals, width=48):
    if not vals:
        return ""
    v = vals[-width:]
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((x - lo) / rng * 7.999))] for x in v)


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4).stdout.strip().splitlines()[0]
        util, used, total = [x.strip() for x in out.split(",")]
        return f"GPU {util}% util   {int(used)/1024:.1f}/{int(total)/1024:.1f} GiB"
    except Exception:
        return "GPU n/a"


def parse(log_path):
    r = parse_train_log(log_path)
    steps = [dict(s, ema=s["ema"] if s["ema"] is not None else s["loss"])
             for s in r["steps"] if s["grad"] is not None]
    return steps, r["epochs"], r["vals"], r["done"]


def render(log_path):
    steps, epochs, vals, done = parse(log_path)

    # Header
    head = Text()
    head.append(f"{log_path}\n", style="dim")
    if steps:
        s = steps[-1]
        head.append(f"epoch {s['epoch']}  ", style="bold cyan")
        head.append(f"step {s['step']}/{s['total']}  ")
        if s["it_s"]:
            head.append(f"{s['it_s']:.2f} it/s  ", style="green")
        head.append(f"ETA {s['eta']}s", style="yellow")
        bar = ProgressBar(total=s["total"], completed=s["step"], width=40)
    else:
        head.append("waiting for first step…", style="yellow")
        bar = ProgressBar(total=1, completed=0, width=40)
    head.append(f"   {gpu_stats()}", style="magenta")
    if done:
        head.append("   ✓ COMPLETE", style="bold green")

    # Loss panel
    loss_body = Text()
    if steps:
        losses = [x["loss"] for x in steps]
        emas = [x["ema"] for x in steps]
        loss_body.append("loss  " + sparkline(losses) + "\n")
        loss_body.append(f"      last={losses[-1]:.4f}  ema={emas[-1]:.4f}  "
                         f"min={min(losses):.4f}  grad={steps[-1]['grad']:.1f}", style="dim")
    else:
        loss_body.append("—", style="dim")
    loss_panel = Panel(loss_body, title="train loss (per logged step)", border_style="blue")

    # Validation table — the key signal
    vt = Table(expand=True, title="held-out validation (P08): does the signal help?")
    vt.add_column("epoch"); vt.add_column("MSE_A (masked)", justify="right")
    vt.add_column("MSE_B (real)", justify="right"); vt.add_column("Δ = A−B", justify="right")
    vt.add_column("verdict")
    prev = None
    for v in vals:
        d = v["delta"]
        dcolor = "green" if d > 0 else "red"
        arrow = "" if prev is None else (" ↑" if d > prev else " ↓")
        vt.add_row(v["tag"], f"{v['mse_A']:.4f}", f"{v['mse_B']:.4f}",
                   Text(f"{d:+.4f}{arrow}", style=dcolor),
                   Text("signal helps" if d > 0 else "no help yet", style=dcolor))
        prev = d
    if vals:
        deltas = [v["delta"] for v in vals]
        vt.caption = "Δ trend: " + sparkline(deltas) + ("   (rising = learning to use gaze/hand)")

    return Group(Panel(Group(head, bar), border_style="cyan"), loss_panel, vt), done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="checkpoints/ego_ft_quick", help="run output dir")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    args = ap.parse_args()

    log_path = Path(args.dir) / "train.log"
    console = Console()

    if args.once:
        group, _ = render(log_path)
        console.print(group)
        return

    with Live(console=console, refresh_per_second=4, screen=False) as live:
        while True:
            group, done = render(log_path)
            live.update(group)
            if done:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
