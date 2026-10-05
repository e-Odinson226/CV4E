"""
Run logs: the log helper the commands write through, and one parser for the train.log
that `python -m ego train` writes. summarize, plot and watch all read runs through it.
"""

import re
from pathlib import Path


class Logger:
    """Print each line and append it to <out_prefix>.log."""

    def __init__(self, out_prefix):
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(f"{out_prefix}.log", "w")

    def __call__(self, msg):
        print(msg, flush=True)
        self._f.write(msg + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# train.log
#
# A step line, in the current and in the older format:
#   epoch 1  step   10/245  loss=0.5856  grad=0.132  lr_proj=1.00e-03  lr_blocks=1.00e-04  ETA=1162s
#   epoch 1  step   10/245 (   4%)  loss=0.5811  ema=0.5993  avg=0.5849  grad=  0.19  it/s=0.17  ...
# ---------------------------------------------------------------------------

STEP_RE = re.compile(r"epoch (\d+)\s+step\s+(\d+)/(\d+)\b")
KV_RE = re.compile(r"([\w/]+)=\s*([-+]?\d[\d.]*(?:e[-+]?\d+)?)")
EPOCH_RE = re.compile(r"\[epoch\s+(\d+)/(\d+)\]\s+loss=([\d.]+)(\*?).*?time=(\d+)s\s+ETA=(\d+)s")
VAL_RE = re.compile(
    r"\[val (\w+)\]\s+MSE_A\(masked\)=([\d.]+)\s+MSE_B\(real\)=([\d.]+)\s+Delta\(A-B\)=([-+\d.]+)"
)
CFG_RE = re.compile(r"INFO\s+  (\w[\w ]*?)\s{2,}(.+)$")
LAYER_RE = re.compile(r"\[layers\] (.+)$")


def parse_train_log(log_path):
    """
    Returns a dict:
      steps   [{epoch, step, total, loss, ema, grad, it_s, eta}]   ema/grad/it_s may be None
      epochs  [{epoch, total, loss, best, time, eta}]
      vals    [{tag, epoch, mse_A, mse_B, delta}]                epoch is None for a non-numeric tag
      config  {name: value} from the block after "Ego predictor fine-tuning"
      layers  the [layers] lines
      done    whether the run finished
    """
    out = {"steps": [], "epochs": [], "vals": [], "config": {}, "layers": [], "done": False}
    log_path = Path(log_path)
    if not log_path.exists():
        return out
    in_cfg = False
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Ego predictor fine-tuning" in line:
            in_cfg = True
            continue
        if in_cfg:
            if "===" in line and out["config"]:
                in_cfg = False
            else:
                m = CFG_RE.search(line)
                if m:
                    out["config"][m.group(1).strip()] = m.group(2).strip()
        m = LAYER_RE.search(line)
        if m:
            out["layers"].append(m.group(1).strip())
            continue
        m = STEP_RE.search(line)
        if m and "loss=" in line:
            kv = {k: float(v) for k, v in KV_RE.findall(line[m.end():])}
            out["steps"].append({"epoch": int(m.group(1)), "step": int(m.group(2)),
                                 "total": int(m.group(3)), "loss": kv["loss"],
                                 "ema": kv.get("ema"), "grad": kv.get("grad"),
                                 "it_s": kv.get("it/s"),
                                 "eta": int(kv["ETA"]) if "ETA" in kv else None})
            continue
        m = EPOCH_RE.search(line)
        if m:
            out["epochs"].append({"epoch": int(m.group(1)), "total": int(m.group(2)),
                                  "loss": float(m.group(3)), "best": m.group(4) == "*",
                                  "time": int(m.group(5)), "eta": int(m.group(6))})
            continue
        m = VAL_RE.search(line)
        if m:
            num = m.group(1).replace("epoch", "")
            out["vals"].append({"tag": m.group(1), "epoch": int(num) if num.isdigit() else None,
                                "mse_A": float(m.group(2)), "mse_B": float(m.group(3)),
                                "delta": float(m.group(4))})
            continue
        if "Training complete" in line:
            out["done"] = True
    return out
