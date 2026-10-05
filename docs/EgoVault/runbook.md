---
type: reference
status: running
created: 2026-09-18
updated: 2026-10-05
---

# Runbook

Commands, data paths, scripts, result files and checkpoints for the tests in [[4-results]].

## Environment

```sh
PY=/mnt/data/home/zj2433/miniconda3/envs/VJEPA2-AC/bin/python
cd /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric
```

- On a shared GPU, put `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the command.
- `vjepa2/` is an ordinary folder in this repo. Our model code is in
  `vjepa2/src/models/ego_predictor.py`, `vjepa2/src/models/ego_finetune.py` and
  `vjepa2/src/datasets/ego_loaders.py`.

## Data

| Path | Contents |
|---|---|
| `data/model_checkpoints/vjepa2-ac-vitg.pt` | Pretrained V-JEPA 2-AC with the ViT-g encoder |
| `data/epic-kitchen/ek100-hd/HD-EPIC/Videos/<P>/` | HD-EPIC videos, P01–P09 |
| `data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze/<P>/` | Gaze and hand CSV files |
| `data/online_calibration.jsonl` | Aria camera calibration. Not used yet. Needed for Coord-PE. |
| `data/ek100/videos/` | EK100 videos |

154 of the 156 HD-EPIC recordings have gaze and hand data. The two without are from P02. The
scripts skip them.

To download and extract the data:

```sh
$PY data/epic-kitchen/hd-epic-downloader/hd-epic-downloader.py \
    data/epic-kitchen/ek100-hd --videos --slam-gaze --participants 1,2,3,4,5,6,7,8,9
$PY scripts/extract_gaze_hand.py --gaze-dir data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze
```

## Training

This command trains `ego_ft_v2`, the model used in most tests:

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/finetune_ego.py \
    --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
    --video-dir  data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
    --gaze-dir   data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
    --participants P01 P02 P03 P04 P05 P06 P07 \
    --val-participants P08 --val-recordings 4 --val-clips 24 \
    --epochs 3 --clips-per-recording 30 --batch-size 16 \
    --num-workers 8 --encode-chunk 48 --save-every 3 \
    --out-dir checkpoints/ego_ft_v2
```

- `ego_sd1p0` uses the same command with `--signal-dropout 1.0 --out-dir checkpoints/ego_sd1p0`.
  It took 38 minutes.
- The planned full run uses `--epochs 8 --clips-per-recording 60 --val-recordings 6
  --val-clips 40`. That gives 240 test clips.
- `--shuffle-signals time` or `--shuffle-signals batch` trains with real signals from the
  wrong time or the wrong clip.
- Each run writes `train.log`, `metrics.jsonl` and checkpoints to `--out-dir`.
- `$PY scripts/watch_progress.py --dir <out-dir>` shows a live view of a running job.

## Test scripts

Most probe scripts take the same base arguments:

```sh
--checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
--predictor-checkpoint checkpoints/ego_ft_v2/best.pt \
--video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
--gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze
```

Without `--predictor-checkpoint`, they use the model before fine-tuning, with random gaze and
hand layers.

| Test | Script | Results |
|---|---|---|
| T2 | `scripts/finetune_ego.py` (checks after each epoch) | `checkpoints/ego_ft_v2/train.log` |
| T4 | `scripts/gaze_recoverability.py --split participant` (or `recording`, `random`) | `results/gaze_recov_*.csv`. The encoder features are cached in `results/gaze_features.npz`. |
| T5 | `scripts/control_recoverability.py` | `results/control_recoverability.csv` |
| T6 | `scripts/probe_sensitivity.py` | `results/sensitivity_ego_ft_v2.csv`, `results/sensitivity_untrained.csv` |
| T7 | `scripts/probe_attention_mass.py` | `results/attention_mass.csv`, `results/attention_mass_untrained.csv` |
| T8 | `scripts/probe_weight_norms.py` | `results/weight_norms.csv` |
| T10 | `scripts/eval_rung_b1.py` | `results/rung_b1.csv`, `results/rung_b1_contrasts.csv` |
| T11 | `scripts/eval_rung_b1.py` with `--predictor-checkpoint checkpoints/ego_sd1p0/best.pt` | `results/rung_b1_sd1p0.csv`. The paired comparison is in `results/signal_dropout_contrasts.csv`. |

- T3, T9 and T12 use Parsa's code. It is not in this repo.
- T4 and T5 use only the frozen encoder. T5 reads the features cached by T4.
- `probe_sensitivity.py`, `probe_attention_mass.py` and `eval_rung_b1.py` use the same P08
  clips as the checks during training (seed 12345).
- `scripts/eval_ego_mse.py` runs the same paired comparison on any participants. It samples
  its own clips, so its numbers differ a little from the 96-clip set.

## Checkpoints

| Folder | Contents |
|---|---|
| `checkpoints/ego_ft_v2/` | `best.pt` (epoch 3). The model used in most tests. |
| `checkpoints/ego_sd1p0/` | `best.pt`, `epoch_003.pt` and `final.pt`. The model trained without signals. |
| `checkpoints/ego_finetune/`, `ego_finetuned_p01_07/`, `ego_ft_quick/` | Logs only, from runs that were stopped. |

Each checkpoint file is 1.22 GB. The trainable weights alone are about 150 MB.

## Practical notes

- If the GPU runs out of memory, lower `--encode-chunk`.
- The frozen encoder runs in bf16 by default. This doubles the speed. `--no-amp` turns it off.
- Video decoding needs a lot of RAM. Parsa's EK100 runs needed `num_workers: 4` and
  `pin_memory: false`.
- `/mnt/data` is shared and has been full before. Write long outputs to scratch space first.
- Scripts that build an untrained model set the random seed. Without it, the random gaze and
  hand layers differ between runs.
