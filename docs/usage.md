# How to Run — Gaze/Hand Conditioning Pipeline

Everything needed to download data, fine-tune the ego predictor on gaze + hand
signals, evaluate it, and produce figures. For *why* the pipeline is built this
way see [experiment_design.md](experiment_design.md); for results so far see
[progress.md](progress.md).

---

## Environment

All commands use the project's conda Python:

```sh
PY=/mnt/data/home/zj2433/miniconda3/envs/VJEPA2-AC/bin/python
cd /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric
```

Notes:
- The env prints a harmless `timm` / `sdpa_kernel` `FutureWarning` on import. Filter it with
  `2>&1 | grep -v FutureWarning | grep -v "warnings.warn\|self.gen"` if you want clean logs.
- For GPU training, prefix with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid
  fragmentation on a shared GPU.
- ViT-g checkpoint: `data/model_checkpoints/vjepa2-ac-vitg.pt` (~11 GB).

---

## Key paths

| Path | What |
|---|---|
| `data/model_checkpoints/vjepa2-ac-vitg.pt` | Pretrained V-JEPA 2-AC checkpoint (encoder + predictor) |
| `data/epic-kitchen/ek100-hd/HD-EPIC/Videos/<P>/` | mp4s + `*_mp4_to_vrs_time_ns.csv` per participant |
| `data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze/<P>/GAZE_HAND/` | Aria MPS zips → gaze/hand CSVs |
| `checkpoints/<run>/` | `best.pt`, `epoch_NNN.pt`, `final.pt`, `train.log`, `metrics.jsonl` |
| `results/` | eval CSVs, run summaries, figures |

---

## The pipeline (end to end)

### 0. (One time) Download HD-EPIC

```sh
$PY data/epic-kitchen/hd-epic-downloader/hd-epic-downloader.py \
    data/epic-kitchen/ek100-hd --videos --slam-gaze --participants 1,2,3,4,5,6,7,8,9
```
Downloads videos + SLAM-and-Gaze (which contains the GAZE_HAND zips). Logs to
`…/HD-EPIC/download.log`. `--participants` takes integers 1–9; `--dry-run` to test.

### 1. Extract the gaze/hand CSVs from the zips

```sh
$PY scripts/extract_gaze_hand.py --gaze-dir data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze
```
Pulls `general_eye_gaze.csv` + `wrist_and_palm_poses.csv` out of each zip, in place.
Idempotent. Prints how many CSVs are present (currently 154 gaze / 154 hand; 2 P02
recordings have no MPS data).

### 2. (Optional) Baseline eval — before any fine-tuning

```sh
$PY scripts/eval_ego_mse.py \
    --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
    --video-dir  data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
    --gaze-dir   data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
    --participants P08 P09 --clips-per-video 20 \
    --out results/mse_baseline.csv
```
Sanity check of the untrained AC predictor (expect Δ ≈ 0 or slightly negative).

### 3. Fine-tune on P01–P07

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/finetune_ego.py \
    --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
    --video-dir  data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
    --gaze-dir   data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
    --participants P01 P02 P03 P04 P05 P06 P07 \
    --val-participants P08 --val-recordings 6 --val-clips 40 \
    --epochs 8 --clips-per-recording 60 \
    --batch-size 16 --encode-chunk 48 --num-workers 8 \
    --out-dir checkpoints/ego_finetune
```
Produces `checkpoints/ego_finetune/best.pt`. Logs `[layers]` (what's trained),
per-step loss, and a `[val]` line per epoch (held-out MSE_A / MSE_B / Δ).

### 4. Monitor (live, optional)

```sh
$PY scripts/watch_progress.py --dir checkpoints/ego_finetune
```
Live dashboard: loss sparkline, the MSE_A/MSE_B/Δ table, throughput, GPU stats.
Reads the log, so it works on an already-running job. `--once` for one snapshot.

### 5. Evaluate the fine-tuned model (paired A/B)

```sh
$PY scripts/eval_ego_mse.py \
    --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
    --predictor-checkpoint checkpoints/ego_finetune/best.pt \
    --video-dir  data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
    --gaze-dir   data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
    --participants P08 P09 --clips-per-video 40 \
    --out results/mse_finetuned.csv
```
For each clip computes MSE_A (signals masked) and MSE_B (real) on the *same* frames →
per-recording `mse_A, mse_B, delta`. Δ > 0 means the signals help.

### 6. Summaries + figures

```sh
$PY scripts/summarize_run.py --dir checkpoints/ego_finetune     # → results/<run>_summary.{json,md}
$PY scripts/plot_results.py  --dir checkpoints/ego_finetune     # → results/<run>_results.png
$PY scripts/plot_results.py  --dir run_a run_b                  # overlay Δ curves across runs
```

---

## What each script does

| Script | Purpose | Key args |
|---|---|---|
| `scripts/extract_gaze_hand.py` | Extract gaze/hand CSVs from MPS zips | `--gaze-dir`, `--participants` |
| `scripts/finetune_ego.py` | **Train** the predictor on gaze+hand | see table below |
| `scripts/eval_ego_mse.py` | **Paired A/B eval** (MSE_A vs MSE_B) | `--predictor-checkpoint`, `--participants`, `--clips-per-video` |
| `scripts/watch_progress.py` | Live training dashboard | `--dir`, `--once` |
| `scripts/summarize_run.py` | Run → JSON + Markdown report | `--dir`, `--out` |
| `scripts/plot_results.py` | Run → PNG figures | `--dir` (1+), `--out-dir` |
| `data/epic-kitchen/hd-epic-downloader/hd-epic-downloader.py` | Download HD-EPIC | `--videos`, `--slam-gaze`, `--participants` |

### `finetune_ego.py` arguments

| Arg | Default | Meaning |
|---|---|---|
| `--checkpoint` | — | AC checkpoint (architecture + pretrained weights) |
| `--video-dir`, `--gaze-dir` | — | HD-EPIC Videos / SLAM-and-Gaze dirs |
| `--participants` | P01–P07 | Training participants |
| `--out-dir` | checkpoints/ego_finetune | Where checkpoints + logs go |
| `--epochs` | 20 | Number of epochs |
| `--batch-size` | 8 | Clips per step (16 fits the GPU comfortably) |
| `--context-steps` | 8 | Temporal steps the predictor sees (world-model length) |
| `--frame-stride` | 8 | Frames between steps (8 @30fps ≈ 4 fps, matches droid) |
| `--clips-per-recording` | 200 | Random clips sampled per recording per epoch |
| `--signal-dropout` | 0.4 | Prob. of fully masking a clip's signals (fair Condition A) |
| `--unfreeze-last-n` | 6 | Transformer blocks to train (fewer ⇒ smaller checkpoint) |
| `--lr-proj` / `--lr-blocks` | 1e-3 / 1e-4 | LR for new projectors / unfrozen blocks |
| `--encode-chunk` | 16 | Frames per ViT-g encode sub-batch (raise to 32–64 for speed) |
| `--no-amp` | off | Disable bf16 autocast (fp32, ~2× slower) |
| `--no-normalize-reps` | off | Disable LayerNorm on reps (config trained with it ON) |
| `--no-standardize` | off | Disable gaze/hand input z-scoring |
| `--val-participants` | None | Held-out participants for per-epoch MSE_A/MSE_B/Δ |
| `--val-recordings` / `--val-clips` | 4 / 8 | Held-out recordings / clips per recording (fixed seed) |
| `--save-every` / `--log-every` | 5 / 10 | Checkpoint / log cadence |

### Checkpoint files (in each `--out-dir`)

- `best.pt` — lowest **training loss** epoch (note: by train loss, not by Δ).
- `epoch_NNN.pt` — periodic snapshot every `--save-every` epochs.
- `final.pt` — last epoch.
Each holds `{predictor state_dict, epoch, loss, config}`. `train.log` (human) and
`metrics.jsonl` (machine: per-step + per-epoch-val records) sit alongside.

---

## Library modules (imported, not run directly)

| Module | What |
|---|---|
| `scripts/ego_common.py` | Shared: `load_models` (encoder=target_encoder + ego predictor), `encode_independent` (per-frame ViT-g encoding, bf16, chunked), VRS-timestamp alignment, CSV discovery |
| `vjepa2/src/models/ego_predictor.py` | `VisionTransformerPredictorEgo` — gaze/hand replace action/state (`cond_tokens=2`) |
| `vjepa2/src/models/ego_finetune.py` | Freeze strategy, AC→ego weight transfer, `log_finetuned_layers` |
| `vjepa2/src/datasets/ego_loaders.py` | `GazeTokenLoader` / `HandTokenLoader` — parse MPS CSVs, align by VRS ns, z-score |

---

## Troubleshooting

- **CUDA OOM** → lower `--batch-size` or `--encode-chunk`; add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **`[dataset] with gaze CSV: 0/N`** → zips not extracted; run step 1 (`extract_gaze_hand.py`).
- **Condition B looks identical to A** → gaze not flowing; check `find_csvs` found the CSVs and that
  the recording's `*_mp4_to_vrs_time_ns.csv` exists.
- **Slow / low GPU use** → ensure bf16 is on (omit `--no-amp`) and raise `--encode-chunk`.
