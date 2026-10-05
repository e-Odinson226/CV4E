# CV4Egocentric

Code for Erfan Yekehzare's master's thesis, part of EgoProject 2026 (University of Rostock and
Babeș-Bolyai University, Cluj). The project asks whether a video world model predicts
egocentric video better when it also receives the person's gaze and hand positions.

The model is Meta's V-JEPA 2-AC with the ViT-g encoder. Its predictor normally receives a
robot's action and state for each frame. Here it receives a gaze token and a hand token
instead. The data is HD-EPIC: kitchen recordings made with Aria glasses, with gaze and hand
tracking.

So far, the gaze and hand inputs do not improve the prediction. A model trained without them
does as well (Δ = +0.0003, p = 0.40). These models were trained for 3 epochs. A full training
run has not been done yet. The thesis notes, with every test and its result, are in
[`docs/EgoVault/`](docs/EgoVault/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Training, evaluation and test scripts. |
| `ego/` | Our model code: the ego predictor, its fine-tuning helpers, and the gaze and hand loaders. |
| `vjepa2/` | Meta's V-JEPA 2 code. Not tracked: clone it here (see Setup). |
| `docs/EgoVault/` | The thesis notes, an Obsidian vault. The paper as submitted and the literature notes are in `docs/EgoVault/papers/`. |
| `notebooks/` | `playground.ipynb`: a first look at the Aria gaze data and at a V-JEPA 2 model from Hugging Face. |
| `data/`, `checkpoints/`, `results/` | Datasets, trained models and result files. Not tracked. |
| `archive/` | Old files. Not tracked. |

## Who wrote the code

- Erfan wrote everything in `scripts/` and `ego/`. Ioana wrote the first gaze and hand
  projection layers. Erfan moved them into the predictor and extended them to all time steps.
- Meta wrote V-JEPA 2, which the code imports from `vjepa2/`. Its licenses are in that
  repository.
- Parsa ran tests T1, T3, T9 and T12 with Parsa's own code. That code is not in this
  repository. The code in Appendix C of the paper is also from Parsa's codebase.

## Setup

The code needs Meta's V-JEPA 2 repository in `vjepa2/`. This work used commit `204698b`
(23 March 2026). Clone it from the repository root:

```sh
git clone https://github.com/facebookresearch/vjepa2 vjepa2
git -C vjepa2 checkout 204698b
```

The environment is the conda environment `VJEPA2-AC`, with Python 3.12. It has the packages in
`vjepa2/requirements.txt`, plus `scipy`, `matplotlib` and `rich`.

```sh
PY=/mnt/data/home/zj2433/miniconda3/envs/VJEPA2-AC/bin/python
cd /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric
```

- Run every script from the repository root. The default input and output paths are relative
  to it.
- The scripts add the repository root and `vjepa2/` to the Python path themselves. Neither
  needs to be installed.
- On a shared GPU, put `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the command.

### Data

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

## How the code fits together

Training and every evaluation script handle a clip in the same steps:

1. **Models.** `ego_common.load_models` reads the V-JEPA 2-AC checkpoint. It builds the ViT-g
   encoder from the EMA target encoder weights and freezes it. It builds the ego predictor and
   copies in all AC predictor weights except the action, state and extrinsics encoders.
2. **Frames.** `ego_common.load_frames` reads the frames of a clip from the mp4 file. The
   scripts take every 8th frame, about 4 frames per second, and use 8 such frames as context.
   Each frame is resized to 256 × 256 pixels and normalized with the ImageNet mean and
   standard deviation.
3. **Signals.** `ego_common.find_csvs` finds a recording's gaze and hand CSV files.
   `ego_common.find_ts_csv` finds the file that gives each video frame's Aria (VRS) timestamp.
   `GazeTokenLoader` and `HandTokenLoader` take the sample nearest to each frame's timestamp,
   scale it, and return a flag that says whether it is valid.
4. **Encoding.** `ego_common.encode_independent` encodes each frame on its own, as a 2-frame
   tubelet of the same image, and layer-normalizes the result. Each frame gives 256 tokens.
5. **Prediction.** The predictor turns the gaze and hand values into one token each. A missing
   signal is replaced by a learned mask token. Each frame becomes [gaze, hand, 256 image
   tokens]. The 24 transformer blocks attend causally over frames. The two signal tokens are
   then dropped, and the predictor returns 256 predicted image tokens per frame.
6. **Error.** The prediction for step t is compared with the encoding of step t + 1. Both are
   layer-normalized. The error is the mean squared error.

All GPU scripts import `scripts/ego_common.py`, so training and testing align, encode and
normalize clips in the same way. Some scripts also import from each other:

- `eval_ego_mse.py` provides the paired evaluation of one clip and the loading of a fine-tuned
  checkpoint. `finetune_ego.py` uses it for the checks after each epoch. The probe scripts use
  it to load models.
- `probe_sensitivity.py` provides the clip sampler. `probe_attention_mass.py` and
  `eval_rung_b1.py` use it too.
- `gaze_recoverability.py` provides the frame sampler and the ridge regression probe.
  `control_recoverability.py` reuses both.

## Model code in `ego/`

| File | Contents |
|---|---|
| `ego_predictor.py` | `VisionTransformerPredictorEgo` and its builder `vit_ego_predictor`. The blocks are the same as in the AC predictor. `gaze_proj` (3 → 1024) and `hand_proj` (12 → 1024) replace the action and state encoders. `gaze_mask` and `hand_mask` are the learned tokens for missing signals. |
| `ego_finetune.py` | `load_ac_weights_into_ego` copies the pretrained AC weights. `freeze_for_ego_finetune` freezes everything except the new layers, the last 6 blocks and the output layers. `get_ego_finetune_param_groups` gives the new layers and the pretrained layers separate learning rates. The other functions list and log the trainable parameters. |
| `ego_loaders.py` | `GazeTokenLoader` (yaw, pitch, depth) and `HandTokenLoader` (wrist and palm positions of both hands). The scaling constants `GAZE_MEAN`, `GAZE_STD` and `HAND_STD` are rough guesses. They need to be recomputed from P01–P07 before the next training run. |

What is trained, and with which learning rates, is in `docs/EgoVault/3-method.md`.

## Scripts

### Shared code and data preparation

| Script | What it does |
|---|---|
| `ego_common.py` | Model loading, per-frame encoding, frame reading, and lookup of the timestamp and CSV files. It is imported by the other scripts and not run directly. |
| `extract_gaze_hand.py` | Extracts the gaze and hand CSV files from each recording's `mps_<rec>_vrs.zip`, in place. It skips zip files that are already extracted, empty or damaged. |

### Training

| Script | What it does |
|---|---|
| `finetune_ego.py` | Fine-tunes the ego predictor. After each epoch it measures Δ on held-out clips. It writes `train.log`, `metrics.jsonl` and checkpoints to `--out-dir`. |

### Evaluation

| Script | Test | What it does |
|---|---|---|
| `eval_ego_mse.py` | | The paired comparison on any participants: the error with the signals hidden and with real signals, on the same clips. It writes `results/mse_paired.csv`. |
| `eval_rung_b1.py` | T10, T11 | Scores the predictor before and after fine-tuning on the same clips, with real signals and with each of the three "no signal" inputs (mask token, zeros, average). |

### Probes on the frozen encoder

| Script | Test | What it does |
|---|---|---|
| `gaze_recoverability.py` | T4 | A linear probe that predicts gaze from the frozen encoder's features. It caches the features in `results/gaze_features.npz`. |
| `control_recoverability.py` | T5 | The same probe with palm position as the target. It reads the features cached by T4. |

### Probes on the trained predictor

| Script | Test | What it does |
|---|---|---|
| `probe_sensitivity.py` | T6 | Measures how much the prediction changes when the gaze or hand input changes. |
| `probe_attention_mass.py` | T7 | Measures how much attention goes to the gaze and hand tokens. |
| `probe_weight_norms.py` | T8 | Divides the norm of each trained parameter by its norm at the start of training. |

### Monitoring and reports

| Script | What it does |
|---|---|
| `watch_progress.py` | A live terminal view of a running training job. It only reads the job's log. |
| `summarize_run.py` | Writes a JSON and a Markdown summary of a training run, to `results/<run>_summary.json` and `.md`. |
| `plot_results.py` | Plots the training loss, the held-out errors and Δ to `results/<run>_results.png`. With several `--dir` arguments it also writes `results/delta_compare.png`. |

### Self-tests

These need no GPU and no data.

| Script | What it checks |
|---|---|
| `test_gaze_recoverability.py` | The numerical parts of `gaze_recoverability.py`. |
| `test_shuffle_signals.py` | That `--shuffle-signals` breaks the match between signals and frames and changes nothing else. |

```sh
$PY scripts/test_gaze_recoverability.py
$PY scripts/test_shuffle_signals.py
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
- `$PY scripts/watch_progress.py --dir <out-dir>` shows a live view of a running job.

## Running the tests

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

- T1, T3, T9 and T12 use Parsa's code. It is not in this repository.
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
- In `finetune_ego.py` and `gaze_recoverability.py`, the frozen encoder runs in bf16 by
  default. This doubles the speed. `--no-amp` turns it off. `probe_sensitivity.py` uses bf16
  only with `--amp`.
- Video decoding needs a lot of RAM. Parsa's EK100 runs needed `num_workers: 4` and
  `pin_memory: false`.
- `/mnt/data` is shared and has been full before. Write long outputs to scratch space first.
- Scripts that build an untrained model set the random seed. Without it, the random gaze and
  hand layers differ between runs.

## Notes

The thesis notes are in `docs/EgoVault/`. Start with `docs/EgoVault/README.md`.
`3-method.md` describes the method in detail. `4-results.md` lists every test, who ran it, and
what it showed.
