# CV4Egocentric

Code for a project that asks whether a video world model predicts egocentric video better when
it also receives the person's gaze and hand positions.

The model is built from Meta's V-JEPA 2-AC: its frozen ViT-g encoder and the pretrained blocks
of its predictor. V-JEPA 2-AC's predictor receives a robot's action and state for each frame.
Here new input layers give it a gaze token and a hand token instead, and its last 6 blocks are
fine-tuned. The data is HD-EPIC: kitchen recordings made with Aria glasses, with gaze and hand
tracking.

So far, the gaze and hand inputs do not improve the prediction. A model trained without them
does as well (Δ = +0.0003, p = 0.40). These models were trained for 3 epochs. A full training
run has not been done yet. The project notes, with every test and its result, are in
`docs/` and detailed in the **Project Notes** section below.

## Repository layout

| Path | Contents |
|---|---|
| `ego/` | The code: the model, the data handling, and one command for training and for each test. |
| `tests/` | Self-tests. They need no GPU and no data. |
| `vjepa2/` | Meta's V-JEPA 2 code. Not tracked: clone it here (see Setup). |
| `docs/` | The project notes, an Obsidian vault. The paper as submitted and the literature notes are in `docs/papers/`. |
| `notebooks/` | `playground.ipynb`: a first look at the Aria gaze data and at a V-JEPA 2 model from Hugging Face. |
| `data/`, `checkpoints/` | Datasets and trained models. Not tracked. |
| `results/` | Result files. The small summaries (CSV, JSON, logs, figures) are tracked; feature caches (`*.npz` at the top level, `cache/` folders) and the per-sample tables of the gaze checks are not. |
| `archive/` | Old files. Not tracked. |

## Code from elsewhere

- Meta's V-JEPA 2, which the code imports from `vjepa2/`. Its licenses are in that repository.
- The paper's Tests 5, 6 and 7, part of its Test 1, and the code in its Appendix C come from a
  separate codebase that is not in this repository.

## Setup

The code needs Meta's V-JEPA 2 repository in `vjepa2/`. This work used commit `204698b`
(23 March 2026). Clone it from the repository root:

```sh
git clone https://github.com/facebookresearch/vjepa2 vjepa2
git -C vjepa2 checkout 204698b
```

The environment is the conda environment `VJEPA2-AC`, with Python 3.12. It has the packages in
`vjepa2/requirements.txt`, plus `scipy`, `matplotlib` and `rich`, and `projectaria-tools`
2.3.0 for the camera model of the glasses. Install it with
`pip install --no-deps projectaria-tools==2.3.0`: its full dependencies add a viewer and
notebook widgets, and they would downgrade `ipykernel` and `pillow`.

```sh
PY=/mnt/data/home/zj2433/miniconda3/envs/VJEPA2-AC/bin/python
cd /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric
$PY -m ego            # lists the commands
```

- Run every command from the repository root, as `$PY -m ego <command>`. The default input
  and output paths are relative to the root.
- The `ego` package puts `vjepa2/` on the Python path itself. Nothing needs to be installed.
- On a shared GPU, put `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the command.

### Data

| Path | Contents |
|---|---|
| `data/model_checkpoints/vjepa2-ac-vitg.pt` | Pretrained V-JEPA 2-AC with the ViT-g encoder |
| `data/epic-kitchen/ek100-hd/HD-EPIC/Videos/<P>/` | HD-EPIC videos, P01–P09 |
| `data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze/<P>/` | Gaze and hand CSV files |
| `data/online_calibration.jsonl` | Calibration of an Aria Gen 2 device, from the AriaGen2 pilot data. It does not apply to HD-EPIC, which was recorded with Aria Gen 1 glasses. |
| `data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze/<P>/SLAM/multi/<n>.zip` | HD-EPIC SLAM output, one zip per group of recordings. Downloaded for P04, P05 (in part), P08 and P09; empty placeholders for the others. `vrs_to_multi_slam.json` gives the group of each recording. |
| `data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze/<P>/SLAM/calibration/<n>.jsonl` | The camera calibration of each group, every 100th record of `slam/online_calibration.jsonl`. Written by `fetch-calibrations`. All 153 groups, covering 153 of the 156 recordings; `P06-20240510-140459` and the two P02 recordings without gaze have none. Needed to project gaze into the image. |
| `data/epic-kitchen/ek100-hd/HD-EPIC/annotations/` | HD-EPIC's annotations: object movements with boxes, gaze priming, narrations, recipes, the VQA benchmark. A clone of https://github.com/hd-epic/hd-epic-annotations at commit `aa1f833` (`git clone --depth 1`). |
| `data/ek100/videos/` | EK100 videos |

154 of the 156 HD-EPIC recordings have gaze and hand data. The two without are from P02. The
test commands skip them. `train` keeps them and trains on their clips with the signals hidden.

For P01–P03, gaze and hand are recorded at 10 Hz. For P04–P09, gaze is recorded at 60 Hz and
hand at 30 Hz.

To download and extract the data:

```sh
$PY data/epic-kitchen/hd-epic-downloader/hd-epic-downloader.py \
    data/epic-kitchen/ek100-hd --videos --slam-gaze --participants 1,2,3,4,5,6,7,8,9
$PY -m ego extract-csvs --gaze-dir data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze
```

## How the code is organized

`ego/` has two kinds of module. The library modules hold the code that more than one
command uses. Each command, in `ego/commands/`, is one thing you run.

| Module | Contents |
|---|---|
| `predictor.py` | `VisionTransformerPredictorEgo` and its builder `vit_ego_predictor`. The blocks are the same as in the AC predictor. `gaze_proj` (3 → 1024) and `hand_proj` (12 → 1024) replace the action and state encoders. `gaze_mask` and `hand_mask` are the learned tokens for missing signals. `gaze_form` sets how gaze enters (Test 9): `angles` (as in `ego_ft_v2`), `pe` (sine and cosine features of the gaze point, scaled to a joint length of 1, plus depth and a flag: 22 inputs to `gaze_proj`), `rope` (the gaze token placed at the gaze point in the attention) or `pe+rope`. |
| `gaze_attention.py` | `GazeRoPEAttention`: V-JEPA 2-AC's attention with one change, the gaze token rotated by the row and column of its gaze point instead of the top-left patch. Used by the `rope` forms. With every position at 0 it equals the upstream attention exactly. |
| `ego_predictor.py` | The predictor of Test 11, trained from the start. `EgoPredictor` adds `exp(-d^2 / 2 sigma^2)` maps of the gaze point and of each palm to the embedded image tokens, so the signals act on the attention between image tokens instead of competing for it as a token of their own. The vectors `e_gaze`, `e_left` and `e_right` start at zero, so an untrained predictor is exactly the matched model, and they still receive a gradient on the first step. Frames to predict are mask tokens at their own time slots, given to RoPE through explicit token ids, so several horizons are predicted in one pass. `signal_strength` reports `alpha * ||e||` per signal. |
| `model.py` | `load_models` builds the frozen ViT-g encoder and the ego predictor from the V-JEPA 2-AC checkpoint, and copies in the AC weights. `load_finetuned` loads a checkpoint written by `train`. `freeze_for_ego_finetune` and `get_ego_finetune_param_groups` choose what trains and at which learning rate. `encode_independent` encodes each frame on its own. `load_trained_predictor` loads a trained predictor with the gaze form saved in its checkpoint, frozen, in eval mode. `build_predictor` builds the predictor alone. |
| `data.py` | `find_recordings` lists recordings in a fixed order, which every seeded sampler depends on. `GazeTokenLoader` and `HandTokenLoader` read the gaze and hand CSVs. `load_frames` reads video frames, and `read_vrs_times` reads each frame's timestamp. The scaling constants `GAZE_MEAN`, `GAZE_STD` and `HAND_STD` are rough guesses. They need to be recomputed from P01–P07 before the next training run. |
| `signals.py` | The gaze and hand inputs of a clip: reading them at the frame timestamps, and every variant the tests use (hidden, zero, average, shifted, swapped with another clip, shuffled in time). The gaze vector holds yaw, pitch, depth and the gaze point (column, row on the patch grid); the point is (-1, -1) when gaze is invalid or uncalibrated. |
| `clips.py` | `fixed_clips`: the fixed evaluation clips (seed 12345) shared by the checks during training, Test 2, and Tests 4a and 4b. With the defaults these are the 96 P08 clips. Each clip also carries the signals one step later (`sig_next`), for the future control of Test 9; the sampling is unchanged. `paired_mse` scores one clip with the signals hidden and with the real signals. |
| `stats.py` | `paired`: the paired comparison behind every Δ. It gives the mean difference, a bootstrap 95% confidence interval and the Wilcoxon p-value. `by_recording`: a mean with a bootstrap 95% interval that resamples whole recordings. |
| `linprobe.py` | The linear probe of Test 3: ridge regression, channel PCA, gaze and palm lookups, and the seeded frame sampler. `point_features` averages the patch features around a point of the grid, for Test 8. The palm control uses the sampler to rebuild the gaze probe's rows from its cache. |
| `gaze_geometry.py` | `GazeProjector` turns gaze (yaw, pitch, depth) of one recording into the pixel where the person looks, in the upright 1408-pixel frame, with the recording's camera calibration and `projectaria_tools`. `to_patch` gives the position on the 16 × 16 patch grid. `projector_for` builds the projector of a recording. |
| `annotations.py` | `object_movements` reads HD-EPIC's object movements: one row per movement of an object, from pick-up to put-down, with the box around the object in the frame where the movement starts and in the frame where it ends. `noun_classes` and `object_class` give each object's free-form name its HD-EPIC noun class ("spoon2" → spoon). It does not read the narrations, which are a pickle file. |
| `runlog.py` | `Logger` prints each line and writes it to `<out>.log`. `parse_train_log` reads a `train.log` for `summarize`, `plot` and `watch`. |

What is trained, and with which learning rates, is in `docs/3-method.md`.

### How a clip is processed

Training and every evaluation command handle a clip in the same steps:

1. **Models.** `model.load_models` reads the V-JEPA 2-AC checkpoint. It builds the ViT-g
   encoder from the EMA target encoder weights and freezes it. It builds the ego predictor and
   copies in all AC predictor weights except the action, state and extrinsics encoders.
2. **Frames.** `data.load_frames` reads the frames of a clip from the mp4 file. The commands
   take every 8th frame, about 4 frames per second, and use 8 such frames as context. Each
   frame is resized to 256 × 256 pixels and normalized with the ImageNet mean and standard
   deviation.
3. **Signals.** `data.find_recordings` finds a recording's gaze and hand CSV files and the file
   that gives each video frame's Aria (VRS) timestamp. `signals.read` takes, through
   `GazeTokenLoader` and `HandTokenLoader`, the sample nearest to each frame's timestamp,
   scaled, with a flag that says whether it is valid.
4. **Encoding.** `model.encode_independent` encodes each frame on its own, as a 2-frame tubelet
   of the same image, and layer-normalizes the result. Each frame gives 256 tokens.
5. **Prediction.** The predictor turns the gaze and hand values into one token each. A missing
   signal is replaced by a learned mask token. Each frame becomes [gaze, hand, 256 image
   tokens]. The 24 transformer blocks attend causally over frames. The two signal tokens are
   then dropped, and the predictor returns 256 predicted image tokens per frame.
6. **Error.** The prediction for step t is compared with the encoding of step t + 1. Both are
   layer-normalized. The error is measured in two ways: the mean squared difference (MSE) and
   the mean absolute difference (L1). Training minimizes one of them (`--loss`, L1 by default,
   as in V-JEPA 2-AC's training). Every model trained before the option existed used MSE.

## Commands

`$PY -m ego <command> --help` shows a command's options.

| Command | Test | What it does |
|---|---|---|
| `extract-csvs` | | Extracts the gaze and hand CSV files from each recording's `mps_<rec>_vrs.zip`, in place. It skips zip files that are already extracted, empty or damaged. |
| `draw-gaze` | | Checks the gaze projection: overlays on frames of every participant, the share of gaze points inside the frame, and the shift caused by the depth. Writes `results/gaze_projection/`. |
| `gaze-at-picks` | | Checks the gaze projection against HD-EPIC's annotations: for each pick, whether the gaze point lies in the box of the object, against chance (the same box with gaze from random moments), without the rotation, at a fixed depth, and before the pick. It also finds the best shift of the gaze points for each participant. Writes `gaze_at_picks.csv`, `.json`, `.log` and `picks.jpg` to `results/gaze_projection/`. |
| `fetch-calibrations` | | Fetches the camera calibration of every HD-EPIC recording group. It reads only the calibration entry of each SLAM zip from the dataset server, with HTTP range requests, or from the local zip where one exists. It writes every 100th record to `SLAM-and-Gaze/<P>/SLAM/calibration/<n>.jsonl`. |
| `train` | 1, 9, 10 | Fine-tunes the ego predictor. After each epoch it measures Δ on the fixed held-out clips, as MSE and as L1. It writes `train.log`, `metrics.jsonl` and checkpoints to `--out-dir`. `--loss` chooses the training loss: `l1` (the default, as in V-JEPA 2-AC's training) or `mse` (every model trained before the option). `--unfreeze-last-n` sets how many of the last predictor blocks train (6 by default); `--unfreeze-last-n 24 --unfreeze-embed` trains the whole predictor (a full fine-tune, Test 10). `--gaze-form` chooses the gaze form of Test 9; `--future-signals` gives each step the signals of the frame it predicts (the positive control). |
| `train-ego` | 11 | Trains `ego/ego_predictor.py` from the start: a predictor of about 22 million parameters, not causal over frames, with gaze and the palms as Gaussian maps added to the image tokens instead of conditioning tokens. `--target-slots` chooses the horizons, predicted together in one forward pass (`2 4` at `--frame-stride 8` is about 0.53 s and 1.07 s). `--arm` chooses the arm of Test 11: `maps` (the design), `none` (the matched model: every signal parameter held at its zero start), `shuffled` (the points of another clip in the batch), `future` (the points of the target frames, the positive control). `head` and `token` raise until they are built. Writes `train.log`, `metrics.jsonl` and `final.pt` to `--out-dir`. |
| `evaluate` | | The paired comparison on any participants: the error with the signals hidden and with real signals, on the same clips. It samples its own clips. It writes `results/mse_paired.csv`. |
| `gaze-forms` | 9, 10 | Evaluates every finished run in the `--runs` folders (`checkpoints/test9` by default; folders `<name>_s<seed>` with `final.pt`), pooled by form: the gaze form, with " l1" for runs trained with L1 and " full" for a full fine-tune (Test 10), with `ego_sd1p0` as "none" and `ego_ft_v2` as a reference. The test set is the fixed clips of all P08 and P09 recordings (600 clips). It measures the error of the next step over the whole frame and within 2 patches of the gaze point, as MSE and as L1, with the model's signals and with them hidden; the comparisons of the plan, seeds averaged, with intervals over recordings; the reproduction check against `ego_ft_v2`. It also scores three references without fine-tuning (repeat the last context frame; a layer-normalized blend, 0.2 × the last context frame + 0.8 × the mean of the 8; V-JEPA 2-AC's predictor before fine-tuning, new layers drawn with seed 0) and compares every form and reference with the blend. The encoder features are cached in `<out>/cache/` (7.8 GB for Test 9; `--cache` reads them from another folder) and each model's scores in `<out>/scores/`, so a rerun scores only new runs, and runs whose scores lack L1. The comparisons of Test 10 are computed when its forms are present. Writes `results/test9/gaze_forms.csv`, `gaze_forms_references.csv`, `gaze_forms_comparisons.csv` (the comparisons with the blend have `b` = "blend of past frames"), `gaze_forms.json` and `gaze_forms.log`. |
| `next-object` | 8 | Does the gaze point tell which object is picked up next? Encodes the frames before each HD-EPIC pick once (`--stage encode`, cached per recording in `results/next_object/cache/`, resumable), then fits linear probes with each input (`--stage fit`). `--weights` sets the range of weights of the added block that cross-validation chooses from; `--cache` reads a cache from another folder, so a refit can write to a new `--out`. Writes `results/next_object/next_object.csv`, `next_object_contrasts.csv`, `next_object.json`, `next_object_predictions.npz` and `next_object.log`. |
| `gaze-probe` | 3a | A linear probe that predicts gaze from the frozen encoder's features. It caches the features in `results/gaze_features.npz`. |
| `control-probe` | 3b | The same probe with palm position as the target. It reads the features cached by `gaze-probe`. |
| `sensitivity` | 4a | Measures how much the prediction changes when the gaze or hand input changes. |
| `attention` | 4b | Measures how much attention goes to the gaze and hand tokens. |
| `weight-norms` | 4c | Divides the norm of each trained parameter by its norm at the start of training. |
| `stock-vs-tuned` | 2 | Scores the predictor before and after fine-tuning on the same clips, with real signals and with each of the three "no signal" inputs (mask token, zeros, average). |
| `signal-dropout` | 2 | Compares `ego_ft_v2` and `ego_sd1p0` clip by clip, from the two `stock-vs-tuned` outputs. It needs no GPU. |
| `summarize` | | Writes a JSON and a Markdown summary of a training run, to `results/<run>_summary.json` and `.md`. |
| `plot` | | Plots the training loss, the held-out errors and Δ to `results/<run>_results.png`. With several `--dir` arguments it also writes `results/delta_compare.png`. |
| `figures` | 3, 4, 8, 9 | Draws the figures of the notes into `docs/figures/` from the result files of the tests (`--only` picks groups: overview, picks, test3, test4, test8, test9). It also writes the Test 9 comparisons without the runs with a loss spike (logged gradient norm above 1.0) to `results/test9/gaze_forms_comparisons_without_spikes.csv`. Two Test 9 figures need the error of every patch: `--stage patches` (GPU, about 40 minutes) runs every Test 9 model over the cached encoder features and keeps the errors in `results/test9/cache/patches/`. |
| `watch` | | A live terminal view of a running training job. It only reads the job's log. |

The self-tests check the numerical parts of the linear probe, that `--shuffle-signals`
breaks the match between signals and frames and changes nothing else, the gaze forms, and the
two error measures (`tests/test_errors.py`: the L1 loss equals V-JEPA 2-AC's, and the scoring of
`gaze-forms` by hand), and what fine-tuning trains and how `gaze-forms` names the forms
(`tests/test_finetune.py`). Run them with `$PY -m tests`.

## Training

This command trains `ego_ft_v2`, the model used in most tests:

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m ego train \
    --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
    --video-dir  data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
    --gaze-dir   data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
    --participants P01 P02 P03 P04 P05 P06 P07 \
    --val-participants P08 --val-recordings 4 --val-clips 24 \
    --epochs 3 --clips-per-recording 30 --batch-size 16 \
    --num-workers 8 --encode-chunk 48 --save-every 3 \
    --loss mse --out-dir checkpoints/ego_ft_v2
```

- `ego_ft_v2`, `ego_sd1p0` and the Test 9 runs were trained with MSE, before `--loss` existed.
  To repeat them, add `--loss mse`; without it, `train` uses L1. The scripts in
  `checkpoints/test9/` that queued the Test 9 runs predate the option.
- `ego_sd1p0` uses the same command with `--signal-dropout 1.0 --out-dir checkpoints/ego_sd1p0`.
  It took 38 minutes.
- The planned full run uses `--epochs 8 --clips-per-recording 60 --val-recordings 6
  --val-clips 40`. That gives 240 test clips.
- `--shuffle-signals time` or `--shuffle-signals batch` trains with real signals from the
  wrong time or the wrong clip.
- `$PY -m ego watch --dir <out-dir>` shows a live view of a running job.

## Running the tests

Most test commands take the same base arguments:

```sh
--checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
--predictor-checkpoint checkpoints/ego_ft_v2/best.pt \
--video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
--gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze
```

Without `--predictor-checkpoint`, they use the model before fine-tuning, with random gaze and
hand layers.

| Test | Command | Results |
|---|---|---|
| 1 | `train` (the training run, and the check after each epoch) | `checkpoints/ego_ft_v2/train.log` |
| 3a | `gaze-probe --split participant` (or `recording`, `random`) | `results/gaze_recov_*.csv`. The encoder features are cached in `results/gaze_features.npz`. |
| 3b | `control-probe` | `results/control_recoverability.csv` |
| 4a | `sensitivity` | `results/sensitivity_ego_ft_v2.csv`, `results/sensitivity_untrained.csv` |
| 4b | `attention` | `results/attention_mass.csv`, `results/attention_mass_untrained.csv` |
| 4c | `weight-norms` | `results/weight_norms.csv`; `results/weight_norms_ft_v2_sd1p0.csv` for `ego_ft_v2` against `ego_sd1p0` |
| 2 | `stock-vs-tuned` | `results/rung_b1.csv`, `results/rung_b1_contrasts.csv` |
| 8 | `next-object` (defaults) | `results/next_object/next_object.csv` (accuracy per input and horizon), `next_object_contrasts.csv` (paired differences), `next_object.json` (the decision). The cache, about 4 GB, is in `results/next_object/cache/`. |
| 2 | `stock-vs-tuned` with `--predictor-checkpoint checkpoints/ego_sd1p0/best.pt --out results/rung_b1_sd1p0`, then `signal-dropout` | `results/rung_b1_sd1p0.csv`, `results/signal_dropout_contrasts.csv` |

- Part of Test 1, and Tests 5, 6 and 7 of the paper, use code that is not in this
  repository.
- Test 3 uses only the frozen encoder. `control-probe` reads the features cached by
  `gaze-probe`.
- The cache `results/gaze_features.npz` was built with `--recordings 12 --windows 10
  --per-window 5 --window-sec 6`. These differ from the defaults. While the cache exists,
  `gaze-probe` reuses it and ignores these options; `--recollect` rebuilds it.
- The stored `results/weight_norms.csv` also lists `ego_ft_v2/final.pt`, `ego_ft_quick` and
  `ego_finetune`, whose checkpoints are no longer on disk.
- `sensitivity`, `attention` and `stock-vs-tuned` use the fixed clips of `ego/clips.py`, the
  same P08 clips as the checks during training.
- `evaluate` runs the same paired comparison on any participants. It samples its own clips,
  so its numbers differ a little from the 96-clip set.

## Checkpoints

| Folder | Contents |
|---|---|
| `checkpoints/ego_ft_v2/` | `best.pt` (epoch 3). The model used in most tests. |
| `checkpoints/ego_sd1p0/` | `best.pt`, `epoch_003.pt` and `final.pt`. The model trained without signals. |
| `checkpoints/test9/` | The 20 runs of Test 9, `<form>_s<seed>/` (pe+rope is `perope`, its shuffled control `peropeshuf`), and the queue that trained them (`queue2.sh`, `todo.txt`, `queue.log`). All trained with MSE. |
| `checkpoints/test10/` | The 12 runs of Test 10: `pe_l1_s<seed>`, `none_l1_s<seed>`, `pe_l1full_s<seed>`, `none_l1full_s<seed>`. `queue.sh` runs the lines of `todo.txt` one after another on one GPU and logs to `queue.log`; its `eval` lines run `gaze-forms` on the Test 9 and Test 10 runs into `results/test10/`. Start it with `setsid nohup bash checkpoints/test10/queue.sh > checkpoints/test10/queue.out 2>&1 &`. |
| `checkpoints/ego_finetune/`, `ego_finetuned_p01_07/`, `ego_ft_quick/` | Logs only, from runs that were stopped. |

Each checkpoint file is 1.22 GB. The trainable weights alone are about 310 MB (77 million
parameters in fp32).

## Using a trained predictor

`docs/howto-predictor-probe.md` explains how to train a probe on a trained predictor: what the
predictor needs as input (encoder, frame format, clip, signals, gaze form), a tested example, the
controls, and a checklist.

## Practical notes

- If the GPU runs out of memory, lower `--encode-chunk`.
- In `train` and `gaze-probe`, the frozen encoder runs in bf16 by default. This doubles the
  speed. `--no-amp` turns it off. `sensitivity` uses bf16 only with `--amp`.
- Video decoding needs a lot of RAM. The EK100 runs needed `num_workers: 4` and
  `pin_memory: false`.
- `/mnt/data` is shared and has been full before. Write long outputs to scratch space first.
- Commands that build an untrained model set the random seed. Without it, the random gaze and
  hand layers differ between runs.

## Project Notes

Notes for a project on egocentric video understanding with JEPA models. The
question is whether a video prediction model predicts the near future better when it also
receives the person's gaze and hand positions.

**Status.** In the tested form, gaze and hand inputs did not improve the prediction: a model
trained without them does as well ([[4-results#^test2|Test 2]]). In that form the gaze token
cannot point at the image. Gaze can now be projected into the image, where it lies on the
object about to be picked up twice as often as by chance
([[3-method#Gaze position in the image]]). What lies at the gaze point tells which object is
picked up next, while the three gaze numbers do not ([[4-results#^test8|Test 8]]). Given to
the predictor as a position, gaze does no better than as three angles: as a position in the
token's content it does as well as the angles, and as a position in the attention it does not
help ([[4-results#^test9|Test 9]]). Test 10 is running: the predictor trained with the loss of
its pretraining (L1), and trained as a whole. Test 11 builds a predictor from V-JEPA 2.1 that
takes gaze and hand. The results of the paper will be measured on two benchmarks with gaze,
HD-EPIC's gaze interaction anticipation and EGTEA Gaze+ (Test 12). The plan is in
[[6-next-steps]].

### Chapters

Read in order. Files 1 to 6 follow the chapters of a report.

| File | Contents |
|---|---|
| [[1-introduction]] | The question, the approach, the three Paths, and the hypotheses with their status. |
| [[2-background]] | The models and papers this work builds on, the two benchmarks, and the terms used. |
| [[3-method]] | Data, model, gaze position in the image, training, evaluation, design choices. |
| [[4-results]] | Tests 1–4, 8 and 9 and what they showed. |
| [[5-discussion]] | What the results mean together, and the weak points of the design. |
| [[6-next-steps]] | Test 10 (running), Tests 11 and 12 (planned) and the other work, in order. |

### Names

| Name | Meaning |
|---|---|
| Path 1–3 | the three ways to use gaze ([[1-introduction]]) |
| M, R, H1–H4 | the hypotheses, each with a short name: M main, R already in the image, H1 horizon, H2 not used, H3 undertrained, H4 gaze form |
| Test 1–4, 8, 9 | the tests that were run; parts are written 4a, 4b |
| Test 10 | the running test: L1 and a full fine-tune of the V-JEPA 2-AC predictor |
| Test 11, 12 | the planned tests: a predictor from V-JEPA 2.1 with gaze and hand; the benchmarks. Numbers 5–7 are tests of the paper made with other code ([[parsa]]). |
| `ego_ft_v2`, `ego_sd1p0` | the trained models: with gaze and hand, and without |
| angles, pe, rope, pe+rope | the gaze forms compared in Test 9 |
| l1, full | added to a form's name in Test 10: trained with L1; the whole predictor trained (for example "pe l1 full") |

### Other files

| File or folder | Contents |
|---|---|
| [[project]] | The work done and its timeline, the paper, and the final presentation. |
| [[parsa]] | Parsa's work for the paper, kept apart: what, why, result, and whether it is valid and useful. |
| [[review]] | A review of the method, new baselines from the Test 9 cache, related work and benchmarks, with recommendations. |
| `docs/papers/` | [[BC-JEPA]], the paper as submitted (never edited), and the literature notes in `docs/papers/literature/`. |
| `docs/templates/` | A template for adding a test to [[4-results]]. |
| `docs/figures/` | The figures embedded in the chapters, drawn by `python -m ego figures` from the result files. |

The literature, in `docs/papers/literature/`, one note per paper:

- [[vjepa]]: V-JEPA 2, the base model.
- [[gazeqwen]]: GazeQwen, which adds gaze to a video-language model.
- [[vl-jepa]]: VL-JEPA, a JEPA model aligned with language. The basis for Path 3.
- [[vla-jepa]]: VLA-JEPA, JEPA-style pretraining for robot policies. Full text, no notes yet.
