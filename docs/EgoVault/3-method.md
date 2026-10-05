---
type: report
status: running
created: 2026-09-18
updated: 2026-10-05
---

# 3. Method

How the model is built, trained and evaluated, and the design choices in force. The code, the
commands and the checkpoints are described in `README.md` at the root of the repository.

## Data

- **HD-EPIC.** Kitchen recordings made with Aria glasses, participants P01–P09. 154 of the 156
  recordings have gaze and hand data. The two without are from P02.
- **Split.** Training uses P01–P07. The standard test set is 96 clips from P08: 4 recordings ×
  24 clips, sampled with seed 12345.
- **EK100.** EPIC-KITCHENS-100 has an action anticipation benchmark. It has no gaze or hand
  data. In Parsa's tests, only P01 was evaluated (870 clips).

## Model

The base model is Meta's V-JEPA 2-AC with the ViT-g encoder ([[2-background]]). The ego
predictor replaces its robot tokens with a gaze token and a hand token:

```
AC predictor:   [action, state, 256 image tokens]   per frame
Ego predictor:  [gaze,   hand,  256 image tokens]   per frame
```

The layout is the same, so all 24 pretrained predictor blocks load without changes. Attention
is causal over frames: a token at frame t sees all tokens of the frames up to t.

Ioana wrote the first gaze and hand loaders and projection layers. Erfan moved the projection
layers into the predictor, extended them to all time steps, and added timestamp matching and
scaling to the loaders. Parsa built a separate ViT-L version, used only for a feasibility
check ([[4-results#^t1|T1]]).

### Inputs

| Signal | Size | Values | Source file |
|---|---|---|---|
| Gaze | 3 | yaw and pitch (radians), depth (metres), in the Aria central pupil frame | `eye_gaze/general_eye_gaze.csv` |
| Hand | 12 | wrist and palm position (x, y, z) for each hand, in the device frame | `hand_tracking/wrist_and_palm_poses.csv` |

- Signals are matched to video frames by timestamp, using
  `<recording>_mp4_to_vrs_time_ns.csv`.
- Gaze is scaled as (gaze − `GAZE_MEAN`) / `GAZE_STD`. Hand is scaled as hand / `HAND_STD`.
- These constants are rough guesses. Measured on P01–P07, scaled gaze has means of (−0.04,
  −0.52, −0.08) and standard deviations of (0.44, 0.55, 0.87). The intended values are 0
  and 1. Scaled hand values have means from −1.0 to 0.87 and standard deviations from 0.31 to
  0.50. The constants need to be recomputed before the next training run.

### Layers

- `gaze_proj`: a linear layer from 3 to 1024 values, with a bias.
- `hand_proj`: a linear layer from 12 to 1024 values, with a bias.
- Both start from small random weights (truncated normal). Their biases start at 0.
- `gaze_mask` and `hand_mask` are learned tokens. They replace a signal when it is missing or
  hidden.
- Gaze has one validity flag per frame. The hand token is replaced only when both hands are
  missing.

### Input format

The AC predictor was trained on frames encoded one at a time. The scripts do the same, in
`scripts/ego_common.py`, which training and testing share:

- Each frame is encoded on its own, as a 2-frame tubelet of the same image.
- The EMA target encoder encodes both the context frames and the target frame.
- Frames are resized to 256 × 256 pixels. This gives 16 × 16 = 256 patches per frame.
- The model uses every 8th frame, about 4 frames per second. It sees 8 such frames.
- Embeddings are layer-normalized before the loss.
- In training, each step predicts the next step.

Encoding a whole clip at once gave an error of 4.82 on P08 before fine-tuning. The format above
gives 0.57.

## Training

| Part | Trained | Learning rate |
|---|---|---|
| Encoder (ViT-g) | no | — |
| `predictor_embed`, blocks 0–17 | no | — |
| `gaze_proj`, `hand_proj`, `gaze_mask`, `hand_mask` | yes | 1e-3 |
| Blocks 18–23, `predictor_norm`, `predictor_proj` | yes | 1e-4 |

- This is 77.0 million of the predictor's 305.2 million parameters (25.2%).
- The optimizer is AdamW with weight decay 0.01.
- `ego_ft_v2`, the model used in most tests: 3 epochs, P01–P07, 30 clips per recording,
  batch size 16.
- Signal dropout: for each training clip, one random draw decides whether the signals are
  hidden. The default rate is 0.4. Gaze and hand are always hidden together.
- `ego_sd1p0`: the same settings with `--signal-dropout 1.0`. This model never sees real
  signals.
- `--shuffle-signals time` shuffles the order of the signals within each clip.
  `--shuffle-signals batch` takes them from another clip. Neither has been used in a run yet.

## Evaluation

There are two evaluation tracks. The main one is the prediction error on HD-EPIC. It needs no
labels, and HD-EPIC has gaze data. The second is the EK100 anticipation probe. Its numbers can
be compared with published V-JEPA 2 results. (Decided by Erfan, May 2026.)

### Prediction error and Δ

- The model sees 8 frames, which cover about 2 seconds. It predicts the embedding of the next
  step, 8 frames (0.27 s) after the last context frame.
- The target is the frozen encoder's embedding of that frame.
- MSE is computed per clip, after layer normalization, and then averaged. Lower is better.
- Δ = MSE with signals hidden − MSE with real signals, on the same clips. A positive Δ means the
  signals helped.
- Δ is tested with a paired Wilcoxon signed-rank test on the per-clip values. It is reported
  with a 95% confidence interval.
- Results are reported as absolute differences. Relative percentages make small effects look
  large. (Asked by Ash, July 2026.)

### "No signal" inputs

There are three ways to give the model no signal. They give different Δ values:

| Input | What the model receives | Δ for `ego_ft_v2` |
|---|---|---|
| Mask token | the learned `gaze_mask` and `hand_mask` | +0.0011 |
| Zeros | zero vectors through the projection layers | +0.0022 |
| Average | the average scaled signal | +0.0007 |

Every Δ states which input it uses. The default is the mask token, which `eval_ego_mse.py` and
the checks during training use. (Decided by Erfan, August 2026.)

### Measuring the value of the signals

- `ego_ft_v2` saw hidden signals in 40% of its training clips. So hiding them at test time is a
  case it knows.
- The model still relies on the signals. Hiding them measures two things together: the value
  of the information and the cost of removing an expected input.
- A matched model trained with `--signal-dropout 1.0` separates the two. The value of the
  information is `ego_ft_v2` with real signals compared with `ego_sd1p0` with hidden signals
  ([[4-results#^t11|T11]]).
- Every future training run includes such a matched model. (Control defined by Parsa, June
  2026. It uses Erfan's `--signal-dropout` option. Run by Erfan, August 2026.)

### Linear probes on the frozen encoder (T4, T5)

- Input: one frame, encoded by the frozen encoder. All 256 patches are kept, because gaze is
  about a place in the image.
- The 1408 values per patch are reduced to 64 with PCA. PCA is fitted on training frames only.
- Model: ridge regression. Its regularization strength is chosen by 5-fold cross-validation on
  the training data.
- Score: skill = 1 − MSE(probe) / MSE(guessing the training mean). 0 is chance and 1 is
  perfect. R² is not used. When the test people differ from the training people, a trivial
  guess can get an R² far below zero.
- Lead: the target is taken 0, 0.25, 0.5, 1 or 2 s after the frame.
- Splits:
  - Random frames. Near-identical frames end up in both sets, so the result is too high.
  - Recordings. 59 training and 20 test recordings, from the same people.
  - Participants. P01–P05 for training, P06–P07 for testing.
- Checks:
  - Predicting gaze from the gaze at the same moment must give skill 1.0. This confirms that
    the timestamps line up.
  - Targets more than 50 ms from a real sample are dropped.
  - With fewer than 500 training samples, the script does not interpret the result.
- T5 reuses the features cached by T4. It finds the frame of each cached row by replaying the
  sampler, and checks every row against the cache.

### Tests on the trained predictor (T6–T8, T10)

- Sensitivity (T6): change = ‖prediction with changed input − prediction with real input‖ /
  ‖prediction with real input‖. It is measured on the last predicted step, after layer
  normalization. The same input run twice must give 0.
- Attention (T7): the attention function is wrapped so that the attention weights can be read.
  Each frame has 258 tokens, so equal attention is 1/258 = 0.39% per token. Both the mean and
  the largest single head are reported.
- Weight norms (T8): each trained parameter's norm is divided by its norm at the start. Frozen
  parameters must give exactly 1.0.
- Untrained model: the gaze and hand layers start random. The scripts set the seed so that runs
  match.

### EK100 probe (T3)

- EK100 has no gaze or hand data, so the predictor always receives its mask tokens there. The
  probe shows what training with the signals left in the model. It cannot show what live
  signals add. (Decided by Parsa, June 2026.)
- Each clip has 16 frames at 4 frames per second. It ends 1 s before the action.
- The encoder and the predictor are frozen. Only the probe is trained.
- The probe is an attentive pooler with one cross-attention block. Linear verb, noun and action
  heads follow it.
- 25 settings of learning rate and weight decay are trained together. The best validation
  score is reported.
- Score: class-mean recall@5.
- Extra tokens can raise the score by themselves. So every control gives the probe the same
  number of tokens. The pooler does not use token positions, so zero padding (6b) and a
  repeated frame (6c) are valid controls. (Decided by Parsa, June 2026.)
- The predictor was trained on 256-pixel frames. This pipeline uses 224-pixel frames.

### Diagnostics on HD-EPIC P01 (T9, T12)

These are Parsa's tests for the paper.

- PEVA-1: is the predictor's output closer to the target with the real signal than with the
  mask, on average?
- PEVA-2: does the real signal increase the margin between the correct candidate and the wrong
  ones?
- Signal-only probe: predicts the action from gaze and hand alone. It is compared with the
  class prior.
- Two code checks. The evaluation path gives exactly the same output as the earlier pipeline
  (largest difference 0). After one backward pass, every module that should train receives a
  gradient.

## Design choices

These ideas came from studying GazeQwen ([[2-background]]). None of them were built. T6–T8
showed that the model already reads gaze. This removed the main reason for them. No
architecture changes are made before the full training run. (Decided by Erfan, August 2026.)

**Zero initialization of the projection layers: rejected.** In this model gaze is a separate
token. A zero token is a new kind of input for the model. If the projection and a gate both
start at 0, both gradients are 0 and the pathway cannot learn. (Decided by Erfan, August 2026.)

**Gate (`alpha_gaze`), if one is added.** A learned number α blends between the "no signal"
token and the projected gaze:

```
token = gaze_mask + α · (gaze_proj(gaze) − gaze_mask)
```

At α = 0 the model behaves like one trained without signals. The projection keeps its normal
random start. α has no weight decay, because weight decay alone would pull it toward 0,
whatever the data. (Decided by Erfan, August 2026.)

**Coord-PE, if it is built.** Encode the gaze position with sine and cosine features at
several frequencies, before the linear layer. This needs gaze as a position in the image,
(u, v). The current gaze values are angles. The steps needed:

1. Recompute the scaling constants.
2. Project the gaze ray into the image with the camera calibration in
   `data/online_calibration.jsonl`.
3. Check the projection by drawing it on sample frames.
4. Extend the gaze vector to [yaw, pitch, depth, u, v, in_frame]. Keep the angles.

**Parsa's stronger pathway.** Bias the image tokens' attention keys with the signal, add a
resampler that combines gaze and hand, and add a gate per block. It was adapted from GazeQwen.
It was designed but not built. The paper dropped it based on [[4-results#^t9|T9]].
