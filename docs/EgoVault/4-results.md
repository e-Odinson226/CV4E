---
type: report
status: running
created: 2026-10-05
updated: 2026-10-07
---

# 4. Results

Erfan's tests, in the order of the argument. Test 1 shows that the predictor can learn with
gaze and hand inputs. Test 2 gives the main result: the inputs do not improve the prediction.
Tests 3 and 4 check two explanations for it. Each test addresses a hypothesis from
[[1-introduction#^hypotheses|the introduction]]. How each measure works is in [[3-method]].
The weak points that limit all four tests are in [[5-discussion#Weak points of the design]].
The planned Tests 8–10 are in [[6-next-steps]].

## Test 1. Can the predictor learn with gaze and hand inputs?

Run by Erfan, 31 May – 1 June 2026. This run produced `ego_ft_v2`, the model behind every
later test. ^test1

**Question.** V-JEPA 2-AC's predictor was trained with a robot's action and state as inputs.
Can it learn when a gaze token and a hand token take their place?

**Hypothesis and prediction.** The gaze and hand tokens fit the predictor's input layout. If
so, fine-tuning lowers the prediction error, and the training is stable.

**Method.**

- Model: V-JEPA 2-AC with the ViT-g encoder, frozen. The new gaze and hand layers, the last 6
  predictor blocks and the output layers are trained ([[3-method#Training]]).
- Data: P01–P07, 30 clips per recording, 3 epochs. The signals are hidden in about 41% of the
  training clips.
- Measure: the prediction error on the 96 P08 test clips, after each epoch.
- Before this run, Erfan changed how frames are encoded: each frame on its own, as in the
  original training. This lowered the error before fine-tuning from 4.82 to 0.58
  ([[3-method#Input format]]).

**Result.**

| Epoch | Training loss | Error on P08, signals hidden | Δ (hidden − real) |
|---|---|---|---|
| 0 (before training) | — | 0.6153 | −0.0082 |
| 1 | 0.5261 | 0.4942 | +0.0009 |
| 2 | 0.5150 | 0.4900 | +0.0015 |
| 3 | 0.5066 | 0.4877 | +0.0011 |

Three longer runs were started (8 epochs, 60 clips per recording, 240 test clips). Before
training, Δ was −0.0066 on these clips. One run trained the last 6 blocks for 2 epochs: Δ was
−0.0002 after epoch 1 and −0.0000 after epoch 2. One trained only the last 3 blocks for 1
epoch: Δ was −0.0003. Erfan stopped these two for this reason. The third stopped before its
first training steps were logged.

**Conclusion.** The predictor learns with gaze and hand inputs, and it improves on a person it
was not trained on. The signals add a small Δ of +0.0011. Test 2 asks what this Δ means.

**Limits.** That the model learns shows that the inputs fit. It does not show that the signals
help.

**Reproduce.** The training command in the repository's `README.md`. Log:
`checkpoints/ego_ft_v2/train.log`.

## Test 2. Do gaze and hand improve the prediction?

Run by Erfan. `ego_ft_v2` was trained on 31 May – 1 June 2026. The model without signals and
the comparisons were run on 19 August 2026. ^test2

**Question.** Does giving the model gaze and hand make its prediction of the next moment
better?

**Hypothesis and prediction.** M: gaze and hand inputs make the prediction better. If M is
true, a model trained with the signals and given them predicts better than an otherwise
identical model that never had them.

**Method.**

- *Two matched models.* `ego_ft_v2` saw real signals in about 59% of its training clips.
  `ego_sd1p0` has the same settings, except that the signals were hidden in every clip, so it
  never saw real gaze or hand.
- *Why a second model.* Comparing one model with and without its signals mixes two things: the
  value of the information, and the cost of removing an input the model expects. The matched
  model removes the second.
- *Reference.* The model before fine-tuning: V-JEPA 2-AC with the new gaze and hand layers at
  their random start.
- *Measure.* The error of the prediction 0.27 s ahead, on the same 96 P08 clips for every model
  and input. Differences are tested clip by clip with a paired Wilcoxon test.

**Result.** Error (MSE):

| Model | Real signals | Mask token | Zeros | Average |
|---|---|---|---|---|
| Before fine-tuning | 0.6183 | 0.6126 | 0.6158 | 0.6182 |
| `ego_ft_v2`, trained with signals | 0.4866 | 0.4877 | 0.4888 | 0.4873 |
| `ego_sd1p0`, trained without signals | 0.5685 | 0.4869 | — | — |

`ego_sd1p0`'s gaze and hand layers were never trained, so for this model only the mask token
is a fair input.

| Comparison | Δ | p | What it measures |
|---|---|---|---|
| `ego_sd1p0` hidden − `ego_ft_v2` real | +0.0003 | 0.40 | the value of the information |
| `ego_ft_v2` hidden − `ego_ft_v2` real | +0.0011 | 0.0015 | that value plus the cost of removing an expected input |
| `ego_sd1p0` hidden − `ego_ft_v2` hidden | −0.0008 | 0.015 | the two models, both without signals |
| Before − after fine-tuning, hidden | +0.125 | < 1e-16 | the gain from adapting to kitchen video |

A positive Δ means the second model or input predicts better.

- The gain from fine-tuning (0.125) improved all 96 clips. The signals account for 0.0011 of
  it, about 0.9%.
- Before fine-tuning, real signals make the prediction worse, because the new layers start
  random.

**Conclusion.** The model trained without signals predicts as well as the model that uses
them: +0.0003, with p = 0.40. So M is not supported in this form: gaze as three angles, 3
epochs, one training run per model. The +0.0011 of `ego_ft_v2` is mostly the cost of removing
an input it expects. Almost all of the gain from fine-tuning is adaptation to the video. Tests
3 and 4 check two explanations: gaze may already be in the image (R), or the model may not use
the signals (H2).

**Limits.**

- Each model was trained once. The p-values cover the 96 clips, not the variation between
  training runs. The two models without signals differ by −0.0008, which comes from training
  alone.
- The 96 clips come from 4 recordings of one person. With a bootstrap over recordings and
  clips, the 95% interval for the value of the information is [−0.0005, +0.0011].
- The prediction looks 0.27 s ahead, and the error is averaged over the whole frame.
- No model receives a signal known to help, so the test does not show how large an effect it
  could detect.
- The numbers before fine-tuning depend on the random start of the new layers. The two stored
  runs of `stock-vs-tuned` were not seeded and drew different starts.

**Reproduce.** `python -m ego stock-vs-tuned` for each model, then
`python -m ego signal-dropout`. Files: `results/rung_b1*.csv`,
`results/signal_dropout_contrasts.csv`.

## Test 3. Is gaze already in the image features?

Run by Erfan: the gaze probe on 28–30 July 2026, the palm control on 19 August 2026. ^test3

**Question.** The model already sees the video. If the video features already contain where
the person looks, a gaze input adds nothing new. (Asked by Ash, 16 July 2026.)

**Hypothesis and prediction.** R: the frozen image features already contain gaze. If R is
true, a simple probe can read the gaze direction from the features, also for a person it has
never seen. If R is false, the probe does no better for new people than guessing the average
gaze.

**Method.**

- *What is read.* The encoder turns one frame into 256 patch vectors of 1,408 numbers each.
  The encoder is frozen and never saw gaze.
- *What is predicted.* The gaze direction as two angles, yaw and pitch, at the moment of the
  frame or 0.25, 0.5, 1 or 2 s later.
- *The probe.* Ridge regression, a weighted sum of the feature values. A weighted sum cannot
  build new features. If it can read gaze, the predictor has easy access to it.
- *Data.* P01–P07, up to 12 recordings per participant: 2,937 training frames and 997 test
  frames in the main split.
- *Splits.* New people (train P01–P05, test P06–P07) answers the question. New recordings of
  the same people, and random frames, are for comparison.
- *Control.* Each person cooks in their own kitchen, so a new person is also a new kitchen.
  The same probe also reads the position of the left and right palm. If palm position can be
  read for new people and gaze cannot, the failure is specific to gaze.

**Result.** Skill (0 = guessing the average, 1 = perfect):

| Lead | New people | New recordings | Random frames |
|---|---|---|---|
| 0 s | 0.001 | 0.116 | 0.273 |
| 0.25 s | 0.004 | 0.071 | 0.230 |
| 0.5 s | −0.015 | 0.055 | 0.219 |
| 1 s | −0.007 | 0.000 | 0.184 |
| 2 s | −0.027 | 0.015 | 0.145 |

| Target, at 0 s | New people | New recordings | Random frames |
|---|---|---|---|
| Left palm | 0.346 | 0.391 | 0.628 |
| Right palm | 0.220 | 0.313 | 0.540 |
| Gaze | 0.001 | 0.116 | 0.273 |

- For new people, the probe's median angle error is 11.2°. Always guessing the average gaze
  gives 11.8°.
- The participant can be identified from one frame in 88.7% of cases (chance 14.3%).

**Conclusion.** For a person it has not seen, a linear probe cannot read gaze from the features
of one frame, while it can read palm position. So the failure is specific to gaze and is not a
general failure of the features. In this form R is not supported, and the gaze input can add
information that the features lack.

**Limits.**

- The test asks whether the current gaze can be read. R needs a different question: does gaze
  add information about the future frame beyond the past frames? Test 8 is planned for this.
- The probe is linear and sees one frame. The predictor is non-linear and sees 8 frames.
- The main split tests on 2 people, without error bars.
- The PCA is fitted on frames of one person (P01).
- The check "skill 1.000 at lead 0" compares a gaze lookup with itself. It does not test the
  video frames. A separate check by Claude Code (October 2026) found frames read by seeking
  identical to frames read in order, at 67 positions in 9 recordings.

**Reproduce.** `python -m ego gaze-probe --split participant` (or `recording`, `random`),
then `python -m ego control-probe`. Files: `results/gaze_recov_*.csv`,
`results/control_recoverability.csv`. The features are cached in `results/gaze_features.npz`,
built with `--recordings 12 --windows 10 --per-window 5 --window-sec 6`.

## Test 4. Does the model use the signals?

Run by Erfan, 19 August 2026, on `ego_ft_v2`, and for comparison on the model before
fine-tuning. ^test4

**Question.** Test 2 found that the signals barely improve the prediction. One explanation is
that the model never learned to use them.

**Hypothesis and prediction.** H2: the model does not use the signals. Three parts, each with
its own prediction:

- 4a: if H2 is true, changing the gaze input does not change the prediction.
- 4b: if H2 is true, the gaze token gets no more attention than any other token.
- 4c: if training switched gaze off, the weights of the gaze layer shrank.

### 4a. Does the prediction change when gaze changes?

**Method.** The video is kept fixed and only the gaze input changes. The test measures how far
the prediction moves, relative to its size, on the 96 P08 clips. "Another clip" is the
neighbouring clip in the list; in 92 of 96 pairs it comes from the same recording.

**Result.**

| Change to the input | `ego_ft_v2` | Before fine-tuning |
|---|---|---|
| None (same input twice) | 0 | 0 |
| Gaze yaw shifted by 1° | 0.0012 | 0.0042 |
| Gaze yaw shifted by 10° | 0.013 | 0.033 |
| Gaze yaw shifted by 45° | 0.081 | 0.104 |
| Gaze from another clip | 0.016 | 0.056 |
| Gaze hidden | 0.084 | 0.133 |
| Gaze and hand hidden | 0.036 | 0.144 |
| Video from another clip | 0.567 | 0.743 |

- Larger changes in gaze give larger changes in the prediction.
- Gaze from another clip moves the prediction 2.9% as much as a different video. Before
  fine-tuning it moved it 7.6% as much. Training reduced the response to the gaze value.
- Hiding gaze alone moves the prediction 5.2 times more than another clip's gaze. But the model
  never saw gaze hidden while the hand was present, because signal dropout always hides both.
  This state raises the error by 0.0074, against 0.0011 when both are hidden. Part of the 5.2
  is a reaction to an unfamiliar input.

### 4b. How much attention goes to the gaze token?

**Method.** In each of the 24 blocks, every token takes in information from the other tokens,
weighted by attention. The image tokens can receive gaze information only through their
attention to the gaze token. Equal attention would give the gaze token 0.39%. 24 clips, three
inputs: real signals, gaze hidden, and both hidden.

**Result.**

- With real signals, image tokens give 2.4% to 26% of their attention to the gaze token,
  depending on the block: 16% on average in the frozen blocks, 9% in the trained blocks.
- One attention head in the first block gives 99.96% of its attention to the gaze token.
- Before fine-tuning, the frozen blocks gave the gaze token 38 times the equal share; after, 42
  times. This attention comes from the robot pretraining, where this position held the action
  token.
- Average share over all blocks: 14.6% with real gaze, 12.1% with gaze hidden.

### 4c. Did training shrink the gaze layer?

**Method.** The size (norm) of each trained parameter is compared with its size at the start.
Frozen parameters must keep exactly their size; they do.

**Result.**

- The gaze layer's weights grew by 15.7%, the hand layer's by 17.5%. The trained blocks
  changed by less than 0.1%.
- In `ego_sd1p0`, which never received real signals, the mask tokens grew by 19–20%. They carry
  no gaze information.
- With Adam, a parameter moves by about the learning rate at every step, whatever the gradient
  carries. The new layers train at 10 times the learning rate of the blocks. So growth of this
  size is expected for any new parameter.

### Conclusion of Test 4

The model reacts to the signals, mainly to whether they are present. Its response to the gaze
value is small, and fine-tuning made it smaller (4a). The gaze position gets much attention, but
it already did before fine-tuning (4b). The gaze layer was not shrunk, but its growth does not
show use (4c). So H2 is partly supported: the model detects whether gaze is present and makes
little use of where the person looks. A likely reason is that the gaze token cannot point at the
image (H4, [[5-discussion#Weak points of the design]]).

**Limits.** One model, trained once for 3 epochs, tested on clips of one person. The test shows
how the model reacts to gaze, not whether gaze helps.

**Reproduce.** `python -m ego sensitivity`, `python -m ego attention`,
`python -m ego weight-norms`. Files: `results/sensitivity_*.csv`, `results/attention_mass*.csv`,
`results/weight_norms_ft_v2_sd1p0.csv`.

## Is the model undertrained? (H3)

No test has addressed H3 yet. All results come from models trained for 3 epochs with 30 clips
per recording. A run at the intended size (8 epochs, 60 clips per recording) has never
finished. Two results bear on it: fine-tuning already lowered the error by 0.125 (Test 2), and
the two longer runs had Δ at or below zero before they were stopped (Test 1). The planned test
is Test 10 in [[6-next-steps]].
