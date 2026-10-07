---
type: report
status: running
created: 2026-10-05
updated: 2026-10-07
---

# 4. Results

The tests, in the order of the argument. Test 1 shows that the predictor can learn with
gaze and hand inputs. Test 2 gives the main result: the inputs do not improve the prediction.
Tests 3 and 4 check two explanations for it. Test 8 asks whether what the person looks at
tells what comes next, before gaze is given to the predictor as a position. Each test addresses
a hypothesis from [[1-introduction#^hypotheses|the introduction]]. How each measure works is in
[[3-method]]. The weak points that limit Tests 1–4 are in
[[5-discussion#Weak points of the design]]. The planned Tests 9 and 10 are in [[6-next-steps]].

## Test 1. Can the predictor learn with gaze and hand inputs?

Run on 31 May – 1 June 2026. This run produced `ego_ft_v2`, the model behind every
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
- Before this run, the frame encoding was changed: each frame on its own, as in the
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
epoch: Δ was −0.0003. These two were stopped for this reason. The third stopped before its
first training steps were logged.

**Conclusion.** The predictor learns with gaze and hand inputs, and it improves on a person it
was not trained on. The signals add a small Δ of +0.0011. Test 2 asks what this Δ means.

**Limits.** That the model learns shows that the inputs fit. It does not show that the signals
help.

**Reproduce.** The training command in the repository's `README.md`. Log:
`checkpoints/ego_ft_v2/train.log`.

## Test 2. Do gaze and hand improve the prediction?

`ego_ft_v2` was trained on 31 May – 1 June 2026. The model without signals and
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

Run on 28–30 July 2026 (the gaze probe) and 19 August 2026 (the palm control). ^test3

**Question.** The model already sees the video. If the video features already contain where
the person looks, a gaze input adds nothing new.

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
  video frames. A separate check (October 2026) found frames read by seeking
  identical to frames read in order, at 67 positions in 9 recordings.

**Reproduce.** `python -m ego gaze-probe --split participant` (or `recording`, `random`),
then `python -m ego control-probe`. Files: `results/gaze_recov_*.csv`,
`results/control_recoverability.csv`. The features are cached in `results/gaze_features.npz`,
built with `--recordings 12 --windows 10 --per-window 5 --window-sec 6`.

## Test 4. Does the model use the signals?

Run on 19 August 2026, on `ego_ft_v2`, and for comparison on the model before
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

## Test 8. Does the gaze point tell what comes next?

Run on 7 October 2026. Uses the frozen encoder of V-JEPA 2-AC
(no predictor), HD-EPIC's annotations of picks and the gaze point in the image
([[3-method#Gaze position in the image]]). ^test8

**Question.** Does what the person looks at tell which object they will pick up next, beyond
what is in view, beyond the three gaze numbers that the model receives now, and beyond where the
head points? It is asked before Test 9 because it needs no training (about 1.5 hours) and shows
whether a gaze position is worth giving the predictor, and over which horizon and history.

**Hypothesis and prediction.** H4 needs the gaze point to carry information about the future.
Written before the run: if it does, a probe that also sees the features at the gaze point names
the next object more often than a probe that sees the three gaze numbers, and more often than a
probe that sees the features where the head points. The decision rule, also fixed before the
run: Test 9 goes ahead if, at 0.5 s or at 1 s before the pick, "+ gaze point" beats both
"+ angles" and "+ head point", or "+ gaze history, 3 s" beats both "+ angles" and "+ head
history, 3 s", in top-1 accuracy, each paired difference with a 97.5% interval above 0. Two
horizons are allowed, so each interval is 97.5% in place of 95%.

**Method.**

- *Picks.* HD-EPIC's annotated picks. The free-form name of each object is mapped to HD-EPIC's
  noun classes ("spoon2" → spoon), which works for 93% of the movements. The 50 most frequent
  classes in training are kept: 15,349 picks in recordings with gaze. Train on P01–P07 (11,305
  picks), test on P08 and P09 (4,044 picks in 25 recordings), people and kitchens the probe has
  not seen.
- *Moments.* For each pick, the frame 0.5, 1, 2 or 4 s before it, encoded on its own by the
  frozen encoder. No predictor is used. A moment counts only if the gaze is valid in its frame.
- *Inputs.* Each input is the scene plus one block:

| Input | What it contains | What it stands for |
|---|---|---|
| Scene | the 256 patches of the frame, each reduced from 1408 to 64 numbers (PCA), plus the patch average of each of the frames 0.5, 1, 1.5 and 2 s earlier | what is in view, where, and the last 2 s |
| + angles | yaw, pitch and depth | what the current model receives |
| + gaze point | the features of the patches around the gaze point, with Gaussian weights of width 1 patch | what the person looks at |
| + head point | the features around a fixed point, the average gaze point of the training moments (column 8.86, row 9.91) | control: where the head points |
| + gaze history, 1, 3 or 6 s | the gaze point, plus the gaze-point features of the frames in the last 1, 3 or 6 s, every 0.5 s, averaged | what the person looked at recently |
| + head history, 1, 3 or 6 s | the head point, plus the head-point features of the same frames, averaged | control: the same frames, without the eyes |

- *Why these controls.* Gaze stays close to one place in the frame, so much of what lies at the
  gaze point also lies where the head points. Only a gain over the head point shows that the
  eyes add something. A longer history also adds more frames, so each gaze history is compared
  with the head history of the same length. The scene keeps the patch grid and the last 2 s,
  because an average over the patches loses where things are, and a weak scene would let any
  local features look useful.
- *The probe.* Ridge regression, a linear map with one output per class. Its guess is the class
  with the highest output; its top-5 guess, the five highest. Each block is scaled to the same
  total variance. The weight of the added block (0.3, 1 or 3 times the scene) and the penalty
  are chosen by 3-fold cross-validation over training recordings, on top-1 accuracy. Every
  input, the controls included, goes through the same procedure.
- *Score.* Top-1 accuracy (the right object) and top-5 accuracy (the right object among five
  guesses) on P08 and P09, with 95% intervals from a bootstrap over the 25 test recordings. The
  floor is the frequency guess, the most frequent training classes: 9.3% top-1, 33.6% top-5.

**Checks.**

- A trial on one recording of each participant ran through every step before the full run.
- All 288,653 frames were read; none was unreadable. The PCA keeps 70.1% of the variance.
- The scene alone beats the frequency guess at 0.5 s and 1 s (13.0% and 11.1% top-1, against
  9.3%), not at 2 s and 4 s.

**Result.** Top-1 accuracy on P08 and P09, in %. 4,044 test picks at each horizon (4,042 at
4 s). Frequency guess: 9.3%.

| Input | 0.5 s | 1 s | 2 s | 4 s |
|---|---|---|---|---|
| Scene | 13.0 | 11.1 | 9.3 | 8.0 |
| + angles | 12.1 | 10.9 | 9.4 | 8.5 |
| + head point | 14.1 | 10.9 | 9.1 | 7.9 |
| + gaze point | **18.5** | **13.3** | 10.4 | 8.6 |
| + head history, 3 s | 13.5 | 10.5 | 8.8 | 8.9 |
| + gaze history, 3 s | **18.8** | **13.4** | 11.1 | 8.9 |

Paired differences in top-1, in percentage points, with 95% intervals:

| Difference | 0.5 s | 1 s | 2 s | 4 s |
|---|---|---|---|---|
| gaze point − head point | +4.3 [+3.0, +5.9] | +2.3 [+1.3, +3.7] | +1.3 [+0.4, +2.2] | +0.7 [−0.3, +1.6] |
| gaze point − angles | +6.4 [+5.2, +7.5] | +2.4 [+1.0, +3.9] | +0.9 [−0.0, +1.8] | +0.2 [−0.9, +1.1] |
| gaze history 3 s − head history 3 s | +5.3 [+3.7, +7.0] | +2.8 [+1.8, +4.0] | +2.3 [+1.3, +3.4] | −0.0 [−1.6, +1.7] |
| gaze history 3 s − angles | +6.7 [+5.7, +7.9] | +2.5 [+1.1, +4.0] | +1.7 [+0.5, +2.9] | +0.4 [−0.7, +1.6] |
| head point − scene | +1.1 [+0.0, +2.0] | −0.2 [−1.1, +0.8] | −0.2 [−0.8, +0.5] | −0.0 [−0.9, +0.7] |
| angles − scene | −0.9 [−1.5, −0.3] | −0.2 [−0.7, +0.3] | +0.2 [−0.2, +0.5] | +0.5 [+0.2, +0.8] |

- *Decision.* At 0.5 s and at 1 s, all four deciding differences have 97.5% intervals above 0.
  The smallest lower bound is +0.9 points (gaze point − angles at 1 s). Test 9 goes ahead.
- *Top-5.* The same pattern, smaller in relative terms: at 0.5 s, 43.5% with the gaze point
  against 37.9% for the scene and 37.0% with the head point.
- *Both test people.* Gaze point − head point at 0.5 s: P08 +3.4 [+1.7, +5.4], P09 +5.4 [+3.7,
  +7.9]. At 1 s: +2.3 and +2.4.
- *A wider range of weights, a refit after the run.* Cross-validation had chosen the largest
  weight on offer (3) for most inputs. A refit from the same cache with weights up to 100, all
  else unchanged, tests whether that limited the gaze inputs. For the gaze point it chose the
  same weight as before at three of the four horizons, and 10 in place of 3 at 1 s. Every top-1
  accuracy changed by at most 1.1 points, and the decision is the same. The deciding differences at 1 s, the closest call, are
  gaze point − angles +2.1 [+0.4, +3.9] and gaze point − head point +2.1 [+0.7, +3.7], with
  97.5% intervals.

| Top-1, refit with weights up to 100 | 0.5 s | 1 s | 2 s | 4 s |
|---|---|---|---|---|
| + gaze point | 18.5 | 13.0 | 10.4 | 8.6 |
| + gaze history, 1 s | 20.0 | 13.4 | 10.3 | 9.0 |
| + gaze history, 3 s | 19.3 | 13.7 | 11.4 | 9.1 |
| + head history, 3 s | 13.4 | 10.2 | 8.3 | 8.7 |

- *History, an analysis after the run.* The gaze history adds little beyond the current gaze
  point. With the original weights, "gaze history, 3 s" − "gaze point" lies between +0.1 and
  +0.8 points at every horizon, with every interval including 0. With the wider weights, a
  short history adds about 1 point: "gaze history, 1 s" − "gaze point" +1.5 [+0.7, +2.3] at
  0.5 s, and "gaze history, 3 s" − "gaze point" +0.8 [+0.1, +1.6] at 0.5 s and +0.8 [+0.0,
  +1.4] at 1 s. A 6 s history never beats a 3 s history in either fit (from −0.8 to +0.3
  points).

**Conclusion.**

- The features at the gaze point tell which object comes next. They add to the scene with its
  full grid and the last 2 s, to the three gaze numbers, and to the features where the head
  points. 0.5 s before the pick they raise the top-1 accuracy from 13–14% to 18.5%.
- R, in the form that matters, is not supported: gaze adds information about the near future
  that the frame and the last 2 s do not show, also for people the probe has not seen.
- H4 is supported in part. The information is in the content at the gaze point, not in the
  three numbers: added as numbers, the angles do not help (−0.9 to +0.5 points). This is the
  difference H4 is about. Whether the predictor can use a gaze position is Test 9.
- The information is short-lived. It is largest 0.5 s before the pick, smaller at 1 s, small at
  2 s and gone at 4 s. This matches the lead of gaze over the hand (0.5–1 s). For the next
  object it does not support the idea that gaze helps more at longer horizons (H1).
- What the person looked at in the last 1–3 s adds at most about 1 point beyond where they look
  now, and a longer history adds nothing more.

**Limits.**

- A linear probe on one frame and summaries of 2 s, not the predictor. It shows that the
  information exists, not that the predictor will use it.
- Every moment is followed by a pick. The test asks which object, not whether a pick comes.
- For new kitchens, the scene alone barely beats the frequency guess, and not at all at 2 s and
  4 s. A stronger probe might take more from the scene, and also more from the gaze point.
- The gaze history is an average. A history that keeps the order of the looks might add more.
- Two test people. Object names that do not map to a class (7% of the movements) are left out.

**Reproduce.** `python -m ego next-object --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos
--gaze-dir data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze --annotations
data/epic-kitchen/ek100-hd/HD-EPIC/annotations`. Files in `results/next_object/`:
`next_object.csv` (accuracy per input and horizon), `next_object_contrasts.csv` (paired
differences), `next_object.json` (the decision), `next_object_predictions.npz` (the result of
every test pick, used for the history and per-person analyses), `next_object.log`. The encoder
features are cached in `results/next_object/cache/` (4.2 GB); `--stage fit` refits from it.
The refit with wider weights: the same command with `--stage fit --cache
results/next_object/cache --weights 0.3 1 3 10 30 100 --out results/next_object_wide`, files
in `results/next_object_wide/`.

## Is the model undertrained? (H3)

No test has addressed H3 yet. All results come from models trained for 3 epochs with 30 clips
per recording. A run at the intended size (8 epochs, 60 clips per recording) has never
finished. Two results bear on it: fine-tuning already lowered the error by 0.125 (Test 2), and
the two longer runs had Δ at or below zero before they were stopped (Test 1). The planned test
is Test 10 in [[6-next-steps]].
