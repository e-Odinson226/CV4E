---
type: report
status: running
created: 2026-10-05
updated: 2026-10-09
---

# 4. Results

The tests, in the order of the argument. Test 1 shows that the predictor can learn with
gaze and hand inputs. Test 2 gives the main result: the inputs do not improve the prediction.
Tests 3 and 4 check two explanations for it. Test 8 asks whether what the person looks at
tells what comes next, before gaze is given to the predictor as a position. Test 9 gives the
predictor gaze as a position in the image. Test 10 trains it with the loss of its pretraining
and as a whole. Test 11 replaces it with a predictor built for the signals; its first runs are
here. Each test addresses a hypothesis from [[1-introduction#^hypotheses|the introduction]]. How
each measure works is in [[3-method]]. The weak points that limit Tests 1–4 are in
[[5-discussion#Weak points of the design]]. The rest of the plan is in [[6-next-steps]].

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

![[figures/t3_probe.png]]

- For new people, the probe's median angle error is 11.2°. Always guessing the average gaze
  gives 11.8°.
- The participant can be identified from one frame in 88.7% of cases (chance 14.3%).

**Conclusion.** For a person it has not seen, a linear probe cannot read gaze from the features
of one frame, while it can read palm position. So the failure is specific to gaze and is not a
general failure of the features. In this form R is not supported, and the gaze input can add
information that the features lack.

**Limits.**

- The test asks whether the current gaze can be read. R needs a different question: does gaze
  add information about the future frame beyond the past frames? Test 8 asks this.
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

![[figures/t4_sensitivity.png]]

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

![[figures/t4_attention.png]]

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

![[figures/t8_accuracy.png]]

Paired differences in top-1, in percentage points, with 95% intervals:

| Difference | 0.5 s | 1 s | 2 s | 4 s |
|---|---|---|---|---|
| gaze point − head point | +4.3 [+3.0, +5.9] | +2.3 [+1.3, +3.7] | +1.3 [+0.4, +2.2] | +0.7 [−0.3, +1.6] |
| gaze point − angles | +6.4 [+5.2, +7.5] | +2.4 [+1.0, +3.9] | +0.9 [−0.0, +1.8] | +0.2 [−0.9, +1.1] |
| gaze history 3 s − head history 3 s | +5.3 [+3.7, +7.0] | +2.8 [+1.8, +4.0] | +2.3 [+1.3, +3.4] | −0.0 [−1.6, +1.7] |
| gaze history 3 s − angles | +6.7 [+5.7, +7.9] | +2.5 [+1.1, +4.0] | +1.7 [+0.5, +2.9] | +0.4 [−0.7, +1.6] |
| head point − scene | +1.1 [+0.0, +2.0] | −0.2 [−1.1, +0.8] | −0.2 [−0.8, +0.5] | −0.0 [−0.9, +0.7] |
| angles − scene | −0.9 [−1.5, −0.3] | −0.2 [−0.7, +0.3] | +0.2 [−0.2, +0.5] | +0.5 [+0.2, +0.8] |

![[figures/t8_differences.png]]

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
  difference H4 is about. Whether the predictor can use a gaze position is
  [[#^test9|Test 9]].
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

## Test 9. Does gaze as a position in the image improve the prediction?

Run on 7–8 October 2026: 20 training runs of about 40 minutes each, one after another on one
GPU, then the evaluation on P08 and P09. ^test9

**Question.** Does the predictor make better predictions when gaze tells it where in the image
the person looks? Test 8 found that the features at the gaze point tell which object comes
next, and the three gaze numbers do not. Test 9 gives the predictor gaze as a position and
measures its prediction.

**Hypothesis and prediction.** H4 (gaze form). Written before the runs: if H4 is true, the
models with a gaze position predict better than the model with angles, and better than the
matched model without signals, on people not in the training data. The gain is larger near the
gaze point than on the whole frame.

The expected order, also written before the runs: rope and pe+rope do better than pe, and pe
does about as well as angles or a little better. The reasoning: the frozen heads of blocks
0–17 relate tokens through the rotations of RoPE, so a gaze token rotated like the patch at the
gaze point can use what they learned at once. Coord-PE puts the position into the token's
content, and the patches carry no position in their content, so the frozen heads have nothing
to match it against ([[3-method#Design choices]]). If pe did better than rope, this reasoning
would be wrong, and the notes would have to say why.

The decision rule, fixed before the runs: an effect counts only if its 95% interval excludes 0
and it is larger than the spread between seeds.

**Method.**

- *Two ways to give gaze a position* ([[3-method#Design choices]]):
  - *pe (Coord-PE), the position in the token's content.* Sine and cosine of the gaze point at
    5 frequencies, the finest repeating once per patch, plus the depth and an inside-the-frame
    flag: 22 inputs to the gaze layer. The 20 waves are scaled to a joint length of 1, so the
    input has about the size of the three scaled angles. Unscaled, it was about twice as large,
    and two of four short trial runs diverged.
  - *rope, the position in the attention.* In every block, the gaze token is rotated in the row
    and column channels of each head like the patch at the gaze point, in place of the top-left
    patch (`GazeRoPEAttention` in `ego/gaze_attention.py`). The position is the gaze point minus
    half a patch, so a point at the centre of a patch gets exactly that patch's position. Gaze without a point, or hidden,
    stays at the top-left patch.
- *Models.* Only the gaze input differs.

| Model | Content of the gaze token | Position of the gaze token | Role |
|---|---|---|---|
| none | no signals (signal dropout 1.0) | — | the matched model without signals; `ego_sd1p0` is its seed 0 |
| angles | yaw, pitch, depth | top-left patch | the form of Tests 1–4; seed 0 retrains `ego_ft_v2` |
| pe | Coord-PE of the gaze point, depth, flag | top-left patch | position in the content only |
| rope | yaw, pitch, depth | the gaze point | position in the attention only |
| pe+rope | Coord-PE of the gaze point, depth, flag | the gaze point | both |
| pe+rope shuffled | as pe+rope, trained with the signals of another clip in the batch | the other clip's gaze point | control: the extra input without the information |
| future | gaze and hand of the frame it predicts, one step later | top-left patch | positive control: a signal known to carry information about the target |

![[figures/t9_forms_diagram.png]]

- *Training.* The recipe of `ego_ft_v2` for every model: 3 epochs, P01–P07, 30 clips per
  recording, signal dropout 0.4, the same trained layers and learning rates
  ([[3-method#Training]]). Three seeds per model: 20 new runs.
- *Test set.* The fixed clips of all 12 P08 and all 13 P09 recordings, 24 per recording: 600
  clips from two people not in the training data. Every clip has a gaze point in its last
  context frame.
- *Measures.* The error of the prediction 0.27 s ahead (one step), over the whole frame (256
  patches) and over the patches within 2 patches of the gaze point of the last context frame
  (about 13 patches). Δ is the error with the signals hidden minus the error with them. Each
  model is tested with the input it was trained for: none with the signals hidden, future with
  the signals one step later, the others with the real signals of their own clip. The shuffled
  control is also tested with the real signals of its own clip.
- *Comparisons, fixed before the runs.* pe+rope, rope and pe each against angles (does a
  position help?); each form against none (the value of the information); rope against pe
  (which way matters); pe+rope against pe+rope shuffled (information, or only an extra
  input?); future against none (can the measure show a known effect?).
- *Statistics.* For each form, the error of each clip is averaged over the three seeds. The 95%
  interval of a comparison comes from a bootstrap over the 25 test recordings. The seed spread
  is the standard deviation of the three seeds' mean errors, the larger one of the two forms
  compared. The plan named a bootstrap over recordings and seeds; the seeds enter only through
  the spread.

**Checks.**

- Self-tests (`tests/test_gaze_forms.py`): with every gaze point at the top-left patch, or
  without a point, the rope model gives exactly the output of the angles model, because a
  rotation by 0 changes nothing. The gaze token gets the rotation of the patch at its gaze
  point, with row and column not swapped. The Coord-PE values are right at known points.
- With the new code, `ego_ft_v2` gives the same outputs as before, bit for bit. The fixed clips
  are sampled at the same positions as before, with the same signals.
- *Reproduction.* Angles at seed 0 retrains `ego_ft_v2` with the new code. Before training, the
  check on the 96 P08 clips gives exactly the values of `ego_ft_v2`: 0.6153 with the signals
  hidden, 0.6235 with them. After each epoch, with `ego_ft_v2` in brackets:

| Epoch | Training loss | Error on P08, signals hidden | Error on P08, real signals |
|---|---|---|---|
| 1 | 0.5264 (0.5261) | 0.4943 (0.4942) | 0.4932 (0.4933) |
| 2 | 0.5152 (0.5150) | 0.4903 (0.4900) | 0.4883 (0.4885) |
| 3 | 0.5067 (0.5066) | 0.4875 (0.4877) | 0.4860 (0.4866) |

  The two runs agree within 0.0006 but not exactly: from the same weights, data and seed, the
  training does not repeat bit for bit. On the 600 test clips they give 0.4998 and 0.5001.

**Result.** Error of the prediction 0.27 s ahead on the 600 test clips, mean of three seeds;
lower is better. Δ is over the whole frame.

| Model | Whole frame | Seeds, lowest – highest | Near the gaze point | Δ (hidden − real) |
|---|---|---|---|---|
| none | 0.5006 | 0.5002 – 0.5009 | 0.3790 | 0 (always hidden) |
| angles | 0.5009 | 0.4998 – 0.5022 | 0.3791 | 0.0009 |
| pe | **0.5002** | 0.4999 – 0.5005 | **0.3783** | 0.0011 |
| rope | 0.5009 | 0.5002 – 0.5017 | 0.3794 | 0.0002 |
| pe+rope | 0.5004 | 0.5001 – 0.5007 | 0.3787 | 0.0010 |
| pe+rope shuffled | 0.5007 | 0.5003 – 0.5009 | 0.3790 | 0.0005 |
| future | 0.4993 | 0.4984 – 0.4998 | 0.3767 | 0.0017 |

The comparisons. The gain of A over B is the error of B minus the error of A, positive when A
predicts better, with 95% intervals over recordings:

| A − B | Whole frame | Seed spread | Near the gaze point | Seed spread | Counts |
|---|---|---|---|---|---|
| pe+rope − angles | +0.0005 [+0.0003, +0.0007] | 0.0010 | +0.0004 [−0.0000, +0.0008] | 0.0012 | no |
| rope − angles | +0.0000 [−0.0001, +0.0002] | 0.0010 | −0.0002 [−0.0006, +0.0001] | 0.0012 | no |
| pe − angles | +0.0007 [+0.0004, +0.0010] | 0.0010 | +0.0008 [+0.0004, +0.0012] | 0.0012 | no |
| angles − none | −0.0004 [−0.0006, −0.0002] | 0.0010 | −0.0001 [−0.0006, +0.0003] | 0.0012 | no |
| pe − none | +0.0003 [+0.0000, +0.0006] | 0.0003 | +0.0007 [+0.0003, +0.0011] | 0.0004 | both; the whole frame narrowly |
| rope − none | −0.0003 [−0.0005, −0.0001] | 0.0007 | −0.0004 [−0.0007, −0.0000] | 0.0007 | no |
| pe+rope − none | +0.0001 [−0.0001, +0.0003] | 0.0003 | +0.0002 [−0.0001, +0.0006] | 0.0004 | no |
| rope − pe | −0.0006 [−0.0009, −0.0003] | 0.0007 | −0.0011 [−0.0015, −0.0006] | 0.0007 | near the gaze point: pe better |
| pe+rope − pe+rope shuffled | +0.0002 [+0.0000, +0.0004] | 0.0003 | +0.0003 [−0.0000, +0.0006] | 0.0003 | no |
| future − none | +0.0013 [+0.0008, +0.0017] | 0.0006 | +0.0023 [+0.0014, +0.0032] | 0.0008 | both |

"Counts" applies the decision rule to the unrounded values.

![[figures/t9_errors.png]]

![[figures/t9_comparisons.png]]

- *The positive control works.* future beats none on both measures, by more than the seed
  spread, and in all 9 pairs of seeds. So the test can show an effect of this size. The effect
  is small: 0.0013 is 0.26% of the error over the whole frame, and 0.0023 is 0.6% of the error
  near the gaze point.
- *Two runs had a loss spike.* At the start of training, the loss of angles seed 1 rose to 1.04
  and that of rope seed 1 to 0.77, within the first 40 steps. Their gradient norms reached 51 and
  6; in every other run they stay at or below 0.2 (training clips them at 1.0). The training has
  no warmup: the new layers start at their full learning rate of $10^{-3}$. Both runs recovered
  within 50 steps, but they are the worst models of their form on the test set (0.5022 and
  0.5017). The analysis fixed before the runs keeps them. The table below repeats the
  comparisons that involve them without the two runs, an analysis chosen after the run.

![[figures/t9_training.png]]

| A − B, without the two runs | Whole frame | Seed spread | Near the gaze point | Seed spread | Counts |
|---|---|---|---|---|---|
| pe+rope − angles | −0.0001 [−0.0003, +0.0000] | 0.0005 | −0.0003 [−0.0007, +0.0001] | 0.0007 | no |
| rope − angles | −0.0001 [−0.0003, −0.0000] | 0.0005 | −0.0005 [−0.0010, −0.0001] | 0.0007 | no |
| pe − angles | +0.0001 [−0.0002, +0.0003] | 0.0005 | +0.0001 [−0.0002, +0.0005] | 0.0007 | no |
| angles − none | +0.0003 [+0.0000, +0.0004] | 0.0005 | +0.0006 [+0.0001, +0.0011] | 0.0007 | no |
| rope − none | +0.0001 [−0.0001, +0.0003] | 0.0003 | +0.0001 [−0.0004, +0.0005] | 0.0004 | no |
| rope − pe | −0.0002 [−0.0005, +0.0001] | 0.0003 | −0.0006 [−0.0010, −0.0002] | 0.0004 | near the gaze point: pe better |

The other comparisons contain neither run and do not change.

- *pe helps a little.* pe beats none near the gaze point by 0.0007, in all 9 pairs of seeds:
  about a third of the positive control's gain there. Over the whole frame its gain is 0.0003
  and passes the rule narrowly: 0.00032 against a spread of 0.00028, and 7 of 9 pairs of seeds.
  No pe run and no none run had a loss spike.
- *pe and angles are equal without the spiked runs.* With all seeds, pe beats angles by 0.0007,
  within the seed spread. Without angles seed 1 the difference is +0.0001 on both measures.
  angles then gains over none about as much as pe: +0.0003 over the whole frame and +0.0006 near
  the gaze point, within its seed spread. So the position in the token's content adds nothing
  measurable to the three angles.
- *rope does not help.* With all seeds, rope is the worst form. Without rope seed 1 it is about
  equal to none (+0.0001), and below angles near the gaze point by 0.0005, within the seed
  spread. pe beats rope near the gaze point in both analyses, by more than the seed spread: by
  0.0011 with all seeds and by 0.0006 without the spiked run.
- *The expected order is wrong.* Written before the runs: rope and pe+rope, then pe, then
  angles. Measured near the gaze point, in the means: with all seeds, pe, then pe+rope, then
  rope and angles; without the spiked runs, pe and angles, then pe+rope, then rope and none.
- *Information or extra input.* pe+rope beats its shuffled control by 0.0002 over the whole
  frame and by 0.0003 near the gaze point, both within the seed spread. pe was not trained with
  shuffled signals.
- *Seeds differ more than forms.* The range of the angles seeds, 0.0024, comes mostly from the
  run with the spike. Without the spiked runs, the seeds of one form still differ by 0.0006 to
  0.0014 over the whole frame, more than most gains of a gaze form, and more than the +0.0003
  that Test 2 measured with one run per model.
- *Δ.* The rope models depend least on their signals: hiding them costs 0.0002 (0.0004 without
  the spiked run), against 0.0009–0.0011 for angles, pe and pe+rope and 0.0017 for future.

**Where in the frame the gain is.** An analysis after the run: the error of every patch,
grouped by its distance from the gaze point of the last context frame. Mean of seeds without
the two runs with a loss spike; 95% intervals over recordings.

![[figures/t9_distance.png]]

![[figures/t9_gain_map.png]]

| Distance from the gaze point (patches) | 0–1 | 1–2 | 2–4 | 4–8 | 8–12 | 12 or more |
|---|---|---|---|---|---|---|
| Patches per clip | 3.2 | 9.4 | 37.2 | 112.9 | 80.2 | 13.2 |
| Error of none | 0.378 | 0.379 | 0.394–0.408 | 0.458–0.520 | 0.577 | 0.427 |
| Gain of future over none (10⁻³) | +2.3 [+1.2, +3.3] | +2.3 [+1.4, +3.1] | +1.9 to +2.2 | +1.3 to +1.7 | +0.6 [+0.2, +1.0] | −0.1 [−0.6, +0.4] |
| Gain of pe over none (10⁻³) | +0.7 [+0.1, +1.3] | +0.7 [+0.2, +1.2] | +0.5 to +0.6 | +0.3 to +0.5 | +0.1 [−0.2, +0.4] | −0.6 [−1.0, −0.2] |
| Gain of angles over none (10⁻³) | +0.5 [−0.2, +1.2] | +0.6 [+0.1, +1.1] | +0.4 to +0.5 | +0.3 | +0.1 [−0.1, +0.3] | −0.4 [−0.7, −0.0] |
| Gain of rope over none (10⁻³) | −0.5 [−1.1, +0.1] | +0.3 [−0.2, +0.7] | +0.2 to +0.3 | +0.1 to +0.2 | +0.0 [−0.1, +0.2] | −0.3 [−0.6, +0.1] |

- The error of none is lowest at the gaze point (0.378) and highest 8–12 patches away (0.577),
  towards the edges of the frame. The gaze point lies near the middle of the frame and below it
  (on average column 8.9, row 11.2 of 16). So "near the gaze point" measures the patches with
  the lowest error.
- The positive control gains most near the gaze point and less with distance: +0.0023 within 2
  patches, +0.0006 at 8–12 patches. pe and angles follow the same shape at a quarter to a third
  of its size.
- In the outermost ring, 12 or more patches away (13 patches per clip, mostly at the top of the
  frame), pe, angles and pe+rope predict worse than none, by 0.0004–0.0006. The shuffled control
  does too (−0.0005), so this cost does not come from the gaze information.
- The three models whose gaze token sits at the gaze point, rope, pe+rope and the shuffled
  control (tested with its own gaze point), predict the patches within 1 patch of the gaze
  point worse than those 1–2 patches away: −0.0005, −0.0002 and −0.0005, against about
  +0.0003, +0.0004 and +0.0001. pe and angles show no such dip. The intervals are wide, because
  the inner ring holds about 3 patches per clip; only that of the shuffled control excludes 0.
  If the dip is real, the rotation does act on the patches at the gaze point, and it harms
  them there.

**Why pe did better than rope.** pe beats rope near the gaze point in both analyses. Why is not
known yet. Three explanations fit the results; none has been tested.

1. *Placing a token does not make the heads attend to it.* In a RoPE head, the attention
   between two tokens depends on their distance only through their content: a head prefers
   nearby tokens only if their queries and keys have suitable content. The frozen heads learned
   this for image patches. The gaze token's content comes from three numbers through a new
   layer and is unlike any patch, so the frozen heads need not prefer the patches near it. The
   reasoning before the runs assumed that they would.
2. *rope moves a token that the frozen blocks rely on.* In pretraining this token was the
   robot's action token, always at the top-left patch, and the frozen blocks give it 38 times
   the equal share of attention ([[#^test4|Test 4b]]). rope moves it in every frame, which
   changes these learned patterns, and 3 epochs may be too few to adapt. The small Δ of the
   rope models fits a model that relies little on its gaze token.
3. *pe gives the position to the trained layers.* In pe, the gaze layer and the 6 trained
   blocks receive the position as numbers and can learn to use them. In rope, the position acts
   only through the rotations, and the token's content still holds only the three numbers.

A check that tells the first explanation apart from the others: in the rope models, measure as
in Test 4b whether image tokens near the gaze point give the gaze token more attention than
tokens far from it. If they do not, the frozen heads do not use the rotation. The dip of the
rope models at the gaze point, above, counts against the first explanation and fits the
second: a patch at the gaze point may take in a token that holds no image content.

**Conclusion.**

- H4 (gaze form), in the predictor, is not supported. With all seeds pe is the best form, but
  its lead over angles rests on one angles run with a loss spike. Without the two such runs,
  gaze as a position in the token's content (pe) does as well as the three angles, not better.
  Gaze as a position in the attention (rope) does no better than no signals, and worse than pe
  near the gaze point. The prediction made before the runs, that rope would do best, was wrong.
- M (main) is not supported at a useful size. Gaze and hand, as angles or as pe, lower the error
  by about 0.0003 over the whole frame (0.06%) and by 0.0006–0.0007 near the gaze point: a
  quarter to a third of what the signals of the target frame give. Only the gain of pe passes
  the decision rule.
- The test detects the positive control and the gain of pe over none. So the gains of the gaze
  forms are small; the measure is able to show effects of this size.
- At 0.27 s, over the whole frame, even the gaze and hand of the target frame lower the error by
  only 0.26%. This puts a number on the weak point "short horizon, coarse measure": the measure
  leaves little room for any signal. In Test 8 the gaze point told most about the next object
  0.5 s ahead (H1 (horizon)).
- The training is not stable for every seed: 2 of 20 runs had a loss spike at the start, and
  without them the seeds of one form still differ by up to 0.0014. One run per model is not
  enough to compare forms.

**Limits.**

- The short recipe: 3 epochs, 30 clips per recording (H3 (undertrained)). A form that needs more
  training to adapt, such as rope, may be judged too early.
- The intervals resample recordings, not training runs. Three seeds give only a rough spread.
- The training has no warmup, and 2 of 20 runs had a loss spike at the start. The analysis
  without them was chosen after the run.
- One horizon, 0.27 s. The error near the gaze point uses the gaze point of the last context
  frame, not the object about to be picked up.
- pe has no shuffled control, so it is not shown that its gain comes from the gaze information
  and not from the extra input.
- Two test people.

**Reproduce.** The runs: `checkpoints/test9/queue2.sh` holds the training recipe, and
`checkpoints/test9/queue.log` the options of each run (`--gaze-form`, `--future-signals`,
`--shuffle-signals batch`, `--signal-dropout 1.0`, `--seed`). Each run is in
`checkpoints/test9/<model>_s<seed>/` (pe+rope is `perope`, its shuffled control `peropeshuf`):
`best.pt`, `final.pt`, `train.log`, `metrics.jsonl`. The evaluation: `python -m ego gaze-forms
--video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos --gaze-dir
data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze --runs checkpoints/test9 --out results/test9`.
Files in `results/test9/`: `gaze_forms.csv` (every model), `gaze_forms_comparisons.csv`,
`gaze_forms.json`, `gaze_forms.log`, and `scores/` (the errors of every clip and model). The
encoder features are cached in `results/test9/cache/` (7.8 GB). The figures and the comparisons
without the runs with a loss spike: `python -m ego figures --only test9`, which writes
`docs/figures/t9_*.png` and `results/test9/gaze_forms_comparisons_without_spikes.csv`
(the spike rule: a logged gradient norm above 1.0).

## Test 10. The loss of the pretraining, and training the whole predictor

Run on 8 October 2026. Twelve models of the V-JEPA 2-AC predictor, four arms with three seeds
each, scored on the 600 clips of Test 9 together with the twenty models of Test 9. ^test10

**Question.** Every model before this one was trained with the mean squared difference (MSE),
while V-JEPA 2-AC was pretrained with the mean absolute difference (L1), and only the last 6 of
its 24 blocks were trained ([[3-method#Training]]). Do the loss of the pretraining, or training
the whole predictor, change how well the model predicts and how much gaze helps?

**Hypothesis and prediction.** H3 says the model is undertrained. If the frozen blocks are what
keeps the predictor from using gaze, then the gain of gaze over the matched model is larger when
the whole predictor trains than when only the last 6 blocks do. Written before the runs: each
model does better on the measure of its own loss; training the whole predictor lowers the error
of the model without signals by more than the spread between seeds; and if the frozen blocks
were the limit, comparison 6 below is larger than comparison 3.

**Method.** Four arms: `pe l1` and `none l1` train the last 6 blocks with L1, and `pe l1 full`
and `none l1 full` train the whole predictor, all 24 blocks and `predictor_embed`, 305.2 million
parameters in place of 77.0 million. `none` is signal dropout 1.0, the matched model. Three seeds
each. The recipe of Test 9 in every other respect: 3 epochs, P01-P07, 30 clips per recording,
batch size 16, no warmup. So each comparison changes one thing. The measures are L1 and MSE over
the whole frame and within 2 patches of the gaze point, on the 600 clips of P08 and P09, against
the references without fine-tuning ([[3-method#Prediction error and Δ]]). Each gain has a 95%
interval from a bootstrap over the 25 test recordings, and an effect counts only if that interval
excludes 0 and the gain is larger than the spread between seeds.

**Result.** The primary comparison was 6, near the gaze point, on L1.

| | A − B | L1 | 95% interval | Seed spread | Counts |
|---|---|---|---|---|---|
| 1 | pe l1 − pe | +0.0110 | 0.0108 to 0.0113 | 0.0001 | yes |
| 2 | none l1 − none | +0.0110 | 0.0108 to 0.0113 | 0.0002 | yes |
| 3 | pe l1 − none l1 | +0.00016 | 0.00009 to 0.00023 | 0.00017 | no |
| 4 | none l1 full − none l1 | +0.0041 | 0.0035 to 0.0047 | 0.0002 | yes |
| 5 | pe l1 full − pe l1 | +0.0041 | 0.0034 to 0.0047 | 0.0001 | yes |
| 6 | pe l1 full − none l1 full | +0.00013 | −0.00004 to 0.00031 | 0.0002 | no |

Near the gaze point, the two comparisons that measure gaze are 3, +0.00019 (0.00001 to 0.00037,
seed spread 0.00029), and 6, +0.00027 (−0.00011 to 0.00062, seed spread 0.00028). Neither counts.

- *The loss.* Training with L1 lowers the L1 error by 0.0110 with gaze and without it, and by
  0.0059 and 0.0060 near the gaze point. The exchange is not even: it raises the MSE error by
  0.0611 and 0.0607. Each model does better on the measure of its own loss, as predicted, but the
  sizes differ by a factor of about five.
- *Training the whole predictor.* It lowers the L1 error by 0.0041 without signals and by 0.0041
  with gaze, and by 0.0043 and 0.0044 near the gaze point. Against the blend of the past frames,
  the model without signals improves from 0.0428 to 0.0469 over the whole frame and from 0.0337 to
  0.0380 near the gaze point. Held-out error fell at every epoch, so 3 epochs did not overfit the
  3,900 training clips.
- *Gaze.* The gain of gaze is +0.00016 with the last 6 blocks and +0.00013 with the whole
  predictor, over the whole frame, and +0.00019 and +0.00027 near the gaze point. Every one of
  these is at or below the spread between seeds, and the two for the full fine-tune have intervals
  that include 0. Training the whole predictor did not make gaze worth more.

**Conclusion.** The loss and the trainable depth both change how well the predictor predicts, and
both by much more than gaze ever has: 0.0110 for the loss and 0.0041 for the depth, against
0.0001 to 0.0003 for gaze. Comparison 4 holds, so later predictors are trained as a whole.
Comparison 6, the primary comparison, does not: gaze is worth no more to a predictor that can
adapt all its blocks than to one with 18 frozen ones.

That answers the part of H3 that this test was built for, and it removes one explanation offered
in [[#^test9|Test 9]] for why the rope form did not help. Explanation 2 there was that rope moves
a token the frozen blocks rely on and 3 epochs are too few to adapt. If frozen blocks were what
held gaze back, the full fine-tune would have released it. It did not. What remains is explanation
1, which is about the attention itself and not about what trains: in a RoPE head the attention
between two tokens depends on their content, not their distance, so a gaze token placed at the
gaze point is not a token the other tokens attend to. This is the finding Test 11 is built on.

**Limits.** Three epochs, as in Tests 9 and 10, so the longer run of H3 is still untested. Three
seeds per arm, and the gaze gains are the same size as the spread between them, so this test can
rule a large gain out but cannot measure a small one. One gaze form, pe, the best of Test 9. The
measure is the error of the predicted embeddings 0.27 s ahead, where even the gaze and hand of
the target frame gain only 0.26% ([[#^test9|Test 9]]).

**Reproduce.** `checkpoints/test10/queue.sh` with `checkpoints/test10/todo.txt`, which holds the
options of each run (`--loss l1`, `--unfreeze-last-n 24 --unfreeze-embed`). The evaluation:
`python -m ego gaze-forms --runs checkpoints/test9 checkpoints/test10 --cache
results/test9/cache --out results/test10`. Files: `results/test10/gaze_forms.csv`,
`gaze_forms_comparisons.csv`, `gaze_forms_references.csv`.

## Test 11. A predictor designed for gaze and hand: first runs

Run on 8–9 October 2026: one run each of two arms, maps and none, 6.3 hours of training on one GPU, then the
evaluation on P08 and P09. These are the first runs of the plan in [[6-next-steps]]. There is
one seed per arm, so the decision rule of the plan cannot be applied. ^test11

**Question.** Tests 1–4, 9 and 10 gave gaze to the V-JEPA 2-AC predictor in the slot its
pretraining built for the robot's action. Does a predictor built for gaze and hand predict
better with them than without, when no pretrained conditioning slot and no pretrained step
constrain the design?

**Hypothesis and prediction.** H4 (gaze form). Written before the runs: if the form of the gaze
input is the reason for the null of Tests 2 and 9, a predictor whose gaze marks the image tokens
gains more from gaze than the V-JEPA 2-AC predictor did, and the gain is larger near the gaze
point than over the whole frame. The runs predict 0.53 s and 1.07 s ahead, so they also bear on
H1 (horizon).

**Method.**

- *The predictor* (`ego/ego_predictor.py`). 12 blocks of width 384, 22.4 million parameters,
  trained from the start on the tokens of the frozen V-JEPA 2 encoder. RoPE over time, row and
  column, and attention over all tokens. The frames to predict are mask tokens at their own
  time slots, so both horizons come from one forward pass.
- *The signals as maps.* The token of patch $p$ at observed step $\tau$ becomes
  $x_{\tau,p} + \alpha_g\, k(p, g_\tau)\, e_g + \alpha_h \left( k(p, l_\tau)\, e_l + k(p, r_\tau)\, e_r \right)$,
  where $k$ is a Gaussian over the patch grid whose width $\sigma$ starts at 1 patch and is
  learned ([[6-next-steps]]). The vectors $e$ start at 0, so every run starts as the model
  without signals.
- *Arms.* The shuffled and token arms of the plan were not run.

| Arm | Signals | Role |
|---|---|---|
| maps | the gaze point and both palms of each observed frame, as maps | the design |
| none | the maps add nothing | the matched model |


- *Training.* P01–P07, 40 clips per recording per epoch (5,240 clips), batch size 16, L1 loss,
  AdamW with a learning rate of $10^{-4}$ and weight decay 0.01. The learning rate rises over 500
  steps and then falls to 0 along a cosine over the planned number of epochs. 8 context frames,
  8 frames apart (2.1 s). The frames to predict are 2 and 4 steps after the last context frame,
  0.53 s and 1.07 s. Seed 0 in every arm. No signal dropout: a signal is hidden only where its
  point is missing in the data. maps and none trained for 13 epochs, 3.2 hours each.
- *Same start, same clips.* maps and none start from the same weights and draw the same
  training clips in the same order. Their training losses agree to the fifth digit over the
  first 50 steps. So maps − none shows what the maps change, with no difference in start or data. It
  does not show how much another seed would change it.
- *Test set.* 24 clips from each of the 25 recordings of P08 and P09, 600 clips. These are the
  people and recordings of Tests 9 and 10, but not their clips, because a clip here spans the
  2.1 s of context and the horizon. Every clip has a gaze point in its last context frame.
- *Measures.* L1 and MSE between the predicted tokens and the frozen encoder's tokens of each
  frame to predict, over the whole frame and within 2 patches of the gaze point of the last
  context frame ("near"), as in Tests 9 and 10. Both token sets are layer-normalized, so the MSE
  equals $2(1-\rho)$, where $\rho$ is the correlation between a predicted and a true token. Each
  model is scored with its points and with the points hidden. The references are repeating the
  last context frame and the blend of the context frames of Tests 9 and 10. Each model is scored
  at its last epoch, which was its best held-out epoch, and at epoch 3.
- *Statistics.* The gain of A over B is the error of B minus the error of A, clip by clip, with
  a 95% interval from a bootstrap over the 25 test recordings. With one seed per arm there is no
  spread between seeds.

**Checks.**

- On the 96 P08 clips of the check after each epoch, the evaluation gives exactly the errors
  that the training logged, with the points and with them hidden.
- No run had a loss spike. The largest gradient norm in any run is 0.87, at the first logged
  step, below the clip at 1.0. In Test 9, without a warmup, 2 of 20 runs had one.

**Result.** Error on the 600 test clips; lower is better.

| Model | L1, 0.53 s | L1, 1.07 s | MSE, 0.53 s | MSE, 1.07 s | MSE near, 0.53 s | MSE near, 1.07 s |
|---|---|---|---|---|---|---|
| repeat last frame | 0.5532 | 0.5778 | 0.8224 | 0.8840 | 0.6482 | 0.7163 |
| blend of past frames | 0.4780 | 0.4940 | 0.6173 | 0.6558 | 0.4946 | 0.5347 |
| none, 13 epochs | 0.4576 | 0.4622 | 0.6720 | 0.6898 | 0.5697 | 0.5866 |
| maps, 13 epochs | 0.4576 | 0.4622 | 0.6720 | 0.6898 | 0.5693 | 0.5862 |

The gains, in units of $10^{-4}$, at 0.53 s / 1.07 s:

| A − B | What it measures | L1 | L1 near | MSE | MSE near |
|---|---|---|---|---|---|
| maps − none | the value of the maps | +0.04 / +0.03 | +0.2 / +0.3 | +0.03 / −0.20 | +3.8 / +3.6 |
| maps − maps with the points hidden | the Δ within maps | +0.07 / +0.08 | +0.5 / +0.7 | +1.4 / +1.3 | +10.5 / +10.3 |
| none − maps with the points hidden | the cost of hiding an input maps expects | +0.03 / +0.06 | +0.3 / +0.4 | +1.3 / +1.5 | +6.7 / +6.8 |

For maps − none, the 95% intervals are, on MSE near, [+3.4, +4.2] and [+3.2, +4.0]; on L1 near,
[+0.1, +0.3] and [+0.2, +0.4]; on MSE over the whole frame, [−0.03, +0.10] and [−0.28, −0.12].
The intervals of the other rows exclude 0 (`results/test11/eval_ego_comparisons.csv`).

![[figures/t11_training.png]]

- *The maps change the prediction near the gaze point, by little.* maps − none is +0.00038 on
  MSE near the gaze point at 0.53 s and +0.00036 at 1.07 s, 0.07% of the error there. maps is
  better in 80% of the clips and in all 25 recordings. On L1, the loss the predictor trains on,
  the gain near the gaze point is 0.00002–0.00003. Over the whole frame it is 0.000004 on L1, and
  on MSE 0 at 0.53 s and −0.00002 at 1.07 s. The gain is the same at both horizons.
- *Hiding the points mostly measures something else.* Within maps, hiding the points raises the
  error near the gaze point by 0.00105 (MSE). Of this, 0.00067 is the cost of hiding an input
  that maps expects: maps with its points hidden predicts worse than none. 0.00038 is the value
  of the maps. Test 2 found the same pattern: +0.0011 within the model, +0.0003 against the
  matched model. maps saw no hidden points in training, except where a point was missing. The
  check after each epoch reports only the Δ within a model, so its rise over the epochs does not
  show that the maps help. Over the epochs that Δ rises from 0 to 0.00014 (MSE, whole frame),
  while maps − none stays between −0.00012 and +0.00008 with no trend.
- *How much the maps move a token.* At the end of training, $\alpha \lVert e \rVert$ is 0.22 for
  gaze and 0.20–0.21 for each palm. The image tokens after the input layer have a mean norm of
  17.5, so the maps change the token at the gaze point by about 1.3% of its length. $\sigma$
  grew from 1 to 1.10 patches.
- *The predictor beats the blend on L1, not on MSE.* Against the blend of the past frames, none
  lowers the L1 error by 0.020 at 0.53 s and 0.032 at 1.07 s, but raises the MSE by 0.055 and
  0.034. Since the MSE is $2(1-\rho)$, the predicted tokens correlate less with the true tokens
  than the blend does: $\rho$ = 0.664 against 0.691 at 0.53 s, 0.655 against 0.672 at 1.07 s.
  Near the gaze point at 0.53 s, the L1 gain over the blend is +0.003, with an interval that
  includes 0. Training with L1 had this cost in Test 10 too, where it raised the MSE of the
  V-JEPA 2-AC predictor by 0.061. There the predictor still beat the blend on both measures at
  0.27 s, by 0.043 on L1 and 0.012 on MSE without signals.

![[figures/t11_decomposition.png]]

![[figures/t11_references.png]]

**Where in the frame the gain is.** The error of every patch, grouped by its distance from the
gaze point of the last context frame, or from the nearest palm in that frame. Gains in units of
$10^{-4}$ MSE, 95% intervals over recordings.

![[figures/t11_distance.png]]

![[figures/t11_gain_map.png]]

| Distance (patches) | 0–1 | 1–2 | 2–3 | 3–4 | 4–6 | 6–8 | 8–12 | 12 or more |
|---|---|---|---|---|---|---|---|---|
| Patches per clip, from the gaze point | 3.2 | 9.4 | 15.8 | 21.4 | 54.7 | 58.2 | 80.2 | 13.2 |
| Error of none, 0.53 s | 0.566 | 0.571 | 0.586 | 0.602 | 0.646 | 0.694 | 0.731 | 0.538 |
| maps − none, from the gaze point, 0.53 s | +7.0 [+6.1, +7.8] | +2.7 [+2.3, +3.0] | +0.1 | −0.5 | −0.4 | −0.2 | −0.1 | +0.1 |
| maps − none, from the gaze point, 1.07 s | +6.4 [+5.6, +7.1] | +2.6 [+2.3, +3.0] | +0.1 | −0.6 | −0.6 | −0.4 | −0.4 | −0.1 |
| maps − none, from the nearest palm, 0.53 s | +4.7 [+4.0, +5.4] | +2.7 [+2.3, +3.1] | +0.5 | −0.3 | −0.4 | −0.2 | −0.1 | +0.2 |
| none − maps with points hidden, from the gaze point, 0.53 s | +8.3 | +6.1 | +3.9 | +2.4 | +1.3 | +0.7 | +0.4 | +0.1 |

- *The gain lies where the maps are drawn.* maps − none is +0.0007 within 1 patch of the gaze
  point and +0.0003 at 1–2 patches, about the width of the kernel ($\sigma$ = 1.1 patches), and
  the same at both horizons. Within 1 patch it is 0.12% of the error there.
- *Farther away the maps cost a little.* From 3 to 8 patches, maps − none is −0.00002 to −0.00006
  per patch, with intervals that exclude 0, and at 1.07 s also beyond 8 patches. 228 of the 256
  patches of a clip lie 3 or more patches from the gaze point. Over the whole frame the local gain
  and this cost cancel, which is why
  maps − none is about 0 over the whole frame, and below 0 at 1.07 s.
- *Around the palms the pattern is the same:* +0.0005 within 1 patch of the nearest palm, +0.0003
  at 1–2 patches. The gaze point lies a median of 2.0 patches from the nearest palm (491 clips
  with a palm), so the gains around gaze and around the palms overlap and are not separated here.
- *The cost of hiding the points spreads wider than the gain:* +0.0008 at the gaze point and
  still +0.0002 at 3–4 patches. Without its points, maps predicts worse over a larger region than
  the one where the points help it.
- The error of none is lowest at the gaze point (0.566) and highest 8–12 patches away (0.731), as
  in Test 9.

**Comparison with Tests 9 and 10.** The gain of gaze against the matched model. Tests 9 and 10:
three seeds each, at 0.27 s, on the clips of Test 9. Test 11: one seed, at 0.53 s / 1.07 s.

| Test | Predictor | L1 | L1 near | MSE | MSE near |
|---|---|---|---|---|---|
| 9: pe − none | V-JEPA 2-AC, last 6 blocks, MSE loss | +0.00017 | +0.00027 | +0.00032 | +0.00070 |
| 9: future − none | the same, positive control | +0.00046 | +0.00081 | +0.00126 | +0.00231 |
| 10: pe l1 − none l1 | V-JEPA 2-AC, last 6 blocks, L1 loss | +0.00016 | +0.00019 | −0.00015 | +0.00016 |
| 10: pe l1 full − none l1 full | V-JEPA 2-AC, whole predictor, L1 loss | +0.00013 | +0.00027 | +0.00043 | +0.00057 |
| 11: maps − none | new, from the start, L1 loss | +0.000004 / +0.000003 | +0.00002 / +0.00003 | +0.000003 / −0.00002 | +0.00038 / +0.00036 |

![[figures/t11_gains.png]]

- On L1, the gain of the maps is much smaller than the gain of pe in Tests 9 and 10: over the
  whole frame 0.000003–0.000004 against 0.00013–0.00017, 30 to 60 times smaller, and near the
  gaze point 0.00002–0.00003 against 0.00019–0.00027, 6 to 13 times smaller.
- On MSE near the gaze point, the gain is of the same size as in Tests 9 and 10 (0.00016–0.00070),
  and smaller than the spread between their seeds there (0.0004–0.0016).
- Over the whole frame on MSE, the maps gain nothing, as pe in Test 10 with the last 6 blocks.
- The positive control of Test 9 was detected on every measure. Test 11 has none.

**Conclusion.**

- In these first runs, gaze and hand as maps over the image tokens lower the error near the gaze
  point by 0.0004 on MSE, consistently over clips and recordings, and change nothing over the
  whole frame or on L1, the loss the predictor trains on. H4's prediction, that this form gains
  more than the V-JEPA 2-AC predictor did, is not supported by them: near the gaze point on MSE
  the gain is the same size as in Tests 9 and 10, and on L1 it is smaller.
- The gain is local. It lies within about 2 patches of the gaze point and of the palms, the width
  of the maps, and a small cost over the rest of the frame cancels it over the whole frame.
- The gain is the same at 0.53 s and 1.07 s. These runs do not show more room for gaze at the
  longer horizon (H1).
- There is no positive control, so these runs cannot say how large a gain the measure can show
  at these horizons.
- Without signals, the predictor trained from the start beats the blend of the past frames on L1
  but not on MSE: its tokens correlate less with the future than an average of the past frames
  does. A gain of gaze is hard to read in a prediction that is still below this reference (H3).
- The Δ within a model again overstates the value of the signals: about two thirds of it near
  the gaze point is the cost of hiding an input the model expects.

**Limits.**

- One seed per arm. The arms share their start and their training clips, which makes maps − none
  precise for this seed, but the spread between seeds is not known.
- No positive control. The shuffled and token arms were not run, so it is not shown that the
  gain of the maps comes from the information of the points and not from the extra input.
- No signal dropout, so the Δ within a model mixes the value of the points with the cost of
  hiding them.
- The measure of the plan on the object about to be picked up is not built; near the gaze point
  is the closest measure available.
- The test clips differ from those of Tests 9 and 10, and the horizons differ, so the errors
  compare across tests only through the gains and the references.
- Two test people.

**Reproduce.** The runs: `checkpoints/test11/queue.sh` with `checkpoints/test11/todo.txt`, which
holds the options of each run (`--arm`, `--seed`, `--epochs`, `--max-hours`); the recipe is in
`queue.sh`. Each run is in `checkpoints/test11/<arm>_s0/`: `best.pt`, `epoch3.pt`, `final.pt`,
`train.log`, `metrics.jsonl`, `run.out`. `checkpoints/test11/future_s0/` holds a run of an arm
that is no longer part of the test; `eval-ego` skips it. The evaluation: `python -m ego eval-ego --checkpoint
data/model_checkpoints/vjepa2-ac-vitg.pt --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos
--gaze-dir data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze --runs checkpoints/test11 --out
results/test11 --patches` (about 12 minutes). Files in `results/test11/`: `eval_ego.csv` (the
error of every model and input), `eval_ego_comparisons.csv`, `scores.npz` (every clip),
`eval_ego.log`, and `cache/patches.npz` (the error of every patch). The figures: `python -m ego
figures --only test11`, which writes `docs/figures/t11_*.png`.

## Is the model undertrained? (H3)

Two tests address parts of H3. Test 10 trained the whole V-JEPA 2-AC predictor for 3 epochs:
the error fell by 0.0041 (L1), and gaze gained no more ([[#^test10|Test 10]]). Test 11 trained a
new predictor from the start for 13 epochs with a warmup: no run had a loss spike, the held-out
error fell at every epoch, and the gain of gaze is no larger than in Tests 9 and 10; without
signals, that predictor is still below the blend of the past frames on MSE
([[#^test11|Test 11]]). A run of the V-JEPA 2-AC predictor at the intended size (8 epochs, 60
clips per recording) has never finished. Earlier results: fine-tuning lowered the error by 0.125
(Test 2), and the two longer runs had Δ at or below zero before they were stopped (Test 1).
