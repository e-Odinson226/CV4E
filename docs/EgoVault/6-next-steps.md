---
type: report
status: running
created: 2026-09-18
updated: 2026-10-07
---

# 6. Next steps

What to do next, in order. Each planned test names the hypothesis it addresses
([[1-introduction#^hypotheses|hypotheses table]]) and the weak points it removes
([[5-discussion#Weak points of the design]]). When a test is run, its section moves to
[[4-results]] under the same number.

The gaze form is tested before the full training run. Test 8 came
first, as a check without training. Its result decided that Test 9 goes ahead.

| Order | Work | Hypothesis | Weak points addressed | Training runs | Status |
|---|---|---|---|---|---|
| 1 | Gaze position in the image | H4 | gaze cannot point | none | Done and checked ([[3-method#Gaze position in the image]]) |
| 2 | Test 8. Does the gaze point tell what comes next? | H4, R | gaze cannot point, short horizon | none | Done ([[4-results#^test8\|Test 8]]): the decision rule was met |
| 3 | Test 9. Does gaze as a position in the image improve the prediction? | H4, M | gaze cannot point, no positive control, too little evidence | 20 runs of about 40 min | Running since 7 October 2026 |
| 4 | Test 10. The full training run | H3 | too little evidence | 3 or more longer runs | Planned |

## 1. Gaze position in the image

Done. How the gaze point is computed, the calibration and the checks are in
[[3-method#Gaze position in the image]]. At the moment of a pick, the gaze point lies in the
box of the object in 38.8% of picks, against 19.7% by chance. Its error is about half a patch
for most people.

## 2. Test 8 (done). Does the gaze point tell what comes next?

Run on 7 October 2026: [[4-results#^test8|Test 8]]. The features at the gaze point tell which
object is picked up next, beyond the scene, the three gaze numbers and the head point. The
decision rule was met at 0.5 s and at 1 s, so Test 9 goes ahead.

## 3. Test 9 (running). Does gaze as a position in the image improve the prediction?

**Question.** Does the predictor make better predictions when gaze tells it where in the image
the person looks?

**What Test 8 decided.** Test 9 goes ahead ([[4-results#^test8|Test 8]]). The features at the
gaze point told which object comes next, and the three gaze numbers did not. This supports the
forms that give gaze a position. The gain was largest 0.5 s before the pick. The gaze history
of the last 1–3 s added at most about 1 point beyond the current gaze point, and a longer
history nothing more. The 8 context frames, each with its own gaze token, already cover the
last 2 s. So the step stays at 0.27 s, the context at 8 frames, and no gaze memory is added
([[3-method#Design choices]]).

**Two ways to give gaze a position** ([[3-method#Design choices]]):

- *Coord-PE, the position in the content.* The token is built from sine and cosine features of
  the gaze point, plus the depth and an inside-the-frame flag. With $u', v'$ the gaze point
  scaled to $[-1, 1]$, the features are
  $\big(\sin(2^k \pi u'),\ \cos(2^k \pi u'),\ \sin(2^k \pi v'),\ \cos(2^k \pi v')\big)$ for
  $k = 0, \dots, K-1$.
- *RoPE position, the position in the attention.* The token is rotated in height and width like
  an image patch at the gaze point.

**Hypothesis and prediction.** H4. If H4 is true, the models with a gaze position predict
better than the model with angles, and better than the matched model without signals, on people
not in the training data. The gain is larger near the gaze point than on the whole frame.

**Expected order, written before the runs.** rope and pe+rope do better than pe, and pe does
about as well as angles or a little better. The reasons:

- The frozen heads of blocks 0–17 relate tokens through the rotations of RoPE. A gaze token
  rotated like the patch at the gaze point can use what they learned at once: it attends to the
  region the person looks at, and the image tokens there attend to it.
- Coord-PE puts the position into the token's content. The patches carry no position in their
  content, so the frozen heads have nothing to match it against. It would have to be learned
  through 18 frozen blocks in 3 epochs.
- In pe+rope the position is in both places. The 6 trained blocks may use the content form, so
  pe+rope can do a little better than rope.

If pe does better than rope, this reasoning is wrong, and the notes must say why.

**Models.** Each model uses the recipe of `ego_ft_v2`: 3 epochs, P01–P07, 30 clips per
recording, signal dropout 0.4, the same layers and learning rates, and the current scaling
constants. Only the gaze input differs. The step stays at 8 frames, so the prediction is
measured 0.27 s ahead and the results compare directly with Tests 1–4.

| Model | Content of the gaze token | Position of the gaze token | Role |
|---|---|---|---|
| none | no signals (signal dropout 1.0) | — | the matched model without signals; `ego_sd1p0` is its seed 0 |
| angles | yaw, pitch, depth (as now) | top-left patch (as now) | the current form; `ego_ft_v2` is its seed 0 |
| pe | Coord-PE of the gaze point, depth, flag | top-left patch | position in the content only |
| rope | yaw, pitch, depth | the gaze point | position in the attention only |
| pe+rope | Coord-PE of the gaze point, depth, flag | the gaze point | both |
| pe+rope shuffled | as pe+rope, with the signals of another clip in the batch | the other clip's gaze point | control: the extra input without the information |
| future | gaze and hand of the target frame, one step later | top-left patch | positive control: a signal known to carry information about the target |

- *Seeds.* Three per model: 20 new runs, about 14 GPU hours. Seed 0 of "none" is
  `ego_sd1p0`. Angles is retrained with seed 0 to check that the new code reproduces
  `ego_ft_v2`: before training, the check on the 96 P08 clips must give exactly 0.6153 (signals
  hidden) and 0.6235 (real signals); after the three epochs, the training loss should be close
  to 0.5261, 0.5150 and 0.5066, and the error with the signals hidden close to 0.4942, 0.4900
  and 0.4877.
- *Test set.* The fixed clips of all 12 P08 and all 13 P09 recordings, 24 clips each: about 600
  clips from two people not in the training data.
- *Measures.* The error of the prediction 0.27 s ahead, on the whole frame and on the patches
  within 2 patches of the gaze point. The Δ of each model.
- *Comparisons, fixed before the runs:* pe+rope − angles (does a position help?); each model −
  none (the value of the information); pe − rope (which way matters); pe+rope − pe+rope
  shuffled (information or only extra input?); future − none (can the measure show a known
  effect?).
- *Statistics.* The mean over seeds, and a 95% interval from a bootstrap over recordings and
  seeds. An effect counts only if the interval excludes 0 and the effect is larger than the
  spread between seeds.

**Implementation.**

- The gaze vector carries the gaze point (column, row on the patch grid) after yaw, pitch and
  depth ([[3-method#Gaze position in the image]]).
- *pe.* Sine and cosine of the gaze point at 5 frequencies ($K = 5$), so the finest wave
  repeats once per patch, plus the depth and the inside-the-frame flag: 22 inputs to the gaze
  layer. The 20 waves are scaled by $1/\sqrt{2K}$ to a joint length of 1, so the input has
  about the size of the three scaled angles. Unscaled, it was about twice as large, and two
  of four short trial runs diverged: the training loss rose from 0.65 to 0.79 in 13 steps.
  With the scaling, the same runs train normally.
- *rope.* The gaze token is rotated in the row and column channels by the gaze point minus
  half a patch, so a point at the centre of a patch gets exactly that patch's position. Hidden
  gaze, or gaze without a point, stays at the top-left patch, as in the angles model. The class
  is `GazeRoPEAttention` in `ego/gaze_attention.py`; Meta's code is unchanged.
- *future.* The angles form, given the gaze and hand of the next step.
- Each run keeps `best.pt` and `final.pt` (2.4 GB).

**Checks, done before the runs.**

- Self-tests (`tests/test_gaze_forms.py`): with every gaze point at the top-left patch, or
  without a point, the rope model gives exactly the output of the angles model, because a
  rotation by 0 changes nothing. The gaze token gets the rotation of the patch at its gaze
  point, with row and column not swapped. The Coord-PE values are right at known points.
- With the new code, `ego_ft_v2` gives the same outputs as before, bit for bit, and the rope
  model with every point at the top-left patch gives the same outputs as the angles model.
- The fixed clips are sampled at the same positions as before, with the same signals.
- Each new form trained for a few steps without error, pe and pe+rope after the scaling
  above.

**Runs.** A queue runs the jobs one after another (`checkpoints/test9/queue.sh`, jobs in
`checkpoints/test9/jobs.txt`), from 7 October 2026, 19:28. Order: rope seed 0 first (a
predictor for an attentive probe), then the reproduction (angles seed 0),
the positive control (future seed 0), seeds 1 and 2 of rope, angles and none, future seeds 1
and 2, and last the pe forms. One run takes about 40 minutes; one GPU fits one run at a time
(31 of 46 GB, at full use).

**Still needed.** An evaluation command for the test set, with the error near the gaze point.

## 4. Test 10 (planned). The full training run

All results so far come from models trained for 3 epochs. Test 10 tests whether more training
changes them (H3). It runs after Test 9, with the best gaze form of Test 9.

- Training: 8 epochs, P01–P07, 60 clips per recording.
- Models: the best gaze form, its shuffled control (`--shuffle-signals batch`), and none.
  `--shuffle-signals time` is not used as the control: a random order of 8 time steps leaves
  one step in place on average, and early steps can receive signals from later frames.
- Recompute the scaling constants `GAZE_MEAN`, `GAZE_STD` and `HAND_STD` from all P01–P07
  recordings before this run.
- Test set, measures and statistics as in Test 9. Write down the stopping rule before the run.

## 5. Later tests

- A test that predicts further ahead than 0.27 s. (H1) Two ways:
  - *A larger step,* for example every 30th frame for 1 s. The pretrained predictor knows steps
    of 0.25 s, so it needs more training to adapt.
  - *Several steps in a row.* The model predicts one step and uses its prediction as the input
    for the next; four steps reach about 1.1 s. This needs no new training and can be run on
    `ego_ft_v2` and `ego_sd1p0`. There is no measured gaze for the predicted steps, so the last
    measured gaze would be reused. The pretraining used two such steps; the fine-tuning here
    uses one. A training loss over several steps is part of the proposal in
    [[3-method#Design choices]].

  In Test 8, the information of the gaze point about the next object was largest 0.5 s ahead
  and gone at 4 s. So this test stays after Test 9.
- Hand position in the image, built like the gaze position. (H4)
- Error bars for Test 3, by resampling recordings. (R)
- Set the gaze layer's bias to zero at test time, to see how much of the effect is a constant.
  (H2)
- Gaze and hand as a training signal: an extra loss that predicts gaze or the future hand
  position from the predictor's states, dropped at test time.

## 6. Path 3: language alignment (postponed)

Postponed until Tests 9 and 10 are done.

- Idea: align the predictor's embedding space with text, so that it predicts concepts such as
  "making pasta sauce".
- Motivation: on EK100, VL-JEPA's advantage over V-JEPA 2 grows with the horizon, from +1.5
  points at 1 s to +4.6 at 10 s ([[vl-jepa]]).
- Text source: HD-EPIC recipe steps and action descriptions. EK100 labels are short verb–noun
  pairs, which are likely too little.
- Loss: start with a single target embedding (L2 distance). Move to a distribution over
  candidate goals only if the model is confidently wrong on ambiguous clips.
- Report absolute differences. Relative percentages make small effects look large.
- Benchmarks that test goals: CrossTask and EgoExo4D.

## 7. Other directions

- A workshop paper on the result, after Test 10.

## 8. Housekeeping

- The final presentation is on 25 November 2026 ([[project#Presentation]]).
- Back up `EgoVault-history-2026-10-05.bundle` (in `Projects/Ego/`) to another machine.
- Find `vjepa2-history.bundle` and `CV4E-full-backup-2026-08-22.bundle`. They are the only
  copies of that git history and are not on this machine.
- Save only the trainable weights in checkpoints: about 310 MB per file in place of 1.22 GB.
