---
type: report
status: running
created: 2026-09-18
updated: 2026-10-08
---

# 6. Next steps

What to do next, in order. Each planned test names the hypothesis it addresses
([[1-introduction#^hypotheses|hypotheses table]]) and the weak points it removes
([[5-discussion#Weak points of the design]]). When a test is run, its section moves to
[[4-results]] under the same number.

Tests 1–4, 8, 9 and 10 all give gaze to the V-JEPA 2-AC predictor through the slot its
pretraining built for the robot action. That slot carries no position in the image, and
[[4-results#^test9|Test 9]] showed why neither repair worked. The question is therefore no
longer whether gaze helps, but where gaze has to enter a predictor for its information to
survive.

Test 11 builds a predictor designed for the signals in place of the V-JEPA 2-AC predictor. Test 13
builds the same design on V-JEPA 2.1 features. Test 14 reads the next object from the prediction.
The results of the paper are measured on benchmarks with gaze (Test 12).

| Order | Work | Hypothesis | Weak points addressed | Status |
|---|---|---|---|---|
| 1 | Gaze position in the image | H4 | gaze cannot point | Done and checked ([[3-method#Gaze position in the image]]) |
| 2 | Test 8. Does the gaze point tell what comes next? | H4, R | gaze cannot point, short horizon | Done ([[4-results#^test8\|Test 8]]): the decision rule was met |
| 3 | Test 9. Does gaze as a position in the image improve the prediction? | H4, M | gaze cannot point, no positive control, too little evidence | Done ([[4-results#^test9\|Test 9]]): no gaze form beats the angles; rope does not help |
| 4 | Test 10. The loss of the pretraining, and training the whole predictor | H3, M | too little evidence | Done ([[4-results#^test10\|Test 10]]): the loss and the trainable depth both matter; gaze does not gain from either |
| 5 | Test 11. A predictor designed for gaze and hand | H4, H3, M | gaze cannot point, short horizon | Planned |
| 6 | Test 13. The same predictor on V-JEPA 2.1 features | H4, M | gaze cannot point | Planned |
| 7 | Test 14. What the prediction says about the next object | M, H1 | coarse measure | Planned |
| 8 | Test 12. The benchmarks | M | short horizon, coarse measure | Planned |

## 1. Gaze position in the image

Done. How the gaze point is computed, the calibration and the checks are in
[[3-method#Gaze position in the image]]. At the moment of a pick, the gaze point lies in the
box of the object in 38.8% of picks, against 19.7% by chance. Its error is about half a patch
for most people.

## 2. Test 8 (done). Does the gaze point tell what comes next?

Run on 7 October 2026: [[4-results#^test8|Test 8]]. The features at the gaze point tell which
object is picked up next, beyond the scene, the three gaze numbers and the head point. The
decision rule was met at 0.5 s and at 1 s, so Test 9 goes ahead.

## 3. Test 9 (done). Does gaze as a position in the image improve the prediction?

Run on 7–8 October 2026: [[4-results#^test9|Test 9]]. The positive control was detected, so
the test can show an effect of 0.0013. Gaze as a position in the token's content (pe) lowered
the error near the gaze point by 0.0007 against the model without signals, and the three
angles did as well once one run with a loss spike is left out. Gaze as a position in the
attention (rope) did not help. The expected order, rope before pe, was wrong; why is open. Two
of the 20 runs had a loss spike at the start of training.

## 4. Test 10 (done). The loss of the pretraining, and training the whole predictor

Run on 8 October 2026: [[4-results#^test10|Test 10]]. The loss and the trainable depth both
lower the error by much more than gaze ever has, and training the whole predictor did not make
gaze worth more. Comparison 6, the primary comparison, did not pass the decision rule.

**Question.** All models so far were trained with the mean squared difference (MSE), while
V-JEPA 2-AC was pretrained with the mean absolute difference (L1), and only the last 6 of its 24
blocks were trained ([[3-method#Training]]). Do the loss of the pretraining, or training the
whole predictor, change how well it predicts and how much gaze helps?

**Models.** Three seeds each. pe is the best form of Test 9: the only form whose gain over none
passed the decision rule, with no loss spike in its 9 runs.

| Model | Loss | What trains | Signals | Runs |
|---|---|---|---|---|
| pe | MSE | last 6 blocks | pe | Test 9 |
| none | MSE | last 6 blocks | none (signal dropout 1.0) | Test 9 (`ego_sd1p0` is seed 0) |
| pe l1 | L1 | last 6 blocks | pe | new |
| none l1 | L1 | last 6 blocks | none | new |
| pe l1 full | L1 | the whole predictor: 24 blocks and `predictor_embed` | pe | new |
| none l1 full | L1 | the whole predictor | none | new |

- *Recipe.* The recipe of Test 9 in every other respect: 3 epochs, P01–P07, 30 clips per
  recording, batch size 16, learning rates $10^{-3}$ for the new layers and $10^{-4}$ for the rest,
  no warmup. So each comparison changes one thing. The full fine-tune trains all 305.2 million
  parameters of the predictor in place of 77.0 million.
- *Test set and measures.* The 600 clips of Test 9. The error of the next step in both measures,
  L1 and MSE, over the whole frame and within 2 patches of the gaze point, and the references
  without fine-tuning ([[3-method#Prediction error and Δ]]).

**Comparisons, fixed before the runs.** The gain of A over B, seeds averaged clip by clip, with
95% intervals from a bootstrap over the 25 test recordings, and the spread between seeds.

| | A − B | What it measures |
|---|---|---|
| 1 | pe l1 − pe | the loss, with gaze |
| 2 | none l1 − none | the loss, without signals |
| 3 | pe l1 − none l1 | the value of gaze with L1 |
| 4 | none l1 full − none l1 | training the whole predictor, without signals |
| 5 | pe l1 full − pe l1 | training the whole predictor, with gaze |
| 6 | pe l1 full − none l1 full | the value of gaze when the whole predictor trains |

- A model trained with L1 has an advantage on the L1 measure, and one trained with MSE on the MSE
  measure. Comparisons 1 and 2 are therefore read on both measures. Comparisons 3 to 6 are read
  on L1, the loss these models were trained with.
- The primary comparison is 6, near the gaze point, on L1. The others are secondary.
- Decision rule, as in Test 9: an effect counts if its 95% interval excludes 0 and it is larger
  than the spread between seeds.

**Predictions, written before the runs.**

- Comparisons 1 and 2: each model does better on the measure of its own loss. The differences are
  small. With L1, gaze still gains about as little as in Test 9 (comparison 3).
- Comparison 4: training the whole predictor lowers the error of none by more than the seed spread.
  Before fine-tuning, the predictor was worse than a blend of the past frames, and its blocks 0–17
  were pretrained on video from a fixed camera ([[review]]).
- Comparisons 5 and 6: if the frozen blocks kept the model from using gaze, the gain of pe over
  none is larger when the whole predictor trains than with the last 6 blocks.
- A risk: the whole predictor may overfit 3,900 training clips per epoch. This shows as a held-out
  error that rises over the epochs in the check after each epoch.

**What follows.** If comparison 4 holds, later predictors are trained as a whole. If comparison 6
passes the decision rule, gaze helps a predictor that can adapt all its blocks, and Test 11 keeps
this form of training.

**Reproduce.** The runs: `checkpoints/test10/queue.sh` with `checkpoints/test10/todo.txt`, which
holds the options of each run (`--loss l1`, `--unfreeze-last-n 24 --unfreeze-embed`). The
evaluation: `python -m ego gaze-forms --runs checkpoints/test9 checkpoints/test10 --cache
results/test9/cache --out results/test10`, after the Test 9 scores are copied to
`results/test10/scores/` so that they are not computed again. The queue does both.

## 5. Test 11 (planned). A predictor designed for gaze and hand

**Why.** Tests 1–4, 8, 9 and 10 give gaze to the V-JEPA 2-AC predictor in the slot its pretraining
built for the robot's action. That token is rotated by the frame index alone, so it holds no row and
no column: it cannot point at the image. Test 9 tested the two repairs and found the reason neither
worked. In a RoPE head the attention between two tokens depends on their content, not on their
distance, so a gaze token placed at the gaze point is not a token the frozen heads attend to
([[4-results#^test9|Test 9]]). The horizon is inherited in the same way: 0.27 s is the step of 8
frames that matches the 4 frames per second of the pretraining ([[3-method#Input format]]). Test 11
replaces that predictor. The signals enter the image tokens, and the horizon is chosen.

**Question.** Does a predictor built for gaze and hand predict better with them than without, when
no pretrained conditioning slot and no pretrained step constrain the design?

**Hypothesis and prediction.** H4 says the gaze input has a form the model cannot relate to the
image. If that is the reason for the null of Tests 2 and 9, then a predictor whose gaze marks the
image tokens gains more from gaze than the V-JEPA 2-AC predictor did, and the gain is larger near
the gaze point and on the object about to be picked than over the whole frame.

**What is kept.** The frozen V-JEPA 2 encoder and its patch grid, HD-EPIC, P01–P07 for training and
P08–P09 for testing, the clip sampling, the gaze position in the image, the three ways to hide the
signals, and the measures. The gain of gaze is then comparable with Tests 9 and 10
([[3-method#Rules for comparing models]]).

**What is dropped.** The V-JEPA 2-AC predictor: its action and state tokens, its frozen blocks, and
its step of 8 frames.

**The predictor.** Trained from the start, not from a pretrained predictor.

- About 12 blocks of width 384, near 20 million parameters. V-JEPA 2.1's predictor is this size.
  The V-JEPA 2-AC predictor holds 305.2 million, too many to train from the start on this data.
- RoPE over the frame index, the row and the column. Attention over all tokens, not causal over
  frames, so each frame to predict is a mask token at its own time.
- It predicts the tokens of the frozen target encoder at the times to predict. L1 loss, the loss of
  the pretraining.

**Design: gaze and hand as maps added to the observed tokens.** The token of patch $p$ at time step
$\tau$ becomes

$$x_{\tau,p} + \alpha_g\, k(p, g_\tau)\, e_g + \alpha_h \left( k(p, l_\tau)\, e_l + k(p, r_\tau)\, e_r \right),
\qquad k(p, q) = \exp\!\left(-\frac{\lVert p - q \rVert^2}{2\sigma^2}\right)$$

- $g_\tau$, $l_\tau$ and $r_\tau$ are the gaze point and the left and right palm in the encoder's
  patch grid at that time step. $e_g$, $e_l$ and $e_r$ are learned vectors of the predictor's
  width. $\sigma$ is 1 patch. $\alpha_g$ and $\alpha_h$ are learned numbers that start at 0.
- A missing signal adds nothing. A dataset without hand tracking gives only gaze.
- Why this form:
  1. It marks the image tokens that the person looks at, which the predictor already relates in
     space and time. What lies at the gaze point carries the information (Test 8), and drawing the
     gaze on the frames helped a video-language model on HD-EPIC (Materia et al., [[review]]).
  2. It adds no tokens and changes no positions, so it does not meet the problem of Test 9: the
     signals act on the attention between image tokens rather than competing for it as a token of
     their own.
  3. An untrained model is exactly the matched model without signals. This is a clean start, and
     a self-test (`tests/test_ego_predictor.py`).
  4. The vectors $e$ start at 0 and the numbers $\alpha$ start at 1, not the other way round. The
     gradient of $\alpha\, k\, e$ with respect to $e$ is $\alpha k$ and with respect to $\alpha$
     is $k \cdot e$. So starting $\alpha$ at 0 would leave $e$ with no gradient at all on the
     first step, and $e$ would begin to learn only once $\alpha$ had moved. With the zero on $e$,
     property 3 still holds and $e$ receives a gradient immediately, because $k$ is not zero
     wherever a point exists. $\alpha$ is then a readable gain, not a gate, and starts to move one
     step after $e$. How much the model uses a signal is $\alpha \lVert e \rVert$, which every run
     starts at 0 ([[3-method#Design choices]]).

**Horizon.** The frames to predict are 0.5 s and 1 s after the last observed frame, both in one
run. 0.5 s is where the gaze point told most in Test 8. 1 s is the horizon of the benchmarks and of
the gaze lead on the hand ([[3-method#Design choices]]). Beyond about 2 s the predictor predicts a
mix of the possible futures. The horizon is no longer fixed by a pretrained step, so H1 is tested
in the same runs.

**Training.** P01–P07. The whole predictor and the signal parameters train. 8 epochs and more clips
per recording than the 30 of Tests 9 and 10, because nothing is pretrained. A warmup of the
learning rate, because 2 of 20 runs in Test 9 had a loss spike at the start. Signal dropout as
before.

**Arms.** Five seeds each. Only the signals differ. The channels that are open to the project,
and the published work behind each, are in [[conditioning-methods]].

| Arm | Signals | Role |
|---|---|---|
| maps | gaze and hand as maps | the design |
| none | $\alpha$ held at 0 | the matched model |
| shuffled | the maps, with gaze and hand from another clip | information against extra input |
| future | the gaze and hand at the target time | the positive control: the ceiling of the measure |
| token | a gaze token placed at the gaze point in RoPE, as the rope form of Test 9 | the second form |

- The arms are the ones that make this predictor comparable with the V-JEPA 2-AC predictors of
  Tests 9 and 10: the same encoder, split, clips and controls, so the gain of gaze can be read
  against theirs ([[3-method#Rules for comparing models]]).
- The future arm is measured again here. Its 0.26% in Test 9 is a property of the V-JEPA 2-AC
  predictor at 0.27 s, not the ceiling of this measure at 0.5 s or 1 s.
- The token arm did not help the V-JEPA 2-AC predictor with frozen blocks. A small predictor
  trained as a whole may use it.

**Measures.** The error of the predicted tokens, L1 and MSE, at each horizon: over the whole frame,
within 2 patches of the gaze point, and on the box of the object about to be picked, against the
references without training ([[3-method#Prediction error and Δ]]). The correlation between the
prediction and the target beside each error.

**Comparisons and the decision rule.** The primary comparison is maps − none, on the patches of the
object about to be picked, at 1 s, on L1. It is written before the runs, with 95% intervals from a
bootstrap over the test recordings and the spread between seeds, as in Tests 9 and 10. The others
are secondary, with a correction for their number. An effect counts if its interval excludes 0 and
is larger than the spread between seeds.

**Steps.**

1. The hand position in the image, built like the gaze position (`project_device_point` in
   `ego/gaze_geometry.py` and `signals.points`), and checked against the pick boxes. The
   projection is built; the check is not.
2. The predictor in `ego/ego_predictor.py`, with the self-test that an untrained model gives the
   matched model exactly, and the command `python -m ego train-ego`. Both built.
3. Targets at both horizons in the training data.
4. Predictions and the decision rule written before the runs.
5. The runs, then the evaluation against the existing references.

**What follows.** If maps − none passes the decision rule, the design carries gaze and Test 13
repeats it on V-JEPA 2.1 features. If it does not, the comparison with Tests 9 and 10 says
whether the channel or the signal is the limit, and Test 14 then asks the question on a measure
that is not the token error.

## 6. Test 13 (planned). The same predictor on V-JEPA 2.1 features

**Why.** V-JEPA 2.1 learns better patch features than V-JEPA 2 ([[2-background#V-JEPA 2.1]]), works
at 384 pixels, and its predictor is the one the strongest EK100 anticipation systems read. Test 13
keeps the design of Test 11 and changes the features, so the two differ in one thing.

**The released model.**

| | ViT-L (distilled from ViT-G) | ViT-G |
|---|---|---|
| Encoder | 305 million parameters, 1,024 numbers per token | 1,845 million, 1,664 per token |
| Predictor | 12 blocks of width 384, 23 million parameters | 24 blocks of width 384, 59 million |
| What the predictor predicts | the features of ViT-G (1,664 numbers) | four layers of its own encoder (4 × 1,664) |

- Both work at 384 pixels: 24 × 24 patches per tubelet of 2 frames.
- The predictor takes the encoder tokens of the observed frames and learned mask tokens at the
  positions to predict. Positions enter through RoPE over time, row and column. Attention spans all
  tokens; it is not causal over frames.
- It was trained with L1.
- For anticipation, the released code feeds 32 frames at 8 frames per second (4 s), and the
  predictor fills in the tokens of the 2 frames at the anticipation time
  (`vjepa2/evals/action_anticipation_frozen`).

**What changes from Test 11.** The encoder; the grid, 24 × 24 in place of 16 × 16, so $\sigma$ and
the radius of "near the gaze point" are counted in the new patches; the clip, 32 frames at 8 frames
per second; and the target. The predictor of ViT-L predicts ViT-G's features, so the targets also
need the ViT-G encoder, which then sets the cost.

**Reading the result.** The deltas within each backbone, never the absolute errors across them.
Report maps − none on V-JEPA 2.1 against maps − none on V-JEPA 2 from Test 11
([[3-method#Rules for comparing models]]). The future arm is measured again.

**Steps.**

1. Load V-JEPA 2.1 in `ego/`. Check that the released predictor predicts the future: its error at
   +0.25, +0.5 and +1 s on HD-EPIC clips, against repeating the last step and against the blend,
   with no training. Measure the time per clip.
2. The maps on the new grid, with the $\alpha = 0$ self-test.
3. First runs with ViT-L to check the pipeline, then ViT-G for the numbers.

**Open choices.** If step 1 shows that ViT-G is too slow, use 16 frames or 256 pixels.

## 7. Test 14 (planned). What the prediction says about the next object

**Why.** The error of the predicted tokens is not the quantity gaze helps. In Test 9 even the gaze
and hand of the target frame lowered it by 0.26%, so the measure leaves little room for any signal.
Test 8 measured gaze where it does tell: which object is picked up next. Test 14 asks that question
of the prediction in place of the frozen encoder, so Tests 8 and 14 differ in one thing and the
difference is what the predictor adds.

**Method.** The attentive probe of V-JEPA 2's anticipation code, on the encoder tokens of the
observed frames and the predicted tokens (`vjepa2/evals/action_anticipation_frozen`). The picks, the
noun mapping and the target of Test 8. The probe trains on P01–P07 and is tested on P08–P09.

**Three arms.** Whether the predictor received gaze and whether the probe receives it are separate
questions.

| Arm | The predictor receives gaze | The probe receives gaze |
|---|---|---|
| A | no | no |
| B | no | yes |
| C | yes | yes |

The value of the conditioning is C − B. C − A adds to it the value of gaze as an input of the
probe, which Test 8 already measured ([[3-method#Rules for comparing models]]).

**Measure.** The share of correct guesses over the noun classes of Test 8, at 0.5 s and 1 s before
the pick, with intervals from a bootstrap over recordings. The same probe on the predictor's output
gives the world-model version of the benchmark question of Test 12.

## 8. Test 12 (planned). The benchmarks

The paper's results. Every predictor is judged by how well a probe reads the next object or action
from it.

**EGTEA Gaze+ action anticipation (Primary Test).**

- Train on the training set of split 1 (8,299 segments) and test on its test set (2,022). The
  protocol of AVT, InAViT and SAGE: 0.5 s ahead, top-1 and mean class accuracy. The protocol of
  RULSTM: 0.25–2 s ahead, top-5 accuracy.
- Gaze is a 2D point in the image, so no projection is needed. There is no hand tracking.
- The data is not on disk. Anticipation needs the seconds before each action; whether the
  download holds full videos or only the trimmed action clips (21.9 GB) has to be checked first.
- This is the main evaluation that will go into comparison tables, as it provides an official training split.

**HD-EPIC gaze interaction anticipation (Secondary Zero-Shot Check).**

- The task: 1,000 questions, "What object will the person interact with next, ignoring ongoing
  interactions?", 5 answers each, chance 20%. Each 10 s clip ends 0.3 s after the person first
  looks at the object (`vqa-benchmark/gaze_interaction_anticipation.json`). 280 questions are on P08
  and P09.
- HD-EPIC has no training split. Every published result comes from models not trained on HD-EPIC
  ([[2-background#Benchmarks]]).
- To compare fairly, the model trained on EGTEA Gaze+ will be evaluated zero-shot on HD-EPIC.
- The clip ends just after the first look at the answer, so a method with gaze has an advantage by
  construction. A baseline without training: the object under the last fixations, matched to the
  five answers.

**MECCANO (both signals).**

- 415 minutes of assembly by 20 people, recorded with a headset that holds gaze and depth. It
  annotates hands with boxes, and defines both action anticipation and next-active-object
  detection.
- It is the only benchmark with an anticipation task where gaze and a hand signal are both
  present, so it is where the conditioning on gaze and hand together is tested. Next-active-object
  detection is the task of Test 8 with a published protocol.
- Its literature is smaller than EGTEA's, so it is reported beside EGTEA, not in place of it.

**EK100 action anticipation (transfer check).** No gaze. Run only to show that the conditioning
does not damage the backbone. The harness is the one Test 14 uses
(`vjepa2/evals/action_anticipation_frozen`).

**Which datasets carry which signals.** No dataset holds gaze, a hand signal, an official training
split and an anticipation task with published baselines at once. EGTEA has the task and 2D gaze but
hand masks on about 14,000 frames only. MECCANO has gaze, hand boxes and the task. Ego-Exo4D and
Nymeria hold gaze and hand in 3D with official splits but define no anticipation task. HD-EPIC
holds both signals and the task but no training split. Each benchmark above therefore answers one
question and not the others.

**The readout.** The attentive probe of V-JEPA 2's anticipation code, on the encoder tokens and the
predicted tokens, as in JFAA and TAP-JEPA for EK100. It needs an adapter for EGTEA's annotations.
Every benchmark is read with the three arms of Test 14: the value of the conditioning is the arm
whose predictor received gaze minus the arm whose probe alone received it
([[3-method#Rules for comparing models]]). The published results are the context.

## 9. Later tests

- A longer horizon on the V-JEPA 2-AC models, if a comparison with Tests 1-10 at 1 s is wanted.
  (H1) Tests 11 and 13 set the horizon by where the mask tokens sit, so they need none of this.
  On the AC predictor there are two ways: a larger step, for example every 30th frame, which the
  pretrained predictor must learn; or several steps in a row, where the prediction is the input of
  the next and four steps reach about 1.1 s, with the last measured gaze reused for the predicted
  steps. The pretraining used two such steps and the fine-tuning here uses one.
- A longer run of the V-JEPA 2-AC models: 8 epochs, 60 clips per recording, with a warmup of the
  learning rate. (H3) Test 11 trains this way from the start.
- A shuffled control for pe (`--shuffle-signals batch`). Test 9 had none.
  `--shuffle-signals time` is not used as the control: a random order of 8 time steps leaves one
  step in place on average, and early steps can receive signals from later frames.
- Recompute the scaling constants `GAZE_MEAN`, `GAZE_STD` and `HAND_STD` from all P01–P07
  recordings.
- Head motion as a conditioning signal, if the pipeline itself is to be calibrated: the relative
  pose of the camera from one step to the next, from HD-EPIC's SLAM trajectories ([[review]]).
  Not an arm of Test 11.
- Why pe did better than rope in Test 9: measure in the rope models, as in Test 4b, whether
  image tokens near the gaze point give the gaze token more attention than tokens far from it
  ([[4-results#^test9|Test 9]]). (H4)
- Hand position in the image, built like the gaze position. (H4) Test 11 needs it; it is step 1
  there.
- Error bars for Test 3, by resampling recordings. (R)
- Set the gaze layer's bias to zero at test time, to see how much of the effect is a constant.
  (H2)
- Gaze and hand as a training signal: an extra loss that predicts gaze or the future hand
  position from the predictor's states, dropped at test time. Four published lines use the signals
  this way and three report that the benefit survives without them at test time, which also makes
  a model comparable with SAGE on EGTEA Gaze+ and usable where there is no hand tracking
  ([[conditioning-methods]]). Worth promoting out of this list.
- Gaze drawn on the frames, the one channel with a published number on HD-EPIC's gaze interaction
  anticipation, 27.5% with no training ([[conditioning-methods]]). It changes the encoder's input,
  so nothing from Tests 1-10 compares with it directly.

## 10. Path 3: language alignment (postponed)

Postponed until the benchmarks of Test 12 are set up.

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

## 11. Other directions

- The paper reports the results of Test 12.

## 12. Housekeeping

- The final presentation is on 25 November 2026 ([[project#Presentation]]).
- The data disk (`/mnt/data`) is 97% full, with 55 GB free. EGTEA Gaze+ needs about 22 GB, and
  feature caches need more.
- Back up `EgoVault-history-2026-10-05.bundle` (in `Projects/Ego/`) to another machine.
- Find `vjepa2-history.bundle` and `CV4E-full-backup-2026-08-22.bundle`. They are the only
  copies of that git history and are not on this machine.
- Save only the trainable weights in checkpoints: about 310 MB per file in place of 1.22 GB.
