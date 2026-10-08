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

The results of the paper are measured on two benchmarks with gaze: HD-EPIC's gaze interaction
anticipation and EGTEA Gaze+ action anticipation (Test 12). Tests 10 and 11 prepare the
predictors these benchmarks judge. Test 10 trains the V-JEPA 2-AC predictor with the loss of its
pretraining and as a whole. Test 11 builds a predictor from V-JEPA 2.1 that also takes gaze and
hand.

| Order | Work | Hypothesis | Weak points addressed | Training runs | Status |
|---|---|---|---|---|---|
| 1 | Gaze position in the image | H4 | gaze cannot point | none | Done and checked ([[3-method#Gaze position in the image]]) |
| 2 | Test 8. Does the gaze point tell what comes next? | H4, R | gaze cannot point, short horizon | none | Done ([[4-results#^test8\|Test 8]]): the decision rule was met |
| 3 | Test 9. Does gaze as a position in the image improve the prediction? | H4, M | gaze cannot point, no positive control, too little evidence | 20 runs of about 40 min | Done ([[4-results#^test9\|Test 9]]): no gaze form beats the angles; rope does not help |
| 4 | Test 10. The loss of the pretraining, and training the whole predictor | H3, M | too little evidence | 12 runs of 40–50 min | Running |
| 5 | Test 11. A predictor from V-JEPA 2.1 that takes gaze and hand | H4, M | gaze cannot point | to be set | Planned |
| 6 | Test 12. The benchmarks: HD-EPIC gaze interaction anticipation and EGTEA Gaze+ | M | short horizon, coarse measure | probes | Planned |

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

## 4. Test 10 (running). The loss of the pretraining, and training the whole predictor

Started on 8 October 2026.

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

## 5. Test 11 (planned). A predictor from V-JEPA 2.1 that takes gaze and hand

**Why.** V-JEPA 2.1 learns better patch features than V-JEPA 2 ([[2-background#V-JEPA 2.1]]).
Its predictor is the one the two strongest EK100 anticipation systems read. It is small, so it
can be trained as a whole at little cost. The V-JEPA 2-AC predictor was pretrained on a fixed
robot camera.

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
  predictor fills in the tokens of the 2 frames at the anticipation time, for example 1 s after the
  last frame. A probe reads the encoder tokens and the predicted tokens
  (`vjepa2/evals/action_anticipation_frozen`).

**Design: gaze and hand as maps added to the observed tokens.** After `predictor_embed`, the token
of patch $p$ at time step $\tau$ becomes

$$x_{\tau,p} + \alpha_g\, k(p, g_\tau)\, e_g + \alpha_h \left( k(p, l_\tau)\, e_l + k(p, r_\tau)\, e_r \right),
\qquad k(p, q) = \exp\!\left(-\frac{\lVert p - q \rVert^2}{2\sigma^2}\right)$$

- $g_\tau$, $l_\tau$ and $r_\tau$ are the gaze point and the left and right palm in the
  24 × 24 grid at that time step. $e_g$, $e_l$ and $e_r$ are learned vectors of 384 numbers.
  $\sigma$ is 1 patch. $\alpha_g$ and $\alpha_h$ are learned numbers that start at 0.
- A missing signal adds nothing. EGTEA Gaze+ has no hand tracking, so there only gaze enters.
- Why this form:
  1. It marks the image tokens that the person looks at, which the predictor already relates in
     space and time. What lies at the gaze point carries the information (Test 8), and drawing the
     gaze on the frames helped a video-language model on HD-EPIC (Materia et al., [[review]]).
  2. It adds no tokens and changes no positions.
  3. With $\alpha = 0$ the model is exactly the released predictor. This is a clean start, and a
     self-test.
  4. The vectors $e$ start random and only the numbers $\alpha$ start at 0, so both receive a
     gradient from the first step ([[3-method#Design choices]]).
- A second form, for comparison: a gaze token per time step, placed at the gaze point in RoPE, like
  the rope form of Test 9. It did not help the V-JEPA 2-AC predictor with frozen blocks; a small
  predictor trained as a whole may use it.

**Training.**

- Clips of 32 frames at 8 frames per second, and the frames at the target time, 0.5 s and 1 s after
  the last observed frame. 0.5 s is where the gaze point told most in Test 8; 1 s is the horizon
  of the benchmarks.
- The frozen encoder reads the observed frames. The frozen target encoder reads the observed and the
  future frames; its tokens of the future frames are the targets. This is a mask over the future,
  as in V-JEPA's pretraining.
- L1 loss. The whole predictor and the gaze and hand parameters train. Signal dropout as before.

**Controls.** The released predictor without training. The same training without gaze and hand
($\alpha$ held at 0): the matched model. Gaze and hand from another clip. A positive control with
the gaze and hand at the target time.

**Measures.** The error of the predicted tokens, L1 and MSE, over the whole frame, near the gaze
point, and on the box of the object about to be picked (clips that end before a pick), against
the references. Then the benchmarks (Test 12).

**Steps.**

1. Load V-JEPA 2.1 in `ego/`. Check that the released predictor predicts the future: its error at
   +0.25, +0.5 and +1 s on HD-EPIC clips, against repeating the last step and against the blend, with
   no training. Measure the time per clip.
2. The hand position in the image, built like the gaze position and checked against the pick boxes.
3. The gaze and hand maps in the predictor, with the self-test that $\alpha = 0$ gives the released
   output.
4. A training command with the controls. First runs with ViT-L to check the pipeline, then ViT-G for
   the numbers.
5. Predictions and the decision rule written before the runs.

**Open choices.** The predictor of ViT-L predicts ViT-G's features, so its targets also need the
ViT-G encoder, which then sets the cost. If step 1 shows that ViT-G is too slow, use 16 frames
or 256 pixels.

## 6. Test 12 (planned). The benchmarks

The paper's results. Every predictor is judged by how well a probe reads the next object or action
from it.

**HD-EPIC gaze interaction anticipation.**

- The task: 1,000 questions, "What object will the person interact with next, ignoring ongoing
  interactions?", 5 answers each, chance 20%. Each 10 s clip ends 0.3 s after the person first
  looks at the object (`vqa-benchmark/gaze_interaction_anticipation.json`). 280 questions are on P08
  and P09.
- HD-EPIC has no training split. Every published result comes from models not trained on HD-EPIC
  ([[2-background#Benchmarks]]). A probe trained on HD-EPIC's own annotations is a different setting
  and is reported as such: trained on P01–P07 and tested on the 280 questions of P08 and P09, or
  trained once per held-out person and tested on all 1,000.
- The clip ends just after the first look at the answer, so a method with gaze has an advantage by
  construction. A baseline without training: the object under the last fixations, matched to the
  five answers.

**EGTEA Gaze+ action anticipation.**

- Train on the training set of split 1 (8,299 segments) and test on its test set (2,022). The
  protocol of AVT, InAViT and SAGE: 0.5 s ahead, top-1 and mean class accuracy. The protocol of
  RULSTM: 0.25–2 s ahead, top-5 accuracy.
- Gaze is a 2D point in the image, so no projection is needed. There is no hand tracking.
- The data is not on disk. Anticipation needs the seconds before each action; whether the
  download holds full videos or only the trimmed action clips (21.9 GB) has to be checked first.

**The readout.** The attentive probe of V-JEPA 2's anticipation code, on the encoder tokens and the
predicted tokens, as in JFAA and TAP-JEPA for EK100. It needs an adapter for EGTEA's annotations.
Each predictor is read with and without measured gaze; the published results are the context.

## 7. Later tests

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
  and gone at 4 s. In Test 9, even the gaze and hand of the target frame lowered the error
  0.27 s ahead by only 0.26%.
- A longer run: 8 epochs, 60 clips per recording, with a warmup of the learning rate, because 2 of
  20 runs in Test 9 had a loss spike at the start. (H3)
- A shuffled control for pe (`--shuffle-signals batch`). Test 9 had none.
  `--shuffle-signals time` is not used as the control: a random order of 8 time steps leaves one
  step in place on average, and early steps can receive signals from later frames.
- Recompute the scaling constants `GAZE_MEAN`, `GAZE_STD` and `HAND_STD` from all P01–P07
  recordings.
- Head motion as a positive control: the relative pose of the camera from one step to the next,
  from HD-EPIC's SLAM trajectories ([[review]]).
- Why pe did better than rope in Test 9: measure in the rope models, as in Test 4b, whether
  image tokens near the gaze point give the gaze token more attention than tokens far from it
  ([[4-results#^test9|Test 9]]). (H4)
- Hand position in the image, built like the gaze position. (H4) Test 11 needs it.
- Error bars for Test 3, by resampling recordings. (R)
- Set the gaze layer's bias to zero at test time, to see how much of the effect is a constant.
  (H2)
- Gaze and hand as a training signal: an extra loss that predicts gaze or the future hand
  position from the predictor's states, dropped at test time.

## 8. Path 3: language alignment (postponed)

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

## 9. Other directions

- The paper reports the results of Test 12.

## 10. Housekeeping

- The final presentation is on 25 November 2026 ([[project#Presentation]]).
- The data disk (`/mnt/data`) is 97% full, with 55 GB free. EGTEA Gaze+ needs about 22 GB, and
  feature caches need more.
- Back up `EgoVault-history-2026-10-05.bundle` (in `Projects/Ego/`) to another machine.
- Find `vjepa2-history.bundle` and `CV4E-full-backup-2026-08-22.bundle`. They are the only
  copies of that git history and are not on this machine.
- Save only the trainable weights in checkpoints: about 310 MB per file in place of 1.22 GB.
