---
created: 2026-10-08
updated: 2026-10-08
---

# Review of the project and of related work

This note checks the project as it stands after Test 9: the question, the method, the code and
the comparisons. It adds three baselines that were computed for this review from the Test 9
feature cache. It then places the project in the published work: what has been done before,
whether the comparisons are adequate, and which benchmarks the project could report on. The
recommendations at the end are ordered by how much they would change the conclusions.

## Summary

- The question is open. No published work was found that gives gaze to a latent (JEPA-style)
  video predictor. The nearest work conditions egocentric world models on head or body motion,
  and gives gaze to video-language models.
- The experimental controls are better than in most of the related papers: a matched model
  without signals, a shuffled control, a positive control, three seeds, held-out people, decision
  rules written before the runs, and a reproduction check.
- The trained predictor does model egocentric dynamics. It beats a "repeat the past" baseline by
  0.073 in error. Gaze and hand lower the error by about 0.0003, 0.4% of that margin. Even the
  gaze and hand of the target frame lower it by only 0.0013, 1.8% of that margin.
- The predictor is the weakest part of the design. Its frozen blocks were pretrained on robot
  video from a fixed camera. Before fine-tuning it was worse than the "repeat the past" baseline.
- The conditioning token does not carry what the robot action carried. In V-JEPA 2-AC the action
  token says how the scene will change in the next step. In egocentric video the closest
  equivalent is the motion of the head (the camera) and of the hands from frame t to frame t+1.
  The project has not tested head motion. This is the most important missing control.
- The measure (one step, 0.27 s, whole frame) is used by no other work, so the results cannot be
  placed next to published numbers. HD-EPIC's own gaze benchmark and EGTEA Gaze+ can serve as
  external measures.

## 1. What is sound

**The chain of tests.** The tests follow one argument. Test 2 finds no measurable value of the
signals against a matched model. Tests 3 and 4 check two explanations: gaze is already in the
image (R), and the model does not use the signals (H2). Test 8 checks, without training,
whether the gaze point carries information about the future. Test 9 gives the predictor gaze as
a position and includes a positive control. Each step was planned from the result of the
previous one. This is a sound way to study a null result.

**The controls.** The matched model without signals separates the value of the information from
the cost of hiding an expected input ([[4-results#^test2|Test 2]]). The shuffled control
separates information from an extra input. The positive control shows how large an effect the
measure can detect ([[4-results#^test9|Test 9]]). Test 8 compares the gaze point with a fixed
head point and gaze history with head history of the same length. Without these controls, a
gain from "features at the gaze point" could come from the centre of the image, where gaze
usually lies.

**The gaze projection.** The projection was checked against HD-EPIC's annotated picks: the
projected point lies in the object's box in 38.8% of picks, against 19.7% for gaze at random
moments ([[3-method#Gaze on the object being picked up]]). The checks without the rotation and
with a fixed depth show that each step of the projection is needed. Few papers that use Aria
gaze check the projection this way.

**The code.** The predictor (`ego/predictor.py`), the signals (`ego/signals.py`), the training
loop (`ego/commands/train.py`) and the Test 9 evaluation (`ego/commands/gaze_forms.py`) do what
the notes say. The encoder setup follows the V-JEPA 2-AC training: the EMA target encoder,
each frame encoded on its own as a 2-frame tubelet, and layer-normalized embeddings. The self-test
that the rope form with every gaze point at the top-left patch gives exactly the angles model is
a good check of `GazeRoPEAttention`.

## 2. New baselines from the Test 9 cache

### Why a baseline is needed

A video predictor should be compared with predictions that use no model at all. The simplest is
to repeat the last context frame. A second is a weighted average of the last frame and the mean
of the 8 context frames. If a trained predictor does not beat these, it has not learned how the
scene changes, and a small effect of a conditioning signal means little. The notes did not report
such baselines. The Test 9 cache holds the encoder features of all 600 test clips
(`results/test9/cache/`), so they can be computed without a GPU.

### The measure is a correlation

The prediction and the target are both layer-normalized before the error is computed. After
layer normalization, the 1,408 numbers of each patch have mean 0 and variance 1. For two such
vectors $p$ and $t$ of length $D$:

$$\text{MSE} = \frac{1}{D}\sum_i (p_i - t_i)^2 = \frac{1}{D}\sum_i p_i^2 + \frac{1}{D}\sum_i t_i^2 - \frac{2}{D}\sum_i p_i t_i = 2\,(1 - \rho)$$

where $\rho$ is the correlation between the predicted and the true patch vector. So an error of
0.50 means a correlation of 0.75 per patch, and every comparison in the notes is a comparison of
correlations.

A prediction that is not layer-normalized can have a smaller length. Under squared error, a
shorter vector in the right direction can score better: averaging several past frames gives such
a vector. An average of past frames that is not normalized scores 0.493, lower than every trained
model. This number is not comparable, because the predictor's output is always normalized. The
baselines below are normalized in the same way as the predictor's output.

### Result

Error on the 600 Test 9 clips (P08 and P09), one step (0.27 s) ahead, whole frame:

| Prediction | Error (MSE) | Correlation $\rho$ |
|---|---|---|
| repeat the last context frame | 0.742 | 0.629 |
| layer-normalized $0.2 \times$ last frame $+ 0.8 \times$ mean of the 8 frames | 0.574 | 0.713 |
| none, seed 0 (`ego_sd1p0`) | 0.5005 | 0.750 |
| angles, seed 0 | 0.4998 | 0.750 |
| pe, seed 0 | 0.5003 | 0.750 |
| future, seed 0 | 0.4984 | 0.751 |

- The weight 0.2 was chosen on these clips from 0, 0.2, 0.4, 0.6, 0.8 and 1 (errors 0.595,
  0.574, 0.586, 0.625, 0.680, 0.742). This favours the baseline slightly.
- Each of the four models checked (none, angles, pe and future, seed 0) beats the best baseline by
  about 0.073 (95% interval over recordings [+0.069, +0.077] for none). Each is better on 94–96% of
  the clips and on all 25 recordings.
- On the 96 P08 clips of Tests 1–4, the baseline scores 0.559. The model before fine-tuning
  scored 0.613–0.615 with the signals hidden, and `ego_ft_v2` scores 0.487.

### What this means

- The trained predictor models how egocentric video changes. It beats any blend of past frames by
  a clear margin.
- V-JEPA 2-AC before fine-tuning was worse than the blend of past frames on egocentric video, by
  about 0.055. Of the fine-tuning gain of 0.125 ([[4-results#^test2|Test 2]]), about 0.055 brings
  the model up to the baseline, and about 0.072 goes beyond it.
- The margin over the baseline, 0.073, is the part of the error the predictor explains. The value
  of the signals measured in Test 2 (0.0003) is 0.4% of it. The best gaze form in Test 9 (pe,
  0.0003 over the whole frame) is 0.4%. The positive control (0.0013) is 1.8%.
- These baselines and the correlation $\rho$ should appear in every results table. They make the
  size of an effect readable without knowing the details of the measure.

## 3. Problems and gaps

The problems are listed in order of how much they limit the conclusions.

### 3.1 The tokens do not carry the robot action's role

In V-JEPA 2-AC, the action token at step $t$ is the change of the robot's end-effector pose from
frame $t$ to frame $t+1$. The camera in DROID is fixed, so the only large change in the scene is
the arm. The action token therefore explains most of what changes in the next frame. The
pretrained blocks learned to read the next frame's change from it.

In egocentric video the camera moves with the head. Between two frames 0.27 s apart, most of the
change in the image comes from head motion. The closest equivalent of the robot action is
therefore the change of the head (camera) pose from $t$ to $t+1$, together with the hand motion
over the same step. The project gives the gaze direction and the hand position at frame $t$.
These describe the present state. The weak point "the signals describe the present" in
[[5-discussion#Weak points of the design]] says this for gaze and hands. It does not name head
motion.

This explains the size of the positive control. The future control gives the predictor the gaze
and the hand position of the target frame. Neither says much about how the head moved, which is
the largest source of error. So the positive control gains only 0.26%, and the notes conclude that
"the measure leaves little room for any signal". That conclusion holds for gaze and hand. It is
not known for a signal that describes the head motion.

Two things follow:

1. **A stronger positive control.** Give the predictor the relative 6-DoF pose of the RGB camera
   from frame $t$ to frame $t+1$: 3 numbers of translation and 3 of rotation. If the error drops
   by several percent, the pipeline (frozen blocks 0–17, a linear projection, 3 epochs) can use a
   conditioning token, and the small gaze effects are a property of gaze. If the error barely
   drops, the pipeline cannot use any conditioning token, and the null for gaze says little about
   gaze. Either result is informative. Today the project cannot tell these two cases apart.
2. **A clearer question for gaze.** Gaze leads head turns by about 0.1–0.5 s and the hand by about
   0.5–1 s ([[3-method#Design choices]]). So gaze can predict the head and hand motion of the next
   step, which the model does not know. The question "does gaze help the prediction" can be split
   in two. Does gaze predict the coming head and hand motion? Does gaze add information beyond
   the head and hand motion? The first part has support in the literature (GazeMotion, below).

The data exists. Each HD-EPIC SLAM zip holds `slam/closed_loop_trajectory.csv`, the 6-DoF pose of
the glasses over time, about 400 MB per group of recordings. The zips are downloaded for P04 (7 of
20), P08 (12 of 14), P09 (13 of 15) and part of P05, and not for P01. `python -m ego fetch-calibrations` already
reads single entries from these zips with HTTP range requests. The same method can fetch only the
trajectory files.

The closest published work uses exactly this signal. TrajPilot (Jun et al., 2026) predicts the
future of egocentric video from a frozen V-JEPA 2.1 encoder with a causal predictor, conditioned
on the relative 6-DoF head trajectory. Shuffling the trajectory raised its error by 0.039, while
shuffling a language input raised it by 0.0006. PEVA (Bai et al., NeurIPS 2025) and Navigation
World Models (Bar et al., CVPR 2025) condition on body pose and on camera motion.

### 3.2 The pretrained predictor comes from a fixed camera

The V-JEPA 2-AC configuration (`vjepa2/configs/train/vitg16/droid-256px-8f.yaml`) uses only
`left_mp4_path`, one of DROID's fixed exterior cameras. Blocks 0–17 of the predictor never saw a
moving camera, and they stay frozen during fine-tuning. Before fine-tuning, the predictor was worse
than a blend of past frames (section 2). Six trained blocks, at a learning rate of 1e-4 for 3
epochs, then learned the egocentric dynamics.

A predictor that has learned little of the domain leaves little room for any input to help. Test 10
as planned trains for 8 epochs with the same frozen blocks. It tests H3 (undertrained) for the
number of epochs only. It does not test whether the frozen blocks limit the model. One added arm
would answer this: the model without signals with all 24 blocks trained. If its error falls well
below 0.500, the frozen blocks limit the model, and the gaze comparisons should be repeated on the
stronger predictor.

### 3.3 The loss differs from the pretraining

The V-JEPA 2-AC pretraining used an L1 loss (`loss_exp: 1.0`) over one step and a second step
predicted from the first (`auto_steps: 2`). The models of Tests 1–9 were fine-tuned with squared
error over one step. The two losses have different optima under uncertainty: squared error
favours the mean of the possible futures, L1 their median. Training now uses L1 by default, and
the evaluation reports both measures ([[3-method#Training]], [[3-method#Prediction error and Δ]]).
No model has been trained with L1 yet.

A recent analysis makes a related point. Wang, Cai and Hong (2026) show that a predictor trained
with a one-step squared-error loss learns the conditional mean of the next embedding. It does not
learn a model that can be rolled out over several steps. This matters for the planned rollouts to
1 s ([[6-next-steps]]): four teacher-forced steps learned this way can drift. The pretraining's
second step, and a multi-step loss in fine-tuning, reduce this problem.

### 3.4 Many comparisons, few seeds

Test 9 tested 10 comparisons on 2 measures, 20 tests in all, each with the rule "the 95% interval
excludes 0 and the effect is larger than the seed spread". The rule has no correction for the number
of tests. Two of the 20 tests are the positive control. Of the other 18, three passed: pe against
none on both measures (over the whole frame narrowly, 0.00032 against a spread of 0.00028), and rope
against pe near the gaze point.

The seed spread is the standard deviation of 3 values. With 2 degrees of freedom, the 95% interval
for a standard deviation estimated from 3 values runs from about 0.5 to 6.3 times the estimate. So
"larger than the seed spread" is a weak criterion. The two runs with a loss spike show the problem:
one bad seed moved the angles mean enough to make pe look better than angles. Without that run, pe
and angles are equal ([[4-results#^test9|Test 9]]). The notes mark this analysis as chosen after the
run, which is correct.

The gain of pe over none near the gaze point holds in all 9 pairs of seeds, and no pe or none run
had a spike. With 18 tests and no correction, it still needs a confirmation in Test 10 with these
changes:

- Name one primary comparison before the run: the chosen form against none, near the gaze point.
- Compute the interval with a bootstrap that resamples both recordings and seeds, or with a
  mixed model with recording and seed as random effects.
- Treat the other comparisons as secondary, with a Holm correction.
- Keep the shuffled control for pe, as planned.

### 3.5 The measure has no external reference

The main measure, the latent error 0.27 s ahead on HD-EPIC P08 and P09, is used by no other work.
A reader cannot compare it with anything. The paper's EK100 results ([[parsa]]) are on a dataset
without gaze, so they cannot test the question. Section 6 lists benchmarks where gaze is available.

### 3.6 Smaller points

- **Test 8 shows a known effect.** That the object at the gaze point tells which object is picked
  up next is expected from HD-EPIC's own priming analysis: 94.8% of the objects that can be checked
  are looked at before they are picked up. The value of Test 8 is that it measures the effect with
  the project's frozen features, for new people, against head-point and angle controls. This is a
  sound check before Test 9. It is not a new finding about gaze.
- **Hands have the same form problem as gaze.** The hand token holds 12 numbers in the device
  frame and cannot point at the image either. Hand position in the image is listed in
  [[6-next-steps]].
- **The scaling constants** for gaze and hand are rough guesses. This is known and planned.
- **Gaze history as an average.** Test 8 averages the gaze history. An ordered history, such as the
  last fixations drawn as a path (as in Materia et al., below), may carry more.

## 4. Is the direction sound?

The question is sound and has not been answered in the literature. The careful diagnosis of the
null result is the main strength of the project.

The weakness lies in the measuring instrument. The project measures the value of gaze with a
one-step predictor 0.27 s ahead, whose frozen blocks come from a fixed robot camera, with an
error over the whole frame. The positive control shows that this instrument gives gaze and hand
little room. So the null result is at present mostly a statement about the instrument. The project
has partly recognised this: the weak point "short horizon, coarse measure" and hypotheses H1
(horizon) and H3 (undertrained) state it.

The next steps should do three things, in this order:

1. Calibrate the instrument with a signal known to matter: head motion (section 3.1), and a
   predictor without frozen blocks (section 3.2).
2. Move the measure to where gaze is known to carry information: the object about to be picked up,
   0.5–1 s ahead, and anticipation of the next object or action. Test 8 found the gaze-point
   information largest 0.5 s ahead.
3. Report on a benchmark that others use (section 6).

Test 10 as planned (8 epochs, frozen blocks, 0.27 s, whole frame) changes only the number of epochs.
It is unlikely to change the conclusion. The arms in sections 3.1 and 3.2 would add more information
for a similar cost.

## 5. Related work

### 5.1 Has this been done before?

No published work was found that gives gaze as an input to a latent (JEPA-style) video predictor or
to an action-conditioned world model. The searches covered gaze-conditioned video prediction,
egocentric world models, V-JEPA 2 with human signals, gaze in video-language models, and gaze for
action anticipation, up to October 2026. The nearest work falls into five groups.

**Egocentric world models conditioned on motion.** These models predict egocentric video given how
the person moves. None of them uses gaze.

| Work | Model | Conditioning | Data |
|---|---|---|---|
| PEVA (Bai et al., NeurIPS 2025) | autoregressive diffusion transformer | 3D whole-body pose, relative per step | Nymeria |
| Navigation World Models (Bar et al., CVPR 2025) | conditional diffusion transformer | camera trajectory | navigation video |
| EgoWM (2026) | pretrained video diffusion model with added action layers | robot and humanoid actions | several |
| EgoControl (2025) | video diffusion | 3D full-body pose | egocentric video |
| TrajPilot (Jun et al., 2026) | causal predictor on frozen V-JEPA 2.1 features | relative 6-DoF head trajectory | Ego-Exo4D, Ego4D, EK100 and others |

CamFormer (Xue et al., CVPR 2026) shows that the camera trajectory alone tells much about what the
person is doing. Nymeria, the dataset of PEVA, records Aria eye gaze with full-body motion. PEVA with gaze added has
not been published. TrajPilot is the closest in design to this project: a frozen V-JEPA encoder, a
learned predictor, and a motion signal as input. It also reports that language as an input adds
nothing, while language as the space of the targets helps. That fits Path 3, which uses language as
the target.

**Gaze in video-language models.** These models answer questions. Gaze enters as an input.

- GazeQwen (Pham, Nguyen and Le, 2026; note [[gazeqwen]]). A small resampler reads V-JEPA 2.1
  features with position encodings of the fixations and adds a correction inside Qwen2.5-VL. It
  scores 63.9% on StreamGaze, 16.1 points above the same model without gaze. Its fixation position
  encoding is close to the pe form of Test 9.
- Materia, Ragusa and Farinella (ICPR 2026). The last 15 fixations are drawn on the frames as a
  path, with numbered object marks. On HD-EPIC's gaze interaction anticipation it reaches 27.2%
  with LLaVA-OneVision and 27.5% with Gemini 2.0 Flash, against 20.4% for LLaVA-OneVision alone.
  With their frame sampling but without gaze or marks, LLaVA-OneVision scores 24.9%, so gaze and
  marks together add 2.3 points. Chance is 20%.
- Pani and Yang (2026). Gaze-based queries and a loss that aligns the model's attention with gaze.
  About 13% better semantic scores when describing future events.
- Benchmarks built for this: EgoGazeVQA (Peng et al., NeurIPS 2025) and StreamGaze (2025).

**Gaze for action recognition and anticipation.**

- Li, Liu and Rehg (ECCV 2018) use gaze as a weighting over the feature map, on EGTEA Gaze+. The
  gaze-point features of Test 8 are a form of this.
- Zhang et al. ("Can Gaze Inform Egocentric Action Recognition?", ETRA 2022) fused gaze with video
  at test time on EGTEA Gaze+ and found no gain. Their analysis: the network's attention and the
  gaze points overlapped little, so the gaze input did not reach what the network used. This is
  the closest published null result to Test 2.
- SAGE (Kuang and Agarwal, ECCV 2026) models current and future actions and gaze together, on EGTEA
  Gaze+, VidHOI and a new Exo-Cook set.
- G3Ego (Haralović et al., ECCV 2026 workshop) builds scene graphs from the objects at the gaze
  point, on EGTEA Gaze+ and MECCANO.
- Ozdel et al. (ETRA 2024) use gaze to build a graph for intention recognition, in simulated
  VirtualHome scenes.
- HoloAssist (Wang et al., ICCV 2023), recorded with HoloLens 2, has gaze, hand pose and head pose.
  Adding hands and gaze to the video improved the prediction of the type of help the person needs.

**Gaze for forecasting motion.**

- GazeMotion (Hu et al., IROS 2024) first forecasts gaze, then uses it to forecast body motion. The
  error falls by up to 7.4% against the best methods without gaze, on MoGaze, ADT and GIMO.
- He, Zhang and Stienen (2025) forecast 3D hand motion with gaze. Gaze helps most when few past
  frames are given.

**The strongest EK100 anticipation systems.** JFAA (Chu et al., 2026), first in the EgoVis 2026
challenge, and TAP-JEPA (Wang and Xu, 2026), second with 27.91% action mean top-5 recall on the test
set, both use a frozen V-JEPA 2.1 ViT-G/384 encoder and predictor with light probes. V-JEPA 2 ViT-g
(384 px) reaches 39.7% action recall@5 on the validation set. None of them uses gaze or hands.

### 5.2 Where the project stands

Across these works a pattern appears. Gaze helps when it selects content: what lies at the gaze
point, or the path of the last fixations drawn on the image. It also helps when the task is about
objects or intentions. It helps little when it enters as a few numbers into a model whose target is
dominated by other changes in the image. Zhang et al. (2022) found this for recognition. GazeQwen
found almost no gain from gaze for future action prediction ([[2-background]]).

The project reproduces this pattern within one setup. The three gaze numbers add nothing to the
probe of Test 8, and the features at the gaze point tell which object comes next. In the predictor
the gain is small whatever the form: angles and pe both lower the error by about 0.0003 over the
whole frame, and rope does not help (Test 9). As far as the
searches show, the project is the first to test gaze as an input to an action-conditioned latent
predictor. It is also the first to compare a content form (pe) with an attention form (rope) of
gaze in such a predictor. These are publishable as a careful negative-to-weak result, once the
instrument is calibrated (section 3.1).

## 6. Benchmarks

| Benchmark | Task | Gaze | Best published result | Fit to the project |
|---|---|---|---|---|
| HD-EPIC VQA, gaze interaction anticipation | Which object will the person use next? 1,000 questions with 5 answers, chance 20%. The 10 s clip ends 0.3 s after gaze first falls on the object. No training split. | Aria, the same data | 27.5% for this question type (Materia et al., ICPR 2026, no training). 52.15% for the two gaze question types together (EgoAdapt, winner of the CVPR 2026 HD-EPIC VQA challenge, Qwen3-VL-8B, no training). | Best. Test 8 is close to this task. |
| HD-EPIC VQA, gaze estimation | What is the person looking at? 1,000 questions. | Aria | see the row above | Medium. |
| EGTEA Gaze+ action anticipation | Name the next action. Three official splits (split 1: 8,299 training and 2,022 test segments, 106 actions). Two protocols: 0.5 s ahead, top-1 and mean class accuracy, split 1 (AVT, InAViT, SAGE); or 0.25–2 s ahead, top-5 accuracy (RULSTM). | 2D gaze from SMI glasses | 58.4% mean class accuracy at 0.5 s (SAGE, ECCV 2026; gaze predicted, not measured, at test time) | Good. The standard dataset for gaze and anticipation. |
| EK100 action anticipation | Name the action 1 s ahead | none | 27.91% action MT5R on the test set (TAP-JEPA, second) | Only to show that the conditioning does not harm transfer. |
| Ego4D short-term object interaction anticipation | Box, noun, verb and time to contact of the next object | for a subset of Ego4D only | EgoVis 2026 reports, for example VISTA with V-JEPA 2.1 | Medium. Whether the gaze subset overlaps this task's clips is not checked. |
| StreamGaze, EgoGazeVQA | Questions about gaze and intent in streaming video | yes | GazeQwen 63.9% on StreamGaze | Only with a language model, so for Path 3. |
| PEVA protocol on Nymeria | Egocentric video prediction from body motion | Aria gaze recorded | PEVA | Good for the world-model question with gaze added to body motion. |

**HD-EPIC gaze interaction anticipation is the nearest benchmark.** The data, the glasses and the
participants are the same as in the project. For this question type alone, the best published
accuracy is 27.5%, 7.5 points above chance. The Test 8 probe can answer the questions directly:
score each of the five candidate objects with the probe's output for its noun class, and choose the
highest. Three points need care:

1. HD-EPIC is meant for evaluation only and has no training split. Every published result on its
   VQA benchmark comes from models that were not trained on HD-EPIC. The probe is trained on picks
   of P01–P07, so only questions on P08 and P09 videos are fair (280 of the 1,000), or the probe must
   be trained once per held-out person. Either way it is a different setting from the published
   results, and must be reported as such.
2. The answers are free-form object names. They must be mapped to noun classes, as Test 8 does
   for the picks.
3. The timing differs from Test 8. The clip ends 0.3 s after the first look at the object. Test 8
   uses fixed times before the pick.

A probe on frozen features with gaze, compared with VLMs without gaze on this benchmark, would be a
direct and cheap external result. The same probe on the predictor's output gives the world-model
version.

## 7. Recommendations

In order of value for the cost:

1. **Head motion as a positive control.** Fetch `closed_loop_trajectory.csv` for every recording
   group. Train one arm with the relative camera pose from $t$ to $t+1$ in place of the hand token,
   and one with it added to gaze and hand. This calibrates the instrument (section 3.1).
2. **Baselines in every table.** The repeat and blend baselines of section 2, and the correlation
   $\rho$ next to the error.
3. **A predictor without frozen blocks.** One run of the model without signals with all 24 blocks
   trained, to measure how much the frozen blocks limit the predictor (section 3.2).
4. **An external benchmark.** Run the Test 8 probe on HD-EPIC's gaze interaction anticipation
   questions for P08 and P09 (section 6). Then consider EGTEA Gaze+.
5. **The measure where gaze matters.** The error on the patches of the object about to be picked
   up, and rollouts to 0.5–1 s, as already planned in [[6-next-steps]].
6. **Statistics for Test 10.** One primary comparison named before the run, a bootstrap over seeds
   and recordings, and a correction for the secondary comparisons (section 3.4).
7. **The loss.** Training now uses L1, as in the pretraining (section 3.3). One short run with L1
   shows whether the loss changes the comparison of gaze forms.

## References

- Bai, Tran, Bar, LeCun, Darrell, Malik. Whole-Body Conditioned Egocentric Video Prediction (PEVA).
  NeurIPS 2025. https://arxiv.org/abs/2506.21552
- Jun, Nguyen-Truong, Seminara, Torresani. TrajPilot: Trajectory-Conditioned Egocentric Prediction
  ("How You Move Tells What You'll Do"). arXiv 2026. https://arxiv.org/abs/2605.20388
- Xue et al. Seeing without Pixels: Perception from Camera Trajectories (CamFormer). CVPR 2026.
  https://arxiv.org/abs/2511.21681
- EgoWM: Walk through Paintings: Egocentric World Models from Internet Priors. arXiv 2026.
  https://arxiv.org/abs/2601.15284
- EgoControl: Controllable Egocentric Video Generation via 3D Full-Body Poses. arXiv 2025.
  https://arxiv.org/abs/2511.18173
- Ma et al. Nymeria: A Massive Collection of Egocentric Multi-modal Human Motion in the Wild. ECCV
  2024. https://arxiv.org/abs/2406.09905
- Pham, Nguyen, Le. GazeQwen: Lightweight Gaze-Conditioned LLM Modulation for Streaming Video
  Understanding. arXiv 2026. https://arxiv.org/abs/2603.25841
- Materia, Ragusa, Farinella. Leveraging Gaze and Set-of-Mark in VLLMs for Human-Object Interaction
  Anticipation from Egocentric Videos. ICPR 2026. https://arxiv.org/abs/2604.03667
- Pani, Yang. Gaze-Regularized VLMs for Ego-Centric Behavior Understanding. arXiv 2026.
  https://arxiv.org/abs/2603.23190
- Peng, Hua, Liu, Lu. In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with
  Gaze-Guided Prompting (EgoGazeVQA). NeurIPS 2025 Datasets and Benchmarks.
  https://arxiv.org/abs/2509.07447
- StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos. arXiv
  2025. https://arxiv.org/abs/2512.01707
- Zhang, Crandall, Proulx, Talathi, Sharma. Can Gaze Inform Egocentric Action Recognition? ETRA
  2022. https://research.facebook.com/publications/can-gaze-inform-egocentric-action-recognition/
- Kuang, Agarwal. SAGE: Synchronized Action-Gaze Recognition and Anticipation for Human Behavior
  Understanding. ECCV 2026. https://arxiv.org/abs/2607.04017
- Haralović, Ramakrishnan, Talavera Martinez. G3Ego: Gaze-Guided Graphs for Egocentric Action
  Understanding. ECCV 2026 workshop. https://arxiv.org/abs/2608.20157
- Ozdel et al. Gaze-Guided Graph Neural Network for Action Anticipation Conditioned on Intention.
  ETRA 2024. https://arxiv.org/abs/2404.07347
- Wang et al. HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in
  the Real World. ICCV 2023. https://arxiv.org/abs/2309.17024
- Hu, Schmitt, Haeufle, Bulling. GazeMotion: Gaze-guided Human Motion Forecasting. IROS 2024.
  https://arxiv.org/abs/2403.09885
- He, Zhang, Stienen. Gaze-Guided 3D Hand Motion Prediction for Detecting Intent in Egocentric
  Grasping Tasks. arXiv 2025. https://arxiv.org/abs/2504.01024
- Chu et al. JFAA: Technical Report for the EPIC-KITCHENS-100 Action Anticipation Challenge at
  EgoVis 2026. https://arxiv.org/abs/2605.20904
- Wang, Xu. TAP-JEPA: Frozen Future-Latent Probing and Two-Stage Score Fusion for EPIC-KITCHENS-100
  Action Anticipation. arXiv 2026. https://arxiv.org/abs/2606.00662
- Wang, Cai, Hong. One-Step Next-Latent Prediction Is Not a World Model. arXiv 2026.
  https://arxiv.org/abs/2609.36227
- Perrett et al. HD-EPIC: A Highly-Detailed Egocentric Video Dataset. CVPR 2025.
  https://arxiv.org/abs/2502.04144
- EgoAdapt: A Multi-Scene Egocentric Adaptation Method for CVPR 2026 HD-EPIC VQA Challenge. arXiv
  2026. https://arxiv.org/abs/2605.24500
- Li, Liu, Rehg. In the Eye of the Beholder: Gaze and Actions in First Person Video (EGTEA Gaze+).
  ECCV 2018, extended in TPAMI. https://arxiv.org/abs/2006.00626
- Girdhar, Grauman. Anticipative Video Transformer. ICCV 2021. https://arxiv.org/abs/2106.02036
- Furnari, Farinella. Rolling-Unrolling LSTMs for Action Anticipation from First-Person Video.
  TPAMI 2020. https://arxiv.org/abs/2005.02190
- Assran et al. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and
  Planning. 2025. https://arxiv.org/abs/2506.09985
