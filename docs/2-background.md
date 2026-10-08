---
type: report
status: running
created: 2026-10-05
updated: 2026-10-08
---

# 2. Background

The models and papers this work builds on, and the terms used in these notes. Each paper has
its own note in `papers/literature/`.

## V-JEPA 2

V-JEPA 2 is a video model from Meta ([[vjepa]]). It learns from video without labels. Part of
a video is hidden, and the model predicts the embeddings of the hidden part. It never predicts
pixels.

It has three parts:

- A student encoder sees the visible parts and turns them into embeddings.
- A teacher encoder sees the whole video and gives the target embeddings. It is an average
  (EMA) of the student and is not trained directly.
- A predictor takes the student's embeddings and predicts what the teacher would output for
  the hidden parts.

V-JEPA 2 was trained on more than 1 million hours of video. Its authors note that its
accuracy in action anticipation drops at horizons longer than 1 s.

## V-JEPA 2-AC

V-JEPA 2-AC is a variant for robots. Its predictor also receives the robot's action and state
for each frame, as two extra tokens next to the image tokens. The action at step t is the
change of the robot's pose from frame t to frame t+1. The predictor then predicts the
embeddings of the next frame, given what the robot does. This project replaces the two robot
tokens with a gaze token and a hand token ([[3-method]]).

V-JEPA 2-AC was pretrained on 62 hours of robot video from one fixed camera of the DROID
dataset (`left_mp4_path` in its configuration), with an L1 loss.

## V-JEPA 2.1

V-JEPA 2.1 (Mur-Labadia et al., 2026) is a newer version of V-JEPA 2, released on 16 March 2026.
Its training puts the loss on every token and at several depths of the encoder, so its patch
features describe the image in more detail. Its checkpoints work at 384 pixels: ViT-B and ViT-L,
both distilled from ViT-G, and ViT-g and ViT-G. Each comes with its predictor. The predictor
fills in learned mask tokens at the positions it has to predict, and attends over all tokens. It
has no action input, and no action-conditioned version exists. The two strongest published EK100
anticipation systems read its predicted tokens of a frame 1 s ahead (below). The ViT-L and
ViT-G checkpoints are in `data/model_checkpoints/vjepa2.1/`. Test 11 builds on them
([[6-next-steps]]).

## Related work

**GazeQwen** ([[gazeqwen]]). Adds gaze to a frozen video-language model (Qwen2.5-VL). A small
resampler module computes a correction from the video features and the gaze, and adds it
inside the language model. Three ideas were taken from it: zero initialization, a learned
gate, and a sine-and-cosine encoding of the gaze position (Coord-PE). None of them were built
([[3-method]]). Two points from studying it:

- Its equation 2 adds the same gaze value to the attention key of every image token. This
  value cancels in the softmax. So gaze works as a gate and does not select image regions.
- Its future action prediction task shows almost no gain from gaze. This agrees with the
  results here.

**VL-JEPA** ([[vl-jepa]]). Predicts the embedding of a text answer instead of generating the
text word by word. Answers with the same meaning get similar embeddings. On EK100, its
advantage over V-JEPA 2 grows with the horizon: +1.5 points at 1 s, +2.6 at 2 s, +3.5 at 4 s
and +4.6 at 10 s. It is the basis for Path 3.


**VLA-JEPA** ([[vla-jepa]]). Pretrains robot policies with a JEPA objective. The target
encoder sees future frames. The student sees only the current frame. So future information is
used only as a target. This paper has not been studied for the project yet.

**Gaze as a training signal.** Some methods use gaze to supervise a model during training and
do not need it at test time. Examples are GABRIL (Banayeeanzade et al., 2025), the
gaze-regularized vision-language models of Pani and Yang (2026), and the goal-consistency
objective of Roy and Fernando (2022).

**Gaze leads the hands.** In studies of eye–hand coordination, gaze moves to the next object
at the start of a reach and leads the grasp by more than 0.5 s. HD-EPIC (Perrett et al., CVPR
2025) reports that 94.8% of the objects that can be looked at in advance are looked at before
they are picked up, on average 4.0 s before, and 88.5% before they are put down, 2.6 s before.
Methods that use gaze as a position in the scene help to anticipate the next object or action:
gaze as a weighting over a feature map (Li, Liu and Rehg, 2018), gaze to select scene parts
for intention recognition (Ozdel et al., 2024), and a position prompt on frozen DINOv2 patch
features to locate gaze targets (Gaze-LLE, 2025). Video-language models without gaze score
20–29% on HD-EPIC's questions about gaze and upcoming interactions; people score 75%.
**The strongest EK100 systems.** The two best published EK100 anticipation systems use frozen
V-JEPA 2.1 features with separate verb and noun probes (Chu et al., 2026a; Wang and Xu, 2026).
They use no gaze or hand data.

## Benchmarks

The results of the paper are measured on two benchmarks with gaze (Test 12 in
[[6-next-steps]]). A wider survey of related work and benchmarks is in [[review]].

**HD-EPIC gaze interaction anticipation.** One of the 30 question types of HD-EPIC's VQA
benchmark (Perrett et al., CVPR 2025): "What object will the person interact with next, ignoring
ongoing interactions?" It has 1,000 questions with 5 answers each, so chance is 20%. Each 10 s
clip ends 0.3 s after the person first looks at the object. HD-EPIC is released for evaluation
only and has no training split. Every published result comes from a model that was not trained
on HD-EPIC:

| Method | Accuracy |
|---|---|
| LLaVA-OneVision 7B, zero-shot | 20.4% |
| Gemini 1.5 Pro, zero-shot (the HD-EPIC paper) | 21.0% |
| Qwen2.5-VL, an entry of the 2025 challenge | 22.0% |
| Materia et al. (ICPR 2026): the last 15 fixations drawn on the frames, numbered object marks, no training | 27.5% |

EgoAdapt, the winner of the CVPR 2026 HD-EPIC VQA challenge (Qwen3-VL-8B, no training), reports
52.15% on the two gaze question types together, gaze estimation and interaction anticipation,
without a figure for each. HD-EPIC defines no split by participant. The split of this project,
P01–P07 for training and P08–P09 for testing, is its own.

**EGTEA Gaze+ action anticipation.** EGTEA Gaze+ (Li, Liu and Rehg, ECCV 2018) has 28 hours of
cooking by 32 people in 86 sessions, at 24 frames per second, with the 2D gaze point from SMI
eye-tracking glasses. It has 10,321 action segments of 106 classes and three official splits;
split 1 has 8,299 training and 2,022 test segments. Methods train on the training set of a split,
with backbones pretrained on other data. Two protocols are in use:

- 0.5 s ahead, top-1 and mean class accuracy, on split 1 (AVT, InAViT, SAGE). In SAGE's table, the
  mean class accuracy is 35.2% for AVT, 58.2% for InAViT and 58.4% for SAGE. SAGE uses gaze labels
  in training and predicts gaze at test time; it does not use measured gaze.
- 0.25–2 s ahead, top-5 accuracy (RULSTM).

## Terms

- **Skill.** 1 − (probe error / error of always guessing the average). 0 is chance and 1 is
  perfect.
- **Paired test.** A comparison of two conditions on the same clips.
- **Wilcoxon signed-rank test.** A paired test. It checks whether the per-clip differences are
  consistently above or below zero.
- **p-value.** The chance of seeing a difference at least this large if there were no real
  effect. A value above 0.05 is usually read as no evidence of an effect.
- **Confidence interval.** A range that likely contains the true value. These notes use 95%
  intervals.

