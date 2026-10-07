---
type: report
status: running
created: 2026-10-05
updated: 2026-10-07
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

