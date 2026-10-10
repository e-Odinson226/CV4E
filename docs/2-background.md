---
type: report
status: running
created: 2026-10-05
updated: 2026-10-09
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

### Ego4D and Ego-Exo4D

Ego4D and Ego-Exo4D are the two largest egocentric datasets with official splits, test servers
and yearly challenges. Unlike HD-EPIC, both have training sets. Neither has an anticipation
benchmark that takes measured gaze as an input. The numbers below are from the papers, the
official documentation and the challenge reports; they are checked against the data only where
this is said.

| Dataset | Glasses | Gaze | Hands | Official training split | Anticipation task |
|---|---|---|---|---|---|
| HD-EPIC | Aria Gen 1 | 3D (Aria MPS), 10 or 60 Hz, all but 2 recordings | MPS wrists and palms, 10 or 30 Hz | no | gaze interaction anticipation (VQA, evaluation only) |
| EGTEA Gaze+ | SMI | 2D point, all videos | masks on about 14,000 frames | yes, 3 splits | action anticipation |
| Ego-Exo4D | Aria Gen 1 | 3D (Aria MPS), 10 Hz | 3D hand pose annotations | yes | none for objects; "next keystep" inside procedure understanding |
| Ego4D | many models | 2D point, only in about 31–45 h of social recordings | boxes on key frames of the hand–object clips | yes | short-term object interaction (STA), long-term action (LTA), both without gaze |

#### Ego4D

**The data.** Ego4D (Grauman et al., CVPR 2022) has 3,670 hours of daily-life video from 931
camera wearers at 74 locations in 9 countries. The scenarios are household, outdoor, work and
leisure activities. The video was recorded with several head-mounted camera models, not with Aria
glasses. Release v2 added videos and annotations for the forecasting and hand–object tasks, and
v2.1 added the Goal-Step annotations.

**Gaze.** Only a small part of Ego4D has gaze. The Ego4D paper lists 45 hours with gaze. The
part that published gaze work uses has 27 videos of 80 participants, about 31 hours, recorded in
the social setting (conversations and games) with Pupil Invisible eye trackers (Lai et al., BMVC
2022). The gaze is a 2D point in normalized image coordinates, sampled faster than the video.
Of the fields in the gaze files, only the frame index, the confidence and the two coordinates
are filled. There is no 3D gaze and no gaze depth. Ego4D has no continuous hand tracking either:
hands are annotated as boxes on the key frames of the hand–object clips.

**The benchmarks.** Ego4D defines five groups of benchmarks.

| Group | Tasks |
|---|---|
| Episodic memory | natural language queries, visual queries (2D and 3D), moment queries |
| Hands and objects | point-of-no-return localization, object state change, active object detection |
| Audio-visual diarization | speaker localization and tracking, active speaker detection, diarization, transcription |
| Social | looking at me, talking to me |
| Forecasting | locomotion, future hand positions, short-term object interaction anticipation (STA), long-term action anticipation (LTA) |

Goal-Step (Song et al., NeurIPS 2023) was added later. It annotates about 48,000 procedural step
segments (430 hours) and the high-level goals of 2,807 hours of video. The 2026 challenges were
natural language queries, Goal-Step and STA for Ego4D.

**Short-term object interaction anticipation (STA).** The model sees the video up to a frame. It
must name the objects that the person will touch next and, for each one, give its box in the
last frame, its noun, the verb of the interaction and the time to contact. Release v2 has 243
hours of annotated clips: 98,276 training, 47,395 validation and 19,780 test examples, with 128
noun and 81 verb classes. The test labels are hidden and scored by a server. The measure is the
top-5 mean average precision. A prediction counts only if its box overlaps the true box with an
IoU above 0.5. "Noun" also needs the right noun, "Noun+Verb" the right noun and verb,
"Noun+TTC" the right noun and a time-to-contact error below 0.25 s, and "Overall" all of them.
The 2026 leaderboard:

| Method | Overall | Noun | Noun+Verb | Noun+TTC |
|---|---|---|---|---|
| Faster R-CNN + SlowFast, baseline v2 | 3.61 | 26.15 | 9.45 | 8.69 |
| StillFast, baseline v2 | 5.12 | 25.06 | 13.29 | 9.14 |
| VISTA, first place (Qiu et al., 2026) | 5.40 | 27.26 | 16.15 | 8.95 |

VISTA adds frozen V-JEPA 2.1 features of the last 8 frames, read by an attentive probe, to an
object detector on the last frame. It uses no gaze and no hand input. Entries of earlier years
report higher Overall scores (6.75 in 2024). The reports do not explain the difference, so the
2026 scores are compared only with each other.

**Long-term action anticipation (LTA).** The model sees a clip and predicts the next 20 actions,
each a verb and a noun. It may give 5 sequences, and the best one is scored by its edit distance
to the true sequence (lower is better). The published edit distances for actions on the test set
are about 0.85–0.88. The 2025 winner passes the predicted verbs and nouns to a fine-tuned
language model (Llama 2, 7B).

**Benchmarks on the gaze part.** Two tasks use the 31 hours with gaze, outside the official
challenges: gaze estimation (GLC, Lai et al., BMVC 2022: 43.1 F1, with 20 videos for training and
7 for testing) and gaze anticipation (CSTS, Lai et al., ECCV 2024). Both predict gaze. Neither uses
gaze to predict something else.

#### Ego-Exo4D

**The data.** Ego-Exo4D (Grauman et al., CVPR 2024; IJCV 2025) records skilled activities from
the wearer's Aria glasses and from 4 or 5 GoPro cameras around the scene at the same time.
Release v2 has 1,286 hours of video, 221 of them from the Aria glasses, in 5,035 takes. The paper
gives 740 participants, 123 scenes and 13 cities. The website gives more than 800 participants,
131 scenes and, in its introduction, 1,422 hours; it does not explain the difference from the
v2 figure. A take lasts 2.6 minutes on average, from 8 s to 42 minutes.

There are 43 activities in 8 domains. Three are procedural: cooking, bike repair and health care
(a COVID-19 test, CPR). Five are physical: soccer, basketball, dance, bouldering and music.
Cooking is the largest domain: nearly 100 hours of Aria video, more than 650 takes, more than 170
cooks and 60 kitchens. The language annotations are keysteps (a taxonomy of 689), narrations of
each action, and commentary from experts on how well the person performs.

**Gaze and hands.** The glasses are Aria Gen 1, the same as in HD-EPIC. The RGB camera records at
30 fps and 1408 × 1408 pixels. The two eye-tracking cameras record at 10 fps, so the gaze is at
10 Hz, as for P01–P03 of HD-EPIC. Each participant did the gaze calibration of the Aria app, so
both the general and the personalized gaze of MPS can be produced. The gaze, the SLAM trajectory
and the semi-dense point cloud are released per take, and pre-computed 2D gaze points are
included. For hands, the dataset has 3D hand pose annotations of 21 joints per hand: 68,000
frames annotated by hand in 3D and 4.3 million produced automatically. Later work uses the MPS
wrist positions as well (Learning Predictive Visuomotor Coordination, CVPR 2026 Findings). How
many takes have MPS hand tracking is not checked.

**The benchmarks.** Four groups. All tasks use one common split, with the counts per task in the
paper. Whether the split separates participants is not checked. Several tasks forbid gaze as an
input at test time.

| Group | Task | Gaze as input | Measure | Result |
|---|---|---|---|---|
| Ego-exo relation | Correspondence: find the object of a mask in one view in the other view | not used | IoU and others | under 30% IoU (paper baselines) |
| Ego-exo relation | Translation: generate the ego view from the exo views | not used | IoU, image similarity | IoU 10.3 (paper baseline) |
| Recognition | Fine-grained keystep recognition: classify a trimmed ego clip into one of 278 keysteps | in training only | top-1 accuracy | 41.5% (paper baseline) |
| Recognition | Energy-efficient keystep recognition: detect keysteps online under a power budget (20 mW or 2.8 W) | not stated | calibrated mAP | 77.9 at 20 mW, 93.2 at 2.8 W (paper baselines) |
| Recognition | Procedure understanding: from the video up to now, mark each keystep as previous, optional, missing, a mistake or next | not stated | calibrated AP, chance 50% | 2026 challenge |
| Proficiency | Demonstrator proficiency: classify the person as novice, early, intermediate or late expert | forbidden | top-1 accuracy | 50.4% against 42.4% for the majority class (paper); 53% (2025 challenge) |
| Proficiency | Demonstration proficiency: find the moments of good execution and of needed improvement | not stated | mAP | about 4 (paper baselines) |
| Ego pose | Body pose: 17 joints of the wearer from the ego video and IMU | forbidden | MPJPE | 18.5 cm (paper baseline) |
| Ego pose | Hand pose: 21 joints per visible hand from the ego frames | not stated | PA-MPJPE | 11.1 mm (paper baseline); 8.3 mm (2025 challenge) |

The 2026 challenges were body pose and procedure understanding.

**Gaze in published work on Ego-Exo4D.** Learning Predictive Visuomotor Coordination forecasts
head pose, gaze and upper-body motion from the ego video and the past motion. It uses gaze as a
target and as an input, but defines its own task, not one of the benchmarks above.

#### What they offer the project

**Ego-Exo4D is the closest match to HD-EPIC with a training set.** It uses the same glasses and
the same MPS outputs, so the gaze projection of [[3-method#Gaze position in the image]] should
apply with each take's calibration; this is not yet tested. Its cooking takes, nearly 100 hours,
are close to HD-EPIC's kitchens. Its official split gives a training set that HD-EPIC lacks. The
measures of Tests 9–11, the prediction error over the frame and near the gaze point, can be
computed on it directly. Three limits follow from the facts above. The gaze is at 10 Hz. There is
no task that asks which object comes next. The benchmarks that forbid gaze at test time would be
reported twice, once by their rule and once with gaze as a separate setting. The "next keystep"
label of procedure understanding is the nearest task to intention.

**Ego4D has the standard short-term anticipation benchmark, but no gaze for it.** STA asks for
the next object, the verb and the time to contact, and its leader already reads frozen V-JEPA 2.1
features. The measured gaze, however, comes only from the social recordings, while the STA clips
come from the hand–object and forecasting videos. Whether any STA clip has gaze is not checked,
since the Ego4D metadata is not on this machine; given the scenarios, little overlap is expected.
STA can therefore be used without gaze, as a transfer check like EK100. Ego4D's scale also makes
it a candidate for pretraining a predictor without signals.

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

