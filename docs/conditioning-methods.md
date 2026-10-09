---
type: report
created: 2026-10-08
updated: 2026-10-08
---

# Ways to introduce gaze and hand into a model

How published work gives a model a person's gaze and hands, and which of those ways are open to
this project. The project's own forms are in [[3-method#Design choices]], its results in
[[4-results]], and the designs considered for the predictor's attention in
[[7-predictor-redesigns]].

The reason to catalogue this. Tests 2, 4, 9 and 10 all put gaze into one conditioning token and
all returned a gain at or below the spread between seeds. [[4-results#^test9|Test 9]] found the
mechanism: in a RoPE head the attention between two tokens depends on their content, not on their
distance, so a token placed at the gaze point is not a token the other tokens attend to.
[[4-results#^test10|Test 10]] then removed the other candidate explanation, because training the
whole predictor instead of its last 6 blocks did not make gaze worth more. So the question is not
how to fix the token. It is which of the other channels to use.

## Gaze

| | Channel | Where the signal acts | Published examples |
|---|---|---|---|
| G1 | A vector inside a token | the token's content | this project, Tests 1-4; V-JEPA 2-AC's action token |
| G2 | A token moved to the gaze point | the attention's positions | this project, the rope form of Test 9 |
| G3 | Choosing which patches the encoder sees | the encoder's input | this project, Path 1 |
| G4 | A soft weight over patch features before pooling | the features | gaze-weighted patch pooling |
| G5 | A map added to the patch features | the features | this project, Test 11 |
| G6 | A term added to the attention scores | the attention's scores | Eyes on Target; [[7-predictor-redesigns]] option 1 |
| G7 | Heatmaps turned into tokens of their own | the token sequence | ARGaze |
| G8 | A loss that matches attention to the gaze heatmap | the training only | Gaze-VLM |
| G9 | A latent variable with an auxiliary loss | the training only | Li, Liu and Rehg |
| G10 | Drawn on the frames as a visible mark | the pixels | Materia et al.; GazeQwen |
| G11 | A node in a graph | the graph | G3Ego |
| G12 | A quantity the model predicts, then uses | the model's own output | SAGE |

**G4, a soft weight over the features.** A fixation heatmap reweights the patch features before
they are pooled. The gaze point does not have to win any attention competition, because it scales
features directly. This is the closest published relative of the Test 11 design.

**G6, a term added to the attention scores.** Eyes on Target injects gaze-derived features into a
ViT's attention so that spatial selection favours the regions the person looked at, and reports
the effect on individual attention heads. It uses the gaze point, its depth, the pupil dilation
and the gaze direction. This is the one channel that writes a distance prior into the scores
themselves, which is exactly what RoPE does not provide.

**G8 and G9, gaze as supervision rather than as input.** Gaze-VLM converts gaze heatmaps into
patch-level distributions and trains the model's attention to match them, with no gaze at
inference. Li, Liu and Rehg treat gaze as a probabilistic variable whose samples form an attention
map that aggregates the visual features; gaze supervises the training and is inferred at test
time. Both report that the model keeps the benefit without gaze at test time.

**G10, drawn on the frames.** Materia et al. draw the last 15 fixations on the frames and number
the candidate objects, with no training, and reach 27.5% on HD-EPIC's gaze interaction
anticipation ([[review]]). The signal reaches the model through the pixels, so the encoder sees
it. This project's encoder is frozen and is not fine-tuned, but nothing stops the pixels from
carrying the mark.

**G12, gaze as a prediction target.** SAGE predicts gaze at test time in place of using the
measured gaze, and holds the best published mean class accuracy on EGTEA Gaze+ at 0.5 s
([[2-background#Benchmarks]]). It matters for Test 12: a model that consumes measured gaze is not
strictly comparable with SAGE, and that has to be stated.

## Hand

| | Channel | Where the signal acts | Published examples |
|---|---|---|---|
| H1 | A pose vector inside a token | the token's content | this project, Tests 1-4 |
| H2 | A body-pose action vector conditioning the predictor | the conditioning | PEVA |
| H3 | Finger keypoints as the action space, with a consistency loss | the conditioning and the training | DexWM |
| H4 | Predicted hand trajectories, which then condition generation | a stage of its own | Ego-PM |
| H5 | An auxiliary loss on the wrist position | the training only | EgoExo-WM |
| H6 | A map added to the patch features | the features | this project, Test 11 |
| H7 | Masks of the hands | the features or the pixels | EGTEA Gaze+'s hand masks |

**H2, body pose as the action.** PEVA conditions an autoregressive diffusion transformer on a
48-number action: the root translation and the joint rotations of 15 upper-body joints, trained on
Nymeria. It has no finger articulation. It is the nearest published analogue of what this project
gives the predictor, and it works on a far larger signal than 12 numbers.

**H3 and H5, hand as an auxiliary objective.** DexWM represents actions as finger keypoints from
egocentric video and reports that predicting visual features alone was not enough, so it adds a
hand-consistency loss. EgoExo-WM adds a wrist-position consistency objective. Both add a loss on
the signal rather than only an input of it.

**H4, predicting the hand first.** Ego-PM forecasts the future hand trajectory, then conditions a
latent diffusion model on that forecast. The conditioning signal is a prediction, not a
measurement, so it needs no hand tracking at test time.

## Egomotion, for comparison

Not a signal this project gives the model, and the one whose absence [[review]] section 3.1 names
as the reason the positive control is small. Between two frames 0.27 s apart most of the change in
an egocentric image comes from the head. TrajPilot conditions a causal predictor on a frozen
V-JEPA 2.1 encoder with the relative 6-DoF head trajectory, and reports that shuffling the
trajectory raised its error by 0.039 while shuffling a language input raised it by 0.0006.
Navigation World Models and CamFormer condition on camera motion, and PEVA includes the root
translation.

## What this leaves open for the project

Four channels survive the mechanism finding of Test 9 and the constraint that the encoder stays
frozen. They are not alternatives to each other; they act in different places and can be
measured in the same experiment.

1. **A map added to the patch features (G5, H6).** Built, as the arms of
   [[6-next-steps#5. Test 11 (planned). A predictor designed for gaze and hand|Test 11]]. Its
   published relative is G4. It needs no new data beyond the palm projected into the image.
2. **A term added to the attention scores (G6).** Not built. The published instance is Eyes on
   Target. It is the only channel that gives the attention a distance prior, which is the thing
   RoPE does not give and the reason the rope form of Test 9 failed. It is the natural second arm
   of Test 11, because Test 9 does not separate the feature channel from the attention channel and
   this project has now tested only one of them properly.
3. **An auxiliary loss on the signals (G8, G9, H3, H5).** Not built, and currently filed under
   "Later tests" in [[6-next-steps]] as an extra loss that predicts gaze or the future hand
   position from the predictor's states. Four independent published lines use the signal this way,
   and three of them report that the benefit survives without the signal at test time. That last
   property is worth more to this project than it first appears: EGTEA Gaze+ has gaze but no hand
   tracking, HD-EPIC has no training split, and SAGE, the best published result on EGTEA, predicts
   gaze rather than measuring it. A model that needs gaze only during training is comparable with
   all of them. This option should move out of "Later tests".
4. **Drawn on the frames (G10).** Not built. It is the only channel with a published number on the
   exact benchmark this project targets, 27.5% with no training at all. It needs no change to the
   predictor. What it does change is the encoder's input, so the frozen features of every earlier
   test are no longer the same features, and nothing from Tests 1 to 10 compares to it directly.

Two channels are closed. G1 and H1, a vector inside a token, were measured four times and are the
subject of the null. G2, a token moved to the gaze point, was measured in Test 9 and has a
mechanism for why it cannot work.

One channel is worth a note but not a test yet. G7, heatmaps as tokens of their own, adds tokens
and therefore meets the same competition that G1 and G2 lost; ARGaze uses it to estimate gaze,
not to consume it.

## References

- Eyes on Target: Gaze-Aware Object Detection in Egocentric Video. arXiv 2511.01237.
- Gaze-VLM: Bridging Gaze and VLMs via Attention Regularization for Egocentric Understanding.
  arXiv 2510.21356.
- Li, Liu, Rehg. In the Eye of the Beholder: Gaze and Actions in First Person Video. ECCV 2018 and
  TPAMI 2021. arXiv 2006.00626.
- ARGaze: Autoregressive Transformers for Online Egocentric Gaze Estimation. arXiv 2602.05132.
- G3Ego: Gaze-Guided Graphs for Egocentric Action Understanding. arXiv 2608.20157.
- SAGE: Synchronized Action-Gaze Recognition and Anticipation. arXiv 2607.04017.
- Bai et al. Whole-Body Conditioned Egocentric Video Prediction (PEVA). NeurIPS 2025.
  arXiv 2506.21552.
- World Models for Learning Dexterous Hand-Object Interactions from Human Videos (DexWM).
  arXiv 2512.13644.
- Ego-centric Predictive Model Conditioned on Hand Trajectories (Ego-PM). arXiv 2508.19852.
- EgoExo-WM: Unlocking Exo Video for Ego World Models. arXiv 2605.15477.
- EggHand: A Multimodal Foundation Model for Egocentric Hand Pose Forecasting. arXiv 2605.07642.
- Materia et al., and TrajPilot, Navigation World Models and CamFormer: see [[review]].
