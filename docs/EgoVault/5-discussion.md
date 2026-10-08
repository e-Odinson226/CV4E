---
type: report
status: running
created: 2026-10-05
updated: 2026-10-08
---

# 5. Discussion

What the tests in [[4-results]] mean together, and the weak points of the design that
limit them. The planned tests that address these weak points are in [[6-next-steps]].

## What the results show

![[figures/overview_effects.png]]

- **Fine-tuning helps mostly by adapting to kitchen video.** It lowered the error by 0.125. The
  signals account for 0.0011 of it, about 0.9% ([[4-results#^test2|Test 2]]).
- **Training with the signals gives no measurable benefit, in the tested form.** A model
  trained without them predicts as well: +0.0003, p = 0.40 (Test 2).
- **The model reacts to whether gaze is present, and little to where the person looks.**
  Another clip's gaze moves the prediction 2.9% as much as a different video, and fine-tuning
  reduced this response ([[4-results#^test4|Test 4]]).
- **For a new person, a linear probe cannot read gaze from one frame.** Palm position can be
  read, so this is specific to gaze ([[4-results#^test3|Test 3]]). The image features describe
  what is where. They do not say which of those things the person looks at.
- **What the person looks at tells what comes next; the three gaze numbers do not.** For new
  people, the features at the gaze point raise the share of correct guesses of the next object
  0.5 s before the pick from 13–14% to 18.5%, beyond the scene and where the head points. The
  angles, added as numbers, do not help. The gain is short-lived: smaller at 1 s, gone at 4 s
  ([[4-results#^test8|Test 8]]).
- **Given to the predictor as a position, gaze does no better than as three angles.** As a
  position in the token's content (pe), it lowers the error near the gaze point by 0.0007
  against the model without signals; the angles do as well once one run with a loss spike is
  left out. As a position in the attention (rope), it does not help. A signal from the target
  frame itself lowers the error by only 0.0013, 0.26% ([[4-results#^test9|Test 9]]).

## Interpretation

- The model makes little use of where the person looks. A likely reason is the form of the
  input: the gaze token cannot point at the image (first weak point below). H4 states this.
  Test 8 supports it: the information is in what lies at the gaze point, and a linear readout
  of the three numbers finds none of it. In Test 9, however, the predictor gained no more from
  gaze as a position in the token's content than from the three angles, and nothing from gaze
  as a position in the attention, which the reasoning before the runs expected to do best. So
  the information that Test 8 found at the gaze point does not reach the prediction 0.27 s
  ahead in any of the forms tried.
- The prediction looks 0.27 s ahead. In HD-EPIC, objects are first looked at on average 4.0 s
  before they are picked up. But in Test 8 the gaze point told the next object best 0.5 s
  ahead, and not at all 4 s ahead. So the useful information of gaze lies within about 1–2 s.
  For the predictor, horizons longer than 0.27 s have not been tested (H1). At 0.27 s, even
  the gaze and hand of the target frame lower the error by only 0.26% (Test 9).
- All conclusions hold for models trained for 3 epochs, once each (H3).

## Weak points of the design

**Gaze cannot point.** Gaze enters as three numbers (yaw, pitch, depth) through one linear
layer, and is never turned into a position in the image. In the predictor, image patches carry
their position only through the rotations of RoPE, and the gaze token is placed at the top-left
patch in every frame ([[3-method#Model]]). To link gaze to the patches the person looks at, the
model would have to learn the mapping from angles to patch positions by itself, through 18
frozen blocks, with new layers that moved only about 0.01–0.02 per weight during training. It
cannot point. In the robot model these tokens held the arm's action and state, global
quantities that need no position in the image. Test 9 gave the gaze token a position in two
ways. The position in the content did as well as the angles, and the position in the attention
did not help.

**The signals describe the present.** In V-JEPA 2-AC, the action token at step t is the change
of the robot's pose from frame t to frame t+1. It carries information about the next frame that
the past frames cannot contain. Here both tokens are measured at frame t, and the hands are
mostly visible in the frame. A small effect is expected from this alone.

**Short horizon, coarse measure.** The error is measured 0.27 s ahead, one step of the
pretrained model ([[3-method#Input format]]), while gaze leads the hand by 0.5–1 s and precedes
a pick-up by about 4 s. In 0.27 s the scene changes little, and the past frames already show
most of that change. A model could use gaze well and still gain almost nothing at this horizon
(H1). The error is averaged over all 256
patches, and most of it comes from head motion and the whole scene; the hands and the target
object are a small part. The Δ of +0.0011 is 0.2% of the error. Test 9 put a number on this
limit: the gaze and hand of the target frame itself lower the error by 0.0013, 0.26%.

**No positive control.** In Tests 1–4, no model receives a signal that is known to carry
information about the target. So a small Δ cannot be told apart from a measure or a training
recipe that cannot show any effect. Test 9 added one: a model given the gaze and hand of the
frame it predicts. Its gain of 0.0013 was detected, so the measure can show an effect of this
size.

**Too little evidence.** Each model was trained once, and the test set has 96 clips from 4
recordings of one person. The two models differ by −0.0008 with the signals hidden, which
comes from training alone and is larger than the +0.0003 value of the information. With a
bootstrap over recordings, the 95% interval of that value is [−0.0005, +0.0011]. P09 has 13
recordings with gaze and is not used. Test 9 used three seeds per model and 600 clips from 25
recordings of P08 and P09. There, 2 of 20 runs had a loss spike at the start of training, and
even without them the seeds of one form differ by up to 0.0014.

The limits of single tests are listed with each test in [[4-results]]. Other known limits: the
scaling constants for gaze and hand are rough guesses, signal dropout always hides gaze and
hand together, and P01–P03 have more missing hand data than the others ([[3-method#Data]]).
