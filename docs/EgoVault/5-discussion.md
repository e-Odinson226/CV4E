---
type: report
status: running
created: 2026-10-05
updated: 2026-10-07
---

# 5. Discussion

What Erfan's Tests 1–4 in [[4-results]] mean together, and the weak points of the design that
limit them. The planned tests that address these weak points are in [[6-next-steps]].

## What the results show

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

## Interpretation

- The model makes little use of where the person looks. A likely reason is the form of the
  input: the gaze token cannot point at the image (first weak point below). H4 states this, and
  Tests 8 and 9 test it.
- The prediction looks 0.27 s ahead. Gaze looks much further ahead: in HD-EPIC, objects are
  looked at on average 4.0 s before they are picked up. Whether the signals help at longer
  horizons (H1) has not been tested.
- All conclusions hold for models trained for 3 epochs, once each (H3).

## Weak points of the design

**Gaze cannot point.** Gaze enters as three numbers (yaw, pitch, depth) through one linear
layer, and is never turned into a position in the image. In the predictor, image patches carry
their position only through the rotations of RoPE, and the gaze token is placed at the top-left
patch in every frame ([[3-method#Model]]). To link gaze to the patches the person looks at, the
model would have to learn the mapping from angles to patch positions by itself, through 18
frozen blocks, with new layers that moved only about 0.01–0.02 per weight during training. It
cannot point. In the robot model these tokens held the arm's action and state, global
quantities that need no position in the image.

**The signals describe the present.** In V-JEPA 2-AC, the action token at step t is the change
of the robot's pose from frame t to frame t+1. It carries information about the next frame that
the past frames cannot contain. Here both tokens are measured at frame t, and the hands are
mostly visible in the frame. A small effect is expected from this alone.

**Short horizon, coarse measure.** The error is measured 0.27 s ahead, while gaze leads the
hand by 0.5–1 s and precedes a pick-up by about 4 s. The error is averaged over all 256
patches, and most of it comes from head motion and the whole scene; the hands and the target
object are a small part. The Δ of +0.0011 is 0.2% of the error.

**No positive control.** No model receives a signal that is known to carry information about
the target. So a small Δ cannot be told apart from a measure or a training recipe that cannot
show any effect.

**Too little evidence.** Each model was trained once, and the test set has 96 clips from 4
recordings of one person. The two models differ by −0.0008 with the signals hidden, which
comes from training alone and is larger than the +0.0003 value of the information. With a
bootstrap over recordings, the 95% interval of that value is [−0.0005, +0.0011]. P09 has 13
recordings with gaze and is not used.

The limits of single tests are listed with each test in [[4-results]]. Other known limits: the
scaling constants for gaze and hand are rough guesses, signal dropout always hides gaze and
hand together, and P01–P03 have more missing hand data than the others ([[3-method#Data]]).
