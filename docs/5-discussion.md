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

### 1. The Spatial Mismatch (Gaze is localized, but the token is global)
As noted, gaze is fundamentally a spatial pointer. However, the architecture treats it like a global robot state:

- **The RoPE Problem:** By rotating the token only by the time step and leaving its spatial coordinates at (0, 0) (the top-left patch), the frozen Transformer—which has never seen a spatial token behave this way—is forced to somehow learn complex 3D-to-2D trigonometric projections just to figure out where the person is looking.
- **The GazeQwen Parallel:** This perfectly aligns with observations on GazeQwen ([[gazeqwen]]). When gaze isn't given a rigorous spatial attention mechanism, it defaults to acting as a "scalar gate" (just telling the model "gaze is present") rather than a spatial selector ("look at this patch"). This is why Test 4 showed the model reacting to the presence of gaze, but not where the gaze was pointing.

**Proposed Solutions:**
- **Spatial Cross-Attention:** Instead of concatenating the gaze token to the visual tokens, project the gaze 3D coordinates into a 2D heat map or Gaussian blob over the 16x16 patch grid, and add it directly to the positional embeddings of the patches. This avoids forcing the transformer to learn trig functions.
- **Gaze-guided Cropping:** Hard-crop the image patches around the projected gaze point (a foveated approach) and feed only those patches (plus a low-res global context) to the model.

### 2. The Temporal Mismatch (Present state vs. Future delta)
- In the original V-JEPA 2-AC pre-training, the action token at step $t$ represented the *change* in the robot's pose from $t \rightarrow t+1$. It was a forward-looking delta that directly leaked the future.
- In this setup, gaze and hand are sampled at time $t$. They describe the present. Because the horizon is only 0.27s ahead (8 frames), the present hand/gaze position provides almost zero new information that isn't already painfully obvious from the visual trajectory of the 8 context frames. This is why even providing the "cheat" target-frame signals in Test 9 only yielded a minuscule 0.26% error reduction.

**Proposed Solutions:**
- **Extend the Prediction Horizon:** Predict 1 to 2 seconds into the future, the time scale where gaze actually leads hand interaction. At 0.27s, visual momentum dominates; at 2.0s, intent (signaled by gaze) becomes necessary.
- **Predict Semantic Intent (VL-JEPA style):** Stop predicting raw pixel/patch embeddings. Instead, follow Path 3 and align the latent space with language, predicting a semantic "intent" vector.
- **Use Gaze Deltas:** Instead of absolute present gaze/hand positions, feed the *delta* (change) of the gaze/hand over the past $N$ frames to explicitly encode momentum, matching the derivative nature of the original robot action tokens.

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
