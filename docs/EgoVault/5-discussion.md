---
type: report
status: running
created: 2026-10-05
updated: 2026-10-05
---

# 5. Discussion

What the results in [[4-results]] mean together, and what limits them.

## What the results show

- **Gaze carries information the image does not have.** For a new person, gaze cannot be read
  from the frozen image features (T4). Palm position can be read, so this is specific to gaze
  (T5).
- **The model uses the signals.** Changing gaze changes the prediction (T6). The gaze token gets
  a large share of the attention (T7). Training made the gaze layer larger (T8).
- **The model reacts mainly to whether the signals are present.** Hiding gaze moves the
  prediction 5.2 times more than swapping in another clip's gaze (T6). Much of what the gaze
  layer learned is a constant bias (T8). Attention on the signal positions barely changes
  between real and hidden signals (T9).
- **Fine-tuning helps mostly by adapting to kitchen video.** The signals account for about 0.9%
  of the gain (T10).
- **Training with the signals gives no measurable benefit.** A model trained without them
  predicts as well (T11). The Δ of +0.0011 in T2 is the cost of removing an input the model
  expects.
- **On their own, the signals carry weak information about actions.** They predict verbs better
  than image features do and nouns worse. They stay below a guess based on class frequency
  (T12).

## Interpretation

- At the horizons tested, the signals add little that helps the prediction (H1). The
  information about behavior in the frozen features concerns the present: the skill for gaze
  and palm position drops to chance within 1 to 2 seconds (T4, T5).
- Feeding the signals in as input tokens may be a weak use of them. The paper suggests using
  them as a training signal toward goal-level targets ([[6-next-steps]]). The idea came from
  Ioana.
- These conclusions hold for models trained for 3 epochs. A longer training run could change
  them (H3).

## Limits

- All models were trained for 3 epochs, with one seed each.
- The prediction-error tests predict only one step ahead (0.27 s). No test has varied this
  horizon.
- The scaling constants for gaze and hand are wrong. After scaling, yaw and pitch have about
  half the intended spread, and the pitch mean is −0.52 instead of 0. All trained models used
  these constants.
- Signal dropout hides gaze and hand together. The model never saw one signal hidden and the
  other present.
- The EK100 probe and the HD-EPIC P01 probes (T3, T9, T12) use one participant, P01. P01 is
  also in the predictor's training data.
- The EK100 pipeline uses 224-pixel frames. The predictor was trained on 256-pixel frames.
- The probes in T4 and T5 are linear and see one frame. There are no error bars yet.
