---
type: report
status: done
created: 2026-10-07
updated: 2026-10-07
---

# Parsa's work

The paper's tests that were made with Parsa's code ([[BC-JEPA]]), kept apart from the rest of
the notes: Tests 1 (ViT-L part), 5, 6 and 7. For each piece: what was done, why, what it showed,
and whether it is valid and useful for the project's question. The numbers come from the
records of these runs and the paper. The code is not in this repository, so none of it can be
rerun from here.

## Summary

| Work | Result | Valid? | Useful for the project? |
|---|---|---|---|
| Matched model without signals (design) | Became the main comparison of Test 2 | Yes | Yes, central |
| ViT-L feasibility run | The predictor learns with gaze and hand inputs | Yes, as a feasibility check | Little; a different model from `ego_ft_v2` |
| Early EK100 pilot | ViT-L 4.27% against ViT-G 2.81% | No; confounded | No |
| EK100 anticipation probe | No difference between inputs (within 0.1 points) | Narrowly | Little; EK100 has no gaze |
| Diagnostics on HD-EPIC P01 | The signal reaches the output and changes no choice | Partly | Partly; overlaps Test 4 |
| Probe on gaze and hand alone | Below a frequency guess at every horizon | Limited | Little; it never sees what is looked at |
| Stronger pathway (design) | Designed, not built, dropped | The decision is sound | As a design record |

## Matched model without signals

- **What.** The proposal to train a second model with the signals hidden in every clip
  (June 2026). It was trained as `ego_sd1p0` and compared.
- **Why.** Comparing one model with and without its signals mixes the value of the information
  with the cost of removing an input the model expects.
- **Result.** It gives the main result of the project: +0.0003, p = 0.40
  ([[4-results#^test2|Test 2]]).
- **Assessment.** Valid and the most useful of this work for the project.

## ViT-L feasibility run

- **What.** A V-JEPA 2 predictor with the smaller ViT-L encoder, in Parsa's own code, trained on
  HD-EPIC P01 (27 recordings, 20 epochs), with a gaze input of 2 values. Run before 16 June
  2026.
- **Why.** To check that the predictor can learn with gaze and hand inputs.
- **Result.** The training loss (smooth L1) fell from 1.207 to 1.019, with little change after
  epoch 5.
- **Assessment.** Valid as a feasibility check. It differs from `ego_ft_v2` in encoder, data,
  loss and gaze input, so its numbers cannot be compared with those of `ego_ft_v2`, and it was not used
  further. The paper's feasibility figure shows this run, although the results come from
  `ego_ft_v2`.

## Early EK100 pilot

- **What.** A ViT-L vision baseline against the ViT-G predictor trained with gaze and hand, on
  EK100, one probe epoch each.
- **Why.** A first comparison under the deadline.
- **Result.** 4.27% against 2.81% action recall@5.
- **Assessment.** Not valid. It compares two encoders and two training runs, and gaze is hidden
  on EK100. The paper itself calls it a confounded pilot. A later preview of 5.83%, from one
  unseeded run without token-count controls, was withdrawn.

## EK100 anticipation probe

- **What.** A small classifier (one attention layer and linear verb, noun and action outputs)
  trained on frozen tokens to name the action 1 s ahead, on 870 EK100 clips of P01. Four
  inputs, with equal token counts as controls:

| Input | Action recall@5 |
|---|---|
| Encoder output only | 3.62% |
| + 196 zero tokens | 3.60% |
| + the last frame repeated | 3.54% |
| + the predictor's output (`ego_ft_v2`) | 3.61% |

- **Why.** To see whether training with gaze and hand left something in the predictor that
  helps a standard benchmark.
- **Result.** The four inputs are within 0.1 points, less than the differences between seeds.
- **Assessment.** Valid only in a narrow sense: training with the signals left nothing useful
  for EK100 anticipation. It cannot test whether gaze helps, because EK100 has no gaze and the
  predictor always received its "no signal" token. Further limits: only P01 was evaluated
  (the other videos were missing and skipped without a warning), P01 is in the predictor's
  training data, the pipeline uses 224-pixel frames while the predictor was trained on 256,
  and the comparison with `ego_sd1p0` was planned but never run. The equal-token controls are
  a good design.

## Diagnostics on HD-EPIC P01

- **What.** Four checks on `ego_ft_v2`, real against hidden signals, on P01:

| Check | Result |
|---|---|
| Change of the output | 9.9% |
| Average error | 0.055% lower with real signals; better on 172 of 220 clips |
| Choice among candidate futures | the true one is first in 210 of 220 clips, with and without signals |
| Attention on the gaze position | 0.0491 real, 0.0492 hidden (28.6 times the equal share) |

- **Why.** To decide whether a stronger gaze pathway was worth building: does the signal reach
  the output, and does it change which future the model picks?
- **Result.** The signal reaches the output and changes no choice. The paper dropped the
  stronger pathway on this basis.
- **Assessment.** Partly valid. The output change and the attention agree with
  [[4-results#^test4|Test 4]]. The choice test is at ceiling: with 95.5% correct either way, it
  cannot show an improvement. P01 is in the training data. The candidates of the choice test,
  and which signals were hidden, are not described. The paper reports "top-1 error 0.9545",
  which is the share correct. The equal share of 0.0017 matches 224-pixel frames, not the
  256-pixel frames the predictor uses.

## Probe on gaze and hand alone

- **What.** Two classifiers on HD-EPIC P01: one sees only gaze and hand, one sees image
  features. They predict the next action, later only the verb, 1–10 s ahead, with about 1,900
  verb examples.
- **Why.** To measure how much the signal itself tells about the coming action.
- **Result.** First version, recall@5: verbs 0.594 (gaze and hand), 0.480 (image), 0.614
  (frequency guess); nouns 0.231, 0.318 and 0.300. Final version, verbs: gaze and hand beat the
  image by 0.035 to 0.121 at 1–10 s, and stay 0.051 to 0.092 below the frequency guess.
- **Assessment.** Limited. Gaze and hand alone do not anticipate the action better than a
  frequency guess. The classifier sees gaze and hand as numbers, never what lies at the gaze
  point, so it cannot test the idea that the looked-at object predicts the next action
  (H4). One participant and a small sample; the inputs, the classifier, the test split and the
  test-set size (762 or 243) are not described. Two earlier versions were discarded for flaws.

## Stronger pathway (design)

- **What.** A design adapted from GazeQwen: a gaze bias on the image tokens' attention keys, a
  resampler that combines gaze and hand, and a gate per block.
- **Why.** In case the token pathway was too weak to carry the signal.
- **Result.** Designed, not built. The paper dropped it after the diagnostics above.
- **Assessment.** Not building it was sound, for a further reason: a bias added equally to
  every key cancels in the softmax, so it cannot select image regions ([[2-background]],
  GazeQwen). The diagnostics did not test gaze as a position in the image, which
  Test 8 did and Test 9 will.

## Other work

The verb and noun analysis and the sample-size analysis of the paper.
