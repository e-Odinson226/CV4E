---
type: reference
status: running
created: 2026-07-26
updated: 2026-10-07
---

# Project notes

Notes for a project on egocentric video understanding with JEPA models. The
question is whether a video prediction model predicts the near future better when it also
receives the person's gaze and hand positions.

**Status.** In the tested form, gaze and hand inputs did not improve the prediction: a model
trained without them does as well ([[4-results#^test2|Test 2]]). In that form the gaze token
cannot point at the image. Gaze can now be projected into the image, where it lies on the
object about to be picked up twice as often as by chance
([[3-method#Gaze position in the image]]). What lies at the gaze point tells which object is
picked up next, while the three gaze numbers do not ([[4-results#^test8|Test 8]]). The next
test gives the predictor gaze as a position (Test 9), then the full training follows
(Test 10). The plan is in [[6-next-steps]].

## Chapters

Read in order. Files 1 to 6 follow the chapters of a report.

| File | Contents |
|---|---|
| [[1-introduction]] | The question, the approach, the three Paths, and the hypotheses with their status. |
| [[2-background]] | The models and papers this work builds on, and the terms used. |
| [[3-method]] | Data, model, gaze position in the image, training, evaluation, design choices. |
| [[4-results]] | Tests 1–4 and 8 and what they showed. |
| [[5-discussion]] | What the results mean together, and the weak points of the design. |
| [[6-next-steps]] | The planned Tests 9 and 10 and the other work, in order. |

## Names

| Name | Meaning |
|---|---|
| Path 1–3 | the three ways to use gaze ([[1-introduction]]) |
| M, R, H1–H4 | the hypotheses, each with a short name: M main, R already in the image, H1 horizon, H2 not used, H3 undertrained, H4 gaze form |
| Test 1–4, 8 | the tests that were run; parts are written 4a, 4b |
| Test 9, 10 | the planned tests. Numbers 5–7 are tests of the paper made with other code ([[parsa]]). |
| `ego_ft_v2`, `ego_sd1p0` | the trained models: with gaze and hand, and without |
| angles, pe, rope, pe+rope | the gaze forms compared in Test 9 |

## Other files

| File or folder | Contents |
|---|---|
| [[project]] | The work done and its timeline, the paper, and the final presentation. |
| [[parsa]] | Parsa's work for the paper, kept apart: what, why, result, and whether it is valid and useful. |
| `papers/` | [[BC-JEPA]], the paper as submitted (never edited), and the literature notes in `papers/literature/`. |
| `templates/` | A template for adding a test to [[4-results]]. |

The code, the commands, the data paths and the checkpoints are described in `README.md` at the
root of the repository.

The literature, in `papers/literature/`, one note per paper:

- [[vjepa]]: V-JEPA 2, the base model.
- [[gazeqwen]]: GazeQwen, which adds gaze to a video-language model.
- [[vl-jepa]]: VL-JEPA, a JEPA model aligned with language. The basis for Path 3.
- [[vla-jepa]]: VLA-JEPA, JEPA-style pretraining for robot policies. Full text, no notes yet.
