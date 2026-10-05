---
type: report
status: running
created: 2026-09-18
updated: 2026-10-05
---

# 1. Introduction

## Motivation

When people work in a kitchen, their eyes and hands show what they will do next. The eyes
look at a knife before the hand reaches for it. Gaze usually leads the hand by about half a
second to a second. Systems that must anticipate human actions could use these cues. Examples
are robots that work with people and assistants in smart glasses.

The videos in this project are egocentric: they are recorded from the person's own head.

## Question

Does a video prediction model predict the near future better when it also receives the
person's gaze and hand positions?

## Approach

The project uses V-JEPA 2-AC, a video model from Meta ([[2-background]]). Its encoder turns
each frame into an embedding and stays frozen. Its predictor predicts the embedding of the
next step and is trained. Embeddings describe the content of a frame, which fits the goal of
predicting intent. (Decided by Erfan, July 2026.)

V-JEPA 2-AC was built for robots. Its predictor receives the robot's action and state as two
extra tokens. We replaced these with a gaze token and a hand token. The data is HD-EPIC, a
set of kitchen recordings made with Aria glasses. The glasses record gaze and hand positions
in every frame. The details are in [[3-method]].

## Three Paths

The project considered two ways to use gaze. They are called Paths. (Decided by Erfan, July
2026.)

| Path | Idea | Status |
|---|---|---|
| Path 1 | Give the predictor gaze and hand tokens. | Tested. See [[4-results]]. |
| Path 2 | Align the model's embeddings with language, so that it predicts concepts such as "making pasta sauce". | Planned, not started. See [[6-next-steps]]. |

## Hypotheses

M is the main hypothesis. R, H1, H2 and H3 are possible reasons why M was not supported. The
tests are in [[4-results]].

| ID | Hypothesis | Status | Tests |
|---|---|---|---|
| M | Gaze and hand inputs make the prediction better. | Not supported for the 3-epoch model. | T2, T3, T11 |
| R | The frozen image features already contain gaze, so a gaze input adds nothing. | Rejected. | T4, T5 |
| H1 | The signals contain information that does not help at the horizons tested: 0.27 s ahead for the prediction error, 1 s ahead for EK100. (Proposed by Erfan, May 2026.) | Supported, if H3 is false. | T9, T10, T11, T12 |
| H2 | The model does not use the signals. | Rejected. The model uses them. It reacts mainly to whether they are present. | T6, T7, T8, T9 |
| H3 | The model is undertrained. A longer run could give a different result. | Open. | No test yet |

^hypotheses

## Main findings so far

- For a new person, gaze cannot be read from the frozen image features (T4, T5). So the gaze
  input carries new information.
- The model reads the gaze token. It reacts mostly to whether gaze is present. The value of
  the gaze matters little (T6–T9).
- A model trained without gaze and hand predicts as well as the model trained with them (T11).
- All results come from models trained for 3 epochs. A longer training run could change them
  (H3).
