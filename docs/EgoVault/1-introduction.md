---
type: report
status: running
created: 2026-09-18
updated: 2026-10-07
---

# 1. Introduction

## Motivation

When people work in a kitchen, their eyes and hands show what they will do next. The eyes
look at a knife before the hand reaches for it. Gaze usually leads the hand by about half a
second to a second, and in HD-EPIC objects are looked at on average 4 s before they are picked
up. Systems that must anticipate human actions could use these cues. Examples are robots that
work with people and assistants in smart glasses.

The videos in this project are egocentric: they are recorded from the person's own head.

## Question

Does a video prediction model predict the near future better when it also receives the
person's gaze and hand positions?

## Approach

The project uses V-JEPA 2-AC, a video model from Meta ([[2-background]]). Its encoder turns
each frame into a grid of embeddings and stays frozen. Its predictor predicts the embeddings of
the next step and is trained. Embeddings describe the content of a frame, which fits the goal
of predicting intent. (Decided by Erfan, July 2026.)

V-JEPA 2-AC was built for robots. Its predictor receives the robot's action and state as two
extra tokens. Erfan replaced these with a gaze token and a hand token. The data is HD-EPIC, a
set of kitchen recordings made with Aria glasses, which record gaze and hand positions. The
details are in [[3-method]].

## Three Paths

The project considered three ways to use gaze, called Paths. (Decided by Erfan, July 2026.)

| Path | Idea | Status |
|---|---|---|
| Path 1 | Use gaze to choose which image patches V-JEPA 2 takes as context. | Tried by Erfan on 1 May 2026. No benefit. Not continued. |
| Path 2 | Give the predictor gaze and hand tokens. | Tested ([[4-results]]); the next tests give gaze a position in the image ([[6-next-steps]]). |
| Path 3 | Align the model's embeddings with language, so that it predicts concepts such as "making pasta sauce". | Postponed ([[6-next-steps]]). |

## Hypotheses

M is the main hypothesis. The others are possible reasons why M was not supported. The tests
that were run are in [[4-results]], the planned ones in [[6-next-steps]].

| ID | Short name | Hypothesis | Status | Tests |
|---|---|---|---|---|
| M | main | Gaze and hand inputs make the prediction better. | Not supported in the tested form: gaze as three angles, 3 epochs, one training run per model. | [[4-results#^test2\|Test 2]]; Test 9 planned |
| R | already in the image | The frozen image features already contain gaze, so a gaze input adds nothing. | Not supported for a linear probe that reads gaze from one frame. Whether gaze adds information about the future is not tested yet. | [[4-results#^test3\|Test 3]]; Test 8 planned |
| H1 | horizon | The signals carry information that does not help 0.27 s ahead. (Proposed by Erfan, May 2026.) | Open. No test has varied the horizon. | Later test |
| H2 | not used | The model does not use the signals. | Partly supported. The model reacts to whether gaze is present. Its response to where the person looks is small, and fine-tuning made it smaller. | [[4-results#^test4\|Test 4]] |
| H3 | undertrained | The model is undertrained. A longer run could give a different result. | Open. | Test 10 planned |
| H4 | gaze form | The gaze input has a form the model cannot relate to the image: three angles, with the token placed at the top-left patch. Given as a position in the image, gaze improves the prediction. (Proposed by Erfan.) | Open. | Tests 8 and 9 planned |

^hypotheses

## Main findings so far

- A model trained without gaze and hand predicts as well as the model trained with them
  ([[4-results#^test2|Test 2]]). Each model was trained once.
- The model reacts to whether gaze is present. Its response to where the person looks is small,
  and fine-tuning made it smaller ([[4-results#^test4|Test 4]]).
- For a new person, a linear probe cannot read gaze from the frozen features of one frame
  ([[4-results#^test3|Test 3]]).
- The gaze token cannot point at the image ([[5-discussion#Weak points of the design]]). Gaze
  can now be projected into the image ([[3-method#Gaze position in the image]]), and Tests 8
  and 9 test gaze as a position (H4).
- All results come from models trained for 3 epochs (H3).
