---
type: report
status: running
created: 2026-09-18
updated: 2026-10-09
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

The question is measured in two ways. The first is the model's own prediction error (Tests 1–10).
The second gives the results of the paper: how well the model's predictions anticipate the next
object or action on two benchmarks with gaze, HD-EPIC's gaze interaction anticipation and EGTEA
Gaze+ action anticipation ([[2-background#Benchmarks]], Test 12 in [[6-next-steps]]).

## Approach

The project uses V-JEPA 2-AC, a video model from Meta ([[2-background]]). Its encoder turns
each frame into a grid of embeddings and stays frozen. Its predictor predicts the embeddings of
the next step and is trained. Embeddings describe the content of a frame, which fits the goal
of predicting intent.

V-JEPA 2-AC was built for robots. Its predictor receives the robot's action and state as two
extra tokens. Here they are replaced with a gaze token and a hand token. The data is HD-EPIC, a
set of kitchen recordings made with Aria glasses, which record gaze and hand positions. The
details are in [[3-method]].

A second predictor, built for gaze and hand and trained from the start on the same frozen
encoder, has had its first runs (Test 11). The same design on V-JEPA 2.1, a newer version with
better patch features, is planned (Test 13 in [[6-next-steps]]).

## Three Paths

The project considered three ways to use gaze, called Paths.

| Path | Idea | Status |
|---|---|---|
| Path 1 | Use gaze to choose which image patches V-JEPA 2 takes as context. | Tried on 1 May 2026. No benefit. Not continued. |
| Path 2 | Give the predictor gaze and hand tokens. | Tested ([[4-results]]), also with gaze as a position in the image (Test 9), with the loss of the pretraining and a full fine-tune (Test 10), and as maps over the image tokens of a predictor built for the signals (Test 11, first runs). Planned: that predictor on V-JEPA 2.1 (Test 13) and the benchmarks (Test 12) ([[6-next-steps]]). |
| Path 3 | Align the model's embeddings with language, so that it predicts concepts such as "making pasta sauce". | Postponed ([[6-next-steps]]). |

## Hypotheses

M is the main hypothesis. The others are possible reasons why M was not supported. The tests
that were run are in [[4-results]], the planned ones in [[6-next-steps]].

| ID | Short name | Hypothesis | Status | Tests |
|---|---|---|---|---|
| M | main | Gaze and hand inputs make the prediction better. | Not supported at a useful size. Gaze and hand, as three angles or as pe, lower the error over the whole frame by about 0.0003 (0.06%) and near the gaze point by 0.0006–0.0007; only the gain of pe passes the decision rule (Test 9). A signal from the target frame itself gains 0.0013. These models were trained for 3 epochs. A predictor built for the signals and trained for 13 epochs gains 0.0004 near the gaze point on MSE and nothing over the whole frame (Test 11, first runs, one seed). | [[4-results#^test2\|Test 2]], [[4-results#^test9\|Test 9]], [[4-results#^test11\|Test 11]]; Test 13 |
| R | already in the image | The frozen image features already contain gaze, so a gaze input adds nothing. | Not supported. A linear probe cannot read gaze from one frame (Test 3), and the features at the gaze point tell which object is picked up next, beyond the frame and the last 2 s (Test 8). | [[4-results#^test3\|Test 3]], [[4-results#^test8\|Test 8]] |
| H1 | horizon | The signals carry information that does not help 0.27 s ahead. | Open for the predictor. In Test 8, the information of the gaze point about the next object is largest 0.5 s ahead and gone at 4 s. In Test 9, even the gaze and hand of the target frame lower the error 0.27 s ahead by only 0.26%, so this horizon leaves little room for any signal. In the first runs of Test 11, the gain of the maps is the same at 0.53 s and 1.07 s; there is no positive control at these horizons. | [[4-results#^test8\|Test 8]], [[4-results#^test9\|Test 9]], [[4-results#^test11\|Test 11]] |
| H2 | not used | The model does not use the signals. | Partly supported. The model reacts to whether gaze is present. Its response to where the person looks is small, and fine-tuning made it smaller. | [[4-results#^test4\|Test 4]] |
| H3 | undertrained | The model is undertrained. A longer run could give a different result. | Partly answered. Training the whole predictor lowers the error by 0.0041 and the loss of the pretraining by 0.0110, both far more than gaze; neither makes gaze worth more (Test 10). A predictor trained from the start for 13 epochs gains no more from gaze, and without signals it is still below the blend of the past frames on MSE (Test 11, first runs). The longer run of the V-JEPA 2-AC predictor is still untested. | [[4-results#^test10\|Test 10]], [[4-results#^test11\|Test 11]] |
| H4 | gaze form | The gaze input has a form the model cannot relate to the image: three angles, with the token placed at the top-left patch. Given as a position in the image, gaze improves the prediction. | Supported for the information, not for the predictor. The features at the gaze point tell which object comes next; the three numbers do not (Test 8). In the predictor, gaze as a position in the token's content (pe) does as well as the three angles, not better, once two runs with a loss spike are left out; as a position in the attention (rope) it does not help (Test 9). As maps over the image tokens of a predictor built for the signals, gaze and hand gain no more than in Tests 9 and 10: the same near the gaze point on MSE, less on L1 (Test 11, first runs, one seed). | [[4-results#^test8\|Test 8]], [[4-results#^test9\|Test 9]], [[4-results#^test11\|Test 11]] |

^hypotheses

## Main findings so far

- A model trained without gaze and hand predicts as well as the model trained with them
  ([[4-results#^test2|Test 2]]). Test 9 repeats this with three seeds per model and 600 clips
  of two people.
- The model reacts to whether gaze is present. Its response to where the person looks is small,
  and fine-tuning made it smaller ([[4-results#^test4|Test 4]]).
- For a new person, a linear probe cannot read gaze from the frozen features of one frame
  ([[4-results#^test3|Test 3]]).
- The gaze token cannot point at the image ([[5-discussion#Weak points of the design]]). Gaze
  can now be projected into the image ([[3-method#Gaze position in the image]]). At the moment
  of a pick, the projected point lies in the object's box in 39% of picks, against 20% by
  chance.
- What the person looks at tells which object they pick up next. For new people, the features
  at the gaze point raise the share of correct guesses 0.5 s before the pick from 13–14% to
  18.5%, beyond the scene, the three gaze numbers and where the head points. The gain is
  smaller at 1 s and gone at 4 s ([[4-results#^test8|Test 8]]).
- Given to the predictor as a position in the image, gaze does no better than as three angles.
  Sine and cosine features of the gaze point (pe) lower the error near the gaze point by 0.0007
  against the model without signals, and the angles by 0.0006 once one run with a loss spike is
  left out. Placing the gaze token at the gaze point in the attention (rope) does not help. A
  signal from the target frame itself lowers the error by only 0.0013 (0.26%), so at 0.27 s the
  measure leaves little room ([[4-results#^test9|Test 9]]).
- All results come from models trained for 3 epochs (H3).
