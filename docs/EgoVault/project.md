---
type: report
status: running
created: 2026-09-18
updated: 2026-10-07
---

# Project

People, Erfan's work and its timeline, the paper, and the final presentation.

## People

| Person | Role | Contribution to this work |
|---|---|---|
| Erfan Yekehzare (University of Rostock) | Author. The project is part of Erfan's master's thesis. | The ego predictor, all code in this repository, and Tests 1–4 and 8. |
| Seyyed Parsa Sharifi (University of Rostock) | Co-author of the paper | Proposed the matched model without signals used in Test 2. Parsa's own work is summarised in [[parsa]]. |
| Ioana Marica (Babeș-Bolyai University, Cluj) | Collaborator | Wrote the first gaze and hand loaders and projection layers. Raised the concern that the frozen features may be weak, which the palm control of Test 3 tests. Suggested using the signals to supervise the representation. |
| Ashwin Nedungadi (Ash) | Supervisor | Asked whether gaze can already be read from the image (Test 3). The Path 3 design choices came from Ash's review. |

The work is part of EgoProject 2026, a joint project of the University of Rostock and
Babeș-Bolyai University.

## Erfan's work

| Test | Question | When |
|---|---|---|
| [[4-results#^test1\|Test 1]] | Can the predictor learn with gaze and hand inputs? | 31 May – 1 June 2026 |
| [[4-results#^test2\|Test 2]] | Do gaze and hand improve the prediction? | 31 May – 1 June and 19 August 2026 |
| [[4-results#^test3\|Test 3]] | Is gaze already in the image features? | 28–30 July and 19 August 2026 |
| [[4-results#^test4\|Test 4]] | Does the model use the signals? | 19 August 2026 |
| [[4-results#^test8\|Test 8]] | Does the gaze point tell what comes next? | 7 October 2026, run by Claude Code at Erfan's request |

Other work:

- The first experiment, on 1 May 2026 (Path 1): using gaze to choose which image patches
  V-JEPA 2 takes as context. No benefit; not continued.
- Encoding each frame on its own, which lowered the error before fine-tuning from 4.82 to
  0.58.
- The models `ego_ft_v2` and `ego_sd1p0`.
- The idea that the signals may help more at longer horizons (H1, May 2026), and the idea that
  gaze needs a position in the image (H4).
- Sections 2.1, 4.4 and 4.8 of the paper.
- With Claude Code, October 2026: a review of the research logic (the weak points in
  [[5-discussion#Weak points of the design]] and the new hypothesis H4), the camera
  calibrations of all recordings, the gaze position in the image, and its check against
  HD-EPIC's annotations of picks ([[3-method#Gaze position in the image]]).

## Timeline

| When | What | Result |
|---|---|---|
| 1 May 2026 | Used gaze to choose which image patches V-JEPA 2 takes as context (Path 1). | No benefit. Not continued. |
| May 2026 | Built the ego predictor. Trained `ego_ft_v2` for 3 epochs ([[4-results#^test1\|Test 1]]). | Real signals lowered the error slightly (Δ = +0.0011). Longer runs showed no gain and were stopped. |
| 16 July | Ash's review. Asked whether gaze can already be read from the image, and where text labels for Path 3 would come from. | This led to Test 3. HD-EPIC recipe steps were chosen as the text source for Path 3. |
| End of July | Tested whether gaze can be read from the frozen image features ([[4-results#^test3\|Test 3]]). | For new people, a linear probe cannot read gaze from one frame. |
| August | Studied GazeQwen. Proposed design changes, based on the idea that the model ignores gaze. | Not built then. Now planned as the gaze forms of Test 9. |
| 19 August | Palm control of Test 3. Tested whether the model uses gaze ([[4-results#^test4\|Test 4]]). Trained the matched model without signals ([[4-results#^test2\|Test 2]]). | The model reacts to whether gaze is present, and little to where the person looks. A model trained without gaze and hand predicts as well. |
| 24 August | The paper was submitted, with Parsa. | |
| October | Review of the research logic. Camera calibrations fetched. Gaze projected into the image and checked. Test 8 run ([[4-results#^test8\|Test 8]]). | What the person looks at tells which object is picked up next; the three gaze numbers do not. Next: Tests 9 and 10 ([[6-next-steps]]). |

## Paper

- Title: *Behavioral Conditioning of Video World-Model Predictors for Egocentric Action
  Anticipation*. Submitted on 24 August 2026 as the EgoProject 2026 report. Authors: Seyyed
  Parsa Sharifi and Erfan Yekehzare. Ioana Marica is a collaborator, not an author.
- The text as submitted is in [[BC-JEPA]]. It is kept as a reference and never edited.
- The paper is derived from the notes and the code. Where they differ, the paper is changed.
- Main result: conditioning on gaze and hand does not improve prediction. A model trained
  without the signals does as well (Δ = +0.0003, p = 0.40, [[4-results#^test2|Test 2]]).
- Erfan wrote section 2.1 (the JEPA background), section 4.4 (varying the training: Tests 2
  and 4) and section 4.8 (the horizon analysis: Test 3).

Open points for the next version of the paper:

1. Appendix A does not credit Erfan for section 4.4, for Tests 4a and 4c, or for the comparison
   with the model before fine-tuning (Test 2).
2. The paper does not say which inputs the attention test hid: real signals, gaze hidden alone,
   and gaze and hand hidden together (Test 4b).
3. Results in the notes that the paper lacks: the encoding fix (4.82 → 0.58), the longer
   training runs with Δ at or below zero, the wrong scaling constants, the worse prediction with
   real signals before training, the larger effect of hiding gaze alone, the right-palm results
   of Test 3, and the weak points of the design ([[5-discussion#Weak points of the design]]).

## Presentation

- The final presentation is on 25 November 2026 at 13:30. Each member speaks for at least 10
  minutes. Erfan presents first:
  - Section 1, background: the JEPA models, related work on conditioning, the project origin
    and the motivation.
  - Section 2, the signal: how far ahead it can be read, the feasibility check, and the result
    that training with the signals does not help.
- To fix before the presentation:
  1. A backup slide says that P02–P08 data was absent. Training used P01–P07, and testing used
     P08. The slide should say that the data was absent on the cluster used for the probes.
  2. The feasibility slide in Erfan's section shows a ViT-L run. Either show the training curve
     of `ego_ft_v2`, or label the slide as an early ViT-L run.


The plan, in order:

- Projection (preparation, about a day). Write the projection with
- projectaria_tools, using each recording's calibration and the gaze depth
- Then check it four ways:
  - drawings on frames from every participant and every pair of glasses;
  - where the gaze points fall in the frame;
  - with depth against without depth, which must differ by the expected parallax;
  - optionally, with HD-EPIC's pick annotations, how often the gaze point sits on the object just before it is picked up.
   
- Test 8 (no training, a few hours). Check whether the features at the gaze point predict the next seconds better than the scene plus the three angles. This is the gate: if they don't, Test 9 waits.

- Test 9 (about 19 runs, roughly 13 GPU hours). The ablation of angles, pe, rope and pe+rope, plus three controls:

  - none, the matched model without signals;
  - shuffled, to separate information from extra input;
  - future, the positive control.

Three seeds each, tested on about 600 clips from P08 and P09, measuring both the whole-frame error and the error near the gaze point.