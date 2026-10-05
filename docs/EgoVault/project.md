---
type: report
status: running
created: 2026-09-18
updated: 2026-10-05
---

# Project

People, timeline and the final presentation. The submitted paper is described in
[[paper-status]].

## People

| Person | Role | Work |
|---|---|---|
| Erfan Yekehzare (University of Rostock) | Author. The project is part of Erfan's master's thesis. | Built the ego predictor (ViT-G) and its training and evaluation code. Trained `ego_ft_v2`, the model used in T2, T3 and T6–T11. Ran T2, T4–T8, T10 and T11. |
| Seyyed Parsa Sharifi (University of Rostock) | Author | Ran the feasibility check, the EK100 probe and the HD-EPIC P01 probes: T1, T3, T9 and T12. Led the writing of the paper. |
| Ioana Marica (Babeș-Bolyai University, Cluj) | Collaborator | Wrote the first gaze and hand projection layers. Raised the concern that the frozen encoder's features may be weak. T5 tests one form of this concern. Suggested the PEVA readout (T9) and the idea of using the signals to supervise the representation. |
| Ashwin Nedungadi (Ash) | Supervisor of the Rostock team | Asked whether gaze can already be read from the image (T4). The Path 3 design choices came from Ash's review. |
| Klára Orbán | Supervisor of the Cluj team | |
| Stefan Lüdtke | Oversees the project | |

The work is part of EgoProject 2026, a joint project of the University of Rostock and
Babeș-Bolyai University. Each test in [[4-results]] names who ran it.

## Timeline

| When | Who | What | Result |
|---|---|---|---|
| May 2026 | Erfan | Built the ego predictor. Trained `ego_ft_v2` for 3 epochs ([[4-results#^t2\|T2]]). | Real signals lowered the error slightly (Δ = +0.0011). Longer runs showed no gain and were stopped. |
| By mid-June | Parsa | Trained a small ViT-L version as a feasibility check ([[4-results#^t1\|T1]]). | The model can learn with these inputs. |
| June–July | Parsa | Tested `ego_ft_v2` on EK100 action anticipation ([[4-results#^t3\|T3]]). | No effect. |
| 16 July | Ash | Reviewed the work. Asked whether gaze can already be read from the image, and where text labels for Path 3 would come from. | This led to T4. HD-EPIC recipe steps were chosen as the text source for Path 3. |
| End of July | Erfan | Tested whether gaze can be read from the frozen image features ([[4-results#^t4\|T4]]). | For new people it cannot. So the gaze input carries new information. |
| August | Erfan | Studied GazeQwen, a related paper. Proposed design changes, based on the idea that the model ignores gaze. | The changes were not built. The tests on 19 August showed that the model uses gaze. |
| 19 August | Erfan | Tested whether the model uses gaze ([[4-results#^t5\|T5]]–[[4-results#^t8\|T8]]). Trained a matched model without signals ([[4-results#^t10\|T10]], [[4-results#^t11\|T11]]). | The model uses gaze. A model trained without gaze and hand predicts as well. |
| July–August | Parsa | Ran diagnostics and probes on HD-EPIC P01 for the paper ([[4-results#^t9\|T9]], [[4-results#^t12\|T12]]). | The signal reaches the output. It carries little information about the next action. |
| 24 August | Parsa and Erfan | Submitted the paper. See [[paper-status]]. | |

## Presentation

- The final presentation is on 25 November 2026 at 13:30. Each member speaks for at least 10
  minutes.
- Parsa's slides (Google Slides, `EgoProject_merged`) list Erfan first and Ioana as a
  collaborator.
- Erfan presents first:
  - Section 1, background: the JEPA models, related work on conditioning, the project origin
    and the motivation.
  - Section 2, the signal: how far ahead it can be read, the feasibility check, and the result
    that training with the signals does not help.
- Parsa presents sections 3–5: the method, the experiments, dropping the stronger pathway, the
  discussion, the limitations and the conclusion.

To fix before the presentation:

1. The code link on the title slide and on the last slide returns 404.
2. A backup slide says that P02–P08 data was absent. Training used P01–P07, and testing used
   P08. The slide should say that the data was absent on the cluster used for the probes.
3. The EK100 scope on the slides differs from the paper (see
   [[paper-status#Points to discuss with Parsa]]).
4. The feasibility slide in Erfan's section shows Parsa's ViT-L run. Either show the training
   curve of `ego_ft_v2`, or label the slide as an early ViT-L run.
