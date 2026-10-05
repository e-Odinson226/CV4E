---
type: report
status: running
created: 2026-09-18
updated: 2026-10-05
---

# Project

People, timeline, the paper and the final presentation.

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
| 24 August | Parsa and Erfan | Submitted the paper. See the Paper section below. | |

## Paper

The paper is derived from the notes and the code in this repository. The notes and the code
are the reference. Where the paper differs from them, the paper is changed.

- Title: *Behavioral Conditioning of Video World-Model Predictors for Egocentric Action
  Anticipation*.
- Submitted on 24 August 2026 as the EgoProject 2026 report, in NeurIPS format.
- Authors: Seyyed Parsa Sharifi and Erfan Yekehzare. Ioana Marica is a collaborator, not an
  author.
- The LaTeX source is in `papers/overleaf/`. The full text, with notes, is in
  [[final-report]].
- Main result: conditioning on gaze and hand does not improve prediction. A model trained
  without the signals does as well (Δ = +0.0003, p = 0.40,
  [[4-results#^t11|T11]]).

### Who wrote what

The paper's contribution statement (Appendix A) says:

- **Parsa:** the EK100 probe and its controls, the HD-EPIC P01 diagnostics, the verb and noun
  analysis, the sample-size analysis, and the stronger pathway that was designed and then
  dropped. Parsa led the writing of the method, experiments, discussion, limitations and
  conclusion.
- **Erfan:** the JEPA background (section 2.1) and the horizon analysis (section 4.8), with the
  gaze and palm probes. The ego predictor and its training and evaluation code. The trained
  model behind every anticipation result. The matched model trained without signals, and the
  attention test on the untrained model.
- **Ioana:** a collaborator, not an author of the report. Ideas, the first projection layers,
  and the PEVA readout.

### Open problems

These are the places where the paper does not yet follow the notes and the code.

1. Parts of the paper come from code that is not in this repository: the results of
   [[4-results#^t1|T1]], [[4-results#^t3|T3]], [[4-results#^t9|T9]] and
   [[4-results#^t12|T12]], and the code in Appendix C. All of it is Parsa's. These parts
   cannot be derived from this repository until the code is here. See
   [[#Points to discuss with Parsa]].
2. A comment in the LaTeX source says the sample size behind Table 3 is not confirmed. The two
   candidate values are 762 and 243. The notes do not give it either. Parsa needs to confirm
   it. Then [[4-results#^t3|T3]] records it, and the paper follows.
3. Section 4.4 reports Erfan's sensitivity, weight-norm and untrained-model tests
   ([[4-results#^t6|T6]], [[4-results#^t8|T8]], [[4-results#^t10|T10]]). Appendix A does
   not list them under Erfan.
4. Neither the paper nor [[4-results#^t7|T7]] says which inputs the attention test hid.
   `scripts/probe_attention_mass.py` runs three variants: real signals, gaze hidden alone,
   and gaze and hand hidden together. T7 should say this first, then the paper.
5. The text says "five diagnostics". Table 2 has six rows.
6. The PDFs in `papers/overleaf/` are older than the LaTeX source. The submitted PDF still
   lists Ioana as an author. `report_PREVIEW.pdf` is an old draft. The PDF needs to be built
   again from the source.

### Results in the notes that the paper lacks

These results are in the notes. The next version of the paper should include them:

- The encoding fix. Encoding each frame on its own lowered the error from 4.82 to 0.57. Every
  error value in the paper depends on this.
- The longer training runs. On 240 test clips, they showed Δ at or below zero before they were
  stopped. This supports the paper's conclusion.
- The wrong scaling constants. After scaling, gaze has about half the intended spread. This is
  a possible reason for a weak signal.
- Before training, real signals made the prediction worse by 0.0057.
- Hiding gaze alone moves the prediction 2.3 times more than hiding both signals.
- The right-palm results of [[4-results#^t5|T5]].

### Points to discuss with Parsa

In these points the paper and the notes differ, or the paper depends on Parsa's work. Each
needs a decision with Parsa. The notes record the decision, and the paper follows.

- Parsa's code. To derive the paper from this repository, the code behind T1, T3, T9, T12 and
  Appendix C needs to be added here. If it stays in Parsa's codebase, the paper needs to say
  where that code is.
- Section 2.3 reports an early pilot on EK100. It compares two different encoders (ViT-L and
  ViT-G), and EK100 has no gaze data. The paper calls it confounded. In the notes, the
  comparison that isolates gaze is [[4-results#^t11|T11]]. T11 could replace the pilot, or
  follow it.
- The feasibility figure in section 4.1 uses Parsa's ViT-L run (T1). The model behind the
  results is `ego_ft_v2` (T2). The figure could show the training curve of `ego_ft_v2`
  instead.
- The scope of the EK100 test is described in two ways. The paper says P01–P05 were requested
  and only P01's videos were on disk. Parsa's slides say all 32 participants were requested,
  with no filter. Both agree that 870 clips from P01 were evaluated. T3 should record the
  confirmed scope.

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

1. A backup slide says that P02–P08 data was absent. Training used P01–P07, and testing used
   P08. The slide should say that the data was absent on the cluster used for the probes.
2. The EK100 scope on the slides differs from the paper (see
   [[#Points to discuss with Parsa]]).
3. The feasibility slide in Erfan's section shows Parsa's ViT-L run. Either show the training
   curve of `ego_ft_v2`, or label the slide as an early ViT-L run.
