---
type: report
status: running
created: 2026-10-05
updated: 2026-10-05
---

# Paper status

The paper submitted for EgoProject 2026: what it is, who wrote what, and what is still open.

## Summary

- Title: *Behavioral Conditioning of Video World-Model Predictors for Egocentric Action
  Anticipation*.
- Submitted on 24 August 2026 as the EgoProject 2026 report, in NeurIPS format.
- Authors on the title page: Seyyed Parsa Sharifi, Erfan Yekehzare, Ioana Marica.
- The submitted PDF is `overleaf/Semantic_Intention___Final_Report.pdf`. The LaTeX source is
  in `overleaf/`. The full text, with notes, is in [[final-report]].
- Main result: conditioning on gaze and hand does not improve prediction. A model trained
  without the signals does as well (Δ = +0.0003, p = 0.40,
  [[4-results#^t11|T11]]).

## Who wrote what

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

Section 4.4 also reports Erfan's sensitivity, weight-norm and untrained-model tests
([[4-results#^t6|T6]], [[4-results#^t8|T8]],
[[4-results#^t10|T10]]). Appendix A does not list them.

## Open problems in the submitted paper

1. The code link on the title page returns 404. The GitHub organization exists, but it has
   only two unrelated public repositories.
2. The title page lists three authors. Appendix A says that only Parsa and Erfan wrote the
   report. The two need to agree.
3. A comment in the LaTeX source says the sample size behind Table 3 is not confirmed. The two
   candidate values are 762 and 243. The comment does not appear in the PDF.
4. The code in Appendix C is from Parsa's codebase. It is not in this repository.
5. The paper does not say whether the attention test hid gaze and hand together or one at a
   time. Training always hid them together.
6. The text says "five diagnostics". Table 2 has six rows.
7. `overleaf/report_PREVIEW.pdf` is an old draft. The submitted file is
   `Semantic_Intention___Final_Report.pdf`.

## Results missing from the paper

These results are in the notes but not in the paper:

- The encoding fix. Encoding each frame on its own lowered the error from 4.82 to 0.57. Every
  error value in the paper depends on this.
- The longer training runs. On 240 test clips, they showed Δ at or below zero before they were
  stopped. This supports the paper's conclusion.
- The wrong scaling constants. After scaling, gaze has about half the intended spread. This is
  a possible reason for a weak signal.
- Before training, real signals made the prediction worse by 0.0057.
- Hiding gaze alone moves the prediction 2.3 times more than hiding both signals.
- The right-palm results of T5.

## Points to discuss with Parsa

- Section 2.3 reports an early pilot on EK100. It compares two different encoders (ViT-L and
  ViT-G), and EK100 has no gaze data. The paper calls it confounded. The comparison that
  isolates gaze is T11. It could replace the pilot, or follow it.
- The feasibility figure in section 4.1 uses Parsa's ViT-L run (T1). The model behind the
  results is `ego_ft_v2` (T2).
- The scope of the EK100 test is described in two ways. The paper says P01–P05 were requested
  and only P01's videos were on disk. Parsa's slides say all 32 participants were requested,
  with no filter. Both agree that 870 clips from P01 were evaluated.
