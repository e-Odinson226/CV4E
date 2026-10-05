---
type: reference
status: running
created: 2026-07-26
updated: 2026-10-05
---

# Thesis notes

Notes for Erfan's master's thesis on egocentric video understanding with JEPA models. The
question is whether a video prediction model predicts the near future better when it also
receives the person's gaze and hand positions.

**Status.** Gaze and hand inputs did not improve the prediction. A model trained without them
does as well ([[4-results#^t11|T11]]). All results come from models trained for 3 epochs. A
full training run will test whether more training changes this. The next steps are in
[[6-next-steps]].

## Chapters

Files 1 to 6 follow the chapters of a thesis. Read them in order.

| File | Contents |
|---|---|
| [[1-introduction]] | The question, the approach, the three Paths, and the hypotheses with their status. |
| [[2-background]] | The models and papers this work builds on, and the terms used. |
| [[3-method]] | Data, model, training, evaluation methods, and design choices. |
| [[4-results]] | Every test, who ran it, and what it showed, grouped by hypothesis. |
| [[5-discussion]] | What the results mean together, and their limits. |
| [[6-next-steps]] | What to do next, in order. |

## Other files

| File or folder | Contents |
|---|---|
| [[project]] | People, timeline, the paper and the final presentation. |
| [[final-report]] | The full text of the paper, with notes. |
| `papers/` | [[submitted-paper]], the paper as submitted (never edited), and the literature notes in `papers/literature/`. |
| `templates/` | A template for adding a test to [[4-results]]. |

The code, the commands, the data paths and the checkpoints are described in `README.md` at the
root of the repository.

## Papers

The paper is derived from these notes and the code. Who wrote it, its open problems and the
results it still lacks are in [[project#Paper]]. Its full text, with notes, is in
[[final-report]]. The text as submitted on 24 August 2026 is kept, unedited, in
[[submitted-paper]].

The literature, in `papers/literature/`, one note per paper:

- [[vjepa]]: V-JEPA 2, the base model.
- [[gazeqwen]]: GazeQwen, which adds gaze to a video-language model.
- [[vl-jepa]]: VL-JEPA, a JEPA model aligned with language. The basis for Path 3.
- [[vla-jepa]]: VLA-JEPA, JEPA-style pretraining for robot policies. Full text, no notes yet.
