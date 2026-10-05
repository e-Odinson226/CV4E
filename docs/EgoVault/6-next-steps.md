---
type: report
status: running
created: 2026-09-18
updated: 2026-10-05
---

# 6. Next steps

What to do next, in order. Each step names the hypothesis it addresses. The hypotheses are in
[[1-introduction#^hypotheses|the hypotheses table]].

## 1. Before the next training run

- Recompute the scaling constants `GAZE_MEAN`, `GAZE_STD` and `HAND_STD` from all P01–P07
  recordings. With the current values, gaze has about half the intended spread.
- Give `checkpoints/ego_sd1p0/best.pt` to Parsa. The EK100 probe (T3) has not been run on the
  model trained without signals.

## 2. The full training run (H3)

All results so far come from models trained for 3 epochs. This run tests whether more
training changes them. No other development comes before it, and no architecture changes
are made until it is done. (Decided by Erfan, July and August 2026.)

- Training: 8 epochs, P01–P07, 60 clips per recording.
- Testing: the same 240 P08 clips for every arm (6 recordings × 40 clips).
- Arms:

| Arm | Signals during training | Flag |
|---|---|---|
| A | real gaze and hand | `--signal-dropout 0.4` |
| B | real signals in a shuffled time order | `--shuffle-signals time` |
| C | none | `--signal-dropout 1.0` |

`--shuffle-signals batch` is a second option for arm B. It takes the signals from another clip.

- Comparisons:
  - A with real signals against C with hidden signals measures the value of the information,
    as in T11.
  - A against B shows whether correctly timed signals do better than mismatched ones.
- Analysis: Δ per clip, a paired Wilcoxon test, and the effect size with a 95% confidence
  interval.
- Write down the stopping rule before the run starts.
- After training, repeat T6 (sensitivity) on arm A.
- If there is time, train a second seed for each arm.
- Add the result to [[4-results]] as a new test.

## 3. Smaller tests

- A test that predicts further ahead than one step (0.27 s). The scripts have no option for
  this yet. (H1)
- Error bars for T4 and T5, by resampling recordings. The features are cached, so this is
  quick. (R)
- A comparison target that is as hard to predict as gaze. This would complete T5. (R)
- A second seed for T11. (M)
- Set the gaze layer's bias to zero at test time. This shows how much of the effect comes from
  the constant part found in T8. (H2)

## 4. Path 3: language alignment

Start after the full training run, so that the comparison has a known baseline. The design
choices below came from Ash's review in July 2026.

- Idea: align the predictor's embedding space with text, so that it predicts concepts such as
  "making pasta sauce".
- Motivation: on EK100, VL-JEPA's advantage over V-JEPA 2 grows with the horizon, from +1.5
  points at 1 s to +4.6 at 10 s ([[vl-jepa]]).
- Text source: HD-EPIC recipe steps and action descriptions. EK100 labels are short verb–noun
  pairs, which are likely too little.
- Design: keep Parsa's probe, conditions and seeds. Change only the embedding space.
- Loss: start with a single target embedding (L2 distance). Move to a distribution over
  candidate goals only if the model is confidently wrong on ambiguous clips.
- Report absolute differences. Relative percentages make small effects look large.
- If HD-EPIC text does no better than verb–noun pairs, Path 3 repeats T3. In that case,
  change direction.
- Benchmarks beyond EK100 that test goals: CrossTask and EgoExo4D.

## 5. Other directions

- The paper suggests a different use of gaze and hand: as an extra training loss toward
  goal-level targets, dropped at test time. This needs a goal probe across P01–P07. The idea
  came from Ioana.
- Coord-PE: only if an architecture test is wanted again. It needs gaze projected into the
  image first. See [[3-method]].
- A workshop paper on the result, after the full training run. (Planned by Erfan, July 2026.)

## 6. Housekeeping

- The final presentation is on 25 November 2026. The slide fixes are in
  [[project]].
- Back up `EgoVault-history-2026-10-05.bundle` (in `Projects/Ego/`) to another machine. It
  holds the notes' git history from when they were a separate repository.
- Find `vjepa2-history.bundle` and `CV4E-full-backup-2026-08-22.bundle`. They are not on this
  machine, and they are the only copies of that git history.
- Save only the trainable weights in checkpoints. This reduces each file from 1.22 GB to about
  150 MB.
