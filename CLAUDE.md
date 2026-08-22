# CLAUDE.md — start here

Master's thesis project on JEPA-based egocentric video understanding. This file is a
**router**, deliberately thin: the real record lives in the notes vault, and duplicating it
here would only let the two drift apart.

## Read this first, in this order

0. **New to the project, or lost in the cross-links?** Read
   `docs/EgoVault/reports/2026-08-19-project-narrative.md` first — the whole story, zero to
   now, in plain language with every term explained. Everything from step 1 onward is a
   precise lab record written for someone who already knows the story; that file is what
   gets you there.
1. **`docs/EgoVault/_INDEX.md`** — current state, the next concrete action, and links to
   everything active. If you read one file *of the lab record*, read this one.
2. **`docs/EgoVault/CLAUDE.md`** — how to work in the vault: where notes go, the frontmatter
   schema, and the hard rules (append-only decision log, mandatory Takeaways, ask before
   deleting).
3. **`docs/EgoVault/worklog/`** — most recent entry. What the last session did and where it
   stopped.
4. **`docs/EgoVault/decisions.md`** — skim for current direction. Append-only, chronological,
   dated back to 2026-05-31.

Then open only the specific notes you need. Don't read the whole vault.

## What's where

| Path | What |
|---|---|
| `docs/EgoVault/` | The notes vault — experiments, papers, topics, concepts, reference, meetings, decisions |
| `docs/EgoVault/archive/originals/` | The 15 pre-migration source notes, untouched. Never edit; read-only history |
| `scripts/` | The pipeline: `finetune_ego.py`, `eval_ego_mse.py`, `ego_common.py`, `extract_gaze_hand.py`, plus monitoring/plotting |
| `vjepa2/` | V-JEPA 2 fork, **vendored** (not a submodule since 2026-07-28). Our `ego_*` modules live inside `src/models/` and `src/datasets/` |
| `checkpoints/`, `results/`, `data/` | Run outputs and datasets. All gitignored |

Commands and environment: `docs/EgoVault/reference/running-the-pipeline.md` and
`reference/environment-and-paths.md`. The latter also lists **what is currently broken** in
the tree — read it before debugging something that was already known to be broken.

## Working rules

The vault's own `CLAUDE.md` governs note-writing and is authoritative there. Two rules worth
repeating because they are easy to violate by accident:

- **`decisions.md` and experiment `## Log` sections are append-only.** Add a dated entry;
  never rewrite or delete past ones. The history of *why* is the point.
- **Every experiment note needs a Takeaway**, especially failures and null results. A negative
  result is a finding — write what it rules out.

Commit per meaningful chunk with a specific message, not per file. At the end of a work
session, write a `worklog/YYYY-MM-DD.md` entry from the template, then remind about push.

## One thing to know before you reason about results

The headline result of the master project — that gaze/hand conditioning improves prediction —
**was withdrawn as provisional on 2026-07-28**. Only a small 3-epoch run ever completed; four
attempts at the intended full scale died partway through with Δ ≤ 0, and an independent probe
found a null at the same horizon. Do not treat "gaze/hand conditioning works" as established.
See `_INDEX.md` and `experiments/EXP-001-gaze-hand-conditioning.md`.
