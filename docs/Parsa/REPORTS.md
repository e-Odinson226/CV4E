# Activity Report — V-JEPA 2 Ego Project

Chronological log of all project activity, successes and setbacks, compiled 2026-07-15 from
CLAUDE.md, DECISIONS.md, ROADMAP.md, git history, and run logs. Source for the final report.

Dates/times are from git commits and log-file timestamps where available; phases 1–4 predate
the timestamped record and are dated approximately.

---

## 1. Background — what this project is testing and why

**Research question.** V-JEPA 2-AC extends the V-JEPA 2 video world model with an
*action-conditioned* predictor: given past video frames and the robot's end-effector
actions/states, it predicts the latent representation of future frames. This works for robot
manipulation — but humans don't emit 7D end-effector vectors. This project asks: can the same
predictor architecture be conditioned on **human** behavioral signals instead — **eye gaze**
(2–3D angular fixation from Aria glasses) and **hand poses** (wrist + palm per hand = 12D) —
and does that conditioning produce internal representations that better encode how human
actions unfold in time?

**Why gaze and hand.** Gaze is a leading indicator of intention (people look at what they are
about to act on, typically ~0.5–1 s before contact); hand pose captures the effector state
directly, analogous to the robot's end-effector. Together they are the closest egocentric
analogue to the robot action/state pair the AC predictor was built around.

**How the hypothesis is measured.** Directly evaluating a world model is hard, so we use the
standard frozen-representation protocol: freeze the encoder (and predictor), extract features
on the EK100 (EPIC-Kitchens-100) **action anticipation** benchmark, train only a lightweight
*attentive probe* on those features, and compare recall@5 for predicting the action that
begins 1 second in the future. If the behaviorally-trained predictor's output tokens carry
information about unfolding actions beyond what the encoder alone provides, a probe given
those tokens should anticipate better. The probe is deliberately weak (1 attentive-pooler
block, one cross-attention query) so the score reflects the representation, not the probe.

**The evaluation pipeline, concretely:**

```
video clip [B, 3, 16, 224, 224]  (16 frames at 4 fps ending 1s before the action)
    → frozen ViT-G encoder                    → 1568 tokens × 1408 dim  (8 temporal blocks × 196 patches)
    → frozen behavioral predictor (Phase 5)   → + 196 predicted future tokens = 1764 total
    → trainable AttentivePooler (1 block, 1 query) + linear heads (verb / noun / action)
    → recall@5 on EK100 P01–P05 validation (2213 clips)
```

A 25-configuration sweep (5 learning rates × 5 weight decays) is trained simultaneously and
the best validation score is reported — standard practice for frozen-probe evals, removes
probe hyperparameters as an excuse.

**A critical protocol detail — masked eval.** EK100 has no gaze or hand data. At evaluation
the predictor therefore receives its learned `gaze_mask`/`hand_mask` tokens instead of real
signals. These mask tokens are *in the sequence* and participate in all 24 predictor blocks'
self-attention; they are stripped only after the final projection. Because the colleague
trained with `signal_dropout: 0.4` (40% of training frames received mask tokens), this
fully-masked condition is **in-distribution** — the predictor was explicitly trained to
predict under signal absence. This was verified 2026-06-17 (see D3) and matters for
interpretation: any Phase 5 effect is "what behavioral *training* left in the weights," not
"what live signals contribute."

**Terminology discipline for the final report:** the predictor predicts the *latent
representation of the future token block*, not "future frames" (it operates in embedding
space). And "no signal at eval" means *mask tokens substituted*, never "tokens absent."

---

## 2. Experimental design — treatment, controls, and what each gap means

The core comparison is 4 conditions × 3 seeds (42/43/44) = 12 runs, all with the **same
frozen ViT-G encoder**, same probe settings, same data. Only the token set fed to the probe
differs:

| Phase | Probe input | Tokens | Question it answers |
|---|---|---|---|
| 6a | encoder output only | 1568 | Baseline: what does the encoder alone give? |
| 6b | encoder + 196 zero tokens | 1764 | Does merely *adding tokens* inflate the score? |
| 6c | encoder + last temporal block repeated | 1764 | Does a *persistence* forecast (future ≈ present) match the predictor? |
| 5 | encoder + predictor's 196 predicted future tokens | 1764 | Treatment: does the behaviorally-trained prediction help? |

Why 6b exists: the AttentivePooler is permutation-invariant content-only cross-attention
(verified for probe depth 1), but more tokens = more keys to attend over, which alone could
raise the score. 6b isolates that. Why 6c exists: at a 1 s horizon the future usually looks
like the present, so a trivial "repeat the last block" forecast is a strong null model; the
predictor must beat it to claim it predicts anything.

The decomposition:

```
gap (5 − 6a) = total effect of appending the predictor block
gap (5 − 6b) = effect beyond token count alone
gap (5 − 6c) = predicted latent block beats persistence   ← strongest claim available this cycle
```

**Claim ceiling (important for the final report).** Even a clean positive 5−6c gap shows only
that *a behaviorally-trained predictor* helps — not that *behavioral conditioning
specifically* is the cause, because prediction training itself might produce useful tokens
regardless of the conditioning signal. Separating those requires **Phase 7**: retraining the
identical architecture with `signal_dropout: 1.0` (never sees a real signal) and measuring
gap (5 − 7). Phase 7 is deferred (colleague dependency); do not write "behavioral
conditioning adds signal" from this cycle's data.

**Standardized probe settings** (all 12 runs): ViT-G 1408-dim, 16 frames @ 4 fps,
`num_probe_blocks: 1`, batch 32 (64 on H200 for seed 44), 10 epochs, EK100 P01–P05
train/val, 25-config LR×WD sweep, action/verb/noun recall@5.

---

## 3. Pre-history — Phases 1–4 (before 2026-06-16)

### Phase 1 — Reference inference (✓ good)
**Purpose:** confirm the toolchain end-to-end before trusting any of our own numbers.
Ran Meta's pre-trained EK100 probe on the ViT-L encoder over P01 validation:
**53.35% action R@5** (74.20% verb, 74.14% noun). The published full-EK100 number is 32.7%;
ours is higher because P01 alone has a narrow vocabulary and the probe was fully trained by
Meta on all participants. Treated strictly as a plumbing check, not a baseline.

### Phase 2 — ViT-L baseline probe (partial — setback)
**Purpose:** establish a self-trained baseline: same protocol we would later apply to
treatment runs. Result after 1 epoch: **4.27% action R@5** (21.60% verb, 11.31% noun).
**Setback:** 5 epochs were planned but the run was killed repeatedly by the Linux OOM killer
— 8 dataloader workers with `pin_memory: true` exhausted host RAM while decoding 5.8 GB
H.264 files. Diagnosis and fix (`num_workers: 4`, `pin_memory: false`) became standard for
all later runs. The number also turned out to be structurally incomparable to the ViT-G
track (ViT-L 1024-dim encoder, probe depth 4, 32-frame clips) — it survives only as rough
orientation and its comparison against Phase 5 was later ruled invalid (see D4).

### Phase 3 — AC predictor architecture study (✓ good)
**Purpose:** before changing anything, map exactly how V-JEPA 2-AC interleaves action and
state tokens with frame patches, which tensors are shape-bound to robot signals, and what a
minimal egocentric substitution looks like. Output: full tensor-level architecture map (in
CLAUDE.md) and the `VisionTransformerPredictorEgo` design — robot action(7D)/state(7D)
projectors replaced by gaze(2D) and hand(12D) linear encoders, interleaved per timestep as
`[gaze_t, hand_t, patches_t, ...]`, causal attention preserved.

### Phase 4 — Our own ego predictor post-training (✓ good)
**Purpose:** prove the training loop works end-to-end on real egocentric data with real
signals. Built `app/vjepa_ego/` (dataset loader for HD-EPIC's Aria gaze/hand CSVs with
VRS-time alignment, training loop adapted from `vjepa_droid`). Trained the ViT-L ego
predictor on HD-EPIC P01 (27 sessions), encoder frozen, 20 epochs, ~3.8 min/epoch:
smooth-L1 latent-prediction loss **2.13 → 1.00**. Checkpoint:
`checkpoints/ego/vitl16-256px-8f/latest.pt`. Not evaluated in the ViT-G track — the
colleague's ViT-G predictor (trained on P01–P07, 7× the data) became the treatment instead.
**Gotcha found:** initializing the predictor with `num_frames=512` builds a 66K×66K boolean
causal mask (~4 GB, 32K-iteration Python loop) that hangs startup; must use
`num_frames = max_num_frames × 2`.

**Division of labor from here on:** the colleague trains the behavioral predictor on HD-EPIC
(her `checkpoints/colleague2/checkpoint.pt`: full 24-block predictor, blocks 18–23
fine-tuned, 3D gaze + 12D hand projectors, `signal_dropout: 0.4`, 3 epochs on P01–P07,
loss 0.507); we build and run the EK100 probe evaluation that tests whether her predictor's
representations transfer.

---

## 4. Timestamped log — 2026-06-16 onward

## 2026-06-16 (Mon)

- **23:57** — Commit `c23db25` feat(ego): egocentric predictor training and evaluation
  pipeline. First project commit on top of the upstream repo; bundles Phases 3–4 code and
  the Phase 5 eval wrapper (`vit_colleague_ego.py` with `IntegratedColleaguePredictor`,
  which restructures her flat 300-key state dict so it loads with `strict=True`).

## 2026-06-17 (Tue) — pivotal day: preview result, then three course corrections

- **00:52** — Commit `ab82f1c`: fixed the `img_size` bug. The config says
  `resolution: 256` but the WebDataset pipeline actually emits 224×224 crops; the predictor
  had been initialized with a 16×16 spatial grid instead of 14×14 (196 patches), silently
  corrupting positional geometry. Also enabled checkpoint resume. (bad → fixed)
- **01:11** — Commit `d56871e`: tqdm progress bar, TF32 matmul, H200 path/batch updates.
- **01:33** — Commit `418867e`: suppressed FutureWarnings, fixed CSVLogger duplicate headers.
- **Phase 5 preview completed** (seed=0, 7 epochs): best **5.83% val action R@5** at epoch 5
  vs 4.27% Phase 2. Two corrupt videos skipped cleanly (`P05_09.MP4`, `P32_10.MP4`).
  Looked like a strong win — the same evening, scrutiny of the comparison dismantled it:
- **D2 — Token-count confound discovered (bad → fixed).** Reading the wrapper code showed
  Phase 5 hands the probe 1764 tokens while the then-Phase 6 control handed 1568. Since the
  pooler simply cross-attends over all tokens, 196 extra keys could raise the score *no
  matter what they contain*. **The 5.83% preview was declared superseded** — not because the
  number is wrong, but because its comparison is uninterpretable. Phase 6 was refactored
  into the 6a/6b/6c family (§2) and runtime asserts were added so the token counts cannot
  silently drift again.
- **D1 — Original Phase 7 design found impossible (bad → redesigned).** The plan was to use
  the *robot* AC predictor as a no-signal control. Inspection showed its
  `forward(x, actions, states)` requires real 7D robot vectors and the checkpoint contains
  no learned mask token — there is no valid way to run it under the masked-eval protocol
  while keeping the architecture identical to Phase 5. Phase 7 was redefined as a matched
  retrain of the colleague's own architecture with `signal_dropout: 1.0`.
- **D3 — GATE 1 resolved (✓ good).** The colleague checkpoint's embedded config revealed
  `signal_dropout: 0.4`, proving masked eval is in-distribution (see §1). This retired the
  "mask token mismatch / OOD" framing and made the horizon sweep interpretable.
- **D4 — Overclaims excised.** The CLAUDE.md analysis sentence "behavioral conditioning adds
  structure even without live signals" was a hypothesis stated as a finding — it rested on
  the confounded 5-vs-old-6 gap and the invalid 5-vs-Phase-2 comparison. Removed from active
  docs, preserved verbatim in DECISIONS.md. Claim-discipline rule instituted (§2 ceiling).

## 2026-06-18 (Wed)

- **~05:02** — Phase 5 **seed42** completed (10 epochs): best **3.58%** val Act@5 at epoch 4
  (tied ep8). Note the drop vs the 5.83% preview: with a fixed seed and 10 full epochs the
  treatment landed in the same 3–4% band as everything else — the preview's peak was partly
  sweep-selection noise on a single seed. Log: `logs/p5_s42.txt`.
- **14:55** — Commit `631430f` feat(eval): Phase 6 control variants (6b/6c wrappers with
  token-count asserts) and seeded-run infrastructure (`--seed` CLI override writing to
  per-seed output folders, so one config file serves all seeds).

## 2026-06-19 (Fri)

- **~02:24–02:35** — Phase **6a/6b/6c seed42** completed (10 epochs each, parallel GPUs):
  6a **3.7%** (ep8), 6b **3.6%** (ep8/10), 6c **3.8%** (ep8).

## 2026-06-22 (Mon)

- **D5 — Seed-42 gaps analyzed: null/negative (bad news, provisional).**
  5−6a = −0.1pp, 5−6b = 0.0pp, 5−6c = −0.2pp. The treatment does not beat any control at
  the 2 s horizon on this seed. Independently, the colleague's paired A/B eval at ~2 s gave
  Δ ≈ +0.001 MSE (~0.2%) — two different measurement methods pointing the same direction
  strengthens the null reading.
- **Interpretation adopted (not a rationalization — pre-registered in her notes):** at ~2 s
  ahead, visual continuity alone predicts the future well, so behavioral conditioning has
  little room to help; if the effect exists it should live at longer horizons where the
  scene changes more. Consequence: the **horizon sweep** (probes retrained per anticipation
  horizon at 1 s/3 s/5 s/10 s, Phase 5 vs 6a, 3 seeds) was elevated from optional extra to
  **primary interpretive experiment**. Configs staged.
- **Pattern flagged, not concluded:** Phase 5 seed42 peaked at epoch 4 while all controls
  peaked at epoch 8 — could indicate the predictor block contains signal the probe latches
  onto early but can't sustain. Marked "watch across seeds."
- **Phase 7 pipeline partially received** from colleague (`~/work/pash/CV4E/`): training
  driver `finetune_ego.py` (already exposes a `--signal-dropout` CLI flag — Phase 7-ready),
  paired A/B evaluator `eval_ego_mse.py`, shared utilities, monitoring scripts, and her
  3-epoch P01–P07 result (Δ flipped from −0.0082 at v1 to **+0.0011** at v2 — her first
  positive, if tiny, signal). Three `src/` files the scripts import were still missing.

## 2026-06-23 (Tue)

- **14:24** — The three missing colleague files **arrived**: `src/models/ego_predictor.py`,
  `src/models/ego_finetune.py`, `src/datasets/ego_loaders.py` (untracked in git as of this
  report). ROADMAP.md/CLAUDE.md still list them as pending — **docs stale on this point**.
  Her pipeline is now runnable locally in principle; the remaining Phase 7 blockers are
  P02–P07 HD-EPIC data (only P01 is local, and a fair matched control needs the same data
  scale as her Phase 5 checkpoint) — or, preferred, she runs `--signal-dropout 1.0` on her
  own machine, which exactly matches the training environment.
- **15:15–15:35** — `evals/action_anticipation_frozen/metrics.py` and `eval.py` modified
  (uncommitted).

## 2026-06-24 (Wed)

- **~02:26–02:47** — All four **seed43** runs completed (10 epochs each, 4 GPUs parallel):
  Phase 5 **3.49%** (ep8), 6a **3.44%** (ep8), 6b **3.59%** (ep8), 6c **3.31%** (ep6).
- **Two-seed gap means** (best epoch per seed): 5−6a = −0.04pp, 5−6b = −0.06pp,
  5−6c = −0.02pp. All four variants sit within 3.3–3.8% across both seeds — the differences
  are the size of the seed-to-seed noise (e.g. 6c itself swings 3.8 → 3.31 between seeds).
  Reading: at the 2 s horizon the probe extracts essentially the same anticipation
  performance from all four token sets. Still provisional until seed 44 gives mean ± std.
- The seed42 early-peak pattern did **not** repeat: seed43 Phase 5 peaked at ep8 like the
  controls — evidence against the "signal the probe can't sustain" hypothesis, consistent
  with plain seed noise.
- **13:39** — Horizon-sweep configs (h3/h5/h10 for both Phase 5 and 6a) modified —
  staging for the sweep (uncommitted).

## 2026-06-25 → 2026-07-14

- No recorded activity (~3-week gap).

## 2026-07-15 (Wed) — seed 44 launch (in progress)

- **13:54** — Phase 5 **seed44** launched on cuda:0 (`logs/p5_seed44.log`), batch_size 64.
  Running on an **H200 (95 GB)** — different hardware than the earlier L40S runs. Same
  configs and probe settings; batch size raised 32→64 because data loading, not GPU memory,
  is the bottleneck. If the final report compares seeds, note the hardware/batch difference
  (bf16 nondeterminism and LR-vs-batch scaling make seeds non-identical replicas anyway;
  the sweep picks the best LR per run, which absorbs most of the batch effect).
- **13:58** — Phase 6a **seed44** launched on cuda:1 (`logs/p6a_seed44.log`).
- **14:35** (as of this report) — both in epoch 1/10, ~65% through, losses converging
  normally, GPU ~15 GB. **Phase 6b and 6c seed44 not yet launched** — commands ready in
  `commands.txt` (cuda:2/cuda:3).

---

## 5. Standing results snapshot (as of 2026-07-15)

| Variant | Seed 42 | Seed 43 | Seed 44 | Mean±std |
|---|---|---|---|---|
| Phase 5 (behavioral predictor, 1764 tok) | 3.58% (ep4) | 3.49% (ep8) | running | pending |
| 6a (encoder only, 1568 tok) | 3.70% (ep8) | 3.44% (ep8) | running | pending |
| 6b (+zeros, 1764 tok) | 3.60% (ep8/10) | 3.59% (ep8) | not launched | pending |
| 6c (+last frame, 1764 tok) | 3.80% (ep8) | 3.31% (ep6) | not launched | pending |

Per-epoch curves for every run are in the respective
`evals/vitg/<variant>/seed<N>/action_anticipation_frozen/<tag>/log_r0.csv`.

**Provisional reading (2 of 3 seeds):** null result at the 2 s horizon — the
behaviorally-trained predictor's tokens neither help nor hurt relative to encoder-only,
zero-padding, or persistence controls. This matches the outcome row "5 ≤ 6a, 6b, 6c" in
ROADMAP.md's interpretation table, whose designated follow-up is the horizon sweep.

**Superseded numbers that must NOT appear as results in the final report:**
- Phase 5 preview 5.83% (seed=0, token-count confound, single unseeded run)
- Any Phase 5 vs Phase 2 comparison (ViT-G vs ViT-L encoders, different probe depth and
  clip length — three confounds at once)

---

## 6. Setbacks summary (for the final report's "limitations / lessons")

1. **Phase 2 truncated to 1 epoch** by host-RAM OOM (8 dataloader workers + pin_memory while
   decoding 5.8 GB H.264 files). Lesson: I/O-bound video pipelines need conservative worker
   counts; fix became standard for all later runs.
2. **`img_size` 256→224 mismatch** silently corrupted the predictor's spatial grid until
   2026-06-17 — the config's `resolution: 256` does not match the pipeline's actual 224 px
   output. Lesson: verify tensor shapes empirically, not from configs.
3. **Token-count confound** invalidated the entire preview cycle: 1764-token treatment vs
   1568-token control rewards sequence length, not content. Caught by code reading, fixed by
   the 6b/6c control family plus runtime asserts. Lesson: with permutation-invariant poolers,
   match token counts across conditions.
4. **Original Phase 7 design architecturally impossible** (robot AC predictor has no mask
   token and demands real robot vectors). Lesson: verify a control can actually execute the
   protocol before scheduling it.
5. **"Mask token mismatch" framing was factually wrong** — `signal_dropout: 0.4` makes
   masked eval in-distribution. The wrong framing had already leaked into analysis text and
   had to be excised (D3/D4). Lesson: check training configs before reasoning about
   distribution shift.
6. **Seeds 42+43 are null/negative at the 2 s horizon** — the treatment does not beat any
   control. Honest result, drives the horizon sweep; the claim ceiling (§2) was set *before*
   this outcome, so the null is reportable without post-hoc goalpost-moving.
7. **Preview-vs-seeded discrepancy** (5.83% → ~3.5%): a single unseeded run with a 25-config
   sweep can overstate performance by >2pp. Lesson: never report single-seed sweep maxima.
8. **Two corrupt EK100 videos** (`P05_09.MP4`, `P32_10.MP4`) — skipped cleanly by
   `filter_annotations()`; negligible impact (2 of ~24k clips).

---

## 7. Open items

- **Finish seed 44**: Phase 5 + 6a running; **6b + 6c still need launching** (commands in
  `commands.txt`). Then compute mean ± std per variant and the three gaps → this closes the
  12-run experiment and produces the citable table.
- **Horizon sweep** (primary interpretive experiment): Phase 5 vs 6a at 1 s/3 s/5 s/10 s,
  matched probe per horizon, seeds 42/43/44. Run 3 s and 5 s first — most likely to show a
  signal if the effect is horizon-dependent. Launch as GPUs free up after seed 44; do not
  block on Phase 7.
- **Phase 7 matched retrain** (`signal_dropout: 1.0`): the only path to "behavioral
  conditioning *specifically* helps." Colleague src/ files received 2026-06-23; blocked on
  P02–P07 data locally or colleague running it on her machine (preferred). Chase this.
- **Doc hygiene**: ROADMAP.md and CLAUDE.md still say the three src/ files are pending —
  update both; fill seed-43 results into ROADMAP.md's phase tables.
- **Uncommitted work**: horizon config diffs, eval.py/metrics.py changes, three colleague
  src/ files, and this REPORTS.md — commit once seed 44 is underway on all four variants.
