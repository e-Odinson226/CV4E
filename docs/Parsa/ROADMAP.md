# Research Roadmap — V-JEPA 2 with Egocentric Human Signals

## Hypothesis

> Teaching a world model to associate gaze and hand signals with future states during training
> produces representations that better encode human intention — measurable as improved action
> anticipation even when those signals are absent at inference time.

To test this, we post-train V-JEPA 2's predictor on egocentric video with gaze and hand conditioning,
then freeze it and evaluate representations by training a lightweight probe on EK100 action anticipation.

### What this cycle can and cannot claim

**This cycle** (Phases 5 / 6a / 6b / 6c) tests whether **a behaviorally-trained predictor improves
the representation over the encoder alone**. The gaps 5−6a/6b/6c measure this directly.

**This cycle cannot** isolate whether the behavioral conditioning specifically is responsible for any
gain. A predictor trained with no behavioral signal might produce an equally useful predicted latent
representation, because prediction itself may help regardless of what conditioned the training.
Distinguishing those two requires a matched no-signal predictor — **Phase 7, deferred this cycle**
due to a missing training script and data (colleague dependency).

The 5−6c gap is the strongest available claim: does the predicted latent representation of the future
token block outperform simply repeating the present encoder block? If yes and consistent across seeds,
something beyond persistence is happening.

**Note on terminology**: the predictor operates in latent/embedding space, not pixels. In methods or
report text, write "predicted latent representation of the future token block" — not "future frame."

**"No signal at eval" is not the same as "tokens absent"**: when `valid=False`, the gaze/hand
*mask tokens* are injected into the sequence and participate fully in all 24 blocks of the
predictor's self-attention. They are stripped only *after* the final norm+projection (line 79 in
`vit_colleague_ego.py`). The probe therefore sees a predicted latent block that was shaped by
mask-token context, not by an absent context. Use this framing in any writeup.

**GATE 1 (resolved 2026-06-17)**: the colleague predictor was trained with `signal_dropout: 0.4`
— 40% of frames receive mask tokens during training. Masked eval (100% mask tokens) is therefore
*in-distribution*, not OOD. This makes the horizon sweep interpretable: the model has been
explicitly trained to predict under partial and full masking.

---

## Design Principle

All probe evaluations use **ViT-G as the encoder throughout**. This eliminates encoder-size as a
confound. This cycle has four conditions:

| Phase | Condition | Tokens | Purpose |
|---|---|---|---|
| Phase 6a | ViT-G encoder only | 1568 | Does the predictor block add anything at all? |
| Phase 6b | encoder + zero pad | 1764 | Does adding tokens help, regardless of content? |
| Phase 6c | encoder + repeated last frame | 1764 | Does prediction beat persistence? |
| Phase 5 | encoder + behavioral predictor | 1764 | Treatment: does a behaviorally-trained predictor help? |

The gaps 5−6a/6b/6c decompose the contribution cleanly. Phase 7 (matched retrain at
`signal_dropout: 1.0`) is deferred — see Planned Phases.

**Standardized probe settings** (inherited from Phase 5, applied to all future phases):

| Setting | Value |
|---|---|
| Encoder | ViT-G (1408-dim) |
| `frames_per_clip` | 16 |
| `frames_per_second` | 4 |
| `num_probe_blocks` | 1 |
| `batch_size` | 32 |
| `num_epochs` | 10 |
| Probe train data | EK100 P01–P05 train |
| Eval data | EK100 P01–P05 val |
| Metric | Action / Verb / Noun Recall@5 |
| LR × WD sweep | 25 configs (5 LR × 5 WD) |

---

## Checkpoints

| Key | Path | Contents |
|---|---|---|
| ViT-G encoder | `checkpoints/ac/vjepa2-ac-vitg.pt` → key `target_encoder` | V-JEPA 2-AC ViT-G, EMA encoder |
| Colleague predictor | `checkpoints/colleague2/checkpoint.pt` → key `predictor` | Behavioral predictor, epoch 3, HD-EPIC P01-P07, loss 0.507 |
| Our ego predictor | `checkpoints/ego/vitl16-256px-8f/latest.pt` | ViT-L ego predictor, epoch 20, HD-EPIC P01 — **not used in ViT-G track** |

---

## Datasets

| Dataset | Location | Used for |
|---|---|---|
| HD-EPIC P01 | `data/hd_epic/HD-EPIC/Videos/P01/` + `SLAM-and-Gaze/P01/` | Predictor post-training (gaze + hand signals) |
| HD-EPIC P01–P07 | Colleague's training data (not local) | Colleague predictor post-training |
| EK100 P01–P05 | `data/ek100/videos/` | Probe training and evaluation |

---

## Completed Phases

### Phase 1 — Reference Inference (Done)

**Purpose:** Reproduce the published EK100 number in our environment to confirm setup is correct.

| Item | Value |
|---|---|
| Encoder | `checkpoints/vjepa2/vitl.pt` → `target_encoder` |
| Probe | `checkpoints/probes/ek100-vitl-256.pt` (Meta pre-trained) |
| Eval data | EK100 P01 val only (870 clips) |
| Config | `configs/inference/vitl/ek100_local.yaml` |

| Metric | Result |
|---|---|
| Action R@5 | 53.35% |
| Verb R@5 | 74.20% |
| Noun R@5 | 74.14% |

> Reference only — inflated because it uses Meta's fully-trained probe on P01's narrow vocabulary.
> Not a target to beat.

---

### Phase 2 — ViT-L Baseline Probe (Done, 1 epoch)

**Purpose:** Establish our own reproducible baseline by training a probe from scratch.

| Item | Value |
|---|---|
| Encoder | `checkpoints/vjepa2/vitl.pt` → `target_encoder` |
| Predictor | Standard V-JEPA predictor (from same checkpoint) |
| Probe train | EK100 P01–P05 train, 1 epoch completed |
| Config | `configs/eval/vitl/ek100_p01p05.yaml` |
| Note | `num_probe_blocks: 4`, `frames_per_clip: 32` — different settings from Phase 5 |

| Metric | Result |
|---|---|
| Action R@5 | 4.27% |
| Verb R@5 | 21.60% |
| Noun R@5 | 11.31% |

> **Not directly comparable to Phase 5** — different encoder size (ViT-L vs ViT-G), different probe
> depth (4 vs 1 blocks), different clip length (32 vs 16 frames). Serves as a rough orientation only.

---

### Phase 3 — AC Predictor Architecture Study (Done)

**Purpose:** Understand how V-JEPA 2-AC interleaves action/state tokens with frame features, map
every tensor that needs to change for egocentric adaptation.

Output: architecture diagram in CLAUDE.md; `VisionTransformerPredictorEgo` design finalized.

---

### Phase 4 — Ego Predictor Post-Training (Done)

**Purpose:** Post-train a ViT-L predictor on HD-EPIC P01 conditioned on gaze (2D) and hand (12D).

| Item | Value |
|---|---|
| Encoder | `checkpoints/vjepa2/vitl.pt` (frozen throughout) |
| Predictor | `VisionTransformerPredictorEgo` — trained from scratch |
| Post-train data | HD-EPIC P01: 27 sessions, video + gaze CSV + hand CSV |
| Config | `configs/train/vitl16/hd_epic_ego.yaml` |
| Output checkpoint | `checkpoints/ego/vitl16-256px-8f/latest.pt` |
| Training | 20 epochs, loss 2.13 → 1.00 |

> This checkpoint is not evaluated in the ViT-G track. It would be used in a future ViT-L ablation.

---

### Phase 5 — ViT-G Behavioral Predictor Probe (Done, seed42) ← Treatment

**Purpose:** Evaluate whether the colleague's behaviorally-conditioned predictor produces richer
representations than the raw encoder.

| Item | Value |
|---|---|
| Encoder | `checkpoints/ac/vjepa2-ac-vitg.pt` → `target_encoder` |
| Predictor | `checkpoints/colleague2/checkpoint.pt` — post-trained on HD-EPIC P01–P07 with gaze + hand |
| Predictor signals at eval | **None** — `gaze_mask` / `hand_mask` tokens substituted for all frames |
| Probe train | EK100 P01–P05 train, 10 epochs total |
| Config | `configs/eval/vitg/ek100_colleague.yaml` |
| Eval wrapper | `evals/action_anticipation_frozen/modelcustom/vit_colleague_ego.py` |

**Preview run (seed=0, superseded — token-count confound):**

| Epoch | Train Act@5 | Val Act@5 | Val Verb@5 | Val Noun@5 |
|---|---|---|---|---|
| 1 | 3.03% | 2.81% | 16.17% | 9.29% |
| 2 | 3.19% | 4.12% | 22.14% | 11.44% |
| 3 | 3.93% | 4.40% | 23.01% | 13.75% |
| 4 | 4.48% | 3.74% | 25.64% | 10.77% |
| **5** | **4.08%** | **5.83%** | **24.29%** | **12.58%** |
| 6 | 4.21% | 3.65% | 24.71% | 9.88% |
| 7 | 3.85% | 3.82% | 25.56% | 12.66% |

**Seed42 (countable run, 2026-06-18):**

| Epoch | Train Act@5 | Val Act@5 | Val Verb@5 | Val Noun@5 |
|---|---|---|---|---|
| 1 | 1.96% | 2.83% | 18.23% | 9.46% |
| 2 | 2.11% | 3.18% | 20.23% | 9.92% |
| 3 | 2.21% | 3.02% | 19.30% | 10.64% |
| **4** | 2.12% | **3.58%** | 21.79% | 11.11% |
| 5 | 2.26% | 3.21% | 19.88% | 11.24% |
| 6 | 2.47% | 3.06% | 20.83% | 10.09% |
| 7 | 2.27% | 3.13% | 21.59% | 10.64% |
| 8 | 2.22% | 3.58% | 22.06% | 10.10% |
| 9 | 2.39% | 3.50% | 22.65% | 9.74% |
| 10 | 2.95% | 3.48% | 21.96% | 10.45% |

---

## Active Phases — The 12-Run Experiment

**Design:** 4 variants × 3 seeds (42 / 43 / 44) = 12 runs. 4 GPUs in parallel, 3 rounds.

### Phase 6a — ViT-G Encoder Only ← Control: no predictor

| Item | Value |
|---|---|
| Encoder | `checkpoints/ac/vjepa2-ac-vitg.pt` → `target_encoder` |
| Predictor | **None** — 1568 encoder tokens fed directly to probe |
| Config | `configs/eval/vitg/ek100_vitg_raw.yaml` |
| Wrapper | `evals/action_anticipation_frozen/modelcustom/vit_vitg_raw.py` |

| Seed | Val Act@5 |
|---|---|
| 42 | **3.7%** (ep8) |
| 43 | **3.44%** (ep8, done 2026-06-24) |
| 44 | running (launched 2026-07-15) |
| **mean ± std** | pending seed 44 |

### Phase 6b — ViT-G Encoder + Zero Pad ← Control: token count

| Item | Value |
|---|---|
| Encoder | same as 6a |
| Extra tokens | 196 zeros appended → 1764 total |
| Config | `configs/eval/vitg/ek100_vitg_6b.yaml` |
| Wrapper | `evals/action_anticipation_frozen/modelcustom/vit_vitg_6b.py` |

| Seed | Val Act@5 |
|---|---|
| 42 | **3.6%** (ep8/10) |
| 43 | **3.59%** (ep8, done 2026-06-24) |
| 44 | not yet launched (commands staged in commands.txt) |
| **mean ± std** | pending seed 44 |

### Phase 6c — ViT-G Encoder + Last Frame Repeat ← Control: persistence

| Item | Value |
|---|---|
| Encoder | same as 6a |
| Extra tokens | last 196 encoder tokens repeated → 1764 total |
| Config | `configs/eval/vitg/ek100_vitg_6c.yaml` |
| Wrapper | `evals/action_anticipation_frozen/modelcustom/vit_vitg_6c.py` |

| Seed | Val Act@5 |
|---|---|
| 42 | **3.8%** (ep8) |
| 43 | **3.31%** (ep6, done 2026-06-24) |
| 44 | not yet launched (commands staged in commands.txt) |
| **mean ± std** | pending seed 44 |

### Phase 5 — ViT-G + Behavioral Predictor ← Treatment (seeded reruns)

Prior numbers (2.81–5.83%) were preview-only (seed=0, no 6b/6c controls). Seeded reruns are the
countable result.

| Item | Value |
|---|---|
| Predictor | `checkpoints/colleague2/checkpoint.pt` — HD-EPIC P01–P07, gaze+hand |
| Extra tokens | predictor's last 196 output tokens → 1764 total |
| Config | `configs/eval/vitg/ek100_colleague.yaml` |
| Wrapper | `evals/action_anticipation_frozen/modelcustom/vit_colleague_ego.py` |

| Seed | Val Act@5 |
|---|---|
| 42 | **3.58%** (ep4) |
| 43 | **3.49%** (ep8, done 2026-06-24) |
| 44 | running (launched 2026-07-15) |
| **mean ± std** | pending seed 44 |

---

## Planned Phases

### Phase 7 — Matched Control: colleague architecture, signal_dropout: 1.0

**Definition (revised 2026-06-17):** Phase 7 is no longer "ViT-G + original robot AC predictor."
That design is **retired**: the original AC predictor's `forward()` requires real 7D robot vectors
and has no learned mask token, so it cannot run the masked-eval protocol at all.

Phase 7 is redefined as a **matched no-signal control**: the colleague's architecture (same
`IntegratedColleaguePredictor`, same ViT-G encoder) retrained from scratch with
`signal_dropout: 1.0` — every frame gets mask tokens during training. This is structurally
identical to Phase 5 except the predictor never sees real behavioral signals.

`gap (5 − 7)` = behavioral conditioning specifically helps (vs. learning to predict under
permanent masking). This is the gap needed to support the core hypothesis.

**Status (updated 2026-07-15): pipeline fully received — remaining blocker is P02–P07 data.**

Received from colleague (at `/trinity/home/kx2428/work/pash/CV4E/`, 2026-06-22):
- `scripts/finetune_ego.py` — training driver; has `--signal-dropout` CLI flag ready for Phase 7
- `scripts/eval_ego_mse.py` — paired A/B evaluation
- `scripts/ego_common.py` — shared building blocks (model loading, encoding, VRS alignment)
- Monitoring scripts (`watch_progress.py`, `summarize_run.py`, `plot_results.py`)
- 3-epoch results on P01–P07 (ego_ft_v2: Δ flipped from −0.0082 → +0.0011)

Received 2026-06-23 (previously missing; now in this repo, untracked):
- `src/models/ego_predictor.py` — `vit_ego_predictor()` factory (3D gaze, validity flags)
- `src/models/ego_finetune.py` — weight loading and freeze utilities
- `src/datasets/ego_loaders.py` — `GazeTokenLoader`, `HandTokenLoader`

Still missing: P02–P07 HD-EPIC data (only P01 local). Phase 7 retrain needs the same data
scale as the colleague's Phase 5 checkpoint for a fair matched control.

Options: (a) colleague sends P02–P07, we run `--signal-dropout 1.0` ourselves; or
(b) colleague runs it on her machine (preferred — matches training environment).
Request pending.

**Phase 7 is still the only path to "behavioral conditioning specifically helps."** Keep as
parallel track. Do not block horizon sweep on it.

---

## Primary Comparison — This Cycle (5 / 6a / 6b / 6c)

**Claim this cycle supports:** does a behaviorally-trained predictor improve the representation
over the encoder alone? **Not:** does behavioral conditioning specifically cause the improvement
(that requires Phase 7, deferred).

```
Phase 6a: ViT-G encoder only               → ?% ± ?   (1568 tokens)
Phase 6b: encoder + zero pad               → ?% ± ?   (1764 tokens — token-count control)
Phase 6c: encoder + repeated last block    → ?% ± ?   (1764 tokens — persistence control)
Phase 5:  encoder + behavioral predictor   → ?% ± ?   (1764 tokens — treatment)
```
3 seeds each (42 / 43 / 44). Report mean ± std. Prior Phase 5 numbers (2.81–5.83%) are
preview-only (seed=0, against 1568-token 6a) and do not carry forward.

```
gap (5 − 6a) = total effect of the predictor block
gap (5 − 6b) = effect beyond token count alone
gap (5 − 6c) = predicted latent block beats persistence  ← strongest available claim
```

| Outcome | Interpretation |
|---|---|
| 5 > 6c > 6b > 6a | Prediction adds real signal beyond persistence and token count (does not isolate behavioral conditioning — see Phase 7) |
| 5 ≈ 6c > 6b | The predictor's latent block is no better than repeating the present — prediction doesn't help beyond content |
| 5 ≈ 6b ≈ 6c > 6a | Token count alone explains the gap — pooler benefits from length regardless of content |
| All ≈ 6a | Predictor output is ignored entirely; encoder alone determines the score |
| **5 ≤ 6a, 6b, 6c (null/negative)** | Behavioral predictor does not help at 2s horizon — consistent with colleague A/B Δ≈0. Effect may be horizon-dependent; see horizon sweep. |

**Seeds 42+43 provisional result (updated 2026-07-15):** gap means across two seeds (best
epoch per seed): 5−6a = −0.04pp, 5−6b = −0.06pp, 5−6c = −0.02pp. All variants within
3.3–3.8% on both seeds — differences are the size of seed-to-seed noise (6c alone swings
3.8 → 3.31 between seeds). Consistent with the last row above. The seed-42 early-peak
pattern (Phase 5 best at ep4) did NOT repeat on seed 43 (best at ep8, like the controls).
Do not conclude until seed 44 is in (mean ± std). See D5/D6 in DECISIONS.md.

## Full Comparison — With Phase 7 (Deferred)

```
Phase 6a: encoder only                                   (control: no predictor)
Phase 7:  encoder + matched predictor, dropout=1.0       (control: same arch, no behavioral signal ever seen)
Phase 5:  encoder + behavioral predictor, dropout=0.4    (treatment)

gap (7 − 6a) = any prediction helps, regardless of training signal content
gap (5 − 7)  = behavioral conditioning specifically helps  ← this supports the hypothesis
```

Note: "robot AC predictor as control" was retired 2026-06-17. Phase 7 is now a matched retrain
of the colleague's architecture with `signal_dropout: 1.0`. See Planned Phases above.

---

## Next Experiment — Horizon Sweep (elevated priority, 2026-06-22)

Seed-42 null result (5 ≤ all controls at 2s) plus colleague's A/B Δ≈0 at 2s both suggest the
behavioral signal's effect, if real, lives at longer anticipation horizons. This is now the
**primary interpretive experiment**, not an optional extra.

**Rationale:** At ~2s ahead, vision alone is already strong — the current frame predicts the next
well without additional conditioning. Gaze and hand signals should matter more at longer horizons
where visual continuity breaks down. If the effect is horizon-dependent, the null at 2s is
*expected*, not disconfirming.

**Design:** Run Phase 5 and 6a at each anticipation horizon using a matched probe per horizon (not
one probe extrapolated). Configs already staged:

| Horizon | Phase 5 config | Phase 6a config |
|---|---|---|
| 1s | `ek100_colleague_h1.yaml` | `ek100_vitg_raw_h1.yaml` |
| 3s | `ek100_colleague_h3.yaml` | `ek100_vitg_raw_h3.yaml` |
| 5s | `ek100_colleague_h5.yaml` | `ek100_vitg_raw_h5.yaml` |
| 10s | `ek100_colleague_h10.yaml` | `ek100_vitg_raw_h10.yaml` |

Each horizon uses seeds 42/43/44 for both variants (8 runs per horizon × 4 horizons = 32 runs
total). At minimum, run the 3s and 5s horizons first — they are most likely to show a signal.

**Launch:** immediately after seeds 43/44 for the 12-run main experiment complete. Do not wait
for Phase 7.

---

## Optional Future Work

These are not required for the primary result but would strengthen the claim:

- **Phase 7 (parallel track):** Matched retrain with `signal_dropout: 1.0`. Unlocks `gap (5−7)` —
  the gap that directly supports the hypothesis. Waiting on colleague's three src/ files and
  P02–P07 data. See Planned Phases above.
- **`num_probe_blocks: 2 or 4`:** Published V-JEPA 2 uses 4 blocks. A more powerful probe may
  reveal richer structure that a 1-block probe misses. **Caveat:** the 6b/6c controls are
  validated only for `num_probe_blocks: 1` (permutation-invariant: cross-attention, no positional
  encoding). At depth ≥ 2 the pooler may introduce token interactions where count/position matters
  differently — re-verify pooler behavior before treating 6b/6c as valid controls at higher depth
  (see DECISIONS.md D2).
- **Signal injection at eval:** If EK100 participants overlap with HD-EPIC (P01–P07), actual gaze/hand
  could be provided at eval instead of mask tokens. Note: GATE 1 means masked eval is already
  in-distribution, so this tests *additional* signal rather than fixing an OOD problem.
- **Gaze correlation metric:** Measure spatial correlation between the model's feature map attention
  and the gaze fixation heatmap. Label-free, independent of action classes — a direct probe of what
  the model has learned to attend to.
