# Decision Log — V-JEPA 2 Ego Project

Append-only. Each entry records: the decision or reversal, what belief was superseded, and
the evidence that forced the change. The value is the diff — never rewrite, never delete.

---

## 2026-06-17

### D1 — Phase 7 redefined: robot AC predictor control retired

**Superseded belief:** Phase 7 would use the original robot AC predictor
(`checkpoints/ac/vjepa2-ac-vitg.pt` → `predictor` key) as a no-signal control, with a learned
`action_mask` token substituted at eval to match Phase 5's `gaze_mask`/`hand_mask` protocol.

**New decision:** Phase 7 is redefined as a matched retrain of the colleague's architecture
(`IntegratedColleaguePredictor`) with `signal_dropout: 1.0` — every frame receives mask tokens
during training, so the predictor never sees real behavioral signals.

**Evidence forcing the change:** inspection of `VisionTransformerPredictorAC.forward()` in
`src/models/ac_predictor.py` showed the signature is `forward(x, actions, states)` — it requires
real 7D robot vectors. There is no `action_mask` parameter and no trained absent-signal token in
the checkpoint. The robot AC predictor therefore cannot run the masked-eval protocol at all;
there is no valid substitution that keeps the architecture identical to Phase 5.

**Status of old design:** retired entirely. "Robot AC predictor as control" appears nowhere in
active plans. Phase 7 as redefined is deferred pending colleague's training pipeline and P02–P07
HD-EPIC data (not held locally).

---

### D2 — Token-count confound found; Phase 6 refactored into 6a/6b/6c

**Superseded belief:** Phase 5 (treatment) vs Phase 6 (control) was a clean comparison — same
encoder, same probe, different predictor. Phase 6 returned `encoder(x)` → [B, 1568, 1408].

**New decision:** Phase 6 refactored into a family:
- 6a — encoder only, 1568 tokens (raw baseline; this was the old Phase 6)
- 6b — encoder + 196 zeros appended, 1764 tokens (token-count control)
- 6c — encoder + last encoder frame repeated, 1764 tokens (persistence baseline)

Phase 5, 6b, and 6c all produce 1764 tokens. Only 6a is 1568.

**Evidence forcing the change:** reading `ColleagueEgoWrapper.forward()` in `vit_colleague_ego.py`
showed Phase 5 returns `cat([x_enc (1568), last_pred (196)])` = 1764 tokens, while the old
Phase 6 returned 1568. The AttentivePooler is permutation-invariant (content-only cross-attention,
no positional encoding) so extra tokens, regardless of content, may inflate the score simply by
giving the pooler more keys to attend over. This makes the gap (5 − old 6) uninterpretable.

**Token counts verified:** `out.shape[1]` asserts in `vit_vitg_6b.py` and `vit_vitg_6c.py` fire
at runtime if 6b/6c diverge from `x_enc.shape[1] + N`.

**Consequence for prior results:** the Phase 5 preview numbers (2.81–5.83%, seed=0) were obtained
against the old 1568-token Phase 6. They are superseded and archived as preview-only. Countable
results are the 12-run seeded experiment (seeds 42/43/44) against 6a/6b/6c.

**Scope of the permutation-invariance guarantee:** the 6b/6c controls rest on the AttentivePooler
being permutation-invariant — verified for `num_probe_blocks: 1` only (`depth=1` → `self.blocks`
is `None`, only a single cross-attention with no positional encoding on keys/values). This
guarantee does NOT automatically extend to `num_probe_blocks: 2 or 4`. At depth ≥ 2, the pooler
applies self-attention blocks on the query tokens after the cross-attention step; those blocks
could learn position-sensitive behaviour depending on how keys/values are structured. Before
trusting 6b/6c as valid controls in any experiment with higher probe depth, re-read
`AttentivePooler.forward()` in `src/models/attentive_pooler.py` and confirm the same invariance
holds for the new depth.

---

### D3 — GATE 1 resolved: masked eval is in-distribution

**Superseded framing:** "mask token mismatch still a factor" — the idea that substituting
`gaze_mask`/`hand_mask` at EK100 eval introduces an OOD condition because the predictor was
trained on real behavioral signals. Phrased in CLAUDE.md analysis as: "At EK100 eval, the
predictor receives gaze_mask/hand_mask tokens (no-signal substitutes). Yet the representation
is still richer than plain ViT-L. This suggests the behavioral conditioning adds structure even
without live signals."

**New finding:** the colleague predictor was trained with `signal_dropout: 0.4` — 40% of frames
receive mask tokens during training (from `config` key in `checkpoints/colleague2/checkpoint.pt`).
Masked eval (100% mask tokens) is therefore in-distribution, not OOD. The predictor was
explicitly trained to predict under full masking as part of its normal regime.

**Evidence:** `checkpoints/colleague2/checkpoint.pt` → `config` key → `signal_dropout: 0.4`.

**Consequence:** the "mismatch" framing is factually wrong and has been removed from CLAUDE.md
analysis. The "mask tokens are informative despite mismatch" interpretation is also removed:
the tokens are in-distribution by design, not informative despite mismatch.

**What this enables:** the horizon sweep is interpretable — the model was trained to predict
under partial and full signal absence, so varying anticipation horizon at eval is not confounded
by an OOD regime shift.

---

### D4 — Two framings removed from CLAUDE.md (preserved here verbatim)

The following phrases appeared in CLAUDE.md's "Analysis — Phase 5 now beats Phase 2" section
and were removed during the 2026-06-17 update. They are preserved here because DECISIONS.md is
where superseded beliefs belong.

**Verbatim superseded text #1:**
> "Mask token mismatch still a factor: At EK100 eval, the predictor receives `gaze_mask`/`hand_mask`
> tokens (no-signal substitutes). Yet the representation is still richer than plain ViT-L. This
> suggests the behavioral conditioning adds structure even without live signals."

**Reason removed:** (a) The "mismatch" label was wrong — see D3, masked eval is in-distribution.
(b) The conclusion "behavioral conditioning adds structure" was not supported by the comparison:
Phase 5 used 1764 tokens, Phase 2 used ViT-L (different encoder). The token-count confound
(D2) and the encoder-size difference together make this inference invalid. The structure claim
requires Phase 7.

**Verbatim superseded text #2:**
> "behavioral conditioning adds structure even without live signals"

**Reason removed:** this is the exact overclaim that Phase 7 is needed to support. It was stated
as a finding; it is a hypothesis. The gap (5 − 6c), not (5 − Phase 2), is the evidence base.
Even that gap does not isolate behavioral conditioning — it shows prediction beats persistence,
which may be true regardless of what conditioned the predictor. Phase 7 (matched retrain,
`signal_dropout: 1.0`) is required to close this.

---

## 2026-06-22

### D5 — Seed-42 results: null/negative gaps at 2s horizon

**Date:** 2026-06-22

**Observation (provisional — one seed only, not a conclusion):**
Seed-42 best-val Act@5: Phase 5 = 3.58% (peak epoch 4), 6a = 3.7% (epoch 8),
6b = 3.6% (epoch 8/10), 6c = 3.8% (epoch 8).

Gaps: 5−6a = −0.1pp, 5−6b = 0.0pp, 5−6c = −0.2pp. All null or negative.

**Consistency with prior evidence:** matches colleague's paired A/B at ~2s horizon
(Δ = +0.001 MSE, ~0.2%) — two independent measurements now point the same direction.
The behavioral predictor does not clearly help at 2s anticipation horizon.

**Pattern to watch across seeds:** Phase 5 peaked at epoch 4 (tied epoch 8); all
three controls peaked at epoch 8. If this early-peak pattern repeats across seeds 43/44,
it may indicate the predictor's predicted latent block contains signal the probe latches
onto early but cannot sustain — a different failure mode than "predictor adds nothing."
Do not interpret until seeds 43/44 are in.

**Implication for horizon sweep:** null result at 2s is consistent with the hypothesis
being horizon-dependent. Colleague's note: "at ~2s, vision alone is already strong; longer
horizon should amplify the signal." Horizon sweep is now the primary interpretive experiment,
not an optional extra. Launch immediately after the 12-run set completes.

**Implication for Phase 7:** the null result makes Phase 7 (matched retrain,
`signal_dropout: 1.0`) more urgent, not less. A positive Phase 7 gap would confirm that
prediction itself helps even without behavioral signal — a prerequisite for interpreting any
future Phase 5 advantage. Do not collapse Phase 7 planning because seed 42 is null.

---

## 2026-07-15

### D6 — Seed-43 results confirm the null; early-peak pattern did not replicate; Phase 7 unblocked on code

**Date of underlying events:** seed-43 runs completed 2026-06-24; colleague src/ files
arrived 2026-06-23; recorded here 2026-07-15.

**Observation 1 — seed-43 results (still provisional, two of three seeds):**
Best-val Act@5: Phase 5 = 3.49% (ep8), 6a = 3.44% (ep8), 6b = 3.59% (ep8), 6c = 3.31% (ep6).
Two-seed gap means (best epoch per seed): 5−6a = −0.04pp, 5−6b = −0.06pp, 5−6c = −0.02pp.
All four variants sit within 3.3–3.8% on both seeds; between-variant differences are the
size of within-variant seed noise (6c swings 3.8 → 3.31 across seeds). The D5 null reading
at the 2s horizon is reinforced, not revised. Conclusion still requires seed 44 (mean ± std).

**Observation 2 — pattern flagged in D5 did not replicate:** seed-42 Phase 5 peaked at
epoch 4 while controls peaked at epoch 8; on seed 43 Phase 5 peaked at epoch 8 like the
controls. The "signal the probe latches onto early but cannot sustain" hypothesis is
weakened — the seed-42 early peak was most plausibly seed noise. Drop this thread unless
seed 44 revives it.

**Observation 3 — Phase 7 code dependency resolved:** the three missing src/ files
(`src/models/ego_predictor.py`, `src/models/ego_finetune.py`, `src/datasets/ego_loaders.py`)
arrived 2026-06-23 and are in the repo (untracked). The colleague pipeline is now runnable
locally in principle. Phase 7's sole remaining blocker is P02–P07 HD-EPIC data (only P01
local); preferred path remains the colleague running `--signal-dropout 1.0` on her machine.

**Process note — seed 44 launched on different hardware (2026-07-15):** Phase 5 and 6a
seed-44 probes launched on an H200 (95 GB) at `batch_size: 64`, vs L40S at `batch_size: 32`
for seeds 42/43. Data loading, not GPU memory, is the bottleneck; the 25-config LR×WD sweep
absorbs most of the effective-LR shift from the batch change. Nevertheless, seed 44 is not a
bit-identical replica of the seed-42/43 protocol — note this in any writeup if seed 44 is an
outlier. 6b/6c seed-44 launches pending.
