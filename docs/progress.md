# Progress Log — Gaze/Hand Conditioning for V-JEPA 2

_Last updated: 2026-05-31_

> See also: [usage.md](usage.md) (how to run every script) · [experiment_design.md](experiment_design.md) (the design rationale).

## Goal

I'm testing whether human **gaze and hand-tracking** signals (from Aria glasses, HD-EPIC dataset) can replace the **robot-action** conditioning in the V-JEPA 2-AC predictor. The original predictor was trained on DROID robot actions; I want to feed it egocentric-native signals instead and see if they help it predict future video.

My test is a clean A/B on the *same* fine-tuned model:
- **Condition A (masked):** signals turned off → model uses its learned mask tokens → `MSE_A`
- **Condition B (real):** real gaze + hand fed in → `MSE_B`
- **Δ = MSE_A − MSE_B.** Positive means the signals help.

---

## What I did this session

### 1. Reviewed my own pipeline and found a critical bug
I went back through the design and the V-JEPA 2-AC source (`app/vjepa_droid/train.py` + the droid config) and realised my evaluation was feeding the predictor the **wrong kind of input**:

- The AC predictor is a **world model over per-frame latents** — each frame is encoded *independently* (duplicated into a 2-frame tubelet) with the **EMA `target_encoder`**, and reps are **LayerNorm'd** (`normalize_reps: true`) before the loss. The world model steps at ~4 fps (8 frames), not on adjacent frames.
- My first version encoded a whole 16-frame clip jointly (cross-frame attention) — out-of-distribution for the predictor.

Fixing this dropped my baseline feature-prediction MSE from **4.82 → 0.57** on P08, because the predictor finally got the inputs it was trained on.

### 2. Implemented the fixes
- **Per-frame-independent encoding** with the target_encoder + frame striding (now the default).
- **`normalize_reps`** applied to predictions and targets.
- **VRS-timestamp alignment in eval** (it had been using a drifting frame/fps lookup while training used absolute VRS time — they now share one code path).
- **Paired A/B evaluation**: both conditions scored on the *same* clips in one pass, so Δ isn't polluted by sampling noise.
- **Signal dropout (0.4)** during training so the mask tokens are well-trained and Condition A is a *fair* baseline (otherwise MSE_A is inflated and Δ is overstated).
- **Input standardisation** for gaze/hand, **bf16 AMP**, and **all-step supervision** (every step predicts the next, not just the last).
- Pulled the shared logic into `scripts/ego_common.py` so train and eval can't drift apart again.

### 3. Sorted out the gaze/hand data
- The `GAZE_HAND` zips finished downloading for **all of P01–P09**.
- Found a bug: my discovery looked for `eye_gaze.csv` / `hand_tracking_results.csv`, but the Aria MPS files are actually `eye_gaze/general_eye_gaze.csv` and `hand_tracking/wrist_and_palm_poses.csv`. Fixed `find_csvs`.
- Extracted all the zips and verified the loader columns match the real CSV headers.
- **154 of 156 recordings have gaze + hand** (2 P02 recordings simply have no MPS data on the portal — they're auto-skipped). Train split **P01–P07** and eval split **P08–P09** are both ready.

### 4. Validated that fine-tuning actually works
- Added a **per-epoch held-out validation** (MSE_A / MSE_B / Δ on P08) so I can watch whether the signals start helping, not just whether the loss drops.
- First training run **OOM'd** — turned out to be GPU contention from another job plus my encode pushing 64 frames through ViT-g at once. Fixed by **chunking the encoder forward** (bounded memory).
- The quick check showed the right behaviour: held-out **Δ moved from −0.0082 (untrained) → −0.0003 after one epoch**, while MSE_A dropped 0.615 → 0.498 (domain adaptation). So the projectors are starting to make the real signals useful. The direction is exactly what "it's learning" should look like.

### 5. Made it use the GPU properly
- The run was at **100% GPU util but the frozen ViT-g encoder was running in fp32** — the dominant cost, wasted. Moved the encoder under **bf16 autocast** (~2× throughput) and added an `--encode-chunk` knob + bigger batch.
- Throughput went from **~1.96 → ~4.0 clips/s**.

### 6. Better logging / monitoring
- `scripts/watch_progress.py` — a live terminal dashboard (loss sparkline, the MSE_A/MSE_B/Δ table, throughput, GPU stats). It reads the log, so it works on an already-running job.
- `scripts/summarize_run.py` — dumps a run to JSON + Markdown (per-epoch loss, times, Δ trend, config).
- Added a machine-readable `metrics.jsonl` per run, and kept the per-step **log line clean** (loss / grad / lr / ETA).

---

## Which layers I'm fine-tuning

I keep the frozen encoder (~1 B params) and the predictor's low-level layers fixed, and only adapt the parts that need to understand the new signals:

| Layer | Status |
|---|---|
| `gaze_proj` (3→1024), `hand_proj` (12→1024), `gaze_mask`, `hand_mask` | **train** @ lr 1e-3 (new, replace robot `action_encoder`/`state_encoder`) |
| `predictor_blocks[18–23]` (last 6 of 24) | **train** @ lr 1e-4 |
| `predictor_norm`, `predictor_proj` | **train** @ lr 1e-4 |
| `predictor_blocks[0–17]`, `predictor_embed` | frozen |
| ViT-g encoder (all) | frozen |

**Trainable = 77 M / 305 M predictor params (25.2%).** The token layout stays `[gaze, hand, visual…]` (`cond_tokens = 2`), identical to the AC `[action, state, visual…]`, so all 24 transformer blocks transfer directly from the pretrained checkpoint.

---

## Current numbers

- **Baseline (untrained) Δ:** MSE_A 0.6153 / MSE_B 0.6235 / **Δ = −0.0082** (signals hurt before training, as expected).
- **After 1 epoch:** MSE_A 0.4975 / MSE_B 0.4977 / **Δ = −0.0003**.
- **Throughput (bf16, batch 16):** ~4 clips/s, ~**16 min/epoch** on the full P01–P07 (129 recordings × 30 clips).
- **Time to fine-tune full P01–P07:** ~2.2 h for 8 epochs at 30 clips/rec (≈4.3 h at 60 clips/rec).
- **Checkpoint size:** currently 1.22 GB (full predictor, fp32). Can drop to ~150 MB (trainable-only, fp16) or ~90 MB with fewer unfrozen blocks.

A run on the full P01–P07 (bf16, batch 16) is training now.

---

## Results so far

First full fine-tune on **P01–P07** (3 epochs, bf16, batch 16, signal_dropout 0.4), held-out validation on P08 (96 paired clips, fixed). Total 40.7 min (~12 min/epoch).

| epoch | train loss | MSE_A (masked) | MSE_B (real) | Δ = A−B |
|---|---|---|---|---|
| 0 (untrained) | — | 0.6153 | 0.6235 | −0.0082 (hurts) |
| 1 | 0.5261 | 0.4942 | 0.4933 | +0.0009 (helps) |
| 2 | 0.5150 | 0.4900 | 0.4885 | +0.0015 (helps) |
| 3 | 0.5066 | 0.4877 | 0.4866 | +0.0011 (helps) |

**Takeaway:** Δ flipped from negative (signals hurt through random projectors) to **positive after one epoch and stayed there** — the gaze/hand projectors learned to make the real signal a useful cue. The effect is small (~0.2% of MSE), as expected at a ~2 s one-step horizon where vision alone is already strong. This is a functional proof-of-concept, not yet a significance-tested headline number — that needs the longer run + full eval + a paired test (below). Checkpoint: `checkpoints/ego_ft_v2/best.pt` (epoch 3, train loss 0.5066).

Figure: `results/ego_ft_v2_results.png` (train loss · MSE_A vs MSE_B · Δ-with-zero-baseline), produced by `scripts/plot_results.py`. Run summary: `results/ego_ft_v2_summary.md`.

### Data status (2026-05-31)
GAZE_HAND downloaded + extracted for **all P01–P09** — **154/156 recordings** have gaze + hand (2 P02 recordings have no MPS data on the portal, auto-skipped). Train split **P01–P07** and eval split **P08–P09** both ready. Aria MPS files are `eye_gaze/general_eye_gaze.csv` and `hand_tracking/wrist_and_palm_poses.csv`.

## Open items / next

- [ ] Let the full P01–P07 run finish and read the Δ trend across 3 epochs.
- [ ] Slim the checkpoint to **trainable-only** (eval reloads the frozen base from the AC checkpoint).
- [ ] Add **per-participant valid-signal tracking** so I can see which users actually update the gaze/hand projectors (a teammate hit a case where one user's missing signal meant its gradients never updated the weights — I want that to be visible, not silent).
- [ ] Recompute the gaze/hand standardisation stats from real P01–P07 data (current constants are approximate).
- [ ] Consider a **longer prediction horizon** — at ~2 s ahead, vision alone is already strong, so the gaze/hand effect may be small; predicting further out should make the signal matter more.

---

## Key files

| File | What it does |
|---|---|
| `scripts/extract_gaze_hand.py` | Extract gaze/hand CSVs from the MPS zips (in place) |
| `scripts/ego_common.py` | Shared: model loading, per-frame encoding, VRS alignment, CSV discovery |
| `scripts/finetune_ego.py` | Fine-tunes the predictor on gaze/hand (the main training script) |
| `scripts/eval_ego_mse.py` | Paired A/B evaluation (MSE_A vs MSE_B vs Δ) |
| `scripts/watch_progress.py` | Live training dashboard |
| `scripts/summarize_run.py` | Run → JSON + Markdown summary |
| `scripts/plot_results.py` | Run → PNG figures (loss · MSE_A/B · Δ) |
| `vjepa2/src/models/ego_predictor.py` | The ego predictor (gaze/hand replace action/state) |
| `vjepa2/src/models/ego_finetune.py` | Freeze strategy, weight transfer, layer reporting |
| `vjepa2/src/datasets/ego_loaders.py` | Gaze / hand CSV loaders + standardisation |
| `docs/usage.md` | How to run every script (commands + args) |
| `docs/experiment_design.md` | Full experiment design |
