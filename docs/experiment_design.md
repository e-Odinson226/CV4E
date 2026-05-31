# Experiment Design: Gaze & Hand Conditioning in Egocentric Video Prediction

## Hypothesis

Gaze and hand tracking signals from egocentric cameras can improve future frame prediction in V-JEPA 2, replacing robot action conditioning with egocentric-native signals.

---

## Pipeline

```
HD-EPIC video clip
       │
       ▼
  ┌─────────┐        context frames (t=1..T-1)
  │ Encoder │ ──────────────────────────────────► encoder tokens  ─────────┐
  └─────────┘                                                               │
                                                                            ▼
  Aria MPS CSVs                                                    ┌──────────────┐
  eye_gaze.csv   ──► GazeTokenLoader ──► [yaw, pitch, depth]  ──► │ Ego Predictor│ ──► predicted embedding
  hand_tracking  ──► HandTokenLoader ──► [wrist/palm xyz x4] ──►  └──────────────┘
                                                                            │
                                                                            │  MSE
  ┌─────────┐       future frame (t=T)                                     │
  │ Encoder │ ──────────────────────────────────► ground truth embedding ◄─┘
  └─────────┘
```

The encoder and both embeddings exist in the same latent space — **no labels are needed**. The loss is the V-JEPA 2 training objective itself.

> **Methodology note (aligned with the original V-JEPA 2-AC droid training).** To keep the pretrained predictor in-distribution and make the numbers meaningful, the pipeline mirrors `configs/train/vitg16/droid-256px-8f.yaml` + `app/vjepa_droid/train.py`:
> - **Per-frame-independent encoding.** Each step is encoded *alone* (one frame duplicated into a 2-frame tubelet), not as one jointly-attended clip — the AC predictor was trained on per-frame latents.
> - **Target encoder.** The EMA `target_encoder` is used for both context and target.
> - **Frame striding.** The world model steps at ~4 fps (8 steps, stride ≈ 8 frames @30 fps), so each step is a real transition rather than a ~33 ms no-op.
> - **`normalize_reps`.** Reps are LayerNorm'd before the loss.
> - **All-step supervision.** Step *t* predicts step *t+1* (teacher forcing) during fine-tuning.
>
> Train and eval share one module (`scripts/ego_common.py`) so the gaze-alignment / encoding paths cannot diverge.

---

## Two Conditions (Same Model, Same Fine-Tuning)

Both conditions use the **same fine-tuned ego predictor** on the same HD-EPIC data. The only difference is what flows through the conditioning tokens at inference time.

### Condition A — Null Signal (Baseline)

```python
gaze_valid       = torch.zeros(B, T, dtype=torch.bool)  # all invalid
hand_left_valid  = torch.zeros(B, T, dtype=torch.bool)
hand_right_valid = torch.zeros(B, T, dtype=torch.bool)

out_A = ego_predictor(enc_tokens, gaze_vecs, gaze_valid,
                      hand_vecs, hand_left_valid, hand_right_valid)
```

The predictor falls back to its learned **mask tokens** (`gaze_mask`, `hand_mask`) — constant vectors that carry no per-frame information. The model predicts using visual context only.

> **Fair-baseline requirement.** For Condition A to be a fair "no signal" baseline rather than an out-of-distribution surprise, the mask tokens must be well trained. Fine-tuning therefore applies **signal dropout** (`--signal-dropout 0.4`): on a fraction of clips all signals are masked, so the model genuinely learns a no-signal mode. Without this, MSE_A is inflated and ΔMSE overstates the effect.

### Condition B — Real Gaze + Hand Signals

```python
out_B = ego_predictor(enc_tokens, gaze_vecs, gaze_valid_real,
                      hand_vecs, hlv_real, hrv_real)
```

Real sensor readings from Aria MPS flow through `gaze_proj` and `hand_proj` into the transformer. The model sees where the person is looking and where their hands are, for each frame.

---

## What the Comparison Measures

> *Does knowing where the person is looking and where their hands are improve prediction of the next frame, compared to a model of the same capacity trained on the same data but given no signal?*

$$\Delta\text{MSE} = \text{MSE}_A - \text{MSE}_B$$

A positive $\Delta$ means gaze + hand conditioning reduces prediction error — the hypothesis holds.

**Paired evaluation.** Both conditions are scored on the *same* sampled clips in a single pass (`eval_ego_mse.py` encodes context once, then runs the predictor twice — null vs real). This yields a per-clip Δ, removing sampling noise and enabling a paired significance test (Wilcoxon / t-test) rather than comparing two independently-sampled means.

---

## Token Layout

The ego predictor uses the same token layout as V-JEPA 2-AC, with gaze and hand replacing action and state:

```
Per frame:  [ gaze_token | hand_token | visual_tokens ... ]
                  ↑             ↑
             (3-d → 1024-d) (12-d → 1024-d)
             gaze_proj       hand_proj
```

`cond_tokens = 2` in both models — all 24 transformer block weights transfer directly from an AC checkpoint.

---

## Fine-Tuning Strategy

Starting from a pretrained V-JEPA 2-AC checkpoint:

| Layer | Status | Reason |
|---|---|---|
| `gaze_proj`, `hand_proj`, `*_mask` | **Train** @ `lr=1e-3` | Randomly initialised — must learn from scratch |
| Last 6 transformer blocks | **Train** @ `lr=1e-4` | Pretrained — adapt to egocentric signals |
| `predictor_norm`, `predictor_proj` | **Train** @ `lr=1e-4` | Output head needs re-calibration |
| `predictor_embed` | Frozen | Maps encoder tokens — no ego-specific change needed |
| First 18 transformer blocks | Frozen | Low-level feature processing stays fixed |

Loss during fine-tuning: `MSE(norm(predictor_output), norm(target_encoder(next_step)))`, summed over **all** steps (step *t* → step *t+1*).

Additional fine-tuning details:
- **Input standardisation.** Gaze `[yaw, pitch, depth]` and hand `[xyz ×4]` are z-scored (`ego_loaders.GAZE_MEAN/STD`, `HAND_STD`) so depth/translation (metres) don't dominate the fresh projectors over yaw/pitch (radians). Stats are approximate — recompute from P01–P07 once data lands.
- **AMP.** Forward runs under `torch.autocast(bfloat16)`; the frozen ViT-g encoder runs under `no_grad`.

---

## Data

**Dataset:** HD-EPIC — 41 hours of Aria egocentric kitchen recordings, 9 participants (P01–P09)

| Split | Participants | Purpose |
|---|---|---|
| Fine-tuning | P01–P07 | Train projectors + last 6 blocks |
| Evaluation | P08–P09 | Measure MSE for both conditions |

**Sensor files per recording:**

Each `GAZE_HAND/mps_<rec>_vrs.zip` extracts to the Aria MPS standard layout:

| File (inside zip) | Content | Used for |
|---|---|---|
| `eye_gaze/general_eye_gaze.csv` | `tracking_timestamp_us, left/right_yaw_rads_cpf, pitch_rads_cpf, depth_m` | Condition B gaze tokens |
| `hand_tracking/wrist_and_palm_poses.csv` | `tracking_timestamp_us, left/right_tracking_confidence, tx/ty/tz_{left,right}_{wrist,palm}_device` | Condition B hand tokens |
| `<rec>_mp4_to_vrs_time_ns.csv` (next to mp4) | Frame index → absolute VRS device ns | Aligning video frames to MPS by timestamp |

> **Download status (2026-05-31):** GAZE_HAND is downloaded, valid, and **extracted for all participants P01–P09** (154/156 recordings; 2 P02 recordings have no MPS data on the portal and are auto-skipped). Full train split **P01–P07** and eval split **P08–P09** are ready. (`find_csvs` matches the MPS filenames above; the loaders' column parsing matches both CSV headers verbatim.)

---

## Evaluation Metric

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \| \hat{z}_i - z_i \|^2$$

where $\hat{z}_i$ is the predictor's output embedding and $z_i$ is the encoder's direct embedding of the ground truth future frame. Computed per clip, averaged over the evaluation split.

---

## Roadmap

```
Step 1 — Baseline sanity check (no fine-tuning)
    eval_ego_mse.py on AC weights, per-frame encoding + normalize_reps
    → MSE_A ≈ 0.57 on P08 (corrected pipeline; the old 4.82 used the
      wrong joint-clip encoding and no rep normalisation)

Step 2 — Download GAZE_HAND zips  (currently 0 bytes — BLOCKER for Condition B)

Step 3 — Fine-tune ego predictor on P01–P07
    Transfer AC weights → freeze strategy above → train projectors + last 6 blocks
    Per-frame encoding, all-step loss, signal_dropout=0.4, AMP

Step 4 — Paired evaluation on P08–P09
    Per clip: MSE_A (signals masked) and MSE_B (real signals), same frames
    → ΔMSE = MSE_A − MSE_B  +  paired Wilcoxon test
```
