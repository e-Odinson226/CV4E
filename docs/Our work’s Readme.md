---
date: 2026-05-31
---
## integrate Ioana’s code

### Data loaders — functionally the same, just merged

|Archive|My code|
|---|---|
|`gaze_frame.py` — `GazeTokenLoader`|ego_loaders.py — `GazeTokenLoader`|
|`hand_frame.py` — `HandTokenLoader`|ego_loaders.py — `HandTokenLoader`|

No logic change. Just consolidated into one file in the right package location.

---

### Projectors — absorbed into the predictor, not standalone

The archive `GazeZProjector` and `HandZProjector` are **external** `nn.Module`s you'd call before passing anything to the predictor:

`# archive approach — two separate modules, called outside the predictor gaze_token = GazeZProjector()(gaze_vec, gaze_valid) # (B, 1024) hand_token = HandZProjector()(hand_vec, left_v, right_v) # (B, 1024) predictor(encoder_tokens, gaze_token, hand_token) # you'd have to modify predictor too`

In ego_predictor.py, the projection is **internal** — `_encode_gaze()` and `_encode_hand()` live inside the predictor and handle the full `(B, T, D)` temporal dimension, not just a single frame:

`# new approach — one call, everything is handled inside predictor(encoder_tokens, gaze_vecs, gaze_valid, hand_vecs, left_valid, right_valid)`

Three concrete differences in the projectors:

||Archive `GazeZProjector` / `HandZProjector`|`ego_predictor.py` internal|
|---|---|---|
|Output shape|`(B, 1024)` — single frame|`(B, T, D)` — full clip|
|`predictor_embed_dim`|Hardcoded `1024`|Parametric — matches whatever the predictor uses|
|Scope|Standalone module, called externally|Private method, fused into the predictor's `forward`|

### ego_finetune.py — entirely new, no archive equivalent

The archive had no fine-tuning logic at all. ego_finetune.py adds:

- `freeze_for_ego_finetune()` — freezes all but new projectors + last N blocks
- `get_ego_finetune_param_groups()` — two LR groups (new params vs. pretrained params)
- `load_ac_weights_into_ego()` — transfers the pretrained AC checkpoint into the ego predictor
- `trainable_parameter_summary()` — diagnostic to verify what's frozen

---

---

## What to measure

**Metric:** Feature prediction loss — L2 distance between the predictor's output and the actual encoder features on held-out egocentric clips. Lower = better predictions.

`loss = F.mse_loss(predicted_features, target_encoder_features)`

This is the direct signal V-JEPA 2 is trained on, so it's the cleanest comparison.

---

## Baselines to run, in order

Each one isolates exactly one variable:

|#|Name|What it isolates|Model|
|---|---|---|---|
|B0|No conditioning|Does any conditioning help at all?|`predictor.py` (standard V-JEPA 2)|
|B1|AC predictor, zeroed actions|Is the AC architecture useful on ego data without signals?|`ac_predictor.py`, actions=0|
|B2|Ego predictor, random init|Is the AC pretraining transfer valuable?|`ego_predictor.py`, no `load_ac_weights_into_ego`|
|B3|Ego predictor, AC weights, projectors only|Do the blocks need to adapt, or is the signal enough?|`ego_predictor.py` + `load_ac_weights_into_ego` + freeze all blocks|
|**H**|**Ego predictor, AC weights, last N blocks fine-tuned**|**Your hypothesis**|`ego_predictor.py` + full `freeze_for_ego_finetune`|

---

## Concrete code for each baseline

**B0 — standard predictor, no conditioning:**

`from src.models.predictor import vit_predictor predictor = vit_predictor(embed_dim=1024, predictor_embed_dim=1024, depth=12, ...) # load pretrained vjepa2 checkpoint, evaluate prediction loss`

**B1 — AC predictor, zeroed actions:**

`from src.models.ac_predictor import vit_ac_predictor predictor = vit_ac_predictor(...) # load pretrained AC checkpoint actions = torch.zeros(B, T, 7).to(device) states = torch.zeros(B, T, 7).to(device) out = predictor(encoder_tokens, actions, states)`

**B2 — ego predictor, random init (no AC weights):**

`from src.models.ego_predictor import vit_ego_predictor predictor = vit_ego_predictor(embed_dim=1024, predictor_embed_dim=1024, depth=24, ...) # do NOT call load_ac_weights_into_ego — train from scratch`

**B3 — ego predictor, AC weights, projectors only:**

`from src.models.ego_predictor import vit_ego_predictor from src.models.ego_finetune import load_ac_weights_into_ego, freeze_for_ego_finetune predictor = vit_ego_predictor(...) load_ac_weights_into_ego(predictor, ac_ckpt['predictor']) freeze_for_ego_finetune(predictor, unfreeze_last_n_blocks=0) # blocks all frozen`

**H — your hypothesis:**

`load_ac_weights_into_ego(predictor, ac_ckpt['predictor']) freeze_for_ego_finetune(predictor, unfreeze_last_n_blocks=6) # adapt last 6 blocks`

---

## What the result pattern tells you

`B0 > B1 → AC architecture doesn't generalize to ego data at all B1 > B2 → AC pretraining transfer is valuable even with wrong signals B2 > B3 → Block adaptation matters more than the signal B3 > H → More fine-tuning hurts (overfitting) — reduce unfreeze_last_n_blocks H < all → Gaze+hand signals actively harm prediction — rethink signal design H best → Hypothesis confirmed ✓`

---

## Practical suggestion

Start with B1 and H only — they're the fastest to run and give you the most important signal: does your new conditioning outperform the AC predictor with no signal on ego data? If yes, then run the full ladder to understand why.

---

|Option|What you measure|Labels needed|Status|
|---|---|---|---|
|**A. Feature prediction loss on HD-EPIC**|How well the predictor reconstructs future encoder features|None|Can run now (videos exist)|
|**B. EK-100 action anticipation**|Class-mean recall on verb/noun/action anticipation|Yes (you have them)|Can run now|

**Recommendation:** Run **both**. B gives you a number comparable to the published V-JEPA 2 paper. A gives you a direct self-supervised signal on the same domain where your gaze conditioning will eventually run.

---

---

  

  

  

## What to run right now (Option B — EK-100 baseline)

You have everything for this already. Fill in configs/eval/vitg-384/ek100.yaml:

`folder: /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric/results/baseline-ek100 model_kwargs: checkpoint: /path/to/vjepa2_vitg.pt pretrain_kwargs: data: base_path: /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric/data/ek100/videos/ dataset_train: /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric/data/ek100/EPIC_100_train.csv dataset_val: /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric/data/ek100/EPIC_100_validation.csv`

Then:

`cd vjepa2 python evals/main.py \ --fname configs/eval/vitg-384/ek100.yaml \ --debugmode True \ --devices cuda:0`

This gives you the published-protocol baseline number to beat.

---

## What to build next (Option A — HD-EPIC feature prediction loss)

Once the gaze zips are extracted, I can write a short evaluation script that:

1. Iterates over HD-EPIC MP4s
2. Uses `mp4_to_vrs_time_ns.csv` to align video frames to gaze timestamps
3. Runs the frozen encoder on observed frames
4. Runs the predictor (standard or ego) on context
5. Measures L2 loss against encoder features of future frames

---

---

  

I suggest we only compare the output of predictor,  
as 1. standard predictor (no conditioning, or zeroed gaze) vs 2. our Ego predictor (real gaze + hand)

in this way we eliminate any chance of performance degrade due to probe, so we can directly analyze the essence of the process which is predcitions in embeddings not the way we translate those embeddings to natural language

aslo we have introduced probe improvement as our secondary goals and our primary was introduction of gaze and … to predictor

---

---

|Component|Status|Needs training?|
|---|---|---|
|Gaze/hand vectors (the data)|Available from HD-EPIC|No — it's input|
|`gaze_proj` weights|Random (no pretrained version exists)|**Yes**|
|`hand_proj` weights|Random (no pretrained version exists)|**Yes**|
|Last N transformer blocks|Pretrained on robot actions (DROID)|**Yes** — adapt to gaze semantics|
|First 18 blocks + `predictor_embed`|Pretrained|No — freeze|

---

---

1. prepare a script to inference hd-epic without gaze or any additional... to get a baseline value to compare when gaze/hand points are fed to predictor  
2. fine-tune projection part weights and some predictor layers to be able to feed feature (gaze and...) to predictor  
3. get the results from inference with features fed to fune-tuned projector and predictor  
  
so at the end we would have:  
1. ground truth (rela frame fed to encoder)  
2. predicted frame without features introduced  
3. predicted frame with features introduced through fine-tuned projector and predictor

---

---

  

  

  

_"Does knowing where the person is looking and where their hands are improve prediction of the next frame, compared to a model of the same capacity trained on the same data but given no signal?"_

  

---

---

# Generate baseline data

**Step 1 — activate the right environment (run once)**

`source /mnt/data/home/zj2433/miniconda3/etc/profile.d/conda.sh && conda activate VJEPA2-AC`

**Step 2 — verify the checkpoint loads cleanly**

`cd /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric && python scripts/eval_ego_mse.py --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos --participants P08 --condition null --clips-per-video 3 --out results/smoke_test.csv`

**Step 3 — full Condition A baseline (P08 + P09)**

`python scripts/eval_ego_mse.py --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos --participants P08 P09 --condition null --clips-per-video 20 --out results/mse_condition_A.csv`

---

---

# fune tuning

  

1. **Loads** the pretrained V-JEPA 2-AC predictor (trained on DROID **robot actions**).
2. **Swaps the conditioning inputs**: the original `action_encoder` (7-dim robot action) and `state_encoder` are _thrown away_; in their place are `**gaze_proj**` (3-dim: yaw, pitch, depth) and `**hand_proj**` (12-dim: wrist+palm xyz × 2 hands) — both randomly initialized. Because the token layout stays `[signal, signal, visual…]` (`cond_tokens=2`), all 24 transformer blocks transfer directly.
3. **Trains** (the layers from your #1 question): `gaze_proj`, `hand_proj`, the mask tokens, the last 6 transformer blocks, and the output head — on real HD-EPIC gaze+hand → next-frame-feature prediction. The frozen encoder + first 18 blocks keep the pretrained visual knowledge.

---

on p01-07 with:

```
cd /mnt/data/home/zj2433/Projects/Ego/CV4Egocentric
rm -rf checkpoints/ego_ft_v2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /mnt/data/home/zj2433/miniconda3/envs/VJEPA2-AC/bin/python scripts/finetune_ego.py \
--checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
--video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
--gaze-dir data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
--participants P01 P02 P03 P04 P05 P06 P07 \
--val-participants P08 --val-recordings 4 --val-clips 24 \
--epochs 3 --clips-per-recording 30 \
--batch-size 16 --encode-chunk 48 --num-workers 8 \
--log-every 10 --save-every 3 --out-dir checkpoints/ego_ft_v2 2>&1 | grep -v FutureWarning | grep -v "warnings.warn\|self.gen"
```

**Layer logging (your #1) is now in the log:**

`[layers] TRAIN new: gaze_proj, hand_proj, gaze_mask, hand_mask [layers] TRAIN blocks 18-23 (last 6 of 24) + predictor_norm, predictor_proj [layers] FROZEN predictor_blocks[0:18], predictor_embed [layers] 82 trainable tensors 77,042,048/305,216,896 params (25.2%)`

---

|Layer|Tensors|Params|Status|
|---|---|---|---|
|`gaze_proj`|2|4,096|**train** (lr 1e-3, new)|
|`hand_proj`|2|13,312|**train** (lr 1e-3, new)|
|`gaze_mask`, `hand_mask`|2|2,048|**train** (lr 1e-3, new)|
|`predictor_blocks[18–23]` (last 6)|72|75,577,344|**train** (lr 1e-4)|
|`predictor_norm`|2|2,048|**train** (lr 1e-4)|
|`predictor_proj`|2|1,443,200|**train** (lr 1e-4)|
|`predictor_blocks[0–17]` (first 18)|—|~227M|❄️ frozen|
|`predictor_embed`|—|1.4M|❄️ frozen|
|**ViT-g encoder (all)**|—|~1.0B|❄️ frozen|

---

  

# MSE_A vs MSE_B

**The metric (shared by both):** feature-prediction MSE, no labels needed.

1. Encode the context frames, run the ego predictor → it outputs a _predicted embedding_ for the next step.
2. Separately encode the actual future frame with the (frozen) encoder → the _true embedding_.
3. `MSE = ‖predicted − true‖²` averaged over the embedding (the V-JEPA 2 training objective itself).

**The only difference between A and B:**

||Condition A|Condition B|
|---|---|---|
|Name|**null / masked**|**real signal**|
|gaze/hand input|`valid=False` everywhere → the model falls back to its learned `gaze_mask`/`hand_mask` tokens|real Aria gaze + hand vectors fed through `gaze_proj`/`hand_proj`|
|What it predicts from|**visual context only**|visual context **+ where the person looks / where their hands are**|
|Result|**MSE_A**|**MSE_B**|

So:

- **MSE_A** = how well the model predicts the next frame with **no signal** (baseline).
- **MSE_B** = how well it predicts **with gaze+hand**.

**Δ = MSE_A − MSE_B** is the whole point:

- **Δ > 0** → B has lower error → the signal _helps_ → your hypothesis holds.
- **Δ ≈ 0 / < 0** → the signal isn't helping (yet).

Both are scored on the _same_ clips in one pass (paired), so Δ isn't polluted by which random clips got sampled.

---

  

finetune_ego.py, does the fine-tuning and adapt the robot-action predictor to human gaze + hand:

1. **Loads** the pretrained V-JEPA 2-AC predictor (trained on DROID **robot actions**).
2. **Swaps the conditioning inputs**: the original `action_encoder` (7-dim robot action) and `state_encoder` are _thrown away_; in their place are `**gaze_proj**` (3-dim: yaw, pitch, depth) and `**hand_proj**` (12-dim: wrist+palm xyz × 2 hands) — both randomly initialized. Because the token layout stays `[signal, signal, visual…]` (`cond_tokens=2`), all 24 transformer blocks transfer directly.
3. **Trains** (the layers from your #1 question): `gaze_proj`, `hand_proj`, the mask tokens, the last 6 transformer blocks, and the output head — on real HD-EPIC gaze+hand → next-frame-feature prediction. The frozen encoder + first 18 blocks keep the pretrained visual knowledge.

That's the whole "replace robot action conditioning with egocentric signals" idea, and it's training on P01–P07 as we speak. The thing that proves it's learning is the **training** `**loss**` **going down** (0.58 → 0.52 …).

## Evaluation (just a progress check)

The `MSE_A` / `MSE_B` lines are **not** the training — they're a periodic _test_, run on held-out P08, asking "now that you've trained a bit, do the real signals actually help predict better than no signal?" It's how we measure whether the fine-tuning succeeded, not how it learns.

So the flow is:

`finetune_ego.py → trains the predictor on gaze+hand → produces best.pt (THIS is your goal) eval_ego_mse.py → loads best.pt, measures MSE_A vs MSE_B on P08–P09 (did it work?)`

The current run interleaves both: it trains each epoch, then quickly evaluates (the `[val …]` line) so you can watch progress live. When it finishes you'll have a fine-tuned `checkpoints/ego_ft_v2/best.pt` — the gaze/hand-conditioned predictor you're after.

---

1. **Training loss** — drops 0.60 → 0.51, confirming the fine-tune optimizes cleanly.
2. **MSE_A vs MSE_B** — both crash from 0.62 → 0.49 (domain adaptation); they look almost identical at this scale, which is the honest picture.
3. **Δ = MSE_A − MSE_B** — the money panel: red fill (signals hurt, untrained) flips to green fill (signals help) after epoch 1, with values annotated.

## How to generate / present

`python scripts/plot_results.py --dir checkpoints/ego_ft_v2 # the 3-panel PNG above python scripts/plot_results.py --dir run_a run_b run_c # overlays the Δ curves to compare runs python scripts/summarize_run.py --dir checkpoints/ego_ft_v2 # the Markdown/JSON table for a report`

So you have three presentation assets, each for a different medium:

- **Figure** (`plot_results.py`) → slides / paper. Lead with the **Δ panel** — panel 2 makes A and B look identical, panel 3 reveals the real difference.
- **Table** (`summarize_run.py` → `.md`) → written report; drops straight into Markdown/Notion.
- **Live dashboard** (`watch_progress.py`) → for monitoring while training, not for presenting.

  

  

  

---

## In progress (just wait)

- [x] **Finish the current 3-epoch run** (~15 min) and read the Δ trend across all 3 epochs. If it keeps climbing, that's a clean "signals help, and more so with training" result.

## Core — to get a publishable result

- [ ] **Run the real fine-tune**: more epochs (8) on full P01–P07, denser coverage (`-clips-per-recording 60`). ~4 h. The 3-epoch run is a sanity check; this is the actual model.
- [ ] **Full paired eval on P08–P09** with `eval_ego_mse.py --predictor-checkpoint best.pt` — many clips, all recordings, not just the 4×24 in-loop val.
- [ ] **Statistical test**: per-clip paired Δ + a Wilcoxon/t-test, so the result is "Δ>0 with p<…", not just a mean. (Need to add the per-clip dump to the eval — small change.)

## Robustness / rigor (reviewers will ask)

- [ ] **Longer prediction horizon.** At ~2 s ahead, vision alone is strong, so Δ is small. Predicting further out (bigger stride / multi-step) should make gaze/hand matter more — and is the stronger test of the hypothesis.
- [ ] **Control: shuffled-signal.** Train/eval with gaze/hand temporally shuffled (signal destroyed, distribution kept). If real beats shuffled, the gain is genuinely from the _information_, not just extra capacity.
- [ ] **Ablations**: gaze-only vs hand-only vs both; `-unfreeze-last-n` sweep.

## Engineering polish (quick, optional)

- [ ] **Slim the checkpoint** to trainable-only (1.22 GB → ~150 MB).
- [ ] **Per-participant valid-signal tracking** (your teammate's "user 3" issue — make it visible which users actually update the projectors).
- [ ] **Recompute gaze/hand standardisation stats** from real P01–P07 (current constants are approximate).
- [ ] Add a **Results** section to `docs/progress.md` once the run finishes.

The single most important next step is the **8-epoch full fine-tune → full P08–P09 eval → significance test** — that's the actual answer to your research question. Everything else is rigor and polish.