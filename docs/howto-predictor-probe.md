# How to train a probe on a trained ego predictor

This note is for anyone who receives one of our trained predictors and wants to train a probe
on its features. It explains what the predictor is, what it needs as input, how to get its
output, and what to watch for. The code is in this repository; `README.md` at the root
describes the setup and every module.

## 1. What you receive

A checkpoint file, `best.pt`, of the **ego predictor**. It is the action-conditioned predictor
of V-JEPA 2-AC (Meta), fine-tuned on HD-EPIC kitchen videos. In each frame, the robot's action
and state tokens are replaced by a **gaze token** and a **hand token**.

The file contains the whole predictor: the 24 transformer blocks, the layers that turn gaze and
hand into tokens (`gaze_proj`, `hand_proj`), the learned tokens used when a signal is missing
(`gaze_mask`, `hand_mask`), and the output layers. It also stores the training options
(`config`), including the **gaze form**:

| Gaze form | How gaze enters |
|---|---|
| `angles` | yaw, pitch and depth through `gaze_proj`; the gaze token sits at the top-left patch |
| `pe` | sine and cosine features of the gaze point in the image, plus depth and a flag |
| `rope` | the `angles` token, placed at the gaze point in the attention |
| `pe+rope` | both |

Some runs are controls: `none` (trained without any signals), `future` (given the signals of
the frame it predicts) and `shuffled` (given the signals of another clip). Use them only as
comparisons, never as the model under study.

## 2. What you need besides the predictor

The predictor's own layers come with the checkpoint, so you do not need a separate projection
module. You need the same **inputs** as in training. Each item below must match; a mismatch
gives features that look plausible but are wrong.

1. **The encoder.** The frozen ViT-g encoder of V-JEPA 2-AC: the EMA target encoder
   (`target_encoder`) in `vjepa2-ac-vitg.pt`. `ego.model.load_models` loads it.
2. **The frame format.**
   - The whole frame resized to 256 × 256 pixels (no crop), RGB, ImageNet mean and standard
     deviation (`ego.data.load_frames`).
   - Each frame encoded **on its own**, as a 2-frame tubelet of the same image: 256 tokens of
     1408 numbers per frame (`ego.model.encode_independent`).
   - Layer norm over the 1408 numbers of each token (the default of `encode_independent`).
3. **The clip.** 8 frames, 8 video frames apart (30 fps, so 0.27 s apart, 2.1 s in all).
4. **The signals of each frame,** looked up by the frame's device timestamp from
   `<recording>_mp4_to_vrs_time_ns.csv`. `ego.signals.read(gaze_loader, hand_loader,
   timestamps, 8, projector)` builds them:
   - gaze: 5 numbers. Yaw, pitch and depth, scaled as `(x - GAZE_MEAN) / GAZE_STD`
     (`ego/data.py`), then the gaze point in the image as column and row on the 16 × 16 patch
     grid. The point needs the recording's camera calibration (`python -m ego
     fetch-calibrations`) and `projectaria-tools`. It is (-1, -1) without one.
   - hand: 12 numbers, the positions of the left wrist, left palm, right wrist and right palm
     (x, y, z in metres, relative to the glasses), divided by `HAND_STD`.
   - one validity flag for gaze and one per hand.
   - A missing signal is **marked invalid**, so the predictor uses its learned mask token. Do
     not pass zeros marked valid.
   - The point matters only for the `pe` and `rope` forms, but always pass it; the `angles`
     form ignores it.
5. **The right gaze form.** `ego.model.load_trained_predictor(path, device)` reads it from the
   checkpoint and builds the matching predictor, frozen and in eval mode.
6. **The output normalisation.** The predictor returns `(batch, 8 × 256, 1408)`: for each of the
   8 steps, the predicted embedding of the **next** frame. Apply layer norm
   (`ego.model.maybe_norm`) as in training. The last 256 tokens are the prediction of the frame
   0.27 s after the last context frame.

## 3. Setup

- Clone this repository and follow "Setup" in `README.md`: Meta's `vjepa2` at commit `204698b`
  in `vjepa2/`, the conda environment, and `pip install --no-deps projectaria-tools==2.3.0`.
- Data: the HD-EPIC videos and `SLAM-and-Gaze` folders (gaze and hand files), and the camera
  calibrations (`python -m ego fetch-calibrations`). The paths are in `README.md` under "Data".
- The V-JEPA 2-AC checkpoint `vjepa2-ac-vitg.pt` (for the encoder) and the predictor's
  `best.pt`.
- One GPU. Inference with the encoder and one predictor needs about 6 GB.

## 4. Example: the predictor's output for one clip

```python
import torch
from ego import signals
from ego.data import find_recordings, load_frames, open_loaders, read_vrs_times
from ego.gaze_geometry import projector_for
from ego.model import encode_independent, load_models, load_trained_predictor, maybe_norm

device = torch.device("cuda")
encoder, _, _ = load_models("data/model_checkpoints/vjepa2-ac-vitg.pt", device, context_steps=8)
predictor, cfg = load_trained_predictor("checkpoints/test9/rope_s0/best.pt", device)
print(cfg["gaze_form"])

rec = find_recordings("data/epic-kitchen/ek100-hd/HD-EPIC/Videos",
                      "data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze", ["P08"],
                      require_signal=True)[0]
vrs = read_vrs_times(rec.ts_csv)                 # device timestamp of every video frame
gaze_loader, hand_loader = open_loaders(rec)     # scaled as in training
projector = projector_for(rec)                   # the recording's camera calibration

start = 3000
idx = [start + 8 * i for i in range(8)]          # 8 frames, 0.27 s apart
frames = load_frames(rec.mp4, idx)               # (8, 3, 256, 256)
sig = signals.read(gaze_loader, hand_loader, [int(vrs[j]) for j in idx], 8, projector)

with torch.no_grad():
    z = encode_independent(encoder, frames.unsqueeze(0), device)        # (1, 2048, 1408)
    out = maybe_norm(predictor(z, *signals.as_batch(sig, device)))      # (1, 2048, 1408)
next_frame = out[:, -256:, :]    # predicted embedding of frame idx[-1] + 8, 16 x 16 tokens
```

`ego/commands/gaze_forms.py` does the same for many clips at once, with the encoder features
cached: a good model for a feature-extraction script.

## 5. What to probe, and the controls

- **Features.** The last step's prediction, `out[:, -256:, :]`: 256 tokens of 1408 numbers, the
  model's expectation of the frame 0.27 s ahead. An attentive probe (a learned query that
  attends over the 256 tokens, then a linear layer) fits this shape. The tokens are in row-major
  order on the 16 × 16 grid.
- **Controls** that tell what the predictor adds:
  - the same probe on the encoder features of the last context frame, `z[:, -256:, :]`: what
    the predictor adds beyond the current frame;
  - the same probe on the predictor with the signals hidden, `signals.mask_both(...)` on the
    batched signals: what gaze and hand add;
  - the same probe on the `none` predictor: a model trained without signals.
- **Internal states,** such as the gaze token after a given block, are not in the output (the
  predictor drops the gaze and hand tokens). Read them with a forward hook on
  `predictor.predictor_blocks[k]`; each frame's tokens are `[gaze, hand, 256 image tokens]`.

## 6. Rules

- **Keep the encoder and predictor frozen.** Eval mode and `torch.no_grad()`; only the probe
  trains.
- **Split by person.** The predictors were trained on P01–P07. Test the probe on P08 and P09.
  Training the probe on P01–P07 is fine, but the predictor has seen those videos, so a score on
  them says little.
- **Cache the encoder features.** The encoder is the slow part and is the same for every
  predictor. Encode each clip once and feed the cached features to each predictor.
- **Compare predictors on the same clips.** Use the same clip list for every model;
  `ego.clips.fixed_clips` gives a fixed list.
- **Report the seeds.** Each form is trained with three seeds (`_s0`, `_s1`, `_s2`). Report the
  spread between them next to any difference.

## 7. Checklist

- [ ] Frames: whole frame to 256 × 256, ImageNet normalisation, each frame encoded alone, layer
      norm.
- [ ] 8 frames, 8 video frames apart.
- [ ] Signals from `ego.signals.read` with the recording's projector; missing ones marked
      invalid.
- [ ] Predictor loaded with `load_trained_predictor` (the right gaze form).
- [ ] Output layer-normed; the last 256 tokens for the next frame.
- [ ] Encoder and predictor frozen; probe tested on P08 and P09.
