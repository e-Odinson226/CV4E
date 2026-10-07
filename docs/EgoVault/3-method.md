---
type: report
status: running
created: 2026-09-18
updated: 2026-10-07
---

# 3. Method

How the data is prepared, how the model is built and trained, and how it is evaluated. The
code, the commands and the checkpoints are described in `README.md` at the root of the
repository.

## Data

- **HD-EPIC.** Kitchen recordings made with Aria Gen 1 glasses by participants P01–P09. 154 of
  the 156 recordings have gaze and hand data. The two without are from P02.
- **Recording settings.** For P01–P03, gaze and hand are recorded at 10 Hz. For P04–P09, gaze
  is recorded at 60 Hz and hand at 30 Hz. Both groups use the same gaze model (Aria MPS 3.0),
  and their gaze values have similar distributions. Hand tracking differs: no hand is tracked
  in 26–27% of the samples of P01–P03, and in 4–14% of the samples of P04–P09.
- **Glasses.** Nine pairs of glasses were used. Each participant used 3 or 4 of them.
- **Split.** Training uses P01–P07. The test set is 96 clips from P08: 4 recordings × 24 clips,
  sampled with seed 12345. P09 is not used yet.

## Model

The base model is Meta's V-JEPA 2-AC with the ViT-g encoder ([[2-background]]). The encoder
turns each frame into 256 embeddings, one for each 16 × 16-pixel patch, kept in row and column
order. The predictor receives these patch embeddings and predicts those of the next step. The
ego predictor replaces the robot tokens of V-JEPA 2-AC with a gaze token and a hand token:

```
AC predictor:   [action, state, 256 image tokens]   per frame
Ego predictor:  [gaze,   hand,  256 image tokens]   per frame
```

The layout is the same, so all 24 pretrained predictor blocks load without changes. Attention
is causal over frames: a token at frame t sees all tokens of the frames up to t.

Positions enter the predictor only through rotary position encoding (RoPE): the queries and
keys of each image patch are rotated by its time step, row and column. There is no position in
the tokens' content. The two signal tokens are rotated only by their time step. In height and
width they are not rotated, which places them at the top-left patch in every frame
(`ACRoPEAttention` in `vjepa2/src/models/utils/modules.py`). For the robot's action and state,
which are global quantities, this did not matter. For gaze it means that the token cannot point
at the image ([[5-discussion#Weak points of the design|weak point "gaze cannot point"]]).

Ioana wrote the first gaze and hand projection layers. Erfan moved them into the predictor and
extended them to all time steps.

### Inputs

| Signal | Size | Values | Source file |
|---|---|---|---|
| Gaze | 3 | yaw and pitch (radians), depth (metres), in the Aria central pupil frame | `eye_gaze/general_eye_gaze.csv` |
| Hand | 12 | wrist and palm position (x, y, z) for each hand, in the device frame | `hand_tracking/wrist_and_palm_poses.csv` |

- Signals are matched to video frames by timestamp, using
  `<recording>_mp4_to_vrs_time_ns.csv`.
- Gaze is scaled as (gaze − `GAZE_MEAN`) / `GAZE_STD`. Hand is scaled as hand / `HAND_STD`.
- These constants are rough guesses. Measured on P01–P07, scaled gaze has means of (−0.04,
  −0.52, −0.08) and standard deviations of (0.44, 0.55, 0.87). The intended values are 0
  and 1. Scaled hand values have means from −1.0 to 0.87 and standard deviations from 0.31 to
  0.50.

### Layers

- `gaze_proj`: a linear layer from 3 to 1024 values, with a bias.
- `hand_proj`: a linear layer from 12 to 1024 values, with a bias.
- Both start from small random weights. Their biases start at 0.
- `gaze_mask` and `hand_mask` are learned tokens. They replace a signal when it is missing or
  hidden.
- Gaze has one validity flag per frame. The hand token is replaced only when both hands are
  missing.

### Input format

The AC predictor was trained on frames encoded one at a time. The code does the same, in
`ego/model.py` and `ego/clips.py`, which training and testing share:

- Each frame is encoded on its own, as a 2-frame tubelet of the same image.
- The EMA target encoder encodes both the context frames and the target frame.
- Frames are resized to 256 × 256 pixels. This gives 16 × 16 = 256 patches per frame.
- The model uses every 8th frame, about 4 frames per second. It sees 8 such frames.
- Embeddings are layer-normalized before the loss.
- In training, each step predicts the next step.

Encoding a whole clip at once gave an error of 4.82 before fine-tuning, on 36 clips from 12 P08
recordings. The format above gives 0.58, on 48 clips from the same recordings. (Found by Erfan,
May 2026.)

## Gaze position in the image

The model receives gaze as two angles and a depth. The encoder's output is a grid of patch
embeddings, one per region of the frame. To relate gaze to this grid, gaze has to be a point in
the frame. `ego/gaze_geometry.py` computes this point, and `python -m ego draw-gaze` checks it.
The trained models `ego_ft_v2` and `ego_sd1p0` do not use it. Tests 8 and 9 will
([[6-next-steps]]). (Built by Claude Code at Erfan's request, October 2026.)

### Steps

1. The gaze point in 3D: the direction from yaw and pitch, at the depth of the gaze file, in
   the central pupil frame (between the eyes).
2. Moved to the frame of the RGB camera, with the camera calibration of the recording and
   Aria's CAD transform from the central pupil frame to the device.
3. Projected with the model of the RGB camera (Aria Gen 1: a fisheye model with radial,
   tangential and thin-prism terms), scaled to the 1408-pixel frame.
4. Rotated by 90° clockwise. The camera sits sideways in the glasses, and HD-EPIC's videos are
   upright. Without this rotation the point lands on walls and shelves. The frames keep the
   fisheye distortion, so no undistortion is needed.
5. Scaled to the 16 × 16 patch grid of the 256-pixel input: the gaze point $(u, v)$ in patch
   units.

### Calibration

- HD-EPIC stores a camera calibration for each group of recordings, inside its SLAM zips.
  `python -m ego fetch-calibrations` downloads only these entries, a few MB per group in place
  of 0.5–4 GB per zip. All 153 groups are in `SLAM-and-Gaze/<P>/SLAM/calibration/`. They cover
  153 of the 156 recordings. The two P02 recordings without gaze have none. Neither has
  `P06-20240510-140459`, which has gaze; it uses the calibration of the P06 recording made
  closest in time, on the same day, most likely with the same glasses.
- The nine pairs of glasses differ by up to 10 pixels in focal length and 19 pixels in image
  centre at the full resolution of 2880 pixels, about 0.1 patch at the 256-pixel input. Within
  a group the calibration does not change.
- The transform from the central pupil frame to the device is not in these files.
  `projectaria_tools` gives it from Aria's CAD values. The two Gen 1 frame sizes place the
  central pupil frame 2.4 mm apart, so the unknown frame size of each recording does not
  matter.
- `data/online_calibration.jsonl` belongs to an Aria Gen 2 device from other data. It does not
  apply to HD-EPIC.

### Depth matters

The eyes are 57–59 mm from the RGB camera. So the point where a gaze direction lands in the
image depends on how far away the person looks. Gaze straight ahead at 0.3 m lands 1.4 patches
away from the same direction at infinity, at 0.5 m 0.8 patch, and at 1 m 0.4 patch. Most
fixations in HD-EPIC are 0.3–0.8 m away. The projection uses the depth of the gaze file, and an
error in that depth moves the point.

### Checks

| Check | Result |
|---|---|
| Overlays on frames of every participant (`results/gaze_projection/<P>.jpg`) | The point lies on what the person handles or is about to handle: the bowl being stirred, the carrot being cut, the phone being typed on. |
| Share of gaze points inside the frame (30,798 samples, all participants) | 100%. |
| Where the points fall | Around column 9 and row 10–12 of the 16 × 16 grid: near the centre and below it, where the hands work. |
| Shift caused by the depth (gaze depth against the same direction at 100 m) | Median 0.9 patch, 90th percentile 1.7 patches. Largest for P08 (median 1.7 patches), who works closest to the camera (median depth 0.26 m). |
| The gaze values of `ego.data` against the two-eye combination of `projectaria_tools` | Yaw differs by 0.05° (median); pitch and depth are the same. |
| Gaze point against the box of the picked object, in the seconds before a pick | Not done. HD-EPIC's annotations of pick events are not on this machine yet. |

## Training

| Part | Trained | Learning rate |
|---|---|---|
| Encoder (ViT-g) | no | — |
| `predictor_embed`, blocks 0–17 | no | — |
| `gaze_proj`, `hand_proj`, `gaze_mask`, `hand_mask` | yes | 1e-3 |
| Blocks 18–23, `predictor_norm`, `predictor_proj` | yes | 1e-4 |

- This is 77.0 million of the predictor's 305.2 million parameters (25.2%).
- The optimizer is AdamW with weight decay 0.01.
- `ego_ft_v2`, the model used in most tests: 3 epochs, P01–P07, 30 clips per recording,
  batch size 16.
- Signal dropout: for each training clip, one random draw decides whether the signals are
  hidden. The rate is 0.4. Gaze and hand are always hidden together.
- The training set includes the two P02 recordings without gaze and hand data. Their clips
  always have the signals hidden. So about 41% of all training clips have hidden signals.
- `ego_sd1p0`: the same settings with `--signal-dropout 1.0`. This model never sees real
  signals.
- `--shuffle-signals batch` gives each clip the signals of another clip. `--shuffle-signals
  time` shuffles their order within the clip. Neither has been used in a run yet.

## Evaluation

The model is evaluated by its prediction error on HD-EPIC. This needs no labels, and HD-EPIC
has gaze data. (Decided by Erfan, May 2026.)

### Prediction error and Δ

- The model sees 8 frames, which cover about 2 seconds. It predicts the embedding of the next
  step, 8 frames (0.27 s) after the last context frame.
- The target is the frozen encoder's embedding of that frame.
- The error (MSE) is computed per clip, after layer normalization, and then averaged. Lower is
  better.
- Δ = error with the signals hidden − error with real signals, on the same clips. A positive Δ
  means the signals helped.
- Δ is tested with a paired Wilcoxon test on the per-clip values and reported with a 95%
  confidence interval.
- Results are reported as absolute differences. Relative percentages make small effects look
  large. (Asked by Ash, July 2026.)

### Three ways to hide the signals

| Input | What the model receives | Δ for `ego_ft_v2` |
|---|---|---|
| Mask token | the learned `gaze_mask` and `hand_mask` | +0.0011 |
| Zeros | zero vectors through the projection layers | +0.0022 |
| Average | the average scaled signal | +0.0007 |

Every Δ states which input it uses. The default is the mask token, which `python -m ego evaluate`
and the checks during training use. (Decided by Erfan, August 2026.)

### Measuring the value of the signals

- `ego_ft_v2` saw hidden signals in about 41% of its training clips. So hiding them at test
  time is a case it knows.
- The model still relies on the signals. Hiding them measures two things together: the value
  of the information and the cost of removing an expected input.
- A matched model trained with `--signal-dropout 1.0` separates the two. The value of the
  information is `ego_ft_v2` with real signals compared with `ego_sd1p0` with hidden signals
  ([[4-results#^test2|Test 2]]). Every future training comparison includes such a matched model.

### Linear probes on the frozen encoder (Test 3)

- Input: one frame, encoded by the frozen encoder. All 256 patches are kept, because gaze is
  about a place in the image.
- The 1408 values per patch are reduced to 64 with PCA, fitted on the first 300 training frames
  the sampler collects. These come from the first 6 recordings of P01.
- Model: ridge regression. Its regularization strength is chosen by 5-fold cross-validation on
  the training data.
- Score: skill = 1 − MSE(probe) / MSE(guessing the training mean). 0 is chance and 1 is
  perfect.
- Lead: the target is taken 0, 0.25, 0.5, 1 or 2 s after the frame.
- Splits: new participants (P01–P05 for training, P06–P07 for testing), new recordings of the
  same people (59 and 20 recordings), and random frames.
- The palm control reuses the features cached by the gaze probe. It finds the frame of each
  cached row by replaying the sampler, and checks every row against the cache.

### Tests on the trained predictor (Test 4)

- Sensitivity (4a): change = ‖prediction with changed input − prediction with real input‖ /
  ‖prediction with real input‖, on the last predicted step, after layer normalization. The
  same input run twice must give 0.
- Attention (4b): the attention function is wrapped so that the attention weights can be read.
  An image token at frame t sees the 258 tokens of each frame from 0 to t. Its attention on the
  gaze tokens of all these frames is added up, so equal attention gives 1/258 = 0.39%. Both the
  mean and the largest single head are reported.
- Weight norms (4c): each trained parameter's norm is divided by its norm at the start. Frozen
  parameters must give exactly 1.0.

## Design choices

These ideas came from studying GazeQwen ([[2-background]]). None of them is built yet. The gaze
form is tested before the full training run, as Test 9 in [[6-next-steps]], because the gaze
token cannot point at the image and the model makes little use of the gaze value. (Decided by
Erfan, October 2026.)

**Gaze as a position in the image.** Planned in Test 9, which compares two ways:

- *Coord-PE: the position in the token's content.* Sine and cosine features of the gaze point
  $(u, v)$ at several frequencies, plus the depth and an inside-the-frame flag, before the
  linear layer.
- *RoPE position: the position in the attention.* The gaze token is rotated in height and width
  like an image patch at the gaze point, in place of the top-left patch. The frozen attention
  heads then relate it to the patches near the gaze point, as they relate nearby patches. A
  rotation by position 0 changes nothing, so this form with every gaze point at the top-left
  patch is exactly the current model.

Coord-PE alone may not be enough. Patches carry their position only through the rotations, not
in their content. To find the patch at $(u, v)$ from content features, the frozen heads would
have to match those features against rotated keys.

**Zero initialization of the projection layers: rejected.** A zero token is a new kind of input
for the model. If the projection and a gate both start at 0, both gradients are 0 and the
pathway cannot learn. (Decided by Erfan, August 2026.)

**Gate (`alpha_gaze`), if one is added.** A learned number α blends between the "no signal"
token and the projected gaze:

$$\text{token} = \text{gaze\_mask} + \alpha \,(\text{gaze\_proj}(\text{gaze}) - \text{gaze\_mask})$$

At α = 0 the model behaves like one trained without signals. α has no weight decay, because
weight decay alone would pull it toward 0, whatever the data. (Decided by Erfan, August 2026.)
