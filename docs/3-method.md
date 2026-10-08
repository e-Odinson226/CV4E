---
type: report
status: running
created: 2026-09-18
updated: 2026-10-08
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
- **Split.** Training uses P01–P07. The test set of Tests 1–4 is 96 clips from P08: 4
  recordings × 24 clips, sampled with seed 12345. Tests 8–10 test on P08 and P09. HD-EPIC
  defines no split by participant; this split is the project's own
  ([[2-background#Benchmarks]]).

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
- The model uses every 8th frame, about 4 frames per second. It sees 8 such frames, about 2 s
  of video.
- Embeddings are layer-normalized before the loss.
- In training, each step predicts the next step: the frame 8 frames later, which at 30
  frames per second is $8 / 30 \approx 0.27$ s ahead. All tests so far measure the prediction
  at this horizon. The pretraining of V-JEPA 2-AC also predicted two steps in a row, each
  prediction the input for the next (`auto_steps: 2` in its configuration). The fine-tuning
  here uses only the one-step loss.

The step of 8 frames matches the pretraining. V-JEPA 2-AC was trained on robot video (DROID)
at 4 frames per second, a step of 0.25 s, and its predictor learned how much a scene changes in
one such step. A step of a different length would have to be learned again
(`--frame-stride` in `train`).

Encoding a whole clip at once gave an error of 4.82 before fine-tuning, on 36 clips from 12 P08
recordings. The format above gives 0.58, on 48 clips from the same recordings.

## Gaze position in the image

The model receives gaze as two angles and a depth. The encoder's output is a grid of patch
embeddings, one per region of the frame. To relate gaze to this grid, gaze has to be a point in
the frame. `ego/gaze_geometry.py` computes this point. `python -m ego draw-gaze` and
`python -m ego gaze-at-picks` check it.
The trained models `ego_ft_v2` and `ego_sd1p0` do not use it. Test 8 and the gaze forms of
Test 9 use it ([[4-results#^test8|Test 8]], [[4-results#^test9|Test 9]]).

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
| Gaze point against the box of the object being picked up (19,324 picks) | Inside the box in 38.8% of picks, against 19.7% by chance. Details below. |

### Gaze on the object being picked up

HD-EPIC's annotators drew a box around each object in the frame where its movement starts. For
each of these picks, `python -m ego gaze-at-picks` projects the gaze at that frame and compares
the point with the box. The data are 19,324 picks with a box and gaze, in 154 recordings of all
nine participants. The median box is 1.6 × 1.8 patches. As chance, the same box is compared
with the gaze of 20 random moments of the same recording, at least 10 s away from the pick.
Gaze and objects both lie mostly near the centre of the frame, so a point can land in a box
without the person looking at the object. The intervals are 95% intervals from a bootstrap
over recordings.

| Gaze point | Inside the box | Within 1 patch of the box | Median distance (patches) |
|---|---|---|---|
| at the pick | 38.8% [36.7, 40.9] | 68.5% [66.8, 70.2] | 0.25 |
| chance: gaze at random moments | 19.7% [18.8, 20.7] | 49.9% [48.5, 51.3] | 0.90 |
| without the 90° rotation | 1.8% | 7.8% | 3.46 |
| depth fixed at 1 m | 33.7% | 66.6% | 0.33 |
| depth fixed at 100 m | 23.2% | 62.7% | 0.62 |
| 0.25 s before the pick | 37.1% | 73.1% | 0.25 |
| 0.5 s before the pick | 30.2% | 68.3% | 0.44 |
| 1 s before the pick | 21.7% | 54.7% | 0.84 |
| 2 s before the pick | 19.4% | 50.3% | 0.99 |

![[figures/picks_gaze_on_object.png]]

- At the pick, the gaze point lies in the box twice as often as by chance: 19.1 percentage
  points more [17.3, 21.0]. Every participant is above chance, from 32.1% (P09, chance 14.7%)
  to 50.9% (P03, chance 19.8%).
- Without the rotation, the point almost never lands in the box. The rotation is right.
- The measured depth gives more hits than a fixed depth of 1 m or 100 m. The depth and the
  distance between eyes and camera are applied correctly.
- In the overlays (`results/gaze_projection/picks.jpg`), most misses lie on what the hands
  work on at that moment: the board being cut on, the other hand, a pan.
- Before the pick, the share inside the box is about the same 0.25 s before, and close to
  chance from 1 s before. The head moves in between, and the box is from the pick frame. So this
  decline mixes gaze moving to the object with the object moving in the image. HD-EPIC's own
  analysis follows the objects in 3D. Of the picks it can check, 94.8% are preceded by a look at
  the object within the 10 s before, on average 4.0 s before (Perrett et al., CVPR 2025). The
  released file, `eye-gaze-priming/priming_info.json`, gives 94.0% of 13,285 picks.

**Offset for each person.** For each participant, the command also finds the shift of all gaze
points that puts the most of them in the boxes:

| | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|---|---|---|---|---|---|---|---|---|---|
| Shift right, in patches | 0.25 | −0.75 | 0 | 0.25 | 0.25 | 0 | 0.5 | −0.5 | 0.25 |
| Shift down, in patches | 0.75 | −0.25 | 0 | 0.5 | 0.25 | 0.25 | 0.25 | 0.25 | 0 |
| Inside, as projected | 44.7% | 36.3% | 50.9% | 34.7% | 41.6% | 43.4% | 32.8% | 29.8% | 32.1% |
| Inside, after the shift | 56.2% | 54.0% | 50.9% | 43.8% | 48.1% | 45.7% | 39.7% | 39.8% | 32.3% |

The shifts are smaller than one patch and point in different directions for different people.
This fits a bias of the eye tracker for each person. HD-EPIC gives the output of Aria's general
gaze model only, not the version calibrated for each person. Six of the nine shifts are
downward and one is upward, so people may also look at the upper part of an object before they
pick it up. The shifts are fitted on the same picks they are scored on, so the gains after the
shift are an upper bound. The projection does not apply them.

**What this means for the tests.** The gaze point is right to within about half a patch for most
people, and it lies on the object about to be picked up far more often than by chance. Test 8
therefore takes the features around the gaze point over a radius of about one patch, not from a
single patch. In Test 9, an error of half a patch moves the gaze token's position at most
to the neighbouring patch.

## Training

| Part | Trained | Learning rate |
|---|---|---|
| Encoder (ViT-g) | no | — |
| `predictor_embed`, blocks 0–17 | no | — |
| `gaze_proj`, `hand_proj`, `gaze_mask`, `hand_mask` | yes | 1e-3 |
| Blocks 18–23, `predictor_norm`, `predictor_proj` | yes | 1e-4 |

- This is 77.0 million of the predictor's 305.2 million parameters (25.2%).
- A full fine-tune trains the whole predictor: all 24 blocks and `predictor_embed` as well, at
  1e-4 (`--unfreeze-last-n 24 --unfreeze-embed`). The encoder stays frozen. Test 10 compares it
  with training the last 6 blocks ([[6-next-steps]]).
- The optimizer is AdamW with weight decay 0.01.
- The loss is the mean absolute difference (L1) between the predicted and the true embeddings,
  as in the pretraining of V-JEPA 2-AC (`loss_exp: 1.0`). Its blocks were trained for this
  loss. `ego_ft_v2`, `ego_sd1p0` and the 20 models of Test 9 were trained with the mean squared
  difference (MSE). With MSE, the model before fine-tuning is scored with a loss it was not
  trained for, so part of the gain from fine-tuning can be adaptation to the new loss. The
  pretraining also predicted a second step from its own first prediction (`auto_steps: 2`); the
  fine-tuning predicts one step.
- `ego_ft_v2`, the model used in most tests: 3 epochs, P01–P07, 30 clips per recording,
  batch size 16.
- Signal dropout: for each training clip, one random draw decides whether the signals are
  hidden. The rate is 0.4. Gaze and hand are always hidden together.
- The training set includes the two P02 recordings without gaze and hand data. Their clips
  always have the signals hidden. So about 41% of all training clips have hidden signals.
- `ego_sd1p0`: the same settings with `--signal-dropout 1.0`. This model never sees real
  signals.
- `--shuffle-signals batch` gives each clip the signals of another clip. It trained the shuffled
  control of Test 9. `--shuffle-signals time` shuffles their order within the clip. It has not
  been used in a run yet.

## Evaluation

The model is evaluated by its prediction error on HD-EPIC. This needs no labels, and HD-EPIC
has gaze data.

### Prediction error and Δ

- The model sees 8 frames, which cover about 2 seconds. It predicts the embedding of the next
  step, 8 frames (0.27 s) after the last context frame.
- The target is the frozen encoder's embedding of that frame.
- The error is computed per clip, after layer normalization, and then averaged. Lower is
  better. It is measured in two ways: the mean squared difference (MSE) and the mean absolute
  difference (L1). Tests 1–4 report MSE only.
- After layer normalization, the 1,408 numbers of each patch have mean 0 and variance 1. For two
  such vectors, $\text{MSE} = 2\,(1 - \rho)$, where $\rho$ is the correlation between the
  predicted and the true patch vector. An MSE of 0.50 is a correlation of 0.75.
- References without fine-tuning show how much of the error a model explains: repeating the last
  context frame; a blend of the past frames, $0.2 \times$ the last context frame $+ 0.8 \times$
  the mean of the 8, layer-normalized like the predictor's output; and the model before
  fine-tuning. The weight 0.2 was chosen on the test clips of Test 9, which favours the blend a
  little.
- Δ = error with the signals hidden − error with real signals, on the same clips. A positive Δ
  means the signals helped.
- Δ is tested with a paired Wilcoxon test on the per-clip values and reported with a 95%
  confidence interval.
- Results are reported as absolute differences. Relative percentages make small effects look
  large.

### Three ways to hide the signals

| Input | What the model receives | Δ for `ego_ft_v2` |
|---|---|---|
| Mask token | the learned `gaze_mask` and `hand_mask` | +0.0011 |
| Zeros | zero vectors through the projection layers | +0.0022 |
| Average | the average scaled signal | +0.0007 |

Every Δ states which input it uses. The default is the mask token, which `python -m ego evaluate`
and the checks during training use.

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

Most of these ideas came from studying GazeQwen ([[2-background]]). The two ways to give gaze
a position were built and compared in Test 9; the others are not built. The gaze form was
tested before the full training run, because the gaze token cannot point at the image and the
model makes little use of the gaze value.

**Gaze as a position in the image.** Built for Test 9 ([[4-results#^test9|Test 9]]), which
compared two ways:

- *Coord-PE: the position in the token's content.* Sine and cosine features of the gaze point
  $(u, v)$ at several frequencies, plus the depth and an inside-the-frame flag, before the
  linear layer.
- *RoPE position: the position in the attention.* The gaze token is rotated in height and width
  like an image patch at the gaze point, in place of the top-left patch. The frozen attention
  heads then relate it to the patches near the gaze point, as they relate nearby patches. A
  rotation by position 0 changes nothing, so this form with every gaze point at the top-left
  patch is exactly the current model.

Before Test 9, Coord-PE alone was expected not to be enough. Patches carry their position only
through the rotations, not in their content. To find the patch at $(u, v)$ from content
features, the frozen heads would have to match those features against rotated keys. Test 9
found the opposite: near the gaze point, Coord-PE (pe) did better than the RoPE position
(rope), and as well as the three angles. The RoPE position did not help. Why is not known; the
candidate explanations are in [[4-results#^test9|Test 9]].

**Step, context and horizon.** (Test 8 settled the context; the rest is open.)

Gaze carries information at three time scales:

- *About 0.1–0.5 s: where the head turns next.* The eyes usually move to a new target first,
  and the head follows. This is known from studies of eye and head movement. It has not been
  measured in HD-EPIC.
- *About 0.5–1 s: which object the hand reaches for.* Gaze leads the hand by about this much
  ([[2-background]]). The gaze point lies on the object at the pick and 0.25 s before
  ([[#Gaze on the object being picked up]]).
- *About 1–10 s: which objects belong to the current task.* A picked object is first looked at
  on average 4 s before the pick.

*Horizon.* At 0.27 s the scene changes little, the past frames already show most of that
change, and the hand has not reached the object yet. Only the first time scale applies. About
1 s covers the second. Beyond about 2 s the model has a difficulty: it predicts one future, not
several, so when the future is uncertain it predicts a mix of the possible futures, and head
motion dominates the error over the whole frame. The best single horizon is therefore about
1 s. The error should also be measured on the patches of the object about to be picked, because
the object covers about 3 of the 256 patches.

*Context.* The third time scale would need about 4–6 s of gaze history. In Test 8, however,
the gaze history of the last 1–3 s added at most about 1 point to the current gaze point in
telling the next object, and 6 s nothing more than 3 s ([[4-results#^test8|Test 8]]). The model
sees 8 frames, about 2 s. Each frame has its own gaze token, so the context length is also the
length of the gaze history. Each further frame adds 258 tokens.

*Options for the step:*

| Step | Context with 8 frames | One step ahead | Fit to the pretraining | Gaze samples per second |
|---|---|---|---|---|
| 8 frames, 0.27 s (now) | 2.1 s | 0.27 s | the same | 3.75 |
| 15 frames, 0.5 s | 4 s | 0.5 s | twice as long | 2 |
| 30 frames, 1 s | 8 s | 1 s | four times as long | 1 |
| 8 frames, with 16 context frames | 4.3 s | 0.27 s | the same, with twice the tokens | 3.75 |

People make about 2–4 fixations per second. The current step samples gaze at about this rate.
Longer steps miss most fixations.

V-JEPA 2's own benchmark of action anticipation on EK100 uses 32 frames at 8 frames per second,
4 s of context (`vjepa2/configs/eval/vitg-384/ek100.yaml`). That setup belongs to the plain
V-JEPA 2 model: the encoder takes the 32 frames as one video and joins every two frames, so it
has 16 time steps of 0.25 s, and an attentive probe names the action 1 s ahead. It has no place
for a gaze or hand token per step, so it does not fit the action-conditioned predictor used
here. In this predictor, 32 frames would be 8,256 tokens in place of 2,064, about 16 times the
cost of attention, with a step of 0.125 s, half the step of the pretraining. It fits a probe on
the plain encoder, where results can be compared with published EK100 numbers.

*One step per fixation.* A step for each fixation, in place of a fixed step, would follow how
people take in a scene. It is not proposed as the step of the model, for five reasons:

1. The predictor's time positions count steps, and its frozen blocks learned that one step is
   0.25 s. Steps of 0.1–1 s would need positions in seconds, which the frozen blocks have not
   seen.
2. The next step would be the next fixation. The time of the target would then depend on the
   future gaze.
3. Fixations cannot be found reliably in the gaze of P01–P03, which is sampled 10 times per
   second. A saccade lasts 20–80 ms.
4. The gaze is given relative to the head. When the head turns while the eyes stay on an
   object, the gaze moves in the head's frame. Finding fixations therefore needs the head
   motion from the SLAM output, which is downloaded for some participants only.
5. Several fixations on one object give nearly identical frames, and with 2–4 fixations per
   second the number of tokens grows.

What a step per fixation offers can be kept another way: fixed steps for the frames, plus a
gaze memory with one token for each recent gaze sample or fixation. Each token holds the
encoder's features at that gaze point, with its position and time. One token per look costs
far less than 258 tokens per frame.

*Proposal:*

- Keep the step of 8 frames and the 8 context frames. This matches the pretraining and samples
  gaze at about the fixation rate.
- Predict 1 s ahead with four steps in a row. Train with a loss over several steps, as the
  pretraining did with two ([[#Input format]]).
- No gaze memory for now. In Test 8 the useful history lay within the last 1–3 s, which the
  8 context frames with their gaze tokens already cover, and the gain of the gaze point was
  largest 0.5 s ahead.
- The simpler alternative is a step of 15 frames (0.5 s). It gives 4 s of context and gaze
  history at the same cost, but differs from the pretraining. The error before fine-tuning at
  both steps shows how large that difference is.

**Zero initialization of the projection layers: rejected.** A zero token is a new kind of input
for the model. If the projection and a gate both start at 0, both gradients are 0 and the
pathway cannot learn.

**Gate (`alpha_gaze`), if one is added.** A learned number α blends between the "no signal"
token and the projected gaze:

$$\text{token} = \text{gaze\_mask} + \alpha \,(\text{gaze\_proj}(\text{gaze}) - \text{gaze\_mask})$$

At α = 0 the model behaves like one trained without signals. α has no weight decay, because
weight decay alone would pull it toward 0, whatever the data.
