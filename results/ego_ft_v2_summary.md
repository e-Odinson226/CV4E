# Run summary: checkpoints/ego_ft_v2

- epochs: **3/3**  | avg epoch: **733.3s**  | total: **2200s**  | throughput: **None it/s**
- best train loss: **0.5066**
- held-out Δ(A−B): **-0.0082 → +0.0011**

## Trainable layers

- TRAIN  new: gaze_proj, hand_proj, gaze_mask, hand_mask
- TRAIN  blocks 18-23 (last 6 of 24)  + predictor_norm, predictor_proj
- FROZEN predictor_blocks[0:18], predictor_embed
- 82 trainable tensors  77,042,048/305,216,896 params (25.2%)

## Per-epoch

| epoch | train_loss | time(s) | MSE_A | MSE_B | Δ(A−B) |
|---|---|---|---|---|---|
| 0 | — | — | 0.6153 | 0.6235 | -0.0082 |
| 1 | 0.5261 | 825 | 0.4942 | 0.4933 | +0.0009 |
| 2 | 0.5150 | 688 | 0.4900 | 0.4885 | +0.0015 |
| 3 | 0.5066 | 687 | 0.4877 | 0.4866 | +0.0011 |

## Config

- `checkpoint` = data/model_checkpoints/vjepa2-ac-vitg.pt
- `video_dir` = data/epic-kitchen/ek100-hd/HD-EPIC/Videos
- `gaze_dir` = data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze
- `participants` = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07']
- `out_dir` = checkpoints/ego_ft_v2
- `epochs` = 3
- `batch_size` = 16
- `num_workers` = 8
- `context_steps` = 8
- `frame_stride` = 8
- `clips_per_recording` = 30
- `signal_dropout` = 0.4
- `unfreeze_last_n` = 6
- `lr_proj` = 0.001
- `lr_blocks` = 0.0001
- `weight_decay` = 0.01
- `no_normalize_reps` = False
- `no_amp` = False
- `encode_chunk` = 48
- `no_standardize` = False
- `val_participants` = ['P08']
- `val_recordings` = 4
- `val_clips` = 24
- `save_every` = 3
- `log_every` = 10
- `seed` = 0
- `device` = cuda
- `normalize_reps` = True
- `amp_dtype` = torch.bfloat16