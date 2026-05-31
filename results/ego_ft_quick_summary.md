# Run summary: checkpoints/ego_ft_quick

- epochs: **1/3**  | avg epoch: **978.0s**  | total: **978s**  | throughput: **None it/s**
- best train loss: **0.5413**
- held-out Δ(A−B): **-0.0082 → -0.0003**

## Trainable layers


## Per-epoch

| epoch | train_loss | time(s) | MSE_A | MSE_B | Δ(A−B) |
|---|---|---|---|---|---|
| 0 | — | — | 0.6153 | 0.6235 | -0.0082 |
| 1 | 0.5413 | 978 | 0.4975 | 0.4977 | -0.0003 |

## Config

- `checkpoint` = data/model_checkpoints/vjepa2-ac-vitg.pt
- `video_dir` = data/epic-kitchen/ek100-hd/HD-EPIC/Videos
- `gaze_dir` = data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze
- `participants` = ['P04', 'P05', 'P06', 'P07']
- `out_dir` = checkpoints/ego_ft_quick
- `epochs` = 3
- `batch_size` = 8
- `num_workers` = 4
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
- `no_standardize` = False
- `val_participants` = ['P08']
- `val_recordings` = 4
- `val_clips` = 24
- `save_every` = 3
- `log_every` = 20
- `seed` = 0
- `device` = cuda
- `normalize_reps` = True
- `amp_dtype` = torch.bfloat16