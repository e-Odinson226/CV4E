# ═══════════════════════════════════════════════════════════
# SCHIMBĂ DOAR ACESTE VALORI:
# ═══════════════════════════════════════════════════════════
VIDEO              = "/home/imarica/clean_0_data/AriaGen2PilotDataset_v1.0_clean_0_preview_rgb.mp4"
CALIB              = "/home/imarica/clean_0_data/slam/online_calibration.jsonl"
GAZE_CSV           = "/home/imarica/clean_0_data/eye_gaze/eye_gaze.csv"
TIMESTAMPS         = "27,38,54,97,115,251,266,307"
OUTPUT             = "/home/imarica/vjepa2_gaze_results_final"
FUTURE_SEC         = 1.0          # predict 1 second into the future
NUM_FRAMES         = 16
CONTEXT_RATIO      = 0.5
GAZE_BOOST_RATIO   = 0.4
GAZE_SIGMA_PATCHES = 3.0
MAX_GAZE_LAG_SEC   = 0.5          # don't use gaze if sample is further than this
# ═══════════════════════════════════════════════════════════

import json, warnings
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")
from transformers import AutoModel, AutoVideoProcessor

# ---- Setup folders ----
out = Path(OUTPUT)
for sub in ["frames", "comparison", "metrics"]:
    (out / sub).mkdir(parents=True, exist_ok=True)

# ---- Calibration ----
with open(CALIB) as fh:
    for line in fh:
        if line.strip():
            calib_data = json.loads(line)
            break
cam = next(c for c in calib_data["CameraCalibrations"] if c["Label"] == "camera-rgb")
p = cam["Projection"]["Params"]
fx_nat, fy_nat, cx_nat = p[0], p[1], p[2]
cy_nat = cx_nat

# ---- Gaze Data with Interpolation ----
df_gaze = pd.read_csv(GAZE_CSV)
gaze_t0 = df_gaze['timestamp_ns'].iloc[0]
df_gaze['ts_sec'] = (df_gaze['timestamp_ns'] - gaze_t0) / 1_000_000.0 # Microseconds to Sec

valid_col = 'gaze_valid' if 'gaze_valid' in df_gaze.columns else (
            'combined_gaze_valid' if 'combined_gaze_valid' in df_gaze.columns else None)

mask = df_gaze[valid_col].astype(bool) if valid_col else np.ones(len(df_gaze), dtype=bool)
v_ts = df_gaze.loc[mask, 'ts_sec'].values
v_yaw = df_gaze.loc[mask, 'yaw'].values
v_pitch = df_gaze.loc[mask, 'pitch'].values

# Interpolators
yaw_fn = interp1d(v_ts, v_yaw, bounds_error=False, fill_value="extrapolate")
pitch_fn = interp1d(v_ts, v_pitch, bounds_error=False, fill_value="extrapolate")

# ---- V-JEPA 2 Load ----
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

cfg = model.config
grid_h = grid_w = cfg.image_size // cfg.patch_size
T_tokens = NUM_FRAMES // cfg.tubelet_size
N_spatial = grid_h * grid_w
N_total = T_tokens * N_spatial
n_ctx = int(N_total * CONTEXT_RATIO)
n_gaze_gtd = int(n_ctx * GAZE_BOOST_RATIO)

# ---- Helpers ----
def get_gaze_pixel(ts, w, h):
    dist = np.abs(v_ts - ts).min()
    if dist > MAX_GAZE_LAG_SEC: return None, None
    y, p = yaw_fn(ts), pitch_fn(ts)
    sx = w / (cx_nat * 2)
    vfx, vfy, vcx, vcy = fx_nat * sx, fy_nat * sx, cx_nat * sx, cx_nat * sx
    cos_y = max(np.cos(y), 1e-9)
    px = vcx + vfx * np.tan(y)
    py = vcy - vfy * (np.tan(p) / cos_y)
    return (px, py) if (0 <= px < w and 0 <= py < h) else (None, None)

def get_clip(ts):
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps) - NUM_FRAMES//2)
    frames = []
    for _ in range(NUM_FRAMES):
        ret, f = cap.read()
        if not ret: break
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((256, 256)))
    cap.release()
    return frames

# ---- Processing ----
ts_list = [float(t) for t in TIMESTAMPS.split(',')]
results = []

for ts in ts_list:
    fut_ts = ts + FUTURE_SEC
    print(f"Processing t={ts}s -> future={fut_ts}s")
    
    frames_ctx = get_clip(ts)
    frames_fut = get_clip(fut_ts)
    
    # Use FUTURE gaze to select PRESENT context
    px, py = get_gaze_pixel(fut_ts, 2560, 1920) # Based on your resolution
    
    # Baseline
    inputs = processor(videos=frames_ctx, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gt_feat = model(**processor(videos=frames_fut, return_tensors="pt").to(model.device), 
                        context_mask=[torch.tensor(list(range(N_total))).unsqueeze(0).to(model.device)], 
                        skip_predictor=True).last_hidden_state.mean(dim=1)
        
        # Baseline pred
        mask_unif = [torch.tensor(np.linspace(0, N_total-1, n_ctx, dtype=int)).unsqueeze(0).to(model.device)]
        out_base = model(**inputs, context_mask=mask_unif, target_mask=[torch.tensor(list(range(N_total))).unsqueeze(0).to(model.device)]).predictor_output.last_hidden_state
        sim_base = F.cosine_similarity(out_base.mean(dim=1), gt_feat).item()

        # Gaze pred
        if px is not None:
            gx, gy = (px/2560)*grid_w, (py/1920)*grid_h
            yy, xx = np.mgrid[0:grid_h, 0:grid_w]
            dist = (xx - gx)**2 + (yy - gy)**2
            score = np.tile(np.exp(-dist/(2*GAZE_SIGMA_PATCHES**2)).flatten(), T_tokens)
            idx_gaze = np.argsort(score)[::-1][:n_gaze_gtd].tolist()
            idx_rand = np.random.choice([i for i in range(N_total) if i not in idx_gaze], n_ctx-n_gaze_gtd, replace=False).tolist()
            mask_gaze = [torch.tensor(sorted(idx_gaze + idx_rand)).unsqueeze(0).to(model.device)]
            
            out_gaze = model(**inputs, context_mask=mask_gaze, target_mask=[torch.tensor(list(range(N_total))).unsqueeze(0).to(model.device)]).predictor_output.last_hidden_state
            sim_gaze = F.cosine_similarity(out_gaze.mean(dim=1), gt_feat).item()
            
            # Visualization
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(frames_ctx[NUM_FRAMES//2]); ax[0].set_title("Context (Present)")
            ax[1].imshow(frames_fut[NUM_FRAMES//2]); ax[1].set_title("Target (Future)")
            diff = (out_gaze - out_base).norm(dim=-1).cpu().numpy()[0, :N_spatial].reshape(grid_h, grid_w)
            ax[2].imshow(diff, cmap='jet'); ax[2].set_title(f"Gaze Effect (Δ={sim_gaze-sim_base:+.4f})")
            plt.savefig(out / "comparison" / f"t{ts}.png")
            plt.close()
            
            results.append({"ts": ts, "base": sim_base, "gaze": sim_gaze, "diff": sim_gaze-sim_base})
            print(f"  Result: Base={sim_base:.4f}, Gaze={sim_gaze:.4f}, Delta={sim_gaze-sim_base:+.4f}")
        else:
            print(f"  Skipping t={ts}: No valid future gaze.")

pd.DataFrame(results).to_csv(out / "metrics" / "results.csv")