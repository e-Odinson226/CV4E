"""
integration.py
==============
Gaze-guided V-JEPA2 integration: compare gaze-biased vs uniform context/target patch
sampling using the V-JEPA2 predictor, across 7 landmark video frames.

Pipeline
--------
1. Load eye-gaze CSV + online_calibration.jsonl
2. Extract 7 landmark frames from the RGB video (0.5 s look-back context clips)
3. For each frame build two patch sets:
   - UNIFORM  (gaze_ratio = 0.00)
   - GAZE-20  (gaze_ratio = 0.20, 80 % uniform + 20 % gaze-biased)
4. Run the V-JEPA2 ViT encoder + predictor on both sets
5. Compute L2 distance and cosine similarity between the predictor output and the
   ground-truth target encoding
6. Save per-frame side-by-side figures to  ./gaze_20/

Paths (edit if needed)
----------------------
VIDEO_PATH    : /home/imarica/clean_0_data/AriaGen2PilotDataset_v1.0_clean_0_preview_rgb.mp4
GAZE_CSV      : /home/imarica/clean_0_data/eye_gaze/eye_gaze.csv
CALIB_JSONL   : /home/imarica/clean_0_data/slam/online_calibration.jsonl
VJEPA2_REPO   : /home/imarica/vjepa2   (must contain configs/ and a checkpoint)
"""

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION  ← edit these
# ──────────────────────────────────────────────────────────────────────────────
VIDEO_PATH = "/home/imarica/clean_0_data/AriaGen2PilotDataset_v1.0_clean_0_preview_rgb.mp4"
GAZE_CSV = "/home/imarica/clean_0_data/eye_gaze/eye_gaze.csv"
CALIB_JSONL = "/home/imarica/clean_0_data/slam/online_calibration.jsonl"
VJEPA2_REPO = "/home/imarica/vjepa2"  # root of the cloned repo
CHECKPOINT = "/home/imarica/vitl.pt"  # ← your checkpoint
OUTPUT_DIR = "./gaze_35"

# Model / sampling hyper-params
GAZE_RATIO = 0.35  # 20 % gaze, 80 % uniform
IMG_SIZE = 224  # ViT input spatial size
PATCH_SIZE = 16  # ViT patch size
FRAMES_PER_CLIP = 8  # temporal context frames fed to the encoder
CONTEXT_RATIO = 0.50  # fraction of spatial patches used as context
SEED = 42

# The 7 landmark video timestamps (seconds) and the label for each
LANDMARKS = {
    "0m27s": 27,
    "0m38s": 38,
    "0m54s": 54,
    "1m37s": 97,
    "1m55s": 115,
    "4m11s": 251,
    "5m07s": 307,
}

# Video / sensor params (from calibration JSON)
VIDEO_W, VIDEO_H = 1280, 960  # preview video resolution
NATIVE_W, NATIVE_H = 2560, 1920  # RGB sensor resolution
# RGB camera intrinsics (Params[0]=fu, Params[1]=fv, Params[2]=cx)
FU = 1112.7323880128738
FV = 1301.5442019394354
CX_NATIVE = 954.9673344498369
CY_NATIVE = 960.0  # half of 1920

# ──────────────────────────────────────────────────────────────────────────────
import os, sys, json, math, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

# Add the vjepa2 repo to sys.path so we can import its modules
sys.path.insert(0, VJEPA2_REPO)

import torch
import torch.nn.functional as F
from torchvision import transforms

# ──────────────────────────────────────────────────────────────────────────────
# 0.  GAZE PROBABILITY MAP  (from gaussian_map.py)
# ──────────────────────────────────────────────────────────────────────────────


def project_gaze_to_native_pixel(yaw, pitch, fu, fv, cx_native, cy_native):
    px = cx_native + fu * np.tan(yaw)
    py = cy_native - fv * (np.tan(pitch) / np.cos(yaw))
    return px, py


def scale_pixel_to_video(px_native, py_native, cx_native, cy_native, video_w, video_h):
    sx = video_w / (2.0 * cx_native)
    sy = video_h / (2.0 * cy_native)
    return px_native * sx, py_native * sy


def normalize_pixel_to_unit(px_video, py_video, video_w, video_h):
    return px_video / video_w, py_video / video_h


def unit_to_grid(gx, gy, grid_w, grid_h):
    return gx * grid_w, gy * grid_h


def compute_elliptic_distance_map(cx_grid, cy_grid, grid_w, grid_h, sigma_x, sigma_y):
    cols = np.arange(grid_w, dtype=np.float32)
    rows = np.arange(grid_h, dtype=np.float32)
    J, I = np.meshgrid(cols, rows)
    dx = (J - cx_grid) / sigma_x
    dy = (I - cy_grid) / sigma_y
    return np.sqrt(dx**2 + dy**2)


def laplacian_weights_from_distance(dist_map):
    return np.exp(-np.sqrt(dist_map)).astype(np.float32)


def normalize_to_probability(weights):
    total = weights.sum()
    if total <= 0.0:
        raise ValueError(f"Weight sum is {total}")
    return (weights / total).astype(np.float32)


def uniform_probability_map(grid_w, grid_h):
    n = grid_h * grid_w
    return np.full((grid_h, grid_w), fill_value=1.0 / n, dtype=np.float32)


def build_gaze_probability_map(
    yaw: Optional[float],
    pitch: Optional[float],
    gaze_valid: bool,
    fu,
    fv,
    cx_native,
    cy_native,
    video_w,
    video_h,
    grid_w,
    grid_h,
    sigma_x_patches=3.0,
    sigma_y_patches=2.0,
) -> np.ndarray:
    if not gaze_valid or yaw is None or pitch is None:
        return uniform_probability_map(grid_w, grid_h)
    if np.isnan(yaw) or np.isnan(pitch):
        return uniform_probability_map(grid_w, grid_h)
    px_native, py_native = project_gaze_to_native_pixel(yaw, pitch, fu, fv, cx_native, cy_native)
    px_video, py_video = scale_pixel_to_video(px_native, py_native, cx_native, cy_native, video_w, video_h)
    if not (0.0 <= px_video < video_w and 0.0 <= py_video < video_h):
        return uniform_probability_map(grid_w, grid_h)
    gx, gy = normalize_pixel_to_unit(px_video, py_video, video_w, video_h)
    cx_grid, cy_grid = unit_to_grid(gx, gy, grid_w, grid_h)
    dist_map = compute_elliptic_distance_map(cx_grid, cy_grid, grid_w, grid_h, sigma_x_patches, sigma_y_patches)
    weights = laplacian_weights_from_distance(dist_map)
    return normalize_to_probability(weights)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PATCH SAMPLING  (from context_patches.py)
# ──────────────────────────────────────────────────────────────────────────────


def compute_patch_budget(total_context, gaze_ratio):
    n_gaze = round(total_context * gaze_ratio)
    n_uniform = total_context - n_gaze
    return n_uniform, n_gaze


def sample_uniform_patches(total_patches, n_samples, rng):
    if n_samples == 0:
        return np.array([], dtype=np.int64)
    return rng.choice(total_patches, size=n_samples, replace=False)


def sample_gaze_patches(prob_map, n_samples, rng):
    total_patches = prob_map.size
    if n_samples == 0:
        return np.array([], dtype=np.int64)
    flat_probs = prob_map.flatten().astype(np.float64)
    flat_probs /= flat_probs.sum()
    return rng.choice(total_patches, size=n_samples, replace=False, p=flat_probs)


def merge_patch_sets(uniform_indices, gaze_indices, total_patches, total_context, rng):
    combined = np.union1d(uniform_indices, gaze_indices)
    if len(combined) == total_context:
        return combined
    if len(combined) > total_context:
        return rng.choice(combined, size=total_context, replace=False)
    n_missing = total_context - len(combined)
    unused = np.setdiff1d(np.arange(total_patches, dtype=np.int64), combined)
    fill = rng.choice(unused, size=n_missing, replace=False)
    return np.concatenate([combined, fill])


def compute_target_indices(context_indices, total_patches):
    return np.setdiff1d(np.arange(total_patches, dtype=np.int64), context_indices)


def select_patches(prob_map, grid_h, grid_w, total_context, gaze_ratio, rng):
    total_patches = grid_h * grid_w
    n_uniform, n_gaze = compute_patch_budget(total_context, gaze_ratio)
    uniform_idx = sample_uniform_patches(total_patches, n_uniform, rng)
    gaze_idx = sample_gaze_patches(prob_map, n_gaze, rng)
    context_idx = merge_patch_sets(uniform_idx, gaze_idx, total_patches, total_context, rng)
    target_idx = compute_target_indices(context_idx, total_patches)
    return context_idx, target_idx


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GAZE DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────


def load_gaze(gaze_csv: str) -> pd.DataFrame:
    df = pd.read_csv(gaze_csv)
    df.columns = df.columns.str.strip()
    
    # timestamp_ns este în nanosecunde, dar valorile sunt absolute
    # Trebuie să le transformăm în secunde relative la începutul înregistrării
    if "timestamp_ns" in df.columns:
        # Prima valoare timestamp_ns
        start_ns = df["timestamp_ns"].iloc[0]
        # Convertim în secunde relative
        df["time_s"] = (df["timestamp_ns"] - start_ns) / 1e9
        print(f"[i] Gaze timestamps: first={df['timestamp_ns'].iloc[0]/1e9:.2f}s, "
              f"last={df['timestamp_ns'].iloc[-1]/1e9:.2f}s")
        print(f"[i] Relative time range: {df['time_s'].min():.2f}s - {df['time_s'].max():.2f}s")
    else:
        print(f"[!] No timestamp column found")
        df["time_s"] = np.arange(len(df)) / 30.0
    
    return df


def get_gaze_for_time(gaze_df: pd.DataFrame, t_sec: float):
    """Return (yaw, pitch, gaze_valid) for the gaze sample nearest to t_sec."""
    
    # Debug first call only
    if not hasattr(get_gaze_for_time, "_debug_printed"):
        print(f"\n[i] GAZE TIMESTAMP DEBUG:")
        print(f"    Looking for t={t_sec}s")
        print(f"    Available range: {gaze_df['time_s'].min():.2f}s - {gaze_df['time_s'].max():.2f}s")
        get_gaze_for_time._debug_printed = True
    
    # Găsește indexul cel mai apropiat
    idx = (gaze_df["time_s"] - t_sec).abs().idxmin()
    row = gaze_df.iloc[idx]
    
    best_match_time = row["time_s"]
    diff = abs(best_match_time - t_sec)
    
    yaw = float(row["yaw"])
    pitch = float(row["pitch"])
    gaze_valid = bool(int(row["gaze_valid"]))
    
    print(f"    t={t_sec:.1f}s → matched gaze at {best_match_time:.3f}s (diff={diff:.4f}s) "
          f"yaw={yaw:.4f}, pitch={pitch:.4f}")
    
    return yaw, pitch, gaze_valid


# ──────────────────────────────────────────────────────────────────────────────
# 3.  VIDEO FRAME EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────


def extract_clip(video_path: str, end_sec: float, n_frames: int, img_size: int) -> Optional[torch.Tensor]:
    """
    Extract n_frames evenly spaced from [end_sec-0.5, end_sec], resize to img_size,
    return tensor shape (1, C, T, H, W) float32 in [0,1].
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_sec = max(0.0, end_sec - 0.5)
    start_f = int(start_sec * fps)
    end_f = min(int(end_sec * fps), total_frames - 1)

    if end_f <= start_f:
        cap.release()
        return None

    frame_indices = np.linspace(start_f, end_f, n_frames, dtype=int)
    frames = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (img_size, img_size))
        frames.append(frame_rgb)
    cap.release()

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,C)
    arr = arr.transpose(3, 0, 1, 2)  # (C,T,H,W)
    return torch.from_numpy(arr).unsqueeze(0)  # (1,C,T,H,W)


def grab_single_frame(video_path: str, t_sec: float, img_size: int) -> np.ndarray:
    """Return a single RGB frame (H,W,3) uint8 for visualisation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame_rgb, (img_size, img_size))


# ──────────────────────────────────────────────────────────────────────────────
# 4.  V-JEPA2 MODEL LOADING (direct from local checkpoint)
# ──────────────────────────────────────────────────────────────────────────────

def load_vjepa2(repo_path: str, checkpoint_path: str, device: torch.device):
    """
    Load V-JEPA2 model directly from local checkpoint file.
    """
    print("[i] Loading V-JEPA2 model from local checkpoint...")
    
    try:
        # Adaugă repo-ul la path pentru importuri
        sys.path.insert(0, os.path.join(repo_path, "src"))
        
        # Importă funcțiile necesare din repo-ul V-JEPA2
        from src.hub.backbones import _make_vjepa2_model, _clean_backbone_key
        
        # Creează modelul (architecture only, fără greutăți)
        encoder, predictor = _make_vjepa2_model(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            num_frames=FRAMES_PER_CLIP,
            pretrained=False,
        )
        
        # Încarcă greutățile din checkpoint
        if os.path.exists(checkpoint_path):
            print(f"[i] Loading checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            
            # Încarcă encoder-ul
            if "target_encoder" in ckpt:
                state_dict = _clean_backbone_key(ckpt["target_encoder"])
                missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"  [i] Encoder missing keys: {len(missing)}")
                if unexpected:
                    print(f"  [i] Encoder unexpected keys: {len(unexpected)}")
            
            # Încarcă predictor-ul
            if "predictor" in ckpt and ckpt["predictor"]:
                pred_state = _clean_backbone_key(ckpt["predictor"])
                missing, unexpected = predictor.load_state_dict(pred_state, strict=False)
                if missing:
                    print(f"  [i] Predictor missing keys: {len(missing)}")
            
            print(f"[✓] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[!] Checkpoint not found at {checkpoint_path} — using random weights")
        
        encoder = encoder.to(device).eval()
        predictor = predictor.to(device).eval()
        
        print(f"[✓] V-JEPA2 model ready on {device}")
        return encoder, predictor, False
        
    except Exception as e:
        print(f"[!] Failed to load V-JEPA2 model: {e}")
        print("[i] Falling back to stub model...")
        return _make_stub_encoder_predictor(device)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  INFERENCE  – run encoder + predictor for one set of patch indices
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_inference(
    encoder,
    predictor,
    clip_tensor: torch.Tensor,
    context_idx: np.ndarray,
    target_idx: np.ndarray,
    device: torch.device,
    is_stub: bool,
):
    """
    Returns (pred_tokens, target_tokens) both shape (N_target, D).
    """
    clip = clip_tensor.to(device)  # (1, C, T, H, W)

    ctx_mask = torch.from_numpy(context_idx).long().unsqueeze(0).to(device)
    tgt_mask = torch.from_numpy(target_idx).long().unsqueeze(0).to(device)

    if is_stub:
        all_tokens = encoder(clip)  # (1, N_all, D)
        ctx_tokens = all_tokens[:, context_idx, :]
        tgt_tokens = all_tokens[:, target_idx, :]
        pred_tokens = predictor(ctx_tokens, ctx_mask, tgt_mask)
        # Align sizes
        min_n = min(pred_tokens.shape[1], tgt_tokens.shape[1])
        return (pred_tokens[0, :min_n, :].cpu().numpy(), tgt_tokens[0, :min_n, :].cpu().numpy())
    else:
        # Real V-JEPA2 API: encoder accepts mask list
        all_tokens = encoder(clip, masks=None)  # (1, N_all, D)
        ctx_tokens = all_tokens[:, context_idx, :]
        tgt_tokens = all_tokens[:, target_idx, :]
        pred_tokens = predictor(ctx_tokens, [ctx_mask], [tgt_mask])
        if isinstance(pred_tokens, (list, tuple)):
            pred_tokens = pred_tokens[0]
        min_n = min(pred_tokens.shape[1], tgt_tokens.shape[1])
        return (pred_tokens[0, :min_n, :].cpu().float().numpy(), tgt_tokens[0, :min_n, :].cpu().float().numpy())


# ──────────────────────────────────────────────────────────────────────────────
# 6.  METRICS
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """pred, target: (N, D) arrays"""
    diff = pred - target
    l2 = float(np.linalg.norm(diff, axis=-1).mean())
    mse = float(np.mean(diff**2))

    p_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    t_norm = target / (np.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    cos_sim = float(np.mean(np.sum(p_norm * t_norm, axis=-1)))

    # per-token variance overlap (how spread the errors are)
    pred_var = float(np.var(pred, axis=0).mean())
    tgt_var = float(np.var(target, axis=0).mean())

    return dict(l2=l2, mse=mse, cos_sim=cos_sim, pred_var=pred_var, tgt_var=tgt_var)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PATCH GRID OVERLAY VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────


def draw_patch_overlay(
    ax,
    frame_img: np.ndarray,
    context_idx: np.ndarray,
    target_idx: np.ndarray,
    grid_h: int,
    grid_w: int,
    title: str,
    pred: np.ndarray,
    target: np.ndarray,
):
    """
    Draw the video frame with a semi-transparent patch grid overlay.
    Context patches = blue, Target patches = red.
    Patch brightness encodes per-patch cosine similarity (brighter = more similar).
    """
    H, W, _ = frame_img.shape
    ph = H / grid_h  # patch height in pixels
    pw = W / grid_w  # patch width  in pixels

    ax.imshow(frame_img)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")

    # per-token cos sim (min-max normalised → alpha)
    p_n = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    t_n = target / (np.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    per_tok_cos = np.sum(p_n * t_n, axis=-1)  # (N_target,)
    mn, mx = per_tok_cos.min(), per_tok_cos.max()
    norm_cos = (per_tok_cos - mn) / (mx - mn + 1e-8)

    # draw context patches
    for idx in context_idx:
        row = idx // grid_w
        col = idx % grid_w
        rect = mpatches.FancyBboxPatch(
            (col * pw, row * ph),
            pw,
            ph,
            boxstyle="square,pad=0",
            linewidth=0.3,
            edgecolor="cyan",
            facecolor=(0.0, 0.5, 1.0, 0.25),
        )
        ax.add_patch(rect)

    # draw target patches coloured by cos-sim quality
    for i, idx in enumerate(target_idx):
        row = idx // grid_w
        col = idx % grid_w
        alpha = 0.15 + 0.55 * norm_cos[i] if i < len(norm_cos) else 0.3
        rect = mpatches.FancyBboxPatch(
            (col * pw, row * ph),
            pw,
            ph,
            boxstyle="square,pad=0",
            linewidth=0.3,
            edgecolor="red",
            facecolor=(1.0, 0.15, 0.1, alpha),
        )
        ax.add_patch(rect)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  METRICS BAR / RADAR SIDE-PANEL
# ──────────────────────────────────────────────────────────────────────────────


def draw_metrics_panel(ax, m_uniform: dict, m_gaze: dict):
    """
    A grouped horizontal bar chart comparing uniform vs gaze metrics.
    """
    metric_keys = ["cos_sim", "l2", "mse"]
    metric_labels = ["Cosine Similarity ↑", "Mean L2 ↓", "MSE ↓"]

    vals_uniform = [m_uniform[k] for k in metric_keys]
    vals_gaze = [m_gaze[k] for k in metric_keys]

    y = np.arange(len(metric_keys))
    height = 0.32

    bars_u = ax.barh(y + height / 2, vals_uniform, height, label="Uniform", color="#4a90d9", alpha=0.85)
    bars_g = ax.barh(y - height / 2, vals_gaze, height, label="Gaze-20%", color="#e87040", alpha=0.85)

    for bar, val in zip(bars_u, vals_uniform):
        ax.text(
            bar.get_width() + 1e-4,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=7,
        )
    for bar, val in zip(bars_g, vals_gaze):
        ax.text(
            bar.get_width() + 1e-4,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=7,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(metric_labels, fontsize=8)
    ax.set_xlabel("Value", fontsize=8)
    ax.set_title("Metrics: Uniform vs Gaze-20%", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_token_distribution(ax, pred_u, tgt_u, pred_g, tgt_g, title="Token Embedding PCA"):
    """
    2D PCA of context + predicted tokens coloured by method.
    """
    try:
        from sklearn.decomposition import PCA

        all_vecs = np.vstack([pred_u, tgt_u, pred_g, tgt_g])
        pca = PCA(n_components=2)
        proj = pca.fit_transform(all_vecs)
        n1, n2, n3 = len(pred_u), len(tgt_u), len(pred_g)
        p_u = proj[:n1]
        t_u = proj[n1 : n1 + n2]
        p_g = proj[n1 + n2 : n1 + n2 + n3]
        t_g = proj[n1 + n2 + n3 :]

        ax.scatter(*t_u.T, s=12, alpha=0.5, color="#4a90d9", label="Target (Uniform)", marker="o")
        ax.scatter(*p_u.T, s=12, alpha=0.5, color="#a0c8f0", label="Pred (Uniform)", marker="^")
        ax.scatter(*t_g.T, s=12, alpha=0.5, color="#e87040", label="Target (Gaze)", marker="o")
        ax.scatter(*p_g.T, s=12, alpha=0.5, color="#f5b08a", label="Pred (Gaze)", marker="^")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.legend(fontsize=6, markerscale=1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except ImportError:
        ax.text(
            0.5,
            0.5,
            "sklearn not available\n(PCA skipped)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.axis("off")


def draw_cosine_histogram(ax, pred_u, tgt_u, pred_g, tgt_g):
    """Per-token cosine similarity histograms for both strategies."""

    def cos_per_tok(p, t):
        pn = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
        tn = t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)
        return (pn * tn).sum(axis=-1)

    n = min(len(pred_u), len(tgt_u), len(pred_g), len(tgt_g))
    cu = cos_per_tok(pred_u[:n], tgt_u[:n])
    cg = cos_per_tok(pred_g[:n], tgt_g[:n])

    bins = np.linspace(-1, 1, 30)
    ax.hist(cu, bins=bins, alpha=0.6, color="#4a90d9", label=f"Uniform  μ={cu.mean():.3f}")
    ax.hist(cg, bins=bins, alpha=0.6, color="#e87040", label=f"Gaze-20% μ={cg.mean():.3f}")
    ax.axvline(cu.mean(), color="#1a5fa8", linestyle="--", linewidth=1.0)
    ax.axvline(cg.mean(), color="#b03010", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Cosine Similarity", fontsize=8)
    ax.set_ylabel("Token count", fontsize=8)
    ax.set_title("Per-token Cos-Sim Distribution", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_gaze_heatmap(ax, prob_map: np.ndarray, frame_img: np.ndarray):
    """Overlay the gaze probability map on the frame thumbnail."""
    H, W, _ = frame_img.shape
    heat = cv2.resize(prob_map, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    ax.imshow(frame_img, alpha=0.6)
    ax.imshow(heat_norm, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
    ax.set_title("Gaze Probability Heatmap", fontsize=9, fontweight="bold")
    ax.axis("off")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  FULL FIGURE FOR ONE LANDMARK FRAME
# ──────────────────────────────────────────────────────────────────────────────


def make_figure(
    label: str,
    t_sec: float,
    frame_img: np.ndarray,
    prob_map: np.ndarray,
    ctx_u,
    tgt_u,
    ctx_g,
    tgt_g,
    pred_u,
    gt_u,
    pred_g,
    gt_g,
    m_uniform,
    m_gaze,
    grid_h,
    grid_w,
    output_dir: str,
    is_stub: bool,
):

    fig = plt.figure(figsize=(20, 13), facecolor="#1a1a2e")

    # ── Layout: top row = frame overlays (2) + gaze heat (1)
    #            mid row = metrics bar   (1) + cos-sim hist (1)
    #            bot row = PCA scatter   (2) + summary table (1)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32, left=0.04, right=0.97, top=0.92, bottom=0.05)

    ax_uni = fig.add_subplot(gs[0, 0])
    ax_gaz = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_pca = fig.add_subplot(gs[1, 2])
    ax_tab = fig.add_subplot(gs[2, :])

    # style axes
    for ax in fig.get_axes():
        ax.set_facecolor("#0d0d1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.tick_params(colors="#aaaacc", labelsize=7)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("#ddddf5")

    # Row 0 ── patch overlays
    draw_patch_overlay(
        ax_uni,
        frame_img,
        ctx_u,
        tgt_u,
        grid_h,
        grid_w,
        f"UNIFORM  |  cos={m_uniform['cos_sim']:.4f}  l2={m_uniform['l2']:.4f}",
        pred_u,
        gt_u,
    )
    draw_patch_overlay(
        ax_gaz,
        frame_img,
        ctx_g,
        tgt_g,
        grid_h,
        grid_w,
        f"GAZE-20% |  cos={m_gaze['cos_sim']:.4f}  l2={m_gaze['l2']:.4f}",
        pred_g,
        gt_g,
    )
    draw_gaze_heatmap(ax_heat, prob_map, frame_img)

    # Row 1 ── metric plots
    draw_metrics_panel(ax_bar, m_uniform, m_gaze)
    draw_cosine_histogram(ax_hist, pred_u, gt_u, pred_g, gt_g)
    draw_token_distribution(ax_pca, pred_u, gt_u, pred_g, gt_g, title="Token PCA (Pred vs Target)")

    # Row 2 ── summary table
    ax_tab.axis("off")
    table_data = [
        ["Metric", "Uniform", "Gaze-20%", "Δ (Gaze−Uni)"],
        [
            "Cosine Sim ↑",
            f"{m_uniform['cos_sim']:.6f}",
            f"{m_gaze['cos_sim']:.6f}",
            f"{m_gaze['cos_sim']-m_uniform['cos_sim']:+.6f}",
        ],
        ["Mean L2 ↓", f"{m_uniform['l2']:.6f}", f"{m_gaze['l2']:.6f}", f"{m_gaze['l2']-m_uniform['l2']:+.6f}"],
        ["MSE ↓", f"{m_uniform['mse']:.6f}", f"{m_gaze['mse']:.6f}", f"{m_gaze['mse']-m_uniform['mse']:+.6f}"],
        [
            "Pred Var",
            f"{m_uniform['pred_var']:.6f}",
            f"{m_gaze['pred_var']:.6f}",
            f"{m_gaze['pred_var']-m_uniform['pred_var']:+.6f}",
        ],
        [
            "Tgt  Var",
            f"{m_uniform['tgt_var']:.6f}",
            f"{m_gaze['tgt_var']:.6f}",
            f"{m_gaze['tgt_var']-m_uniform['tgt_var']:+.6f}",
        ],
    ]
    tbl = ax_tab.table(cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#0d0d1a" if r > 0 else "#22224a")
        cell.set_edgecolor("#333355")
        cell.set_text_props(color="#ddddf5")

    stub_note = "  [stub encoder — random weights, results not meaningful]" if is_stub else ""
    fig.suptitle(
        f"V-JEPA2 Gaze vs Uniform  |  Frame: {label}  (t={t_sec:.1f}s){stub_note}",
        fontsize=12,
        fontweight="bold",
        color="#ffffff",
        y=0.975,
    )

    out_path = os.path.join(output_dir, f"{label}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Saved: {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 10.  SUMMARY ACROSS ALL FRAMES
# ──────────────────────────────────────────────────────────────────────────────


def make_summary_figure(all_labels, all_uniform, all_gaze, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#1a1a2e")
    metrics = ["cos_sim", "l2", "mse"]
    titles = ["Cosine Similarity ↑", "Mean L2 ↓", "MSE ↓"]
    x = np.arange(len(all_labels))

    for ax, mk, mt in zip(axes, metrics, titles):
        ax.set_facecolor("#0d0d1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.tick_params(colors="#aaaacc", labelsize=7)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("#ddddf5")

        yu = [m[mk] for m in all_uniform]
        yg = [m[mk] for m in all_gaze]
        ax.plot(x, yu, "o-", color="#4a90d9", linewidth=1.8, label="Uniform", markersize=6)
        ax.plot(x, yg, "s-", color="#e87040", linewidth=1.8, label="Gaze-20%", markersize=6)
        ax.fill_between(x, yu, yg, alpha=0.12, color="#ffffff")
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(mt, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Summary: Gaze-20% vs Uniform across all landmark frames", fontsize=12, fontweight="bold", color="#ffffff"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, "SUMMARY.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Summary saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 11.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device}")

    # ── grid dimensions (spatial patches only; temporal handled by clip)
    grid_h = IMG_SIZE // PATCH_SIZE  # 14
    grid_w = IMG_SIZE // PATCH_SIZE  # 14
    total_patches = grid_h * grid_w  # 196
    total_context = int(total_patches * CONTEXT_RATIO)

    print(f"[i] Patch grid: {grid_h}×{grid_w}  |  context={total_context}  target={total_patches-total_context}")

    # ── load data
    print("[i] Loading gaze data …")
    gaze_df = load_gaze(GAZE_CSV)

    # ── load model (pass device parameter)
    print("[i] Loading V-JEPA2 …")
    encoder, predictor, is_stub = load_vjepa2(VJEPA2_REPO, CHECKPOINT, device)

    all_labels, all_uniform_metrics, all_gaze_metrics = [], [], []

    for label, t_sec in LANDMARKS.items():
        print(f"\n── Frame: {label}  (t={t_sec}s) ──")

        # 1. video clip
        clip = extract_clip(VIDEO_PATH, t_sec, FRAMES_PER_CLIP, IMG_SIZE)
        if clip is None:
            print(f"  [!] Could not extract clip at t={t_sec}s — skipping")
            continue
        frame_img = grab_single_frame(VIDEO_PATH, t_sec, IMG_SIZE)

        # 2. gaze
        yaw, pitch, gaze_valid = get_gaze_for_time(gaze_df, t_sec)
        print(f"  Gaze: yaw={yaw:.4f} pitch={pitch:.4f} valid={gaze_valid}")

        prob_map = build_gaze_probability_map(
            yaw,
            pitch,
            gaze_valid,
            FU,
            FV,
            CX_NATIVE,
            CY_NATIVE,
            VIDEO_W,
            VIDEO_H,
            grid_w,
            grid_h,
        )

        rng = np.random.default_rng(SEED)

        # 3. UNIFORM patches
        ctx_u, tgt_u = select_patches(
            uniform_probability_map(grid_w, grid_h), 
            grid_h, grid_w, total_context, 0.0, rng
        )

        rng = np.random.default_rng(SEED)  # same seed → fair comparison

        # 4. GAZE-20 patches
        ctx_g, tgt_g = select_patches(
            prob_map, grid_h, grid_w, total_context, GAZE_RATIO, rng
        )

        # 5. inference
        print("  Running encoder + predictor …")
        pred_u, gt_u = run_inference(encoder, predictor, clip, ctx_u, tgt_u, device, is_stub)
        pred_g, gt_g = run_inference(encoder, predictor, clip, ctx_g, tgt_g, device, is_stub)

        # 6. metrics
        m_u = compute_metrics(pred_u, gt_u)
        m_g = compute_metrics(pred_g, gt_g)
        print(f"  Uniform  → cos={m_u['cos_sim']:.4f}  l2={m_u['l2']:.4f}  mse={m_u['mse']:.6f}")
        print(f"  Gaze-20% → cos={m_g['cos_sim']:.4f}  l2={m_g['l2']:.4f}  mse={m_g['mse']:.6f}")

        # 7. figure
        make_figure(
            label,
            t_sec,
            frame_img,
            prob_map,
            ctx_u,
            tgt_u,
            ctx_g,
            tgt_g,
            pred_u,
            gt_u,
            pred_g,
            gt_g,
            m_u,
            m_g,
            grid_h,
            grid_w,
            OUTPUT_DIR,
            is_stub,
        )

        all_labels.append(label)
        all_uniform_metrics.append(m_u)
        all_gaze_metrics.append(m_g)

    # 8. summary
    if all_labels:
        print("\n── Generating summary figure …")
        make_summary_figure(all_labels, all_uniform_metrics, all_gaze_metrics, OUTPUT_DIR)

    print(f"\n[✓] All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
