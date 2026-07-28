import argparse
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. CALIBRARE ȘI PROIECȚIE GAZE
# ==========================================

def load_rgb_calib(jsonl_path):
    """Încarcă calibrarea camerei RGB din JSONL"""
    with open(jsonl_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                break
            except: continue
    
    cam = next(c for c in data["CameraCalibrations"] if "camera-rgb" in c["Label"].lower())
    p = cam["Projection"]["Params"]
    
    # Detectare format parametri
    if len(p) >= 4:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    
    # Coeficienți distorsiune (pentru corecție)
    dist = np.array(p[4:8], dtype=np.float32) if len(p) > 8 else np.zeros(4)
    
    return fx, fy, cx, cy, dist

def project_gaze_to_pixel(yaw, pitch, fx, fy, cx, cy):
    """Proiectează unghiurile gaze în coordonate pixel"""
    px = cx + fx * np.tan(yaw)
    py = cy - fy * (np.tan(pitch) / np.cos(yaw))
    return px, py

def create_gaze_bias_heatmap(px, py, width, height, grid_size=14, sigma=1.5, strength=0.3):
    """
    Creează o hartă de bias Gaussiană.
    strength = 0.3 înseamnă că zona privirii este doar cu 30% mai importantă.
    """
    # Mapează pixelul în grid
    grid_x = (px / width) * grid_size
    grid_y = (py / height) * grid_size
    
    x = np.arange(0, grid_size, 1, float)
    y = np.arange(0, grid_size, 1, float)[:, np.newaxis]
    
    # Gaussian 2D centrat pe privire
    gaussian = np.exp(-((x - grid_x)**2 + (y - grid_y)**2) / (2 * sigma**2))
    
    # Bias subtil: 1 + strength * gaussian (nu 0 în afara zonei!)
    # Astfel, zona privirii e doar "sugerată", nu impusă
    bias = 1.0 + strength * gaussian
    
    # Normalizare pentru a păstra media ~1
    bias = bias / bias.mean()
    
    return torch.from_numpy(bias).float()

def apply_gaze_bias_to_patches(patch_importance, gaze_bias, alpha=0.3):
    """
    Aplică bias-ul subtil asupra importanței patch-urilor.
    alpha = 0.3 înseamnă 30% bias, 70% păstrează din importanța originală.
    """
    # Bias ponderat: păstrează informația originală, doar o ajustează ușor
    biased_importance = (1 - alpha) * patch_importance + alpha * (patch_importance * gaze_bias)
    
    # Renormalizare
    biased_importance = biased_importance / (biased_importance.sum() + 1e-8)
    
    return biased_importance

# ==========================================
# 2. PROCESARE VIDEO ȘI FRAME-URI
# ==========================================

def undistort_frame(frame, K, D):
    """Corecție fisheye pentru camera Aria"""
    h, w = frame.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.5)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

def extract_clip(video_path, start_sec, num_frames, K=None, D=None):
    """Extrage un clip video ca PIL Images"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, start_sec * 1000))
    
    frames = []
    display_frame = None
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if K is not None:
            frame_rgb = undistort_frame(frame_rgb, K, D)
        
        if i == num_frames // 2:
            display_frame = frame_rgb.copy()
        
        resized = cv2.resize(frame_rgb, (224, 224))
        frames.append(Image.fromarray(resized))
    
    cap.release()
    
    # Padding dacă nu sunt suficiente frame-uri
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return frames, display_frame

# ==========================================
# 3. VIZUALIZARE COMPARATIVĂ
# ==========================================

def create_comparative_visualization(original_importance, biased_importance, 
                                       gaze_bias, frame, ts, future_sec, output_path):
    """
    Creează o vizualizare comparativă între atenția originală și cea biasată.
    """
    fig = plt.figure(figsize=(20, 10), facecolor='black')
    
    # Grid: 2 rânduri x 3 coloane
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.8], wspace=0.1, hspace=0.1)
    
    # Rândul 1: Vizualizări
    # Coloana 1: Frame original
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(frame)
    ax0.set_title(f"SCENE CONTEXT\nt={ts:.1f}s", color='white', fontsize=12, fontweight='bold')
    ax0.axis('off')
    
    # Coloana 2: Atenție originală V-JEPA
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(original_importance, cmap='hot', interpolation='bilinear')
    ax1.set_title("ORIGINAL V-JEPA ATTENTION", color='#FFFF00', fontsize=11, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Coloana 3: Atenție cu Gaze Bias
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(biased_importance, cmap='plasma', interpolation='bilinear')
    ax2.set_title("GAZE-GUIDED ATTENTION", color='#00FF00', fontsize=11, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Rândul 2: Overlay-uri
    # Coloana 1: Gaze bias heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(gaze_bias, cmap='viridis', interpolation='bilinear')
    ax3.set_title("GAZE PRIOR (Gaussian Bias)", color='cyan', fontsize=11)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Coloana 2: Overlay atenție originală pe frame
    ax4 = fig.add_subplot(gs[1, 1])
    # Upscale importance la dimensiunea frame-ului
    imp_resized = cv2.resize(original_importance, (frame.shape[1], frame.shape[0]))
    overlay_orig = np.clip(0.6 * frame/255.0 + 0.4 * plt.cm.hot(imp_resized)[:, :, :3], 0, 1)
    ax4.imshow(overlay_orig)
    ax4.set_title("ORIGINAL ATTENTION OVERLAY", color='#FFFF00', fontsize=11)
    ax4.axis('off')
    
    # Coloana 3: Overlay atenție biasată pe frame
    ax5 = fig.add_subplot(gs[1, 2])
    imp_biased_resized = cv2.resize(biased_importance, (frame.shape[1], frame.shape[0]))
    overlay_biased = np.clip(0.6 * frame/255.0 + 0.4 * plt.cm.plasma(imp_biased_resized)[:, :, :3], 0, 1)
    ax5.imshow(overlay_biased)
    ax5.set_title("GAZE-GUIDED OVERLAY", color='#00FF00', fontsize=11)
    ax5.axis('off')
    
    # Titlu principal
    fig.suptitle(f"V-JEPA 2 + Gaze Guidance | Prediction Horizon: +{future_sec}s", 
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='black', dpi=150)
    plt.close()

# ==========================================
# 4. PIPELINE PRINCIPAL
# ==========================================

def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    
    print("="*70)
    print("V-JEPA 2 + GAZE GUIDANCE PIPELINE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Gaze bias strength: {args.gaze_strength}")
    print(f"Gaze sigma: {args.gaze_sigma}")
    print(f"Bias alpha: {args.bias_alpha}")
    
    # 1. Încărcare date gaze
    print("\n📊 Loading gaze data...")
    df_gaze = pd.read_csv(args.gaze_csv)
    
    # Detectare coloane timestamp
    if 'timestamp_ns' in df_gaze.columns:
        df_gaze['ts_us'] = (df_gaze['timestamp_ns'] / 1000.0).astype(np.int64)
    elif 'tracking_timestamp_us' in df_gaze.columns:
        df_gaze['ts_us'] = df_gaze['tracking_timestamp_us'].astype(np.int64)
    else:
        raise ValueError("No timestamp column found in gaze CSV")
    
    # Detectare coloane yaw/pitch
    if 'yaw' in df_gaze.columns and 'pitch' in df_gaze.columns:
        df_gaze['yaw_use'] = df_gaze['yaw']
        df_gaze['pitch_use'] = df_gaze['pitch']
    elif 'left_yaw_rads_cpf' in df_gaze.columns:
        df_gaze['yaw_use'] = (df_gaze['left_yaw_rads_cpf'] + df_gaze['right_yaw_rads_cpf']) / 2
        df_gaze['pitch_use'] = df_gaze['pitch_rads_cpf']
    else:
        raise ValueError("No yaw/pitch columns found in gaze CSV")
    
    df_gaze = df_gaze.sort_values('ts_us').reset_index(drop=True)
    gaze_start_us = df_gaze['ts_us'].iloc[0]
    print(f"   Gaze samples: {len(df_gaze):,}")
    print(f"   Start time: {gaze_start_us:,} us")
    
    # 2. Încărcare model V-JEPA 2
    print("\n🤖 Loading V-JEPA 2 model...")
    model_id = "facebook/vjepa2-vitl-fpc64-256"
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    if device.type == "cuda":
        model = model.to(torch.float16)
    print("   ✓ Model loaded")
    
    # 3. Încărcare calibrare
    print("\n📐 Loading calibration...")
    fx_nat, fy_nat, cx_nat, cy_nat, dist_coeffs = load_rgb_calib(args.calib)
    
    # 4. Informații video
    cap_temp = cv2.VideoCapture(args.video)
    vw = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
    cap_temp.release()
    
    # Construim K matrix pentru undistort
    K = np.array([[fx_nat, 0, cx_nat], [0, fy_nat, cy_nat], [0, 0, 1]], dtype=np.float32)
    
    # Scalare calibrare la rezoluția video
    native_w = cx_nat * 2.0
    native_h = cy_nat * 2.0
    sx, sy = vw / native_w, vh / native_h
    fx, fy = fx_nat * sx, fy_nat * sy
    cx, cy = cx_nat * sx, cy_nat * sy
    
    print(f"   Video: {vw}x{vh} @ {fps:.1f}fps")
    print(f"   Scaled intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # 5. Parse timestamps
    ts_list = [float(x.strip()) for x in args.timestamps.split(',')]
    print(f"\n🎯 Timestamps to analyze: {ts_list}")
    
    # 6. Pipeline principal
    results = []
    
    for idx, ts in enumerate(ts_list):
        print(f"\n[{idx+1}/{len(ts_list)}] Processing t={ts:.1f}s...")
        
        context_ts = ts - args.future_sec
        
        # Extrage clip context
        frames, display_frame = extract_clip(args.video, context_ts, args.num_frames, K, dist_coeffs)
        
        if not frames or display_frame is None:
            print(f"   ⚠️ Skipping - insufficient frames")
            continue
        
        # Sincronizare gaze pentru momentul context
        cur_us = gaze_start_us + int(context_ts * 1e6)
        gaze_idx = (df_gaze['ts_us'] - cur_us).abs().idxmin()
        gaze_row = df_gaze.iloc[gaze_idx]
        
        # Creează gaze bias heatmap
        gaze_bias_2d = torch.ones((14, 14))  # Bias neutru (1.0 peste tot)
        
        # Verifică dacă gaze-ul este valid
        gaze_valid = gaze_row.get('gaze_valid', 1)
        if gaze_valid == 1:
            yaw = float(gaze_row['yaw_use'])
            pitch = float(gaze_row['pitch_use'])
            
            if not (np.isnan(yaw) or np.isnan(pitch)):
                try:
                    px, py = project_gaze_to_pixel(yaw, pitch, fx, fy, cx, cy)
                    
                    if 0 <= px < vw and 0 <= py < vh:
                        gaze_bias_2d = create_gaze_bias_heatmap(
                            px, py, vw, vh, 
                            grid_size=14, 
                            sigma=args.gaze_sigma,
                            strength=args.gaze_strength
                        )
                        print(f"   👁️ Gaze at ({px:.0f}, {py:.0f}) → bias applied")
                    else:
                        print(f"   👁️ Gaze outside frame: ({px:.0f}, {py:.0f})")
                except Exception as e:
                    print(f"   👁️ Error projecting gaze: {e}")
            else:
                print(f"   👁️ Invalid gaze angles (NaN)")
        else:
            print(f"   👁️ Gaze invalid (valid={gaze_valid}) → using neutral bias")
        
        # Rulează V-JEPA
        inputs = processor(videos=frames, return_tensors="pt").to(device)
        if device.type == "cuda":
            inputs = {k: v.to(torch.float16) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, skip_predictor=False)
            z_pred = outputs.last_hidden_state  # [1, tokens, dim]
            
            # Calculează importanța patch-urilor (norma fiecărui patch)
            patch_importance = torch.norm(z_pred[0, 1:, :], dim=-1).cpu().numpy()
            patch_importance_2d = patch_importance.reshape(14, 14)
            patch_importance_2d = (patch_importance_2d - patch_importance_2d.min()) / (patch_importance_2d.max() - patch_importance_2d.min() + 1e-8)
            
            # Aplică gaze bias subtil
            biased_importance = apply_gaze_bias_to_patches(
                torch.from_numpy(patch_importance_2d), 
                gaze_bias_2d, 
                alpha=args.bias_alpha
            ).numpy()
            
            # Calculează metrici de comparație
            # Cât de mult s-a schimbat atenția?
            attention_shift = np.abs(biased_importance - patch_importance_2d).mean()
            
            # Câtă atenție a ajuns în zona privirii?
            gaze_bias_np = gaze_bias_2d.numpy()
            gaze_attention_original = (patch_importance_2d * gaze_bias_np).sum()
            gaze_attention_biased = (biased_importance * gaze_bias_np).sum()
            
            results.append({
                'timestamp': ts,
                'gaze_valid': int(gaze_valid),
                'attention_shift': attention_shift,
                'gaze_attention_original': gaze_attention_original,
                'gaze_attention_biased': gaze_attention_biased,
                'attention_boost': gaze_attention_biased - gaze_attention_original
            })
            
            print(f"   📊 Attention shift: {attention_shift:.4f}")
            print(f"   📊 Gaze attention boost: +{results[-1]['attention_boost']:.4f}")
        
        # Vizualizare comparativă
        vis_path = output_dir / "visualizations" / f"gaze_guidance_t{int(ts)}.png"
        create_comparative_visualization(
            patch_importance_2d, biased_importance, 
            gaze_bias_np, display_frame, 
            ts, args.future_sec, vis_path
        )
        print(f"   💾 Saved: {vis_path}")
    
    # 7. Salvare rezultate și raport
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "gaze_guidance_metrics.csv", index=False)
    
    # Raport sumar
    with open(output_dir / "gaze_guidance_report.txt", "w") as f:
        f.write("="*70 + "\n")
        f.write("V-JEPA 2 + GAZE GUIDANCE REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Gaze strength: {args.gaze_strength}\n")
        f.write(f"Gaze sigma: {args.gaze_sigma}\n")
        f.write(f"Bias alpha: {args.bias_alpha}\n")
        f.write(f"Prediction horizon: +{args.future_sec}s\n")
        f.write(f"Samples analyzed: {len(results)}\n\n")
        
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        if len(results) > 0:
            f.write(f"Mean attention shift: {df_results['attention_shift'].mean():.4f}\n")
            f.write(f"Mean gaze attention (original): {df_results['gaze_attention_original'].mean():.4f}\n")
            f.write(f"Mean gaze attention (biased): {df_results['gaze_attention_biased'].mean():.4f}\n")
            f.write(f"Mean attention boost: {df_results['attention_boost'].mean():.4f}\n\n")
        
        f.write("PER-TIMESTAMP RESULTS\n")
        f.write("-"*70 + "\n")
        for _, row in df_results.iterrows():
            f.write(f"t={row['timestamp']:.1f}s: shift={row['attention_shift']:.3f}, "
                   f"boost={row['attention_boost']:.3f}\n")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   📊 Metrics: {output_dir / 'gaze_guidance_metrics.csv'}")
    print(f"   📄 Report: {output_dir / 'gaze_guidance_report.txt'}")
    print(f"   🖼️ Visualizations: {output_dir / 'visualizations/'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V-JEPA 2 with Gaze Guidance')
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--gaze_csv", required=True, help="Path to gaze CSV file")
    parser.add_argument("--calib", required=True, help="Path to calibration JSONL")
    parser.add_argument("--timestamps", required=True, help="Comma-separated timestamps")
    parser.add_argument("--output", default="./vjepa_gaze_results", help="Output directory")
    parser.add_argument("--future_sec", type=float, default=2.0, help="Prediction horizon")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per clip")
    parser.add_argument("--gaze_strength", type=float, default=0.3, 
                        help="Strength of gaze bias (0=no bias, 1=strong bias)")
    parser.add_argument("--gaze_sigma", type=float, default=1.5, 
                        help="Sigma for Gaussian gaze bias")
    parser.add_argument("--bias_alpha", type=float, default=0.3, 
                        help="Weight of bias vs original attention (0=no bias, 1=full bias)")
    
    args = parser.parse_args()
    run_pipeline(args)