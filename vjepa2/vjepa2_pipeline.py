import argparse
import json
import os
import csv
import warnings
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor

warnings.filterwarnings("ignore", category=FutureWarning)

def load_aria_calibration(calib_path):
    with open(calib_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                break
            except: continue
    cam = next(c for c in data["CameraCalibrations"] if "camera-rgb" in c["Label"].lower())
    p = cam["Projection"]["Params"]
    K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float32)
    D = np.array(p[4:8], dtype=np.float32)
    return K, D

def undistort_frame(frame, K, D):
    h, w = frame.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.5)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def extract_clip(video_path, start_sec, num_frames, K, D):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, start_sec * 1000))
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clean = undistort_frame(frame_rgb, K, D)
        frames.append(Image.fromarray(cv2.resize(clean, (224, 224))))
    cap.release()
    return frames

def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    
    print(f"--- Deep Metrics Mode: V-JEPA 2 on {device} ---")
    model_id = "facebook/vjepa2-vitl-fpc64-256"
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    
    K, D = load_aria_calibration(args.calib)
    ts_list = [float(x.strip()) for x in args.timestamps.split(',')]
    metrics_list = []

    for ts in ts_list:
        context_ts = ts - args.future_sec
        print(f"Crunching: Target {ts}s...")
        
        f_now = extract_clip(args.video, context_ts, args.num_frames, K, D)
        f_fut = extract_clip(args.video, ts, args.num_frames, K, D)
        
        if len(f_now) < args.num_frames or len(f_fut) < args.num_frames: continue

        in_now = processor(videos=f_now, return_tensors="pt").to(device)
        in_fut = processor(videos=f_fut, return_tensors="pt").to(device)

        with torch.no_grad():
            z_pred_raw = model(**in_now, skip_predictor=False).last_hidden_state
            z_gt_raw = model(**in_fut, skip_predictor=True).last_hidden_state
            
            # Vectori globali (pooling pe patch-uri)
            v_pred = z_pred_raw.mean(dim=1)
            v_gt = z_gt_raw.mean(dim=1)
            
            # 1. Cosine Similarity (Semantic direction)
            cos_sim = F.cosine_similarity(v_pred, v_gt).item()
            
            # 2. L2 Distance (Magnitude error)
            l2_dist = torch.dist(v_pred, v_gt).item()
            
            # 3. Latent Variance (Cât de "împrăștiate" sunt trăsăturile prezise)
            pred_variance = torch.var(z_pred_raw).item()
            
            # 4. Signal-to-Noise Ratio (Aproximativ)
            snr = (torch.mean(v_gt**2) / (torch.mean((v_gt - v_pred)**2) + 1e-8)).item()

        metrics_list.append({
            'target_ts': ts,
            'cos_sim': cos_sim,
            'l2_dist': l2_dist,
            'latent_var': pred_variance,
            'snr_latent': snr
        })

    # Calcule statistice finale
    all_cos = [m['cos_sim'] for m in metrics_list]
    all_l2 = [m['l2_dist'] for m in metrics_list]
    
    csv_path = os.path.join(args.output, 'deep_metrics_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_list[0].keys())
        writer.writeheader()
        writer.writerows(metrics_list)
        
        # Adăugăm rânduri de sumar la final
        f.write("\n--- STATISTICS ---\n")
        f.write(f"Mean CosSim,{np.mean(all_cos):.4f}\n")
        f.write(f"Std Dev CosSim,{np.std(all_cos):.4f}\n")
        f.write(f"Mean L2 Dist,{np.mean(all_l2):.4f}\n")
        f.write(f"Max L2 Error,{np.max(all_l2):.4f}\n")

    print(f"\n[COMPLETE] Raport detaliat generat în: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--timestamps", required=True)
    parser.add_argument("--output", default="./vjepa_deep_analysis")
    parser.add_argument("--future_sec", type=float, default=2.0)
    parser.add_argument("--num_frames", type=int, default=16)
    run_pipeline(parser.parse_args())