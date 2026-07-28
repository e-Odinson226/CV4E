import cv2
import pandas as pd
import numpy as np
import os
from gaussian_map import build_gaze_probability_map

# PATH-URI (Verifică să fie corecte pentru sistemul tău)
VIDEO_PATH = "/home/imarica/clean_0_data/AriaGen2PilotDataset_v1.0_clean_0_preview_rgb.mp4"
GAZE_PATH = "/home/imarica/clean_0_data/eye_gaze/eye_gaze.csv"
OUTPUT_DIR = "/home/imarica/vjepa2/debug_frames"

def run_debug_images():
    print("--- START DEBUG GAZE IMAGES ---")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Încărcare Gaze CSV
    df_gaze = pd.read_csv(GAZE_PATH)
    
    # 2. Configurare Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    
    # Sincronizare: Gaze e la 30Hz, Video la 10Hz => Stride 3
    stride = 3 
    
    # Frame-urile pe care vrem să le verificăm
    target_frames = [10, 50, 100]
    
    # Intrinsics Placeholder (Ajustează dacă ai valorile reale)
    fu, fv = 1500.0, 1500.0
    cx_n, cy_n = 1280.0, 960.0

    while cap.isOpened():
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret: break

        if frame_idx in target_frames:
            print(f"Procesez frame-ul {frame_idx}...")
            
            # Luăm datele de gaze corespunzătoare
            gaze_idx = frame_idx * stride
            if gaze_idx < len(df_gaze):
                row = df_gaze.iloc[gaze_idx]
                yaw, pitch = row['yaw'], row['pitch']
                valid = row['gaze_valid'] == 1
            else:
                yaw, pitch, valid = 0, 0, False

            # Generăm harta de probabilitate (grid 24x24)
            prob_map = build_gaze_probability_map(
                yaw, pitch, valid,
                fu, fv, cx_n, cy_n,
                video_w, video_h, 24, 24
            )

            # --- VIZUALIZARE ---
            # Normalizăm masca pentru a fi vizibilă (0-255)
            mask_vis = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)
            mask_vis = (mask_vis * 255).astype(np.uint8)
            
            # Redimensionăm la mărimea video-ului
            mask_res = cv2.resize(mask_vis, (video_w, video_h), interpolation=cv2.INTER_LINEAR)
            heatmap = cv2.applyColorMap(mask_res, cv2.COLORMAP_JET)

            # Suprapunem heatmap-ul peste imaginea originală (70% original, 30% heatmap)
            result = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

            # Desenăm un cerc verde în punctul central calculat pentru verificare maximă
            if valid:
                from gaussian_map import project_gaze_to_native_pixel, scale_pixel_to_video
                px_n, py_n = project_gaze_to_native_pixel(yaw, pitch, fu, fv, cx_n, cy_n)
                px_v, py_v = scale_pixel_to_video(px_n, py_n, cx_n, cy_n, video_w, video_h)
                cv2.circle(result, (int(px_v), int(py_v)), 15, (0, 255, 0), -1)
                cv2.putText(result, "GAZE CENTER", (int(px_v)+20, int(py_v)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Salvare imagine
            img_path = os.path.join(OUTPUT_DIR, f"debug_gaze_frame_{frame_idx}.jpg")
            cv2.imwrite(img_path, result)
            print(f"Salvat la: {img_path}")

        if frame_idx > max(target_frames):
            break

    cap.release()
    print("--- DEBUG COMPLETAT ---")

if __name__ == "__main__":
    run_debug_images()