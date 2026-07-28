import numpy as np
import pandas as pd
class HandTokenLoader:
    def __init__(self, hand_csv_path: str, video_fps: float):
        self.fps = video_fps
        self.df, self._t0 = self._load(hand_csv_path)

    def get_token_for_frame(self, frame_index: int):
        target_us = self._t0 + int((frame_index / self.fps) * 1_000_000)
        idx = (self.df['ts_us'] - target_us).abs().idxmin()
        row = self.df.iloc[idx]

        left_valid  = bool(row['left_valid'])
        right_valid = bool(row['right_valid'])

        hand_vec = np.array([
            row['tx_lw'], row['ty_lw'], row['tz_lw'],
            row['tx_lp'], row['ty_lp'], row['tz_lp'],
            row['tx_rw'], row['ty_rw'], row['tz_rw'],
            row['tx_rp'], row['ty_rp'], row['tz_rp'],
        ], dtype=np.float32)

        return hand_vec, left_valid, right_valid

    def get_tokens_for_clip(self, start_frame: int, num_frames: int):
        vecs   = np.zeros((num_frames, 12), dtype=np.float32)
        l_vals = np.zeros(num_frames, dtype=bool)
        r_vals = np.zeros(num_frames, dtype=bool)
        for i in range(num_frames):
            vecs[i], l_vals[i], r_vals[i] = self.get_token_for_frame(start_frame + i)
        return vecs, l_vals, r_vals

    def _load(self, path):
        df  = pd.read_csv(path)
        out = pd.DataFrame()
        out['ts_us']      = df['tracking_timestamp_us'].astype(np.int64)
        out['left_valid']  = df['left_tracking_confidence']  != -1
        out['right_valid'] = df['right_tracking_confidence'] != -1
        out['tx_lw'] = df['tx_left_wrist_device'].astype(np.float32)
        out['ty_lw'] = df['ty_left_wrist_device'].astype(np.float32)
        out['tz_lw'] = df['tz_left_wrist_device'].astype(np.float32)
        out['tx_lp'] = df['tx_left_palm_device'].astype(np.float32)
        out['ty_lp'] = df['ty_left_palm_device'].astype(np.float32)
        out['tz_lp'] = df['tz_left_palm_device'].astype(np.float32)
        out['tx_rw'] = df['tx_right_wrist_device'].astype(np.float32)
        out['ty_rw'] = df['ty_right_wrist_device'].astype(np.float32)
        out['tz_rw'] = df['tz_right_wrist_device'].astype(np.float32)
        out['tx_rp'] = df['tx_right_palm_device'].astype(np.float32)
        out['ty_rp'] = df['ty_right_palm_device'].astype(np.float32)
        out['tz_rp'] = df['tz_right_palm_device'].astype(np.float32)
        # zero out invalid coords
        for col in ['tx_lw','ty_lw','tz_lw','tx_lp','ty_lp','tz_lp']:
            out.loc[~out['left_valid'], col] = 0.0
        for col in ['tx_rw','ty_rw','tz_rw','tx_rp','ty_rp','tz_rp']:
            out.loc[~out['right_valid'], col] = 0.0
        out = out.sort_values('ts_us').reset_index(drop=True)
        return out, int(out['ts_us'].iloc[0])