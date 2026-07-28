import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Input standardisation (z-score) for the conditioning signals.
#
# A freshly-initialised Linear projector learns far faster when its inputs are
# on a common scale. Gaze mixes radians (yaw/pitch ~ O(0.3)) with metres
# (depth up to 10), and hand coordinates are metres in the device frame — so
# without this, depth/translation dominate the projection.
#
# These are APPROXIMATE population stats for Aria CPF gaze + device-frame hands.
# Recompute them from P01-P07 once the GAZE_HAND data is downloaded and update
# here (or load from a stats file) for best results. Hand mean is kept at 0 so
# that zeroed (invalid) hand coordinates stay zero after standardisation.
# ---------------------------------------------------------------------------
GAZE_MEAN = np.array([0.0, -0.25, 1.0], dtype=np.float32)   # yaw, pitch(rad), depth(m)
GAZE_STD  = np.array([0.35, 0.30, 1.00], dtype=np.float32)
HAND_MEAN = np.float32(0.0)                                 # keep 0: preserves zeroed-invalid
HAND_STD  = np.float32(0.30)                                # metres, device frame


class GazeTokenLoader:
    """
    Loads gaze CSV and maps samples to video frames by timestamp alignment.
    Supports Aria (timestamp_ns, left/right_yaw_rads_cpf) and generic formats.
    Returns [yaw, pitch, depth] vectors; invalid frames return zeros + valid=False.

    With standardize=True (default) valid vectors are z-scored using the module
    GAZE_MEAN/GAZE_STD so the gaze_proj sees inputs on a common scale.
    """

    DEPTH_MIN = 0.05    # metres — closer is sensor noise
    DEPTH_MAX = 10.0    # metres — further is unreliable
    DEPTH_FILL = 1.0    # fallback when depth is 0 or missing

    def __init__(self, gaze_csv_path: str, video_fps: float, standardize: bool = True):
        self.fps = video_fps
        self.standardize = standardize
        self.df = self._load(gaze_csv_path)

    def get_token_for_vrs_ns(self, vrs_ns: int):
        """Look up gaze by absolute VRS device timestamp (nanoseconds) from mp4_to_vrs_time_ns.csv."""
        return self._lookup(vrs_ns // 1000)

    def get_token_for_frame(self, frame_index: int):
        target_us = self._gaze_start_us + self._frame_to_us(frame_index)
        return self._lookup(target_us)

    def _lookup(self, target_us: int):
        idx = (self.df['ts_us'] - target_us).abs().idxmin()
        row = self.df.iloc[idx]

        if not bool(row['gaze_valid']):
            return np.zeros(3, dtype=np.float32), False

        depth = float(row['depth'])
        if depth == 0.0 or np.isnan(depth):
            depth = self.DEPTH_FILL
        else:
            depth = float(np.clip(depth, self.DEPTH_MIN, self.DEPTH_MAX))

        vec = np.array([float(row['yaw']), float(row['pitch']), depth], dtype=np.float32)
        if np.any(np.isnan(vec)):
            return np.zeros(3, dtype=np.float32), False
        if self.standardize:
            vec = (vec - GAZE_MEAN) / GAZE_STD
        return vec, True

    def get_tokens_for_clip(self, start_frame: int, num_frames: int):
        """Returns (num_frames, 3) float32 and (num_frames,) bool."""
        vecs = np.zeros((num_frames, 3), dtype=np.float32)
        valids = np.zeros(num_frames, dtype=bool)
        for i in range(num_frames):
            vecs[i], valids[i] = self.get_token_for_frame(start_frame + i)
        return vecs, valids

    def _load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        cols = list(df.columns)
        out = pd.DataFrame()

        if 'timestamp_ns' in cols:
            out['ts_us'] = (df['timestamp_ns'] / 1000.0).astype(np.int64)
        elif 'tracking_timestamp_us' in cols:
            out['ts_us'] = df['tracking_timestamp_us'].astype(np.int64)
        else:
            raise ValueError(f"No timestamp column found. Got: {cols}")

        if 'yaw' in cols:
            out['yaw'] = df['yaw'].astype(np.float32)
        elif 'left_yaw_rads_cpf' in cols:
            out['yaw'] = ((df['left_yaw_rads_cpf'] + df['right_yaw_rads_cpf']) / 2.0).astype(np.float32)
        else:
            raise ValueError(f"No yaw column found. Got: {cols}")

        if 'pitch' in cols:
            out['pitch'] = df['pitch'].astype(np.float32)
        elif 'pitch_rads_cpf' in cols:
            out['pitch'] = df['pitch_rads_cpf'].astype(np.float32)
        else:
            raise ValueError(f"No pitch column found. Got: {cols}")

        if 'depth' in cols:
            out['depth'] = df['depth'].astype(np.float32)
        elif 'depth_m' in cols:
            out['depth'] = df['depth_m'].astype(np.float32)
        else:
            out['depth'] = np.float32(self.DEPTH_FILL)

        if 'gaze_valid' in cols:
            out['gaze_valid'] = df['gaze_valid'].astype(bool)
        else:
            out['gaze_valid'] = (
                out['depth'].notna()
                & (out['depth'] > self.DEPTH_MIN)
                & (out['depth'] < self.DEPTH_MAX)
            )

        out = out.sort_values('ts_us').reset_index(drop=True)
        self._gaze_start_us = int(out['ts_us'].iloc[0])
        return out

    def _frame_to_us(self, frame_index: int) -> int:
        return int((frame_index / self.fps) * 1_000_000)


class HandTokenLoader:
    """
    Loads Aria hand-tracking CSV and maps samples to video frames by timestamp.
    Returns a 12-vector: [tx,ty,tz] for left wrist, left palm, right wrist, right palm.
    Invalid hands (confidence == -1) are zeroed; returns per-hand validity flags.
    """

    def __init__(self, hand_csv_path: str, video_fps: float, standardize: bool = True):
        self.fps = video_fps
        self.standardize = standardize
        self.df, self._t0 = self._load(hand_csv_path)

    def get_token_for_vrs_ns(self, vrs_ns: int):
        """Look up hand tracking by absolute VRS device timestamp (nanoseconds)."""
        return self._lookup(vrs_ns // 1000)

    def get_token_for_frame(self, frame_index: int):
        target_us = self._t0 + int((frame_index / self.fps) * 1_000_000)
        return self._lookup(target_us)

    def _lookup(self, target_us: int):
        idx = (self.df['ts_us'] - target_us).abs().idxmin()
        row = self.df.iloc[idx]
        hand_vec = np.array([
            row['tx_lw'], row['ty_lw'], row['tz_lw'],
            row['tx_lp'], row['ty_lp'], row['tz_lp'],
            row['tx_rw'], row['ty_rw'], row['tz_rw'],
            row['tx_rp'], row['ty_rp'], row['tz_rp'],
        ], dtype=np.float32)
        if self.standardize:
            hand_vec = (hand_vec - HAND_MEAN) / HAND_STD   # mean 0 keeps zeroed-invalid at 0
        return hand_vec, bool(row['left_valid']), bool(row['right_valid'])

    def get_tokens_for_clip(self, start_frame: int, num_frames: int):
        """Returns (num_frames, 12) float32, (num_frames,) bool, (num_frames,) bool."""
        vecs = np.zeros((num_frames, 12), dtype=np.float32)
        l_vals = np.zeros(num_frames, dtype=bool)
        r_vals = np.zeros(num_frames, dtype=bool)
        for i in range(num_frames):
            vecs[i], l_vals[i], r_vals[i] = self.get_token_for_frame(start_frame + i)
        return vecs, l_vals, r_vals

    def _load(self, path: str):
        df = pd.read_csv(path)
        out = pd.DataFrame()
        out['ts_us'] = df['tracking_timestamp_us'].astype(np.int64)
        out['left_valid'] = df['left_tracking_confidence'] != -1
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
        for col in ['tx_lw', 'ty_lw', 'tz_lw', 'tx_lp', 'ty_lp', 'tz_lp']:
            out.loc[~out['left_valid'], col] = 0.0
        for col in ['tx_rw', 'ty_rw', 'tz_rw', 'tx_rp', 'ty_rp', 'tz_rp']:
            out.loc[~out['right_valid'], col] = 0.0
        out = out.sort_values('ts_us').reset_index(drop=True)
        return out, int(out['ts_us'].iloc[0])
