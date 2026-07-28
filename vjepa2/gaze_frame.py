
import numpy as np
import pandas as pd


class GazeTokenLoader:

    DEPTH_MIN  = 0.05   # metres — closer is sensor noise
    DEPTH_MAX  = 10.0   # metres — further is unreliable
    DEPTH_FILL = 1.0    # fallback when depth is 0 or missing

    def __init__(self, gaze_csv_path: str, video_fps: float):
     
        self.fps = video_fps
        self.df  = self._load(gaze_csv_path)

    def get_token_for_frame(self, frame_index: int):
        target_us = self._gaze_start_us + self._frame_to_us(frame_index)
        idx = (self.df['ts_us'] - target_us).abs().idxmin()
        row = self.df.iloc[idx]

        if not bool(row['gaze_valid']):
            return np.zeros(3, dtype=np.float32), False

        depth = float(row['depth'])
        if depth == 0.0 or np.isnan(depth):
            depth = self.DEPTH_FILL
        else:
            depth = float(np.clip(depth, self.DEPTH_MIN, self.DEPTH_MAX))

        vec = np.array([float(row['yaw']), float(row['pitch']), depth],
                       dtype=np.float32)

        if np.any(np.isnan(vec)):
            return np.zeros(3, dtype=np.float32), False

        return vec, True

    def get_tokens_for_clip(self, start_frame: int, num_frames: int):
        vecs   = np.zeros((num_frames, 3), dtype=np.float32)
        valids = np.zeros(num_frames, dtype=bool)
        for i in range(num_frames):
            vecs[i], valids[i] = self.get_token_for_frame(start_frame + i)
        return vecs, valids

   

    def _load(self, path: str) -> pd.DataFrame:
        df   = pd.read_csv(path)
        cols = list(df.columns)
        out  = pd.DataFrame()

        # Timestamp → microseconds
        if 'timestamp_ns' in cols:
            out['ts_us'] = (df['timestamp_ns'] / 1000.0).astype(np.int64)
        elif 'tracking_timestamp_us' in cols:
            out['ts_us'] = df['tracking_timestamp_us'].astype(np.int64)
        else:
            raise ValueError(f"No timestamp column found. Got: {cols}")

        # Yaw
        if 'yaw' in cols:
            out['yaw'] = df['yaw'].astype(np.float32)
        elif 'left_yaw_rads_cpf' in cols:
            out['yaw'] = ((df['left_yaw_rads_cpf'] +
                           df['right_yaw_rads_cpf']) / 2.0).astype(np.float32)
        else:
            raise ValueError(f"No yaw column found. Got: {cols}")

        # Pitch
        if 'pitch' in cols:
            out['pitch'] = df['pitch'].astype(np.float32)
        elif 'pitch_rads_cpf' in cols:
            out['pitch'] = df['pitch_rads_cpf'].astype(np.float32)
        else:
            raise ValueError(f"No pitch column found. Got: {cols}")

        # Depth
        if 'depth' in cols:
            out['depth'] = df['depth'].astype(np.float32)
        elif 'depth_m' in cols:
            out['depth'] = df['depth_m'].astype(np.float32)
        else:
            out['depth'] = np.float32(self.DEPTH_FILL)

        # Validity
        if 'gaze_valid' in cols:
            out['gaze_valid'] = df['gaze_valid'].astype(bool)
        else:
            out['gaze_valid'] = (
                out['depth'].notna() &
                (out['depth'] > self.DEPTH_MIN) &
                (out['depth'] < self.DEPTH_MAX)
            )

        out = out.sort_values('ts_us').reset_index(drop=True)
        self._gaze_start_us = int(out['ts_us'].iloc[0])
        return out

    def _frame_to_us(self, frame_index: int) -> int:
        return int((frame_index / self.fps) * 1_000_000)