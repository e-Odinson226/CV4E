"""
Check the gaze projection (ego/gaze_geometry.py) before any test uses it.

Three checks, from the preparation step in docs/EgoVault/6-next-steps.md:

  * Overlays. For each participant, the gaze point is drawn on frames of a few recordings,
    with a circle of one patch radius (88 px, one cell of the encoder's 16 x 16 grid). The
    point should lie on what the person handles or is about to handle. Written to
    <out>/<participant>.jpg.
  * Where the points fall. Over many gaze samples per participant (no video decoded): the
    share that projects into the frame, and the median position in patch units.
  * Depth. The shift between the projection at the gaze depth and the same direction at
    100 m, in patches. It shows how much the depth matters for each participant.

Usage
-----
    python -m ego draw-gaze \
        --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --out results/gaze_projection
"""

import argparse
from pathlib import Path

import numpy as np

from ego.data import find_recordings, read_vrs_times
from ego.gaze_geometry import FRAME, GRID, GazeProjector, to_patch
from ego.linprobe import GazeSeries
from ego.runlog import Logger

PATCH_PX = FRAME // GRID


def overlay(rec, proj, gs, vrs, n_frames, rng):
    """n_frames frames of one recording with the gaze point drawn, as small tiles."""
    import cv2
    cap = cv2.VideoCapture(rec.mp4)
    tiles = []
    for fi in sorted(rng.choice(np.arange(300, len(vrs) - 300), size=n_frames, replace=False)):
        g = gs.at_ns(vrs[fi])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        xy = proj.project(*g) if g is not None else None
        if xy is not None:
            c = (int(xy[0]), int(xy[1]))
            cv2.circle(img, c, PATCH_PX, (0, 0, 255), 6)
            cv2.circle(img, c, 12, (0, 0, 255), -1)
        label = f"{rec.stem[4:]} f{fi}" + ("" if g is not None else " no gaze")
        cv2.putText(img, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 5)
        tiles.append(cv2.resize(img, (400, 400)))
    cap.release()
    return tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--participants", nargs="+",
                    default=["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"])
    ap.add_argument("--overlay-recordings", type=int, default=2)
    ap.add_argument("--overlay-frames", type=int, default=4)
    ap.add_argument("--stat-samples", type=int, default=200, help="gaze samples per recording")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gaze_projection")
    args = ap.parse_args()

    import cv2
    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "draw_gaze")
    rng = np.random.default_rng(args.seed)
    rows = []
    for p in args.participants:
        recs = [r for r in find_recordings(args.video_dir, args.gaze_dir, [p]) if r.gaze_csv]
        tiles = []
        for k, rec in enumerate(recs):
            proj = GazeProjector(args.gaze_dir, p, rec.stem)
            gs = GazeSeries(rec.gaze_csv, 30.0)
            vrs = read_vrs_times(rec.ts_csv)
            if k < args.overlay_recordings:
                tiles += overlay(rec, proj, gs, vrs, args.overlay_frames, rng)
            for fi in rng.choice(len(vrs), size=min(args.stat_samples, len(vrs)), replace=False):
                g = gs.at_ns(vrs[fi])
                if g is None:
                    continue
                xy = proj.project(*g)
                far = proj.project(g[0], g[1], 100.0)
                row = {"participant": p, "recording": rec.stem, "own_calibration": proj.own_calibration,
                       "serial": proj.serial, "depth_m": float(g[2]), "in_frame": xy is not None}
                if xy is not None:
                    row["col"], row["row"] = to_patch(xy)
                    if far is not None:
                        row["depth_shift_patch"] = float(np.hypot(xy[0] - far[0], xy[1] - far[1]) / PATCH_PX)
                rows.append(row)
        if tiles:
            per_row = args.overlay_frames
            grid = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles) - per_row + 1, per_row)]
            cv2.imwrite(str(out / f"{p}.jpg"), np.vstack(grid))
        log(f"{p}: {len(recs)} recordings, overlay -> {out / f'{p}.jpg'}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "gaze_projection_samples.csv", index=False)
    s = df.groupby("participant").agg(
        samples=("in_frame", "size"), in_frame=("in_frame", "mean"),
        col_median=("col", "median"), row_median=("row", "median"),
        depth_median_m=("depth_m", "median"),
        shift_median_patch=("depth_shift_patch", "median"),
        shift_p90_patch=("depth_shift_patch", lambda v: v.quantile(0.9)),
        devices=("serial", "nunique"))
    log("\nGaze points per participant (patch units: 0-16, the centre is 8):")
    log(s.to_string(float_format=lambda v: f"{v:7.2f}"))
    log(f"\nall: {len(df)} samples, {df.in_frame.mean():.1%} inside the frame, median shift from "
        f"depth {df.depth_shift_patch.median():.2f} patch (90th percentile "
        f"{df.depth_shift_patch.quantile(0.9):.2f})")
    no_own = sorted(df[~df.own_calibration].recording.unique())
    log(f"recordings using the closest recording's calibration: {no_own or 'none'}")
    log(f"[out] {out}/<participant>.jpg  {out}/gaze_projection_samples.csv  {out}/draw_gaze.log")
    log.close()


if __name__ == "__main__":
    main()
