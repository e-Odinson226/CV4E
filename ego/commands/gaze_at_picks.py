"""
Does the projected gaze point lie on the object that is about to be picked up?

The last check of the gaze projection (ego/gaze_geometry.py), described in
docs/EgoVault/3-method.md. HD-EPIC's annotators drew a box around each object in the frame
where its movement starts (ego/annotations.py). For each pick, the gaze at that frame is
projected into the frame and compared with the box:

  * inside    the gaze point lies in the box
  * near      the gaze point lies within one patch (88 px) of the box
  * distance  from the gaze point to the box, in patches (0 inside)

Comparisons:

  * Chance. The same box with the gaze of 20 random frames of the same recording, at least
    10 s away from the pick. Gaze and objects both lie mostly near the centre of the frame,
    so a gaze point can land in a box without the person looking at the object.
  * Projection variants, at the pick frame: without the 90-degree rotation of the camera
    model, and with the depth fixed at 1 m or at 100 m in place of the measured depth.
  * Earlier gaze: the gaze 0.25, 0.5, 1 and 2 s before the pick, projected into its own
    frame, against the box of the pick frame. The head moves in between, so this mixes
    where the person looked with how far the object moved in the image.
  * Offset. For each participant, the shift of all gaze points (in quarter patches, up to
    one patch each way) that puts the most of them inside the boxes. A shift that differs
    between people points to a bias of the eye tracker for each person; a shift common to
    everyone can also come from where people look on an object.

The 95% intervals come from a bootstrap over recordings (ego.stats.by_recording).

Outputs in --out: gaze_at_picks.csv (one row per pick), gaze_at_picks.json (the summary),
gaze_at_picks.log, and picks.jpg (frames of random picks, the box in green and the gaze
point in red with a circle of one patch radius).

Usage
-----
    python -m ego gaze-at-picks \
        --video-dir   data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir    data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --annotations data/epic-kitchen/ek100-hd/HD-EPIC/annotations \
        --out results/gaze_projection
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ego.annotations import FPS, object_movements
from ego.data import find_recordings, read_vrs_times
from ego.gaze_geometry import FRAME, GRID, GazeProjector
from ego.linprobe import GazeSeries
from ego.runlog import Logger
from ego.stats import by_recording

PATCH_PX = FRAME / GRID            # 88 px
BEFORE_S = (0.25, 0.5, 1.0, 2.0)
CHANCE_FRAMES = 20
CHANCE_GAP_S = 10.0


def box_distance(xy, box):
    """Distance from the point to the box in patches, 0 inside; NaN without a point."""
    if xy is None:
        return np.nan
    x0, y0, x1, y1 = box
    dx = max(x0 - xy[0], 0.0, xy[0] - x1)
    dy = max(y0 - xy[1], 0.0, xy[1] - y1)
    return float(np.hypot(dx, dy) / PATCH_PX)


def unrotated(xy):
    """Where the point would land without the 90-degree clockwise rotation of the camera."""
    return None if xy is None else (xy[1], FRAME - 1 - xy[0])


def measure(proj, gs, vrs, f, box, rng):
    """All distances for one pick at frame f, or None when there is no gaze at f."""
    g = gs.at_ns(vrs[f])
    if g is None:
        return None
    xy = proj.project(*g)
    row = {"depth_m": float(g[2]), "gaze_x": xy[0] if xy else np.nan,
           "gaze_y": xy[1] if xy else np.nan,
           "dist": box_distance(xy, box),
           "dist_unrotated": box_distance(unrotated(xy), box),
           "dist_depth_1m": box_distance(proj.project(g[0], g[1], 1.0), box),
           "dist_depth_100m": box_distance(proj.project(g[0], g[1], 100.0), box)}
    for s in BEFORE_S:
        fb = f - int(round(s * FPS))
        gb = gs.at_ns(vrs[fb]) if fb >= 0 else None
        row[f"dist_{s:g}s_before"] = box_distance(proj.project(*gb), box) if gb is not None else np.nan
    gap = int(CHANCE_GAP_S * FPS)
    cand = np.r_[0:max(f - gap, 0), min(f + gap, len(vrs)):len(vrs)]
    d = []
    for i in rng.choice(cand, size=min(CHANCE_FRAMES, len(cand)), replace=False):
        gc = gs.at_ns(vrs[i])
        if gc is not None:
            d.append(box_distance(proj.project(*gc), box))
    d = np.array(d)
    row["chance_inside"] = float(np.mean(d == 0)) if len(d) else np.nan
    row["chance_near"] = float(np.mean(d <= 1)) if len(d) else np.nan
    row["chance_dist"] = float(np.median(d)) if len(d) else np.nan
    return row


def overlay(df, video_dir, n, rng):
    """n random picks: the frame with the box (green) and the gaze point (red)."""
    import cv2
    tiles = []
    for _, r in df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)].iterrows():
        cap = cv2.VideoCapture(str(Path(video_dir) / r.participant / f"{r.video}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.pick_frame))
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        x0, y0, x1, y1 = (int(v) for v in (r.box_x0, r.box_y0, r.box_x1, r.box_y1))
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 6)
        if np.isfinite(r.gaze_x):
            c = (int(r.gaze_x), int(r.gaze_y))
            cv2.circle(img, c, int(PATCH_PX), (0, 0, 255), 6)
            cv2.circle(img, c, 12, (0, 0, 255), -1)
        cv2.putText(img, f"{r.video[:3]} {r['name'][:22]}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    2.2, (0, 255, 255), 5)
        tiles.append(cv2.resize(img, (400, 400)))
    per_row = 6
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles) - per_row + 1, per_row)]
    return np.vstack(rows) if rows else None


def best_shift(g, step=0.25, reach=1.0):
    """(dx, dy, inside share): the shift in patches that puts the most gaze points in the boxes."""
    g = g[np.isfinite(g.gaze_x)]
    best = (0.0, 0.0, -1.0)
    for dx in np.arange(-reach, reach + 1e-9, step):
        for dy in np.arange(-reach, reach + 1e-9, step):
            x, y = g.gaze_x + dx * PATCH_PX, g.gaze_y + dy * PATCH_PX
            ins = float(((x >= g.box_x0) & (x <= g.box_x1) & (y >= g.box_y0) & (y <= g.box_y1)).sum() / len(g))
            if ins > best[2]:
                best = (float(dx), float(dy), ins)
    return best


def share(df, col, test):
    """Share of picks meeting `test`, with its interval over recordings; a missing point is a miss."""
    v = df[col].to_numpy()
    hit = np.where(np.isfinite(v), test(np.nan_to_num(v, nan=np.inf)), False).astype(float)
    return by_recording(hit, df.video)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--annotations", required=True, help="the cloned hd-epic-annotations repository")
    ap.add_argument("--participants", nargs="+",
                    default=["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"])
    ap.add_argument("--overlay-picks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gaze_projection")
    args = ap.parse_args()

    import cv2
    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "gaze_at_picks")
    rng = np.random.default_rng(args.seed)
    moves = object_movements(args.annotations)
    moves = moves[moves.participant.isin(args.participants)]
    skipped = Counter({"no box at the pick": int(moves.pick_frame.isna().sum())})
    picks = moves[moves.pick_frame.notna()]
    rows = []
    for rec in find_recordings(args.video_dir, args.gaze_dir, args.participants):
        mine = picks[picks.video == rec.stem]
        if rec.gaze_csv is None:
            skipped["recording without gaze"] += len(mine)
            continue
        if mine.empty:
            continue
        proj = GazeProjector(args.gaze_dir, rec.participant, rec.stem)
        gs = GazeSeries(rec.gaze_csv, FPS)
        vrs = read_vrs_times(rec.ts_csv)
        for _, p in mine.iterrows():
            f = int(p.pick_frame)
            if f >= len(vrs):
                skipped["frame beyond the video"] += 1
                continue
            r = measure(proj, gs, vrs, f, p.pick_box, rng)
            if r is None:
                skipped["no gaze at the pick"] += 1
                continue
            x0, y0, x1, y1 = p.pick_box
            rows.append({"participant": rec.participant, "video": rec.stem,
                         "own_calibration": proj.own_calibration, "name": p["name"],
                         "track_id": p.track_id, "pick_frame": f,
                         "box_x0": x0, "box_y0": y0, "box_x1": x1, "box_y1": y1,
                         "box_w_patch": (x1 - x0) / PATCH_PX, "box_h_patch": (y1 - y0) / PATCH_PX,
                         **r})
        log(f"{rec.stem}: {len(mine)} picks")
    df = pd.DataFrame(rows)
    df.to_csv(out / "gaze_at_picks.csv", index=False)

    measures = [("at the pick", "dist"),
                ("chance (gaze at random times)", None),
                ("without the 90-degree rotation", "dist_unrotated"),
                ("depth fixed at 1 m", "dist_depth_1m"),
                ("depth fixed at 100 m", "dist_depth_100m")] + \
               [(f"gaze {s:g} s before the pick", f"dist_{s:g}s_before") for s in BEFORE_S]
    summary = {"picks": len(df), "recordings": int(df.video.nunique()), "skipped": dict(skipped),
               "picks_without_chance": int(df.chance_inside.isna().sum()),
               "box_median_patch": [float(df.box_w_patch.median()), float(df.box_h_patch.median())],
               "measures": {}}
    log(f"\n{len(df)} picks in {df.video.nunique()} recordings; skipped: {dict(skipped)}")
    log(f"picks without a chance value (no gaze 10 s away): {int(df.chance_inside.isna().sum())}")
    log(f"median box {df.box_w_patch.median():.1f} x {df.box_h_patch.median():.1f} patches")
    log(f"\n{'':34} {'inside':>22} {'within 1 patch':>22} {'median dist':>12}")
    for label, col in measures:
        if col is None:
            c = df[df.chance_inside.notna()]
            ins = by_recording(c.chance_inside, c.video)
            near = by_recording(c.chance_near, c.video)
            med = float(c.chance_dist.median())
        else:
            ins = share(df, col, lambda v: v == 0)
            near = share(df, col, lambda v: v <= 1)
            med = float(df[col].median())
        summary["measures"][label] = {"inside": ins, "near": near, "median_dist_patch": med}
        log(f"{label:34} {ins['mean']:6.1%} [{ins['ci_lo']:5.1%}, {ins['ci_hi']:5.1%}]  "
            f"{near['mean']:6.1%} [{near['ci_lo']:5.1%}, {near['ci_hi']:5.1%}]  {med:9.2f}")
    inside = (df.dist == 0).astype(float)
    c = df.chance_inside.notna()
    diff = by_recording((inside - df.chance_inside)[c], df.video[c])
    summary["inside_minus_chance"] = diff
    log(f"\ninside minus chance: {diff['mean']:+.1%} [{diff['ci_lo']:+.1%}, {diff['ci_hi']:+.1%}]")

    per = df.assign(inside=inside, near=(df.dist <= 1).astype(float)).groupby("participant").agg(
        picks=("inside", "size"), inside=("inside", "mean"), chance_inside=("chance_inside", "mean"),
        near=("near", "mean"), chance_near=("chance_near", "mean"), median_dist=("dist", "median"))
    shifts = {p: best_shift(g) for p, g in df.groupby("participant")}
    per["shift_x"] = [shifts[p][0] for p in per.index]
    per["shift_y"] = [shifts[p][1] for p in per.index]
    per["inside_shifted"] = [shifts[p][2] for p in per.index]
    summary["per_participant"] = per.reset_index().to_dict(orient="records")
    log("\nPer participant (shift_x, shift_y: the best shift in patches, + is right and down):")
    log(per.to_string(float_format=lambda v: f"{v:6.3f}"))

    img = overlay(df, args.video_dir, args.overlay_picks, rng)
    if img is not None:
        cv2.imwrite(str(out / "picks.jpg"), img)
    (out / "gaze_at_picks.json").write_text(json.dumps(summary, indent=2))
    log(f"[out] {out}/gaze_at_picks.csv  {out}/gaze_at_picks.json  {out}/picks.jpg")
    log.close()


if __name__ == "__main__":
    main()
