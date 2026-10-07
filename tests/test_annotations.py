"""Self-test for the HD-EPIC annotation reader, the pick check's geometry and the recording bootstrap (no GPU, no data)."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from ego.annotations import object_class, object_movements
from ego.linprobe import point_features
from ego.commands.gaze_at_picks import box_distance, unrotated
from ego.stats import by_recording

ok = True
def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    ok &= bool(cond)

# --- 1. object_movements: boxes matched to the start and the end of a movement ---
tmp = Path(tempfile.mkdtemp()) / "scene-and-object-movements"
tmp.mkdir()
assoc = {"P01-x": {"a": {"name": "cup", "tracks": [
    {"track_id": "t1", "time_segment": [10.0, 20.0], "masks": ["m1", "m2"]},     # both boxes
    {"track_id": "t2", "time_segment": [30.0, 31.0], "masks": ["m3"]},           # start box only
    {"track_id": "t3", "time_segment": [40.0, 50.0], "masks": ["m4", "m5"]}]}}}  # box far from start; bad box
masks = {"P01-x": {
    "m1": {"frame_number": 300, "bbox": [1, 2, 3, 4], "fixture": "f1", "3d_location": [0, 0, 0]},
    "m2": {"frame_number": 600, "bbox": [5, 6, 7, 8], "fixture": "f2", "3d_location": [0, 0, 0]},
    "m3": {"frame_number": 901, "bbox": [1, 1, 2, 2], "fixture": None, "3d_location": [0, 0, 0]},
    "m4": {"frame_number": 1230, "bbox": [1, 1, 2, 2], "fixture": None, "3d_location": [0, 0, 0]},
    "m5": {"frame_number": 1500, "bbox": [5, 5, 5, 9], "fixture": None, "3d_location": [0, 0, 0]}}}
(tmp / "assoc_info.json").write_text(json.dumps(assoc))
(tmp / "mask_info.json").write_text(json.dumps(masks))
df = object_movements(tmp.parent).set_index("track_id")
check("pick and put boxes of a full track", df.loc["t1", "pick_frame"] == 300 and df.loc["t1", "put_box"] == (5, 6, 7, 8))
check("start box within 0.1 s is the pick, not the put", df.loc["t2", "pick_frame"] == 901 and pd.isna(df.loc["t2", "put_frame"]))
check("box 1 s after the start is not the pick", df.loc["t3", "pick_box"] is None, str(df.loc["t3", "pick_box"]))
check("a zero-width box is dropped", df.loc["t3", "put_box"] is None)

# --- 2. box_distance and unrotated --------------------------------------------
box = (100, 100, 200, 200)
check("inside -> 0", box_distance((150, 150), box) == 0)
check("88 px to the right -> 1 patch", abs(box_distance((288, 150), box) - 1.0) < 1e-9)
check("no point -> NaN", np.isnan(box_distance(None, box)))
check("unrotated undoes a clockwise quarter turn", unrotated((1407 - 10, 20)) == (20, 10))

# --- 3. by_recording -------------------------------------------------------------
rng = np.random.default_rng(0)
v = rng.random(300); g = np.repeat(np.arange(30), 10)
r = by_recording(v, g)
check("mean is the plain mean", abs(r["mean"] - v.mean()) < 1e-12)
check("interval contains the mean", r["ci_lo"] < r["mean"] < r["ci_hi"])
r2 = by_recording(np.repeat(rng.random(30), 10), g)          # all spread lies between recordings
check("interval reflects spread between recordings", r2["ci_hi"] - r2["ci_lo"] > 0.1,
      f"[{r2['ci_lo']:.3f}, {r2['ci_hi']:.3f}]")

# --- 4. object_class -------------------------------------------------------------
classes = {"spoon": "spoon", "teaspoon": "spoon", "bowl": "bowl", "board:chopping": "board:chopping"}
check("trailing number dropped", object_class("Spoon2", classes) == "spoon")
check("listed instance", object_class("teaspoon", classes) == "spoon")
check("last word", object_class("juicer bowl", classes) == "bowl")
check("plural last word", object_class("small bowls", classes) == "bowl")
check("unknown -> None", object_class("Track 12 (skipped)", classes) is None)

# --- 5. point_features ---------------------------------------------------------
grid = np.zeros((256, 3)); grid[5 * 16 + 9] = [1.0, 2.0, 3.0]        # patch at row 5, column 9
f = point_features(grid, 9.5, 5.5, sigma=0.2)                          # centre of that patch
check("a narrow kernel at a patch centre returns that patch", np.allclose(f, [1, 2, 3], atol=1e-3), str(f.round(4)))
check("columns and rows are not swapped", point_features(grid, 5.5, 9.5, sigma=0.2).max() < 1e-3)
check("a wide kernel averages", abs(point_features(np.ones((2, 256, 3)), 3.0, 4.0, sigma=50).mean() - 1) < 1e-9)
check("batch dimensions kept", point_features(np.ones((4, 7, 256, 3)), 8, 8).shape == (4, 7, 3))

sys.exit(0 if ok else 1)
