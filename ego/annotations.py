"""
HD-EPIC annotations, from https://github.com/hd-epic/hd-epic-annotations, cloned to
data/epic-kitchen/ek100-hd/HD-EPIC/annotations.

Two parts are read here.

The object movements (scene-and-object-movements/). Each movement of an object, from pick-up
to put-down, is a track with a start and an end time in seconds. The annotators drew a box
around the object in the frame where the movement starts and in the frame where it ends. The
boxes are in pixels of the upright 1408 x 1408 video frame. HD-EPIC notes that the boxes and
the masks were made by different teams and may disagree in places.

The noun classes of the narrations (HD_EPIC_noun_classes.csv). They give the objects'
free-form names a class shared across kitchens: "spoon2" and "teaspoon" are both "spoon".

The narrations (HD_EPIC_Narrations.pkl) are a pickle file. Loading a pickle runs code from
the file, so they are not read here.
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd

FPS = 30.0        # HD-EPIC videos; a time in seconds times FPS gives the MP4 frame number
MATCH_S = 0.1     # a box belongs to the start (end) of a movement if its frame is this close


def _box_ok(b):
    return b is not None and len(b) == 4 and b[2] > b[0] and b[3] > b[1]


def object_movements(ann_dir):
    """
    One row per object movement, sorted by video and start time. Columns: video,
    participant, object_id and name (the object, the same across its movements), track_id,
    start_s, end_s, and for the start (pick) and the end (put) of the movement: the frame,
    the box (x0, y0, x1, y1) and the fixture. These are missing when no box was drawn
    within MATCH_S of that time.
    """
    base = Path(ann_dir) / "scene-and-object-movements"
    assoc = json.loads((base / "assoc_info.json").read_text())
    masks = json.loads((base / "mask_info.json").read_text())
    rows = []
    for video, objects in assoc.items():
        vm = masks.get(video, {})
        for oid, obj in objects.items():
            for tr in obj["tracks"]:
                start, end = tr["time_segment"]
                ms = [vm[m] for m in tr["masks"] if m in vm and _box_ok(vm[m].get("bbox"))]
                row = {"video": video, "participant": video[:3], "object_id": oid,
                       "name": obj["name"], "track_id": tr["track_id"],
                       "start_s": float(start), "end_s": float(end)}
                for kind, t in (("pick", start), ("put", end)):
                    m = min(ms, key=lambda m: abs(m["frame_number"] / FPS - t), default=None)
                    ok = m is not None and abs(m["frame_number"] / FPS - t) <= MATCH_S
                    row[f"{kind}_frame"] = m["frame_number"] if ok else None
                    row[f"{kind}_box"] = tuple(m["bbox"]) if ok else None
                    row[f"{kind}_fixture"] = m["fixture"] if ok else None
                rows.append(row)
    df = pd.DataFrame(rows).sort_values(["video", "start_s"]).reset_index(drop=True)
    for kind in ("pick", "put"):
        df[f"{kind}_frame"] = df[f"{kind}_frame"].astype("Int64")
    return df


def noun_classes(ann_dir):
    """{name: class} from HD_EPIC_noun_classes.csv, every listed instance and the class itself."""
    f = Path(ann_dir) / "narrations-and-action-segments" / "HD_EPIC_noun_classes.csv"
    out = {}
    for _, r in pd.read_csv(f).iterrows():
        for name in [r.key] + ast.literal_eval(r.instances):
            out.setdefault(name.lower().strip(), r.key)
    return out


def object_class(name, classes):
    """
    The noun class of an object's name, or None. The name is matched without a trailing
    number ("spoon2" -> "spoon"), then by its last word ("juicer bowl" -> "bowl").
    """
    n = re.sub(r"\s*\d+$", "", name.lower().strip())
    if n in classes:
        return classes[n]
    last = n.split()[-1] if n.split() else ""
    for w in (last, last.rstrip("s")):
        if w in classes:
            return classes[w]
    return None
