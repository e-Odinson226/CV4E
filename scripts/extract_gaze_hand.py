"""
Extract the gaze + hand CSVs from the HD-EPIC GAZE_HAND zips, in place.

Each <participant>/GAZE_HAND/mps_<rec>_vrs.zip is an Aria MPS bundle; we pull out
just the two CSVs the loaders need:
    eye_gaze/general_eye_gaze.csv
    hand_tracking/wrist_and_palm_poses.csv
Already-extracted, empty, or corrupt zips are skipped safely.

    python scripts/extract_gaze_hand.py --gaze-dir data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze
    python scripts/extract_gaze_hand.py --gaze-dir <dir> --participants P01 P02 P03
"""

import argparse
import zipfile
from pathlib import Path

WANT = ("general_eye_gaze.csv", "wrist_and_palm_poses.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze-dir", required=True, help="…/HD-EPIC/SLAM-and-Gaze")
    ap.add_argument("--participants", nargs="+",
                    default=["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"])
    args = ap.parse_args()

    gd = Path(args.gaze_dir)
    done = skipped = empty = err = 0
    for p in args.participants:
        d = gd / p / "GAZE_HAND"
        if not d.exists():
            continue
        for z in sorted(d.glob("*.zip")):
            if z.stat().st_size == 0:
                empty += 1
                continue
            try:
                zf = zipfile.ZipFile(z)
                members = [n for n in zf.namelist() if Path(n).name in WANT]
                if members and all((d / m).exists() and (d / m).stat().st_size > 0 for m in members):
                    skipped += 1
                    continue
                for m in members:
                    zf.extract(m, d)
                done += 1
            except Exception as e:
                print(f"  ERROR {z.name}: {e}")
                err += 1
    print(f"extracted={done}  already_done={skipped}  empty_zips(not downloaded)={empty}  errors={err}")
    g = len(list(gd.glob("*/GAZE_HAND/**/general_eye_gaze.csv")))
    h = len(list(gd.glob("*/GAZE_HAND/**/wrist_and_palm_poses.csv")))
    print(f"total CSVs now present:  gaze={g}  hand={h}")


if __name__ == "__main__":
    main()
