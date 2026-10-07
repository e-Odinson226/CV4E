"""
Fetch the camera calibration of every HD-EPIC recording, without downloading the SLAM zips.

Projecting gaze into the video frame needs the calibration of the RGB camera of the glasses
that made the recording. HD-EPIC stores it in its SLAM output, one zip per group of
recordings: SLAM-and-Gaze/<P>/SLAM/multi/<n>.zip, file <n>/slam/online_calibration.jsonl.
The zips are 0.5-4 GB each, but the calibration file inside is about 3 MB compressed.

The server supports HTTP range requests, so this command reads the zip's directory from the
end of the remote file and downloads only the calibration entry. Where a zip is already on
disk, it reads the local copy. vrs_to_multi_slam.json gives the group of each recording.

Each line of online_calibration.jsonl is one calibration record (about 6000 per group). The
command keeps every --every-th record and the last one, which is enough to measure how much
the calibration drifts, and writes them to SLAM-and-Gaze/<P>/SLAM/calibration/<n>.jsonl.

Usage
-----
    python -m ego fetch-calibrations --gaze-dir data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze
"""

import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/SLAM-and-Gaze/"
MEMBER = "slam/online_calibration.jsonl"


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, enough for zipfile."""

    def __init__(self, url, retries=4):
        self.url, self.retries, self.pos = url, retries, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.requests = 0
        self.bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        self.pos += len(data)
        self.requests += 1
        self.bytes += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def read_member(zf):
    """The text of the calibration entry of an open zip."""
    names = [n for n in zf.namelist() if n.endswith(MEMBER)]
    if len(names) != 1:
        raise ValueError(f"expected one {MEMBER}, found {names}")
    with zf.open(names[0]) as f:
        return f.read().decode()


def group_map(multi, out_dir, p):
    """
    {"<P>/<recording>.vrs": "<group>"}. The local vrs_to_multi_slam.json is an empty
    placeholder where the SLAM data was not downloaded; then the file is fetched and saved
    next to the calibrations.
    """
    for f in (multi / "vrs_to_multi_slam.json", out_dir / "vrs_to_multi_slam.json"):
        if f.exists() and f.stat().st_size > 0:
            return json.loads(f.read_text())
    with urllib.request.urlopen(BASE_URL + f"{p}/SLAM/multi/vrs_to_multi_slam.json",
                                timeout=60) as r:
        text = r.read().decode()
    (out_dir / "vrs_to_multi_slam.json").write_text(text)
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaze-dir", required=True, help=".../HD-EPIC/SLAM-and-Gaze")
    ap.add_argument("--participants", nargs="+",
                    default=["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"])
    ap.add_argument("--every", type=int, default=100, help="keep every N-th calibration record")
    args = ap.parse_args()

    root = Path(args.gaze_dir)
    total_remote = 0
    for p in args.participants:
        multi = root / p / "SLAM" / "multi"
        out_dir = root / p / "SLAM" / "calibration"
        out_dir.mkdir(parents=True, exist_ok=True)
        groups = sorted(set(group_map(multi, out_dir, p).values()), key=int)
        for g in groups:
            out = out_dir / f"{g}.jsonl"
            if out.exists() and out.stat().st_size > 0:
                print(f"{p}/{g}: already fetched")
                continue
            local = multi / f"{g}.zip"
            t0 = time.time()
            text = None
            if local.exists() and local.stat().st_size > 0:
                try:
                    with zipfile.ZipFile(local) as zf:
                        text = read_member(zf)
                    how = "local zip"
                except zipfile.BadZipFile:
                    print(f"{p}/{g}: local zip is damaged (an interrupted download?), "
                          "using the server")
            if text is None:
                hf = HTTPRangeFile(BASE_URL + f"{p}/SLAM/multi/{g}.zip")
                with zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 20)) as zf:
                    text = read_member(zf)
                total_remote += hf.bytes
                how = f"remote, {hf.bytes / 1e6:.1f} MB in {hf.requests} requests"
            lines = text.splitlines()
            keep = lines[::args.every] + ([lines[-1]] if (len(lines) - 1) % args.every else [])
            out.write_text("\n".join(keep) + "\n")
            print(f"{p}/{g}: {len(lines)} records, kept {len(keep)} ({how}, {time.time() - t0:.0f}s)",
                  flush=True)
    print(f"downloaded {total_remote / 1e6:.0f} MB in total")


if __name__ == "__main__":
    main()
