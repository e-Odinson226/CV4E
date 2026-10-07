"""
Gaze as a point in the video frame.

The gaze file gives a direction (yaw and pitch, in the central pupil frame of the glasses,
CPF) and a depth. GazeProjector turns them into the pixel where the person looks, in the
upright 1408 x 1408 HD-EPIC frame:

  1. the 3D gaze point in the CPF, at the given depth (projectaria_tools)
  2. moved to the RGB camera: T_camera_cpf = T_device_camera^-1 * T_device_cpf
  3. projected with the RGB camera model (Fisheye624), scaled to 1408 pixels and rotated
     90 degrees clockwise, because the camera sits sideways in the glasses and HD-EPIC's
     videos are upright

The camera calibration comes from HD-EPIC's SLAM output of each recording group
(`python -m ego fetch-calibrations`). The CPF-to-device transform is not in it; it comes from
Aria's CAD values for Gen 1 glasses. The two Gen 1 frame sizes place the CPF 2.4 mm apart, so
DVT-S is used for every recording.

The eyes are about 6 cm from the RGB camera, so the pixel depends on the depth: at 0.3 m the
point is 1.4 patches away from where the same direction at infinity would land.

A recording without a calibration of its own (P06-20240510-140459) uses the calibration of
the participant's recording made closest in time, most likely with the same glasses.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

FRAME = 1408        # HD-EPIC video frames are 1408 x 1408
GRID = 16           # the encoder's patch grid at 256-pixel input


def group_map(gaze_dir, participant):
    """{recording stem: SLAM group} from vrs_to_multi_slam.json (multi/ or calibration/)."""
    base = Path(gaze_dir) / participant / "SLAM"
    for f in (base / "multi" / "vrs_to_multi_slam.json", base / "calibration" / "vrs_to_multi_slam.json"):
        if f.exists() and f.stat().st_size > 0:
            return {Path(k).stem: v for k, v in json.loads(f.read_text()).items()}
    raise FileNotFoundError(f"no vrs_to_multi_slam.json for {participant}; run fetch-calibrations")


def _stem_time(stem):
    return datetime.strptime("-".join(stem.split("-")[1:3]), "%Y%m%d-%H%M%S")


def calibration_group(gaze_dir, participant, stem):
    """(group, own) - own is False when the recording has no group and the closest in time is used."""
    m = group_map(gaze_dir, participant)
    if stem in m:
        return m[stem], True
    t = _stem_time(stem)
    nearest = min(m, key=lambda s: abs((_stem_time(s) - t).total_seconds()))
    return m[nearest], False


class GazeProjector:
    """Gaze (yaw, pitch, depth) of one recording -> pixel (x, y) in the upright 1408-px frame."""

    def __init__(self, gaze_dir, participant, stem, size=FRAME):
        from projectaria_tools.core import calibration as cal
        from projectaria_tools.core import mps

        self._mps = mps
        self.group, self.own_calibration = calibration_group(gaze_dir, participant, stem)
        f = Path(gaze_dir) / participant / "SLAM" / "calibration" / f"{self.group}.jsonl"
        online = mps.read_online_calibration(str(f))[0]     # the RGB calibration does not drift
        cams = {c.get_label(): c for c in online.camera_calibs}
        device = cal.DeviceCalibration(
            camera_calibs=cams,
            device_cad_extrinsics=cal.DeviceCadExtrinsics(cal.DeviceVersion.Gen1, "DVT-S",
                                                          "camera-slam-left"),
            device_subtype="DVT-S", origin_label="camera-slam-left",
            device_version=cal.DeviceVersion.Gen1)
        rgb = cal.rescale_camera_calibration(cams["camera-rgb"], np.array([size, size], np.int32),
                                             cal.DeviceVersion.Gen1)
        self.camera = cal.rotate_camera_calib_cw90deg(rgb)
        self.T_camera_cpf = self.camera.get_transform_device_camera().inverse() @ \
            device.get_transform_device_cpf()
        self.serial = cams["camera-rgb"].get_serial_number()
        self.size = size

    def project(self, yaw, pitch, depth):
        """Pixel (x, y) as floats, or None if the point does not project into the image."""
        p = self.T_camera_cpf @ self._mps.get_eyegaze_point_at_depth(float(yaw), float(pitch),
                                                                     float(depth))
        uv = self.camera.project(p)
        if uv is None:
            return None
        x, y = np.asarray(uv, dtype=float).ravel()
        return (x, y) if (0 <= x < self.size and 0 <= y < self.size) else None


def to_patch(xy, size=FRAME, grid=GRID):
    """Pixel (x, y) -> continuous patch coordinates (column, row) on the encoder's grid."""
    return xy[0] / size * grid, xy[1] / size * grid
