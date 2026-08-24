"""Per-video camera calibration for pipeline/speed.py.

A `<video_stem>_calib.json` file sitting next to a video's REAL
(symlink-resolved) path -- same lookup convention as
pipeline/roi_utils.py's `<video_stem>_roi.json` -- gives the pixel height
of the near and far sets of stumps in a representative frame:

    {"near_stump_height_px": 374.0, "far_stump_height_px": 46.4,
     "frame_size": [1920, 1080]}

Both stump sets are the standard 0.711m height (config.yaml:
calibration.stump_height_m) and sit config.yaml: calibration.pitch_length_m
apart (the regulation 20.12m stumps-to-stumps pitch length). Given their
apparent pixel heights, the pinhole model gives two equations in two
unknowns (focal length f_px, and the camera's distance to the near
stumps):

    near_stump_height_px = f_px * stump_height_m / d_near
    far_stump_height_px  = f_px * stump_height_m / d_far
    d_far = d_near + pitch_length_m

Solving for f_px is what `focal_length_px()` does below. Once known, any
frame's ball distance-from-camera follows directly from its own apparent
diameter (see speed.py): z = f_px * ball_diameter_m / diam_px -- no
assumption about how much of the flight was actually captured is needed,
which is what makes this far more robust than dividing a fixed total
flight distance by the tracked duration (this project's original
approach, still used as a fallback here when no calibration file exists).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_calibration(video_path: str) -> dict[str, Any] | None:
    """Return the parsed `<video_stem>_calib.json` next to `video_path`'s
    real (symlink-resolved) path, or None if it doesn't exist."""
    real_path = Path(video_path).resolve()
    calib_path = real_path.with_name(real_path.stem + "_calib.json")
    if not calib_path.is_file():
        return None
    with calib_path.open() as f:
        return json.load(f)


def focal_length_px(calib: dict[str, Any], config: dict[str, Any]) -> float | None:
    """Derive the camera's focal length in pixels from `calib`'s near/far
    stump measurements, or None if the inputs are degenerate (e.g. near
    and far heights too close together to localize a finite distance)."""
    cfg_calib = config["calibration"]
    near_h = calib["near_stump_height_px"]
    far_h = calib["far_stump_height_px"]
    if near_h <= far_h:
        return None

    pitch_length_m = cfg_calib["pitch_length_m"]
    stump_height_m = cfg_calib["stump_height_m"]

    d_near = far_h * pitch_length_m / (near_h - far_h)
    if d_near <= 0:
        return None
    return near_h * d_near / stump_height_m
