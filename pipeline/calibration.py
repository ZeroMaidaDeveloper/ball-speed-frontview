"""Per-video camera calibration for pipeline/speed.py.

A `<video_stem>_calib.json` file sitting next to a video's REAL
(symlink-resolved) path -- same lookup convention as
pipeline/roi_utils.py's `<video_stem>_roi.json` -- gives what's needed to
derive that video's focal length in pixels (f_px), in one of two ways:

1. PREFERRED -- a directly measured camera-to-near-stumps distance (e.g.
   from a phone AR measuring app: Google's "Measure" on Android/ARCore,
   Apple's "Measure" on iOS/ARKit, optionally LiDAR-assisted on Pro
   models -- or just a laser rangefinder/tape measure), plus the near
   stumps' pixel height:

       {"near_stump_height_px": 374.0, "measured_near_distance_m": 3.0,
        "frame_size": [1920, 1080]}

   f_px follows directly from one similar-triangles equation:
   near_stump_height_px = f_px * stump_height_m / measured_near_distance_m.
   No far-stump measurement needed at all -- which also sidesteps this
   project's recurring pain point of the far stumps being small, distant,
   and often partly hidden behind a batsman.

2. FALLBACK -- when no direct measurement was taken, the near AND far
   stump pixel heights in a representative frame:

       {"near_stump_height_px": 374.0, "far_stump_height_px": 46.4,
        "frame_size": [1920, 1080]}

   Both stump sets are the standard 0.711m height (config.yaml:
   calibration.stump_height_m) and sit config.yaml: calibration.
   pitch_length_m apart (the regulation 20.12m stumps-to-stumps pitch
   length). Given their apparent pixel heights, the pinhole model gives
   two equations in two unknowns (f_px and the camera's distance to the
   near stumps, d_near):

       near_stump_height_px = f_px * stump_height_m / d_near
       far_stump_height_px  = f_px * stump_height_m / d_far
       d_far = d_near + pitch_length_m

   Less accurate than (1): it inherits whatever error is in the assumed
   stump_height_m, and the far stumps are hard to measure precisely at a
   distance (see e.g. this project's teevra.mov and delivery_4.mp4 calib
   files, both of which needed extra care because of exactly this).

Either way, once f_px is known, any frame's ball distance-from-camera
follows directly from its own apparent diameter (see speed.py):
z = f_px * ball_diameter_m / diam_px -- no assumption about how much of
the flight was actually captured is needed, which is what makes this far
more robust than dividing a fixed total flight distance by the tracked
duration (this project's original approach, still used as a fallback here
when no calibration file exists at all).

THIRD, weaker option -- `<video_stem>_wickets_calib.json`, checked only
when no `_calib.json` exists at all: a single-plane pixels-per-meter
conversion, ported from the source project's own standalone ball_speed.py
calibration tool (its "click the two wickets" two-point calibration --
see that script's WICKET_DIST/save_calibration). Used for footage where
not even the far-stump-height fallback (method 2 above) is possible
because no far-end wicket is set up at all (confirmed by inspection for
this project's clip_144626/144954/145619.mp4 nets session -- only a near/
bowling-end set of stumps exists in frame). Rather than clicking two
wickets 20.12m apart, it clicks the near stump's own known height
(stump_height_m) instead:

    {"points": [[900, 678], [900, 1076]], "pixel_distance": 398.0,
     "pixels_per_meter": 559.77, "wicket_distance_m": 0.711,
     "frame_size": [1920, 1080]}

This gives NO distance-from-camera model at all (unlike f_px above) --
just a fixed pixels-per-meter scale, only strictly valid at the near
stump's own distance from the camera. speed.py's planar_pixel_speed
method uses it to convert the ball's tracked pixel path length directly
to metres, which is known to drift for whatever fraction of the flight
happens further down the pitch than the calibration plane (perspective
makes the same real distance cover fewer pixels the farther away it is,
which this method can't account for) -- see that function's docstring.
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


def load_wickets_calibration(video_path: str) -> dict[str, Any] | None:
    """Return the parsed `<video_stem>_wickets_calib.json` next to
    `video_path`'s real (symlink-resolved) path, or None if it doesn't
    exist (see module docstring, third/weakest calibration option)."""
    real_path = Path(video_path).resolve()
    calib_path = real_path.with_name(real_path.stem + "_wickets_calib.json")
    if not calib_path.is_file():
        return None
    with calib_path.open() as f:
        return json.load(f)


def pixels_per_meter_from_wickets(calib: dict[str, Any]) -> float | None:
    """Extract the single-plane pixels-per-meter scale from a
    `_wickets_calib.json` (see module docstring), or None if it's missing
    or non-positive."""
    ppm = calib.get("pixels_per_meter")
    if not ppm or ppm <= 0:
        return None
    return float(ppm)


def focal_length_px(calib: dict[str, Any], config: dict[str, Any]) -> float | None:
    """Derive the camera's focal length in pixels from `calib`, or None if
    the inputs are degenerate (e.g. near and far heights too close
    together to localize a finite distance). Prefers a direct
    `measured_near_distance_m` (see module docstring, method 1) over the
    near/far stump-height solve (method 2) when both are present, since
    it doesn't depend on the assumed stump_height_m being exactly right."""
    cfg_calib = config["calibration"]
    near_h = calib["near_stump_height_px"]
    stump_height_m = cfg_calib["stump_height_m"]

    measured_d_near = calib.get("measured_near_distance_m")
    if measured_d_near:
        return near_h * measured_d_near / stump_height_m

    far_h = calib.get("far_stump_height_px")
    if far_h is None or near_h <= far_h:
        return None

    pitch_length_m = cfg_calib["pitch_length_m"]
    d_near = far_h * pitch_length_m / (near_h - far_h)
    if d_near <= 0:
        return None
    return near_h * d_near / stump_height_m
