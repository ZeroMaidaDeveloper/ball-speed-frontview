"""Shared per-video ROI polygon loading.

A `<video_stem>_roi.json` file sitting next to a video (the convention
used by the user's own "Auto labelling usin LAB Scale" tools) restricts
detection to a hand-drawn polygon (typically the pitch corridor),
eliminating most background clutter a detector would otherwise have to
filter out algorithmically. Shape: {"polygon": [[x,y],...], "frame_size":
[w,h]}.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_roi_mask(video_path: str, frame_shape: tuple[int, ...]) -> np.ndarray:
    """Return a single-channel 0/255 mask for `frame_shape`, built from
    `<video>_roi.json` next to the video's REAL (symlink-resolved) path if
    it exists and matches this frame size, else an all-255 (unrestricted)
    mask."""
    real_path = Path(video_path).resolve()
    roi_path = real_path.with_name(real_path.stem + "_roi.json")

    polygon = None
    if roi_path.is_file():
        with roi_path.open() as f:
            data = json.load(f)
        size = data.get("frame_size")
        h, w = frame_shape[:2]
        if not size or (size[0] == w and size[1] == h):
            polygon = [tuple(p) for p in data.get("polygon", [])]

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if polygon and len(polygon) >= 3:
        cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
    else:
        mask[:] = 255
    return mask
