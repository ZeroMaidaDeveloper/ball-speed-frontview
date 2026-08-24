"""LAB-scale ball color detector, ported from the user's interactive
labeling tool ("Auto labelling usin LAB Scale/Auto_Model_Training/
ball_label.py" and "ball_label_white.py").

This is a DIFFERENT design tradeoff from pipeline/classical_detect.py,
kept as its own module so the two can be compared side by side in the
viewer rather than one silently replacing the other:

  - classical_detect.py uses LAB color as a soft confidence NUDGE on top
    of motion+shape filtering, because on the original test footage a
    hard color threshold passed a player's pink shirt just as happily as
    the ball (see that module's docstring).
  - This module uses LAB color as a HARD per-contour gate (mean a* over
    the contour must clear a threshold for a red ball, or mean L* must be
    high with low chroma for a white ball) -- ball_color and the exact
    thresholds are config.yaml: lab_scale_detect.*.

It also differs in a second way: MOG2 runs on a CLAHE-contrast-enhanced
L channel rather than raw BGR. CLAHE boosts LOCAL contrast, which can
make a low-contrast object visible against a busy background even when
its raw pixel values don't stand out (relevant here: diagnostics on this
project's footage showed the ball has almost no raw LAB a*/b* signal --
CLAHE may change that by the time MOG2 sees it).

Blob filtering (before the color gate) uses a fill-ratio roundness check
(contour area / minimum-enclosing-circle area) and an aspect-ratio cap,
rather than classical_detect.py's circularity (4*pi*area/perimeter^2).

The ported tool's own footage was short (~6s), single-delivery clips with
a human-drawn ROI polygon restricting detection to the pitch corridor, so
it never needed classical_detect.py's activity-based recurring-motion
suppression. Our footage is long and unrestricted, so this module reuses
that same suppression (config.yaml: detection.activity_decay/activity_max)
-- without it, net/leaf jitter floods every frame with dozens of
contours, which is both a quality problem (noise competing with the real
ball) and a performance one (the per-contour color-gate work below scales
with contour count).

Also ported: the two techniques that turned out to matter most once we
actually A/B'd this module's output against the full auto-label pipeline
on real clips (both found in ball_label.py) --
  - a `<video>_roi.json` polygon (pipeline/roi_utils.py), when one exists
    next to the video, restricting detection to the pitch corridor;
  - YOLO person-instance-segmentation masks, excluding any candidate
    whose center falls inside a detected person -- a moving player was
    consistently the dominant source of false candidates once color
    stopped being a hard requirement.
Unlike ball_label.py's choose() (which collapses to a single best
candidate per frame with its own simple linear-prediction gate), this
module still emits every surviving candidate per frame and leaves
temporal association to pipeline/track.py's Kalman tracker.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from roi_utils import load_roi_mask

_MEDIAN_KSIZE = 3
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_COCO_PERSON_CLASS = 0


def _clahe_l_channel(frame_bgr: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    return clahe.apply(l_chan)


def _clean_motion_mask(fg: np.ndarray) -> np.ndarray:
    fg = cv2.medianBlur(fg, _MEDIAN_KSIZE)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _MORPH_KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    return fg


def _fill_ratio(contour: np.ndarray, area: float) -> float:
    (_, _), radius = cv2.minEnclosingCircle(contour)
    enclosing_area = math.pi * radius * radius
    return area / enclosing_area if enclosing_area > 0 else 0.0


def _passes_color_gate(
    frame_lab: np.ndarray, mask: np.ndarray, cfg_lab: dict[str, Any]
) -> tuple[bool, float]:
    """Hard color gate over the pixels in `mask` (a contour-filled binary
    mask, same shape as frame_lab's first two dims). Returns (passes, a
    0..1-ish confidence score for the surviving candidate)."""
    l_mean, a_mean, b_mean = cv2.mean(frame_lab, mask=mask)[:3]

    if cfg_lab["ball_color"] == "white":
        chroma = math.hypot(a_mean - 128.0, b_mean - 128.0)
        passes = l_mean >= cfg_lab["white_l_min"] and chroma <= cfg_lab["white_chroma_max"]
        # Higher lightness and lower chroma both indicate a better match;
        # combine into a single 0..1-ish score.
        whiteness = max(0.0, (l_mean - cfg_lab["white_l_min"])) / 65.0
        return passes, min(1.0, 0.5 + whiteness)

    passes = a_mean >= cfg_lab["red_mean_a_min"]
    redness = (a_mean - 128.0) / 40.0
    return passes, min(1.0, max(0.0, redness))


def _person_masks(model, frame: np.ndarray, cfg_lab: dict[str, Any]) -> list[np.ndarray]:
    """YOLO instance-segmentation masks for people in `frame`, filtered to
    ones tall enough to plausibly be a player rather than someone far in
    the background (mirrors ball_label.py's own filter)."""
    height, width = frame.shape[:2]
    results = model(frame, classes=[_COCO_PERSON_CLASS], conf=cfg_lab["person_conf_threshold"], verbose=False)
    if not results or results[0].masks is None:
        return []

    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        boxes.append((x1, y1, x2, y2, y2 - y1))
    if not boxes:
        return []

    max_h = max(b[4] for b in boxes)
    min_height = max(cfg_lab["person_min_abs_height_px"], cfg_lab["person_min_height_ratio"] * max_h)
    masks_data = results[0].masks.data.cpu().numpy()

    person_masks = []
    for i, (_x1, _y1, _x2, _y2, h) in enumerate(boxes):
        if h < min_height:
            continue
        mask = (masks_data[i] * 255).astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        person_masks.append(mask)
    return person_masks


def _inside_any_mask(cx: int, cy: int, masks: list[np.ndarray]) -> bool:
    return any(mask[cy, cx] > 0 for mask in masks)


class LabScaleFrameDetector:
    """Stateful per-frame LAB-scale detector -- the same MOG2/CLAHE/color-
    gate/ROI/person-mask pipeline as detect_candidates() below, factored
    out so pipeline/zoom_track_detect.py can drive it one frame at a time
    (interleaved with its own zoom-tracking logic) instead of only being
    usable as a single batch pass over a whole video."""

    def __init__(self, video_path: str, config: dict[str, Any]):
        self.cfg_det = config["detection"]
        self.cfg_lab = config["lab_scale_detect"]
        self.video_path = video_path

        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg_det["mog2_history"],
            varThreshold=self.cfg_det["mog2_var_threshold"],
            detectShadows=self.cfg_det["mog2_detect_shadows"],
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=self.cfg_lab["clahe_clip_limit"],
            tileGridSize=(self.cfg_lab["clahe_tile_grid"], self.cfg_lab["clahe_tile_grid"]),
        )
        self.person_model = None
        if self.cfg_lab["exclude_people"]:
            from ultralytics import YOLO  # lazy import: heavy dependency, only needed here

            self.person_model = YOLO(self.cfg_lab["person_seg_model"])

        self.activity: np.ndarray | None = None
        self.roi_mask: np.ndarray | None = None
        self.person_masks: list[np.ndarray] = []
        self.frame_idx = -1

    def process_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Advance internal state by one frame and return this frame's
        surviving candidates as {"cx","cy","diam_px","bbox","conf"} dicts
        (no "frame"/"t"/"source" -- the caller attaches those)."""
        cfg_det, cfg_lab = self.cfg_det, self.cfg_lab
        self.frame_idx += 1

        if self.roi_mask is None and cfg_det["use_roi"]:
            self.roi_mask = load_roi_mask(self.video_path, frame.shape)

        enhanced_l = _clahe_l_channel(frame, self.clahe)
        fg = self.mog2.apply(enhanced_l)
        if self.activity is None:
            self.activity = np.zeros(fg.shape, dtype=np.float32)
        if self.frame_idx < cfg_det["warmup_frames"]:
            self.activity = self.activity * cfg_det["activity_decay"] + (fg > 0).astype(np.float32)
            return []

        fg = _clean_motion_mask(fg)
        if self.roi_mask is not None:
            fg = cv2.bitwise_and(fg, self.roi_mask)

        if self.person_model is not None and self.frame_idx % cfg_lab["person_mask_interval"] == 0:
            self.person_masks = _person_masks(self.person_model, frame, cfg_lab)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: list[dict[str, Any]] = []
        frame_lab = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < cfg_lab["min_area_px"] or area > cfg_lab["max_area_px"]:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if self.activity[y : y + h, x : x + w].mean() > cfg_det["activity_max"]:
                continue

            long_side, short_side = max(w, h), max(1, min(w, h))
            if long_side / short_side > cfg_lab["max_aspect_ratio"]:
                continue

            if _fill_ratio(contour, area) < cfg_lab["min_fill_ratio"]:
                continue

            if self.person_masks and _inside_any_mask(x + w // 2, y + h // 2, self.person_masks):
                continue

            if frame_lab is None:
                frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            # Mask + mean only over the contour's own bounding box -- see
            # detect_candidates() below for why this matters for speed.
            lab_crop = frame_lab[y : y + h, x : x + w]
            local_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(local_mask, [contour], -1, 255, thickness=cv2.FILLED, offset=(-x, -y))

            passes, conf = _passes_color_gate(lab_crop, local_mask, cfg_lab)
            if not passes:
                continue

            out.append(
                {
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "cx": x + w / 2.0,
                    "cy": y + h / 2.0,
                    "diam_px": (w + h) / 2.0,
                    "conf": float(conf),
                }
            )

        # Update activity AFTER using it this frame, so a blob is judged
        # against its history up to (not including) the frame it's in.
        self.activity = self.activity * cfg_det["activity_decay"] + (fg > 0).astype(np.float32)
        return out


def detect_candidates(video_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan `video_path` end to end and return a flat list of per-frame
    LAB-scale candidates (0, 1, or several per frame)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    detector = LabScaleFrameDetector(video_path, config)

    candidates: list[dict[str, Any]] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        for c in detector.process_frame(frame):
            candidates.append(
                {
                    "frame": detector.frame_idx,
                    "t": t,
                    "bbox": c["bbox"],
                    "conf": c["conf"],
                    "source": "lab_scale",
                }
            )

    cap.release()
    return candidates


if __name__ == "__main__":
    import sys

    import yaml

    video_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else __file__.rsplit("/", 1)[0] + "/../config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cands = detect_candidates(video_path, cfg)
    by_frame: dict[int, int] = {}
    for c in cands:
        by_frame[c["frame"]] = by_frame.get(c["frame"], 0) + 1
    print(f"total candidates: {len(cands)} across {len(by_frame)} frames")
    if cands:
        confs = [c["conf"] for c in cands]
        print(f"conf range: {min(confs):.2f}..{max(confs):.2f}, mean={sum(confs) / len(confs):.2f}")
