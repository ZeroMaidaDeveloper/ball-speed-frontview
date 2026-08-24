"""Zoom-and-track ball detector.

Motivation: pipeline/yolo_detect.py's full-frame YOLO scan, even ROI-
restricted, still contributes a lot of noise relative to
pipeline/lab_scale_detect.py's much cleaner motion+color candidate
stream (see pipeline/run.py). This module instead:

  1. Scans every frame with LabScaleFrameDetector (lab_scale_detect.py's
     own per-frame MOG2+CLAHE+color-gate+ROI+person-mask logic) --
     cheap, and already the cleanest signal we have.
  2. Watches consecutive LAB hits for implied speed exceeding
     `fast_speed_px_s` -- the hallmark of a real ball in flight, since
     noise and slow-moving players don't move nearly that fast frame to
     frame. This is the "fast moving LAB object" trigger.
  3. Once triggered, switches into a tracking state: each subsequent
     frame, predicts the ball's next position from its last known
     position + velocity, and looks for a LAB candidate near that
     prediction (a tight gate, not the whole frame) to confirm and
     continue the track.
  4. If no LAB candidate matches the gate, falls back to a tight
     three-frame differencing search around the prediction (see
     `_motion_rescue` below) -- motion evidence only, no color
     requirement at all. Added after visually confirming (delivery_4.mp4,
     frames 137-149) that the real ball keeps moving as a clean, compact,
     consistently-sized blob well past the point where
     lab_scale_detect.py's hard color gate starts failing -- because the
     ball desaturates/darkens rapidly once it's in flight against the
     dark green netting/foliage background in this footage, even though
     its round shape and motion stay perfectly trackable. This recovers
     several times as many real frames of a delivery as LAB color alone.
  5. Only when NEITHER LAB nor the motion-diff rescue matches does this
     fall back further to the crop-and-zoom idea from heatmap_refine.py:
     crop+upscale a window around the prediction and run YOLO on just
     that crop. Tried last, not first, because empirically (see scratch
     investigation on delivery_4.mp4) a fast-moving ball is often
     motion-blurred into a streak that stock YOLO's "sports ball" class
     doesn't recognize at all, zoomed in or not, while both LAB and the
     motion-diff rescue keep tracking it fine -- YOLO confirmation alone
     would have dropped the track completely.
  6. If NONE of the three match for too many consecutive frames, drops
     back to plain LAB scanning to look for the next fast-moving object.

Emits "lab_scale"-sourced candidates every frame (same as
lab_scale_detect.py) PLUS "zoom_track"-sourced (gate-confirmed LAB hits),
"motion_rescue"-sourced (three-frame-diff confirmed), or
"zoom_track"-sourced YOLO-crop-rescue candidates while a confirmed
fast-moving trajectory is being actively tracked.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from lab_scale_detect import LabScaleFrameDetector

_COCO_SPORTS_BALL_CLASS = 32
_MOTION_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


def _motion_rescue(
    gray: np.ndarray,
    prev_gray: np.ndarray,
    prev_gray2: np.ndarray,
    pred_cx: float,
    pred_cy: float,
    frame_w: int,
    frame_h: int,
    cfg_zoom: dict[str, Any],
) -> dict[str, Any] | None:
    """Three-frame-differencing rescue: look for a small, compact, isolated
    moving blob near (pred_cx, pred_cy), with NO color requirement at all.

    Uses bitwise_and of two consecutive absdiffs (gray_t vs gray_t-1, and
    gray_t-1 vs gray_t-2) so a pixel only counts as motion if it changed on
    both steps -- suppresses single-frame sensor noise while still
    responding to a ball that's desaturated/darkened past
    lab_scale_detect.py's hard color gate (see module docstring). Returns
    the best {"bbox", "conf"} within `track_gate_px` of the prediction, or
    None.
    """
    half = cfg_zoom["motion_rescue_half_size_px"]
    x0, y0 = max(0, int(pred_cx) - half), max(0, int(pred_cy) - half)
    x1, y1 = min(frame_w, int(pred_cx) + half), min(frame_h, int(pred_cy) + half)
    if x1 <= x0 or y1 <= y0:
        return None

    d1 = cv2.absdiff(gray[y0:y1, x0:x1], prev_gray[y0:y1, x0:x1])
    d2 = cv2.absdiff(prev_gray[y0:y1, x0:x1], prev_gray2[y0:y1, x0:x1])
    motion = cv2.bitwise_and(d1, d2)
    _, motion = cv2.threshold(motion, cfg_zoom["motion_rescue_diff_threshold"], 255, cv2.THRESH_BINARY)
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, _MOTION_MORPH_KERNEL)

    contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_dist = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg_zoom["motion_rescue_min_area_px"] or area > cfg_zoom["motion_rescue_max_area_px"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        long_side, short_side = max(w, h), max(1, min(w, h))
        if long_side / short_side > cfg_zoom["motion_rescue_max_aspect_ratio"]:
            continue
        cx, cy = x0 + x + w / 2.0, y0 + y + h / 2.0
        dist = math.hypot(cx - pred_cx, cy - pred_cy)
        if dist > cfg_zoom["track_gate_px"]:
            continue
        if best is None or dist < best_dist:
            best = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
            best_dist = dist

    if best is None:
        return None
    return {"bbox": [float(v) for v in best], "conf": cfg_zoom["motion_rescue_conf"]}


def detect_candidates(video_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    from ultralytics import YOLO  # lazy import: heavy dependency, only needed here

    cfg_zoom = config["zoom_track_detect"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lab_detector = LabScaleFrameDetector(video_path, config)
    yolo_model = YOLO(config["paths"]["yolo_weights"])

    half = cfg_zoom["roi_half_size_px"]
    upscale = cfg_zoom["upscale_factor"]

    tracking = False
    last_lab_hit: dict[str, float] | None = None  # {"t","cx","cy"}
    track_pos: tuple[float, float] | None = None
    track_vel: tuple[float, float] | None = None
    track_t: float | None = None
    missed_in_track = 0
    # Grayscale history for _motion_rescue's three-frame differencing --
    # maintained unconditionally every frame (not just while tracking) so
    # it's already warm the moment tracking starts.
    prev_gray: np.ndarray | None = None
    prev_gray2: np.ndarray | None = None

    candidates: list[dict[str, Any]] = []
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lab_cands = lab_detector.process_frame(frame)
        for c in lab_cands:
            candidates.append(
                {"frame": frame_idx, "t": t, "bbox": c["bbox"], "conf": c["conf"], "source": "lab_scale"}
            )

        if not tracking:
            best = max(lab_cands, key=lambda c: c["conf"]) if lab_cands else None
            if best is not None and last_lab_hit is not None:
                dt = t - last_lab_hit["t"]
                if 0 < dt <= cfg_zoom["max_search_gap_s"]:
                    dist = math.hypot(best["cx"] - last_lab_hit["cx"], best["cy"] - last_lab_hit["cy"])
                    if dist / dt >= cfg_zoom["fast_speed_px_s"]:
                        # Confirmed fast-moving object -- start tracking.
                        tracking = True
                        track_pos = (best["cx"], best["cy"])
                        track_vel = ((best["cx"] - last_lab_hit["cx"]) / dt, (best["cy"] - last_lab_hit["cy"]) / dt)
                        track_t = t
                        missed_in_track = 0
            if best is not None:
                last_lab_hit = {"t": t, "cx": best["cx"], "cy": best["cy"]}
        else:
            # --- tracking state: predict, then confirm ------------------
            dt = t - track_t if track_t is not None else 0.0
            pred_cx = track_pos[0] + track_vel[0] * dt
            pred_cy = track_pos[1] + track_vel[1] * dt

            # 1) Prefer a LAB candidate near the prediction -- already
            # computed this frame, and (see module docstring) more reliable
            # than YOLO for a motion-blurred fast ball.
            confirmed = None
            confirmed_source = None
            gated = [
                c for c in lab_cands if math.hypot(c["cx"] - pred_cx, c["cy"] - pred_cy) <= cfg_zoom["track_gate_px"]
            ]
            if gated:
                best_gated = max(gated, key=lambda c: c["conf"])
                confirmed = {"bbox": best_gated["bbox"], "conf": best_gated["conf"]}
                confirmed_source = "zoom_track"

            # 2) Only if LAB found nothing near the prediction, try a
            # tight three-frame-differencing search around it -- motion
            # evidence only, no color requirement (see _motion_rescue and
            # module docstring for why this recovers many real frames LAB
            # alone loses once the ball desaturates against the netting).
            if confirmed is None and prev_gray is not None and prev_gray2 is not None:
                rescued = _motion_rescue(gray, prev_gray, prev_gray2, pred_cx, pred_cy, frame_w, frame_h, cfg_zoom)
                if rescued is not None:
                    confirmed = rescued
                    confirmed_source = "motion_rescue"

            # 3) Last resort: crop+upscale around the prediction and ask
            # YOLO -- kept as the final fallback since (see module
            # docstring) it empirically underperforms both LAB and the
            # motion-diff rescue on this footage's small, motion-blurred
            # ball.
            if confirmed is None:
                x0 = max(0, int(pred_cx) - half)
                y0 = max(0, int(pred_cy) - half)
                x1 = min(frame_w, int(pred_cx) + half)
                y1 = min(frame_h, int(pred_cy) + half)
                if x1 > x0 and y1 > y0:
                    crop = frame[y0:y1, x0:x1]
                    crop_up = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
                    results = yolo_model.predict(
                        crop_up, verbose=False, conf=cfg_zoom["yolo_conf_threshold"], classes=[_COCO_SPORTS_BALL_CLASS]
                    )
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        best_box = max(boxes, key=lambda b: float(b.conf[0]))
                        bx1, by1, bx2, by2 = best_box.xyxy[0].tolist()
                        confirmed = {
                            "bbox": [x0 + bx1 / upscale, y0 + by1 / upscale, x0 + bx2 / upscale, y0 + by2 / upscale],
                            "conf": float(best_box.conf[0]),
                        }
                        confirmed_source = "zoom_track"

            if confirmed is not None:
                bx1, by1, bx2, by2 = confirmed["bbox"]
                new_cx, new_cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if dt > 0:
                    track_vel = ((new_cx - track_pos[0]) / dt, (new_cy - track_pos[1]) / dt)
                track_pos = (new_cx, new_cy)
                track_t = t
                missed_in_track = 0
                candidates.append(
                    {
                        "frame": frame_idx,
                        "t": t,
                        "bbox": confirmed["bbox"],
                        "conf": confirmed["conf"],
                        "source": confirmed_source,
                    }
                )
            else:
                missed_in_track += 1
                if missed_in_track > cfg_zoom["max_missed_in_track"]:
                    tracking = False
                    track_pos = None
                    track_vel = None
                    track_t = None
                    last_lab_hit = None

        prev_gray2 = prev_gray
        prev_gray = gray

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
    by_source: dict[str, int] = {}
    for c in cands:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    print(f"total candidates: {len(cands)} -- {by_source}")
