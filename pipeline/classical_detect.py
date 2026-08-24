"""Classical (non-ML) ball-candidate detector.

Produces per-frame candidates matching the "Per-frame candidate detection"
contract in SCHEMA.md, consumed by `pipeline/track.py` (optionally alongside
`pipeline/yolo_detect.py`).

Pipeline: MOG2 background-subtraction blobs, filtered by size/circularity
(config.yaml: detection.min_ball_area_px/max_ball_area_px/min_circularity),
then by a recurring-motion "activity" filter (config.yaml:
detection.activity_decay/activity_max) that drops blobs sitting in a spot
that's been flickering foreground for a while -- netting and leaves flap in
place across many consecutive frames, unlike the ball which is only ever
briefly at any given position. Without this, raw MOG2 blobs on this footage
run ~24 candidates/frame (mostly net-mesh texture jitter), which overwhelms
track.py's single-active-track gating.

Color (LAB a*-channel "red-ness") is deliberately used only as a soft
confidence nudge, NOT a hard gate. Probing this footage
(scratch/probe_topcolor.py output against Takneek/1000313863.mp4) showed
the top "most red" MOG2 blobs were a player's pink/magenta shirt seen
through the net, and scratch/probe/handball_zoom.jpg confirms the actual
ball's worn-leather color sits in the same LAB a* range as that shirt --
a hard `lab_a_channel_min` threshold would just as happily pass the shirt
as the ball (or reject a real ball frame that happens to read a bit dull).
Real discrimination instead comes from pipeline/track.py's downstream
Kalman motion-continuity gating: a shirt patch doesn't trace a multi-frame
ballistic trajectory the way the ball does, so letting more candidates
through here and filtering on motion later is the more reliable split.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Frames to let the MOG2 background model stabilize before trusting any
# foreground blob (matches the warmup used in scratch/probe_motion.py /
# probe_topcolor.py while exploring this footage). Not exposed via
# config.yaml since it's a fixed startup transient, not a tunable behavior.
_WARMUP_FRAMES = 40

_MORPH_OPEN_KERNEL = np.ones((3, 3), np.uint8)
_MORPH_CLOSE_KERNEL = np.ones((5, 5), np.uint8)


def _circularity(contour: np.ndarray) -> tuple[float, float]:
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return area, 0.0
    return area, 4.0 * np.pi * area / (perim * perim)


def _color_bonus(frame_lab: np.ndarray, x: int, y: int, w: int, h: int, cfg_det: dict[str, Any]) -> float:
    """Small multiplicative nudge toward candidates that DO look red, without
    ever excluding candidates that don't (see module docstring)."""
    patch = frame_lab[max(0, y) : y + h, max(0, x) : x + w]
    if patch.size == 0:
        return 1.0
    l_mean, a_mean, _b_mean = patch.reshape(-1, 3).mean(axis=0)
    if not (cfg_det["lab_l_channel_min"] <= l_mean <= cfg_det["lab_l_channel_max"]):
        return 1.0
    if a_mean >= cfg_det["lab_a_channel_min"]:
        return 1.15
    return 1.0


def _color_only_mask(frame_lab: np.ndarray, cfg_det: dict[str, Any]) -> np.ndarray:
    """Binary mask of pixels matching the ball's expected LAB range -- the
    same range `_color_bonus` nudges toward, but used here as the ONLY
    signal (no motion), for the "what does color alone find" diagnostic."""
    l_chan = frame_lab[:, :, 0]
    a_chan = frame_lab[:, :, 1]
    mask = (
        (l_chan >= cfg_det["lab_l_channel_min"])
        & (l_chan <= cfg_det["lab_l_channel_max"])
        & (a_chan >= cfg_det["lab_a_channel_min"])
    )
    return (mask.astype(np.uint8)) * 255


def detect_candidates(
    video_path: str, config: dict[str, Any], mode: str = "hybrid"
) -> list[dict[str, Any]]:
    """Scan `video_path` end to end and return a flat list of per-frame
    classical candidates (0, 1, or several per frame -- pipeline/track.py
    resolves which, if any, belong to the ball).

    `mode`:
      "hybrid" (default) -- MOG2 motion blobs, LAB color as a soft
        confidence nudge. This is what the real pipeline uses.
      "motion_only" -- MOG2 motion blobs, color ignored entirely. Useful to
        see what the motion signal alone finds.
      "color_only" -- pure LAB-range pixel mask, no motion/MOG2 at all.
        Useful to see what the color signal alone finds (expect this to be
        sparse/noisy on footage where the ball's color isn't distinctive --
        see module docstring).
    """
    if mode not in ("hybrid", "motion_only", "color_only"):
        raise ValueError(f"unknown mode: {mode!r}")

    cfg_det = config["detection"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    if mode == "color_only":
        return _detect_color_only(cap, cfg_det)
    return _detect_motion_based(cap, cfg_det, use_color_bonus=(mode == "hybrid"))


def _detect_color_only(cap: cv2.VideoCapture, cfg_det: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = _color_only_mask(frame_lab, cfg_det)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_OPEN_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_CLOSE_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        for contour in contours:
            area, circularity = _circularity(contour)
            if area < cfg_det["min_ball_area_px"] or area > cfg_det["max_ball_area_px"]:
                continue
            if circularity < cfg_det["min_circularity"]:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append(
                {
                    "frame": frame_idx,
                    "t": t,
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "conf": float(circularity),
                    "source": "classical",
                }
            )

    cap.release()
    return candidates


def _detect_motion_based(
    cap: cv2.VideoCapture, cfg_det: dict[str, Any], use_color_bonus: bool
) -> list[dict[str, Any]]:
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=cfg_det["mog2_history"],
        varThreshold=cfg_det["mog2_var_threshold"],
        detectShadows=cfg_det["mog2_detect_shadows"],
    )

    activity_decay = cfg_det["activity_decay"]
    activity_max = cfg_det["activity_max"]
    activity: np.ndarray | None = None

    candidates: list[dict[str, Any]] = []
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        fg = mog2.apply(frame)
        if activity is None:
            activity = np.zeros(fg.shape, dtype=np.float32)
        if frame_idx < _WARMUP_FRAMES:
            continue

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _MORPH_OPEN_KERNEL)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _MORPH_CLOSE_KERNEL)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_lab = None

            for contour in contours:
                area, circularity = _circularity(contour)
                if area < cfg_det["min_ball_area_px"] or area > cfg_det["max_ball_area_px"]:
                    continue
                if circularity < cfg_det["min_circularity"]:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if activity[y : y + h, x : x + w].mean() > activity_max:
                    continue

                conf = circularity
                if use_color_bonus:
                    if frame_lab is None:
                        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    bonus = _color_bonus(frame_lab, x, y, w, h, cfg_det)
                    # circularity alone (0.55..1.0 after the filter above) is
                    # the base confidence signal; the color bonus only ever
                    # scales it up a little, never gates it out.
                    conf = min(1.0, circularity * bonus)

                candidates.append(
                    {
                        "frame": frame_idx,
                        "t": t,
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "conf": float(conf),
                        "source": "classical",
                    }
                )

        # Update activity AFTER using it this frame, so a blob is judged
        # against its history up to (not including) the frame it's in.
        activity = activity * activity_decay + (fg > 0).astype(np.float32)

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
    by_frame = {}
    for c in cands:
        by_frame.setdefault(c["frame"], 0)
        by_frame[c["frame"]] += 1
    print(f"total candidates: {len(cands)} across {len(by_frame)} frames")
    if cands:
        confs = [c["conf"] for c in cands]
        print(f"conf range: {min(confs):.2f}..{max(confs):.2f}, mean={sum(confs) / len(confs):.2f}")
