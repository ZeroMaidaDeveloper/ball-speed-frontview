"""Heatmap-guided ROI zoom-in for YOLO, targeting the small/far-away ball.

Motivation: pipeline/yolo_detect.py runs YOLO on the FULL frame, where a
far-away ball is only a few pixels across -- too small, relative to the
image, for YOLO to reliably see. This is a general small-object-detection
limitation (the same footage tracks well once the ball is close/large),
not a model-quality issue. This module instead:

  1. builds a per-frame "ball probability" heatmap by combining MOG2
     motion evidence and LAB color-match evidence (the same two raw
     signals pipeline/classical_detect.py's motion_only/color_only modes
     expose), WITHOUT the blob-shape/size filtering classical_detect.py
     applies -- a far-away ball's blob is often too small/faint to pass
     those filters even though there IS a faint concentration of
     motion+color evidence at its true location;
  2. finds the heatmap's LOCAL maxima (connected components of the
     thresholded evidence map, not just the single global peak) as
     several "most probable ball position" proposals this frame. A
     single global-max approach was tried first and failed badly: it
     gets dominated by whatever produces the loudest evidence anywhere in
     the frame (almost always a player moving, not the much fainter
     far-away ball), so it would confidently zoom into the wrong spot for
     seconds at a time and never even look near the real ball. Proposing
     several regions and letting YOLO itself filter them is much more
     robust;
  3. crops a small window around each proposal from the FULL-RESOLUTION
     frame and upscales it (config.yaml: heatmap_refine.upscale_factor),
     so the ball occupies a much larger fraction of what YOLO actually
     sees, then runs YOLO on that crop;
  4. maps any detection back to full-frame coordinates.

Candidates from this module use source="yolo_refined" (see SCHEMA.md) so
downstream code / the viewer can tell them apart from a plain full-frame
YOLO hit.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_MORPH_OPEN_KERNEL = np.ones((3, 3), np.uint8)
_MORPH_CLOSE_KERNEL = np.ones((5, 5), np.uint8)
_WARMUP_FRAMES = 40  # let the MOG2 background model stabilize (see classical_detect.py)
_COCO_SPORTS_BALL_CLASS = 32


def _color_mask(frame_lab: np.ndarray, cfg_det: dict[str, Any]) -> np.ndarray:
    l_chan = frame_lab[:, :, 0]
    a_chan = frame_lab[:, :, 1]
    mask = (
        (l_chan >= cfg_det["lab_l_channel_min"])
        & (l_chan <= cfg_det["lab_l_channel_max"])
        & (a_chan >= cfg_det["lab_a_channel_min"])
    )
    return (mask.astype(np.uint8)) * 255


def _heatmap_peaks(
    heatmap: np.ndarray, blur_ksize: int, min_peak: float, max_peaks: int, max_component_area: float
) -> list[tuple[float, int, int]]:
    """Gaussian-blur the combined evidence map, threshold it, and return up
    to `max_peaks` (peak_value, x, y) proposals -- one per connected region
    of above-threshold evidence, sorted strongest first. Deliberately NOT
    just the single global maximum (see module docstring).

    Components larger than `max_component_area` are dropped entirely: a
    person moving produces a much bigger blob than a small/far ball ever
    would, and would otherwise crowd out real ball-sized proposals."""
    blurred = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), 0)
    _, thresh = cv2.threshold(blurred, min_peak, 255, cv2.THRESH_BINARY)
    n_labels, labels = cv2.connectedComponents(thresh.astype(np.uint8))

    peaks: list[tuple[float, int, int]] = []
    for label in range(1, n_labels):  # label 0 is the background
        component = labels == label
        area = int(component.sum())
        if area > max_component_area:
            continue
        peak_val = float(blurred[component].max())
        ys, xs = np.nonzero(component)
        # Centroid of the region weighted by evidence strength, rather than
        # a plain geometric centroid, so an asymmetric blob's proposal
        # point favors where the evidence is actually strongest.
        weights = blurred[ys, xs]
        wx = float(np.average(xs, weights=weights))
        wy = float(np.average(ys, weights=weights))
        peaks.append((peak_val, int(round(wx)), int(round(wy))))

    peaks.sort(key=lambda p: -p[0])
    return peaks[:max_peaks]


def detect_candidates(video_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan `video_path` end to end and return a flat list of per-frame
    heatmap-refined YOLO candidates (0, 1, or several per frame -- up to
    heatmap_refine.max_peaks ROI zoom-in attempts per frame, skipped
    entirely when there's no evidence worth zooming into)."""
    from ultralytics import YOLO  # lazy import: heavy dependency, only needed here

    cfg_det = config["detection"]
    cfg_hm = config["heatmap_refine"]
    model = YOLO(config["paths"]["yolo_weights"])

    half = cfg_hm["roi_half_size_px"]
    upscale = cfg_hm["upscale_factor"]
    blur_ksize = cfg_hm["heatmap_blur_ksize"]
    min_peak = cfg_hm["min_heatmap_peak"]
    max_peaks = cfg_hm["max_peaks"]
    max_component_area = cfg_hm["max_component_area_px"]
    conf_threshold = cfg_hm["yolo_conf_threshold"]
    activity_decay = cfg_det["activity_decay"]
    activity_max = cfg_det["activity_max"]
    activity: np.ndarray | None = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=cfg_det["mog2_history"],
        varThreshold=cfg_det["mog2_var_threshold"],
        detectShadows=cfg_det["mog2_detect_shadows"],
    )

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

        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        color_mask = _color_mask(frame_lab, cfg_det)

        # Combined evidence: motion OR color, each contributing up to 255,
        # so a spot with BOTH weak motion and weak color hints still adds
        # up, even if neither alone would pass classical_detect.py's
        # size/circularity gate.
        heatmap_raw = fg.astype(np.float32) + color_mask.astype(np.float32)

        # Same recurring-motion suppression as classical_detect.py's
        # activity filter, judged against history up to (not including)
        # this frame: without it, netting/leaf jitter and static color
        # patches flood the heatmap with dozens of spots all tied at the
        # 255 ceiling, and the real (transient) ball evidence never rises
        # above the noise floor to get picked as a top peak.
        heatmap = np.where(activity <= activity_max, heatmap_raw, 0.0)

        peaks = _heatmap_peaks(heatmap, blur_ksize, min_peak, max_peaks, max_component_area)
        activity = activity * activity_decay + (heatmap_raw > 0).astype(np.float32)
        if not peaks:
            continue

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Batch all of this frame's ROI crops into a SINGLE model.predict()
        # call rather than one call per peak -- per-call Python/ultralytics
        # overhead dominated wall-clock time far more than raw inference
        # cost did (a naive one-call-per-peak version took over an hour on
        # the full video and was killed for it); batching amortizes that
        # overhead across up to max_peaks images at once.
        crops_up = []
        origins = []  # (x0, y0) for each entry in crops_up, same order
        for _peak_val, px, py in peaks:
            x0 = max(0, px - half)
            y0 = max(0, py - half)
            x1 = min(frame_w, px + half)
            y1 = min(frame_h, py + half)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame[y0:y1, x0:x1]
            crops_up.append(cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC))
            origins.append((x0, y0))

        if not crops_up:
            continue

        results = model.predict(
            crops_up,
            verbose=False,
            conf=conf_threshold,
            classes=[_COCO_SPORTS_BALL_CLASS],
        )

        for (x0, y0), result in zip(origins, results):
            boxes = result.boxes
            if len(boxes) == 0:
                continue

            # Multiple ball-class detections in one crop would almost
            # always be the same object detected twice (this ROI is
            # small) -- keep only the most confident one per proposal.
            best = max(boxes, key=lambda b: float(b.conf[0]))
            bx1, by1, bx2, by2 = best.xyxy[0].tolist()
            candidates.append(
                {
                    "frame": frame_idx,
                    "t": t,
                    "bbox": [
                        x0 + bx1 / upscale,
                        y0 + by1 / upscale,
                        x0 + bx2 / upscale,
                        y0 + by2 / upscale,
                    ],
                    "conf": float(best.conf[0]),
                    "source": "yolo_refined",
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
    print(f"total candidates: {len(cands)}")
    if cands:
        confs = [c["conf"] for c in cands]
        print(f"conf range: {min(confs):.2f}..{max(confs):.2f}, mean={sum(confs) / len(confs):.2f}")
