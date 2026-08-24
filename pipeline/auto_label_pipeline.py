"""Headless runner for the user's own auto-labeling detection pipeline
("Auto labelling usin LAB Scale/Auto_Model_Training/ball_label.py"),
imported and called directly (not reimplemented) -- the interactive GUI
main() loop in that script needs a human at the keyboard/mouse, so this
strips that away and drives the exact same detection functions
(preprocess, clean_motion, find_candidates, choose) frame-by-frame
instead, using the video's own saved ROI polygon
(<video>_roi.json next to it) and the same person-mask exclusion via
YOLO segmentation.

Produces one candidate per frame at most (source="auto_label") --
ball_label.py's choose() already IS a simple single-target tracker
(linear-extrapolation gate from the last 2 accepted points), so this is
their whole pipeline's output, not just a raw per-frame detector.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from roi_utils import load_roi_mask

_AUTO_LABEL_DIR = Path(__file__).resolve().parent.parent / "Auto labelling usin LAB Scale" / "Auto_Model_Training"
sys.path.insert(0, str(_AUTO_LABEL_DIR))
import ball_label as bl  # noqa: E402  (path must be inserted first)

_YOLO_INTERVAL = 2  # matches ball_label.py's own YOLO_INTERVAL


def detect_candidates(video_path: str) -> list[dict[str, Any]]:
    """Run ball_label.py's actual detection+choose() pipeline over the
    whole video and return one candidate per frame where it found a ball
    (bbox as [x1, y1, x2, y2], source="auto_label")."""
    model = bl.YOLO(bl.YOLO_MODEL)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    ok, frame0 = cap.read()
    if not ok:
        raise ValueError(f"could not read first frame: {video_path}")
    height, width = frame0.shape[:2]
    roi_mask = load_roi_mask(video_path, frame0.shape)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    mode = 1  # preprocess mode 1 = CLAHE-enhanced LAB-L, ball_label.py's own default

    ftrack: deque = deque(maxlen=bl.TRAIL_LEN)
    missed = 0
    yolo_counter = 0
    person_masks: list[np.ndarray] = []

    candidates: list[dict[str, Any]] = []
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        a_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1]
        motion = bl.clean_motion(bg.apply(bl.preprocess(frame, mode)))
        motion_roi = cv2.bitwise_and(motion, roi_mask)

        if yolo_counter % _YOLO_INTERVAL == 0:
            person_masks = []
            results = model(frame, classes=[0], conf=bl.CONF_PERSON, verbose=False)
            if results and results[0].masks is not None:
                all_boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    all_boxes.append((x1, y1, x2, y2, y2 - y1))
                if all_boxes:
                    max_h = max(b[4] for b in all_boxes)
                    min_height = max(bl.PERSON_MIN_ABS_HEIGHT, bl.PERSON_MIN_HEIGHT_RATIO * max_h)
                    masks = results[0].masks.data.cpu().numpy()
                    for i, (x1, y1, x2, y2, h) in enumerate(all_boxes):
                        if h >= min_height:
                            mk = (masks[i] * 255).astype(np.uint8)
                            if mk.shape != (height, width):
                                mk = cv2.resize(mk, (width, height), interpolation=cv2.INTER_NEAREST)
                            person_masks.append(mk)
        yolo_counter += 1

        cands = bl.find_candidates(motion_roi, a_channel, bl.MIN_FILL, bl.RED_MEAN_MIN, bl.USE_REDNESS, person_masks)

        predicted = None
        if len(ftrack) >= 2:
            (x1, y1) = ftrack[-2]["pt"]
            (x2, y2) = ftrack[-1]["pt"]
            predicted = (2 * x2 - x1, 2 * y2 - y1)

        ball = bl.choose(cands, predicted, True)
        if ball is not None:
            ftrack.append({"pt": ball["center"], "pred": False})
            missed = 0
            x, y, w, h = ball["bbox"]
            redness = min(1.0, max(0.0, (ball["mean_a"] - 128.0) / 40.0))
            conf = min(1.0, (redness + ball["fill"]) / 2.0)
            candidates.append(
                {
                    "frame": frame_idx,
                    "t": t,
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "conf": float(conf),
                    "source": "auto_label",
                }
            )
        else:
            missed += 1
            if missed > bl.MAX_MISSED:
                ftrack.clear()

    cap.release()
    return candidates


def save_candidates(video_path: str, runs_dir: str) -> Path:
    """Run detect_candidates() and write runs/<stem>/candidates_auto_label.json
    in the same {video,fps,width,height,duration_s,mode,candidates} shape
    as pipeline/run_diagnostics.py's other candidate dumps, so the viewer's
    source filter can load it identically."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cands = detect_candidates(video_path)

    stem = Path(video_path).stem
    out_dir = Path(runs_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates_auto_label.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "video": Path(video_path).name,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_s": frame_count / fps if fps else 0.0,
                "mode": "auto_label",
                "candidates": cands,
            },
            f,
        )
    return out_path


if __name__ == "__main__":
    video_path = sys.argv[1]
    runs_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parent.parent / "runs")
    out_path = save_candidates(video_path, runs_dir)
    print(f"wrote {out_path}")
