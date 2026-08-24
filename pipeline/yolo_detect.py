"""YOLO-based ball-candidate detector.

Thin wrapper around an Ultralytics YOLO model (config.yaml:
paths.yolo_weights) filtered to the COCO "sports ball" class, producing
per-frame candidates matching the "Per-frame candidate detection" contract
in SCHEMA.md.

`models/ball_yolo26s.pt` is currently just the stock COCO-pretrained
yolo26s checkpoint, copied in as-is -- NOT fine-tuned on cricket balls yet.
Before assuming that needs fixing: its generic "sports ball" class already
tracks the ball reasonably well in this project's footage at low confidence
(~0.10-0.5). `training/train_yolo.py` fine-tunes this checkpoint further
once enough labeled data exists in data/frames + data/labels -- until then
this stock-weights fallback is a real, working detector, not a placeholder.

Restricted to `<video>_roi.json`'s polygon when one exists next to the
video (config.yaml: detection.use_roi) -- a full-frame YOLO pass with no
person exclusion of its own routinely outnumbered the much cleaner
lab_scale_detect.py candidates 10-30x on real clips, and was the dominant
remaining noise source once lab_scale_detect.py gained ROI + person-mask
filtering (see pipeline/run.py).
"""

from __future__ import annotations

from typing import Any

import cv2

from roi_utils import load_roi_mask

_COCO_SPORTS_BALL_CLASS = 32


def detect_candidates(video_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan `video_path` end to end and return a flat list of per-frame
    YOLO candidates (0 or more per frame)."""
    from ultralytics import YOLO  # lazy import: heavy dependency, only needed here

    cfg_det = config["detection"]
    model = YOLO(config["paths"]["yolo_weights"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    use_roi = cfg_det["use_roi"]
    roi_mask = None

    candidates: list[dict[str, Any]] = []
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if roi_mask is None and use_roi:
            roi_mask = load_roi_mask(video_path, frame.shape)

        results = model.predict(
            frame,
            verbose=False,
            conf=cfg_det["yolo_conf_threshold"],
            classes=[_COCO_SPORTS_BALL_CLASS],
        )
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if roi_mask is not None:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if roi_mask[cy, cx] == 0:
                    continue
            candidates.append(
                {
                    "frame": frame_idx,
                    "t": t,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(box.conf[0]),
                    "source": "yolo",
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
