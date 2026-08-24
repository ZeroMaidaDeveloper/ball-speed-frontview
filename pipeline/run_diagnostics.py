"""Generate raw per-detector candidate dumps for the viewer's source filter.

Unlike pipeline/run.py (which fuses detectors -> tracks -> deliveries),
this writes each detector's UNFILTERED-BY-TRACKING candidate stream
straight to runs/<stem>/candidates_<mode>.json, so the viewer can overlay
"what did MOG2 alone see" / "what did LAB alone see" / "what did YOLO
alone see" independently for visual comparison.

Usage:
    python3 pipeline/run_diagnostics.py <video_path> [config_path]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml

from classical_detect import detect_candidates as classical_detect_candidates
from heatmap_refine import detect_candidates as heatmap_refine_candidates
from lab_scale_detect import detect_candidates as lab_scale_detect_candidates
from yolo_detect import detect_candidates as yolo_detect_candidates
from zoom_track_detect import detect_candidates as zoom_track_detect_candidates


def _video_meta(video_path: str, config: dict[str, Any]) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config["video"]["fallback_fps"]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "video": Path(video_path).name,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": frame_count / fps if fps else 0.0,
    }


def _write(out_dir: Path, mode: str, meta: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    out_path = out_dir / f"candidates_{mode}.json"
    with out_path.open("w") as f:
        json.dump({**meta, "mode": mode, "candidates": candidates}, f)
    print(f"[diagnostics] wrote {out_path} ({len(candidates)} candidates)")


def main() -> None:
    video_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parent.parent / "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = Path(config_path).resolve().parent
    config["paths"] = {k: str(project_root / v) for k, v in config["paths"].items()}

    meta = _video_meta(video_path, config)
    stem = Path(video_path).stem
    out_dir = Path(config["paths"]["runs_dir"]) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[diagnostics] motion_only (MOG2, no color) ...", flush=True)
    _write(out_dir, "motion_only", meta, classical_detect_candidates(video_path, config, mode="motion_only"))

    print("[diagnostics] color_only (LAB range, no motion) ...", flush=True)
    _write(out_dir, "color_only", meta, classical_detect_candidates(video_path, config, mode="color_only"))

    print("[diagnostics] lab_scale (CLAHE motion + hard LAB color gate) ...", flush=True)
    _write(out_dir, "lab_scale", meta, lab_scale_detect_candidates(video_path, config))

    print("[diagnostics] yolo ...", flush=True)
    _write(out_dir, "yolo", meta, yolo_detect_candidates(video_path, config))

    print("[diagnostics] yolo_refined (heatmap-guided ROI zoom-in) ...", flush=True)
    _write(out_dir, "yolo_refined", meta, heatmap_refine_candidates(video_path, config))

    print("[diagnostics] zoom_track (fast-LAB-triggered prediction-gated tracking) ...", flush=True)
    _write(out_dir, "zoom_track", meta, zoom_track_detect_candidates(video_path, config))

    print("[diagnostics] done")


if __name__ == "__main__":
    main()
