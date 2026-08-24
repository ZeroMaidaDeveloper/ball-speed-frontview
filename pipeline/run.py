"""End-to-end pipeline driver: detect -> track -> speed -> runs/<stem>/detections.json.

Usage:
    python3 pipeline/run.py <video_path> [config_path]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml

from calibration import focal_length_px, load_calibration
from speed import compute_delivery_speed
from track import segment_deliveries
from zoom_track_detect import detect_candidates as zoom_track_detect_candidates

# classical_detect.py (soft color nudge, no ROI/person-mask),
# heatmap_refine.py (yolo_refined: heatmap-guided ROI zoom-in), and a
# plain full-frame yolo_detect.py scan all fed into this fused pipeline
# at earlier points in this project. Discarded:
#   - classical_detect.py's candidates were consistently dominated by
#     background clutter without a per-video ROI or person exclusion;
#   - heatmap_refine.py never finished a full run on a long, unrestricted
#     video in under an hour even after optimization;
#   - a plain full-frame yolo_detect.py scan, even ROI-restricted, still
#     routinely outnumbered lab_scale_detect.py's much cleaner
#     motion+color candidates 10-30x on real clips -- most of it noise.
# zoom_track_detect.py (this module) instead watches lab_scale_detect.py's
# candidates for a fast-moving one (the hallmark of a real ball, vs.
# noise/players), then tracks it via prediction-gated LAB continuation,
# falling back to a YOLO crop-zoom only as a rescue -- YOLO's "sports
# ball" class alone often misses a motion-blurred fast ball entirely,
# zoomed in or not, while LAB's motion+color signal keeps tracking it
# fine. All three discarded modules are still kept for the viewer's
# diagnostic source filter, just no longer part of the "real" fused
# pipeline.


def _detect_or_reuse_cached(
    mode: str, detect_fn, video_path: str, config: dict[str, Any], cache_dir: Path | None
) -> list[dict[str, Any]]:
    """pipeline/run_diagnostics.py may have already computed the exact same
    detect_fn(video_path, config) call for this video (raw per-detector
    dumps for the viewer's source filter) -- reuse it instead of re-running
    a slow detection pass over every frame a second time."""
    cache_path = cache_dir / f"candidates_{mode}.json" if cache_dir else None
    if cache_path and cache_path.is_file():
        print(f"[run] reusing cached {mode} candidates from {cache_path}", flush=True)
        with cache_path.open() as f:
            candidates = json.load(f)["candidates"]
    else:
        print(f"[run] detecting candidates ({mode}) ...", flush=True)
        candidates = detect_fn(video_path, config)
    print(f"[run] {mode} candidates: {len(candidates)}", flush=True)
    return candidates


def run(video_path: str, config: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config["video"]["fallback_fps"]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps if fps else 0.0
    cap.release()

    candidates = _detect_or_reuse_cached("zoom_track", zoom_track_detect_candidates, video_path, config, cache_dir)

    calib = load_calibration(video_path)
    f_px = focal_length_px(calib, config) if calib else None
    print(
        f"[run] calibration: {'f_px=' + format(f_px, '.0f') if f_px else 'none (using flight_distance_m fallback)'}",
        flush=True,
    )

    print(f"[run] segmenting deliveries from {len(candidates)} total candidates ...", flush=True)
    deliveries = segment_deliveries(candidates, config)
    print(f"[run] raw deliveries: {len(deliveries)}", flush=True)

    out_deliveries = []
    for i, delivery in enumerate(deliveries):
        d = compute_delivery_speed(delivery, config, f_px)
        d["id"] = i
        out_deliveries.append(d)

    return {
        "video": Path(video_path).name,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": duration_s,
        "deliveries": out_deliveries,
    }


def main() -> None:
    video_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parent.parent / "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # config.yaml paths are relative to the project root (config.yaml's own
    # directory), not to whatever the caller's cwd happens to be -- resolve
    # them to absolute paths up front so every module downstream (which
    # just reads config["paths"][...] directly) gets something unambiguous.
    project_root = Path(config_path).resolve().parent
    config["paths"] = {k: str(project_root / v) for k, v in config["paths"].items()}

    stem = Path(video_path).stem
    out_dir = Path(config["paths"]["runs_dir"]) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run(video_path, config, cache_dir=out_dir)

    out_path = out_dir / "detections.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=1)
    print(f"[run] wrote {out_path} ({len(result['deliveries'])} deliveries)")


if __name__ == "__main__":
    main()
