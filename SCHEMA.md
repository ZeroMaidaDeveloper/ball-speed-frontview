# Shared interfaces for the ball-speed pipeline

All modules must conform to these shapes so pieces built independently
(detector, tracker/speed, viewer) integrate without changes.

## `runs/<video_stem>/detections.json` (final pipeline output, one per source video)

```json
{
  "video": "1000313863.mp4",
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "duration_s": 39.3,
  "deliveries": [
    {
      "id": 0,
      "start_frame": 120,
      "end_frame": 145,
      "start_t": 4.00,
      "end_t": 4.83,
      "speed_kmh": 118.4,
      "speed_confidence": "high",
      "speed_method": "calibrated_size",
      "speed_fit_r2": 0.52,
      "speed_fit_frames_used": 34,
      "quality_flags": ["occlusion_gap"],
      "frames": [
        {
          "frame": 120,
          "t": 4.00,
          "bbox": [940.0, 210.0, 958.0, 228.0],
          "cx": 949.0,
          "cy": 219.0,
          "diam_px": 18.0,
          "conf": 0.81,
          "source": "yolo"
        }
      ],
      "size_speed_curve": [
        { "t": 4.02, "speed_kmh": 121.0 }
      ]
    }
  ]
}
```

Notes:
- `t` values are seconds from the START of the video, taken from the
  actual container timestamp (`cv2.CAP_PROP_POS_MSEC / 1000`), NOT
  `frame_index / fallback_fps` — real phone footage can have variable
  frame timing.
- `bbox` is `[x1, y1, x2, y2]` in source-video pixel coordinates (no
  resizing baked in).
- `source` on a per-frame detection is one of `"yolo"`, `"yolo_refined"`
  (YOLO run on a heatmap-guided ROI zoom-in crop rather than the full
  frame -- see `pipeline/heatmap_refine.py`, for the small/far-away ball a
  full-frame pass misses), `"classical"`, `"lab_scale"` (CLAHE-enhanced
  motion + a hard LAB color gate -- see `pipeline/lab_scale_detect.py`, a
  deliberately different tradeoff from `classical`'s soft color nudge),
  `"zoom_track"` (a LAB-scale hit gated against a predicted position, or
  the rarer YOLO-crop rescue, while a fast-moving object is being actively
  tracked -- see `pipeline/zoom_track_detect.py`; this is what
  `pipeline/run.py`'s fused pipeline actually runs on now),
  `"motion_rescue"` (a three-frame-differencing hit near the predicted
  position, tried when the LAB gate misses and before the YOLO-crop
  rescue -- see `pipeline/zoom_track_detect.py`: `_motion_rescue()`; this
  recovers real frames where the ball has desaturated/darkened past LAB's
  hard color gate but is still a clean, compact, consistently-sized
  moving blob),
  `"kalman_predicted"` (bridged gap) — the viewer should render these
  distinctly (e.g. predicted frames dashed/lower-opacity).
- `speed_confidence` is `"high" | "medium" | "low"` — driven by presence
  of `quality_flags` and whether `speed_kmh` falls inside
  `config.yaml: speed.min_plausible_kmh/max_plausible_kmh`.
- `speed_method` is `"calibrated_size"` (preferred -- fits the ball's
  real-world distance-from-camera, derived every frame from its own
  apparent diameter via a per-video calibrated focal length, over time;
  see `pipeline/calibration.py` and `pipeline/speed.py`:
  `_trim_and_fit_distance`) or `"flight_distance_fallback"` (used only
  when the video has no `<video_stem>_calib.json` sidecar -- divides
  `config.yaml: geometry.flight_distance_m` by the tracked duration, which
  is only accurate if the track happens to span the whole release-to-
  arrival flight; flagged via `quality_flags: "uncalibrated_speed_estimate"`).
- `speed_fit_r2` / `speed_fit_frames_used` are present only when
  `speed_method` is `"calibrated_size"` -- the linear fit's R² (see
  config.yaml: calibration.min_fit_r2 for what counts as a poor fit,
  flagged `"noisy_size_fit"`) and how many of the track's real frames
  survived trimming to the physically-consistent monotonic prefix (see
  `quality_flags: "trimmed_non_monotonic_tail"` below).
- `quality_flags` is a list of short strings, e.g. `"occlusion_gap"`,
  `"short_track"`, `"speed_out_of_range"`, `"few_frames"`,
  `"low_diam_change"` (apparent size neither grew nor shrank much over the
  track -- e.g. a delivery only captured partway through its flight; this
  used to silently drop the delivery entirely, now it's reported with
  lower confidence instead -- see `config.yaml:
  tracking.min_diam_growth_ratio/max_diam_shrink_ratio`). Growth OR shrink
  both count as evidence of a real flight -- which one to expect depends
  on camera placement (see that config entry).
  Speed-fit-specific flags: `"uncalibrated_speed_estimate"` (see
  `speed_method` above), `"few_calibrated_frames"` (fewer than
  `config.yaml: calibration.min_frames_for_fit` real frames survived
  trimming), `"trimmed_non_monotonic_tail"` (part of the track was
  excluded from the speed fit -- it's still present in `frames` for
  display, just not trusted for `speed_kmh`), `"noisy_size_fit"` (the fit
  itself has a low R², see `speed_fit_r2`).
- `diam_growth_ratio` (float) is attached by `track.py` alongside `frames`
  -- (median diam_px of the track's last third of real frames) / (median
  of its first third); this is what `low_diam_change` is driven by. A
  ratio > 1 means the ball grew (camera near the batting end, ball
  approaching); < 1 means it shrank (camera near the bowling end, ball
  receding -- true for this project's delivery_*.mp4 nets footage).
- `size_speed_curve` is OPTIONAL/secondary (approximate, size-based
  instantaneous speed) — omit entirely if not computed rather than
  sending an empty misleading array... actually an empty array is fine
  too, viewer should just hide the chart when it's empty.

## Per-frame candidate detection (internal contract between detector and tracker)

A single detection candidate for one frame, produced by
`pipeline/classical_detect.py`, `pipeline/yolo_detect.py`,
`pipeline/heatmap_refine.py`, `pipeline/lab_scale_detect.py`, and
`pipeline/zoom_track_detect.py`, consumed by `pipeline/track.py`:

```python
{
    "frame": int,          # frame index, 0-based
    "t": float,            # seconds, from CAP_PROP_POS_MSEC/1000
    "bbox": [x1, y1, x2, y2],  # floats, pixel coords
    "conf": float,         # 0..1
    "source": "yolo" | "yolo_refined" | "classical" | "lab_scale" | "zoom_track" | "motion_rescue",
}
```

A frame may have zero, one, or multiple candidates (tracker resolves to
at most one active ball track at a time; extra candidates are noise to be
gated out by size/motion continuity).

## Config

All tunable constants live in `config.yaml` at the project root, loaded via
`yaml.safe_load`. Do not hardcode thresholds that already have a
`config.yaml` entry — read them from there so tuning doesn't require code
edits.

## Directory conventions

- `pipeline/*.py` — importable modules, no video-path assumptions baked
  in; every function takes paths/config as arguments.
- `runs/<video_stem>/detections.json` — final output consumed by the
  viewer, `<video_stem>` = filename without extension, e.g. `1000313863`.
- The videos directory (`config.yaml: paths.videos_dir`) may contain
  duplicate files with `(1)` suffixes (same content, re-uploaded) —
  de-duplicate by comparing filesize before processing so the same
  session isn't processed twice.
- `<video_stem>_calib.json` sitting next to a video's REAL (symlink-
  resolved) path -- same lookup convention as `<video_stem>_roi.json`
  (`pipeline/roi_utils.py`) -- gives that video's near/far stump pixel
  heights for `pipeline/calibration.py`'s focal-length derivation, used by
  `pipeline/speed.py`'s primary speed estimate. See that module's
  docstring for the shape. Optional: a video without one falls back to
  `speed_method: "flight_distance_fallback"`.
- This pipeline handles both a camera near the batting end (ball
  approaches, grows in apparent size) and one near the bowling end (ball
  recedes, shrinks) -- see config.yaml: tracking.min_diam_growth_ratio /
  max_diam_shrink_ratio. The delivery_4/7/19/21.mp4 nets footage is
  confirmed to be the latter (visual inspection: the bowler fills the
  frame at release, the striker is small and distant at the far end).
  Camera position is NOT auto-detected across videos -- both thresholds
  are applied to every track, so a video that's neither (e.g. a side-on
  angle) will still get flagged low-confidence rather than silently
  misread as one or the other.
- `pipeline/run.py`'s fused pipeline (the one that writes
  `detections.json`) uses `zoom_track_detect.py` only -- it internally
  runs `lab_scale_detect.py` every frame and adds `zoom_track`-sourced
  candidates while a fast-moving object is confirmed and being tracked.
  `classical_detect.py`, `heatmap_refine.py`, and a plain full-frame
  `yolo_detect.py` scan are kept for the viewer's diagnostic source filter
  but are no longer part of the fused pipeline -- all three were
  consistently outperformed once `lab_scale_detect.py` gained ROI
  restriction + person-mask exclusion and `zoom_track_detect.py` added
  fast-motion-triggered prediction-gated tracking on top of it.
