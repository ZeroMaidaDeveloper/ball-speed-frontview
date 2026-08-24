"""Single-object-focused ball tracker.

Consumes per-frame candidate detections (see SCHEMA.md: "Per-frame candidate
detection") and produces a list of "deliveries" (see SCHEMA.md:
`detections.json` -> `deliveries[]`, minus the speed fields which are
computed by `pipeline/speed.py`).

Does not depend on the detector implementation at all -- it only consumes
dicts shaped like:

    {"frame": int, "t": float, "bbox": [x1, y1, x2, y2], "conf": float,
     "source": "yolo" | "classical"}

Pipeline:
  1. Build raw tracks frame-by-frame using a constant-velocity Kalman filter
     on (cx, cy, diam), gating candidates by predicted-position distance and
     size consistency, bridging short gaps (net occlusion) with
     Kalman-predicted frames.
  2. Discard tracks shorter than `min_track_len_frames`.
  3. Merge tracks that are separated by a short gap AND whose Kalman
     state at the end of the first track is consistent with the start of
     the second (this is the net-occlusion-fragmentation case). Gaps at or
     beyond `delivery_boundary_gap_s` are always treated as a genuine
     delivery boundary and never merged.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _bbox_center_diam(bbox: list[float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    diam = ((x2 - x1) + (y2 - y1)) / 2.0
    return cx, cy, diam


def _make_kalman(cx: float, cy: float, diam: float) -> cv2.KalmanFilter:
    """Constant-velocity Kalman filter, state = [cx, cy, diam, vcx, vcy, vdiam]."""
    kf = cv2.KalmanFilter(6, 3, 0, cv2.CV_64F)
    kf.transitionMatrix = np.eye(6, dtype=np.float64)  # dt filled in per-step
    kf.measurementMatrix = np.zeros((3, 6), dtype=np.float64)
    kf.measurementMatrix[0, 0] = 1.0
    kf.measurementMatrix[1, 1] = 1.0
    kf.measurementMatrix[2, 2] = 1.0
    # Process noise: allow velocity/size to change moderately (ball
    # decelerates and can jitter a bit frame to frame).
    kf.processNoiseCov = np.eye(6, dtype=np.float64) * 1.0
    kf.processNoiseCov[3:, 3:] *= 4.0
    # Measurement noise: detector bboxes are somewhat noisy.
    kf.measurementNoiseCov = np.eye(3, dtype=np.float64) * 4.0
    kf.errorCovPost = np.eye(6, dtype=np.float64) * 10.0
    kf.statePost = np.array([cx, cy, diam, 0.0, 0.0, 0.0], dtype=np.float64).reshape(6, 1)
    return kf


def _set_dt(kf: cv2.KalmanFilter, dt: float) -> None:
    dt = max(dt, 1e-3)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


class _ActiveTrack:
    """Internal mutable state for one in-progress raw track."""

    __slots__ = ("frames", "kf", "last_t", "missed_streak")

    def __init__(self, frame_rec: dict[str, Any]):
        cx, cy, diam = _bbox_center_diam(frame_rec["bbox"])
        self.kf = _make_kalman(cx, cy, diam)
        self.frames: list[dict[str, Any]] = [frame_rec]
        self.last_t = frame_rec["t"]
        self.missed_streak = 0

    def predict(self, t: float) -> np.ndarray:
        _set_dt(self.kf, t - self.last_t)
        return self.kf.predict()

    def update(self, cx: float, cy: float, diam: float) -> None:
        meas = np.array([[cx], [cy], [diam]], dtype=np.float64)
        self.kf.correct(meas)

    def state(self) -> np.ndarray:
        return self.kf.statePost


def _build_raw_tracks(candidates: list[dict[str, Any]], cfg_tracking: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Greedy single-active-track builder with Kalman gating + gap bridging.

    Returns a list of raw tracks (each a list of per-frame dicts, possibly
    containing "kalman_predicted" bridge frames already spliced in).
    """
    max_jump = cfg_tracking["max_center_jump_px"]
    max_bridge = cfg_tracking["max_missed_frames_bridge"]

    # Group candidates by frame index for easy per-frame lookup.
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for c in candidates:
        by_frame.setdefault(c["frame"], []).append(c)

    if not by_frame:
        return []

    # IMPORTANT: iterate over the full contiguous range of frame indices
    # (not just frames that happen to have a candidate) -- a net-occlusion
    # gap means zero candidates for a run of real video frames, and those
    # frames must still pass through the missed/bridge logic below. For
    # frames with no candidate we don't have a real timestamp, so
    # interpolate one from the surrounding known detection timestamps
    # (falls back to boundary-value extrapolation at the very ends, which
    # only affects predicted frames that get trimmed off anyway).
    known_frames = sorted(by_frame.keys())
    known_ts = [by_frame[f][0]["t"] for f in known_frames]
    full_range = range(known_frames[0], known_frames[-1] + 1)

    raw_tracks: list[list[dict[str, Any]]] = []
    active: _ActiveTrack | None = None

    for fidx in full_range:
        cands = by_frame.get(fidx, [])
        if fidx in by_frame:
            t = cands[0]["t"]
        else:
            t = float(np.interp(fidx, known_frames, known_ts))

        if active is None:
            if not cands:
                continue
            # Start a new track with the highest-confidence candidate.
            best = max(cands, key=lambda c: c["conf"])
            active = _ActiveTrack(_frame_record(best))
            continue

        pred = active.predict(t)
        pred_cx, pred_cy, pred_diam = float(pred[0]), float(pred[1]), float(pred[2])
        pred_diam = max(pred_diam, 1e-3)

        # Find best candidate within gating window.
        best_c = None
        best_score = None
        for c in cands:
            cx, cy, diam = _bbox_center_diam(c["bbox"])
            dist = math.hypot(cx - pred_cx, cy - pred_cy)
            if dist > max_jump:
                continue
            # Size consistency: relative diameter change must be plausible
            # (ball only grows/shrinks gradually frame-to-frame).
            size_ratio = diam / pred_diam if pred_diam > 0 else float("inf")
            if size_ratio < 0.4 or size_ratio > 2.5:
                continue
            # Lower score is better: normalized distance + size deviation,
            # penalized by lower confidence.
            score = (dist / max_jump) + abs(1.0 - size_ratio) - 0.2 * c["conf"]
            if best_score is None or score < best_score:
                best_score = score
                best_c = c

        if best_c is None:
            # YOLO hijack: nothing matched the active track's gate this
            # frame, but a YOLO/yolo_refined hit did show up and is
            # wildly different in size from what the active track
            # expects. That's almost certainly a different, more likely
            # real object -- e.g. the active track locked onto a small
            # noise blob early on, and a legitimate much-larger ball
            # detection later would otherwise just get silently discarded
            # (it fails this track's gate, and a track never yields the
            # "active" slot to let a rival start on its own). The size
            # mismatch itself is the real signal here, not raw confidence
            # -- a YOLO candidate already cleared its own detector's
            # confidence threshold to exist at all. Abandon the weak
            # active track for it rather than losing the signal.
            trusted = [c for c in cands if c["source"] in ("yolo", "yolo_refined")]
            if trusted:
                hijacker = max(trusted, key=lambda c: c["conf"])
                _, _, hijack_diam = _bbox_center_diam(hijacker["bbox"])
                size_mismatch = hijack_diam / pred_diam if pred_diam > 0 else float("inf")
                if size_mismatch > 3.0 or size_mismatch < 1.0 / 3.0:
                    raw_tracks.append(_trim_trailing_predictions(active.frames))
                    active = _ActiveTrack(_frame_record(hijacker))
                    continue

        if best_c is not None:
            cx, cy, diam = _bbox_center_diam(best_c["bbox"])
            active.update(cx, cy, diam)
            active.frames.append(_frame_record(best_c))
            active.last_t = t
            active.missed_streak = 0
        else:
            # No matching candidate this frame -- bridge with prediction if
            # still within the allowed missed-frame budget.
            active.missed_streak += 1
            if active.missed_streak <= max_bridge:
                active.frames.append(
                    {
                        "frame": fidx,
                        "t": t,
                        "bbox": _diam_to_bbox(pred_cx, pred_cy, pred_diam),
                        "cx": pred_cx,
                        "cy": pred_cy,
                        "diam_px": pred_diam,
                        "conf": 0.0,
                        "source": "kalman_predicted",
                    }
                )
                active.last_t = t
                # Kalman internal state already advanced via predict(); keep
                # statePost aligned with the prediction since there was no
                # correction this step.
                active.kf.statePost = pred.copy()
            else:
                # Gap too long -- close out this track (without the
                # trailing unmatched predictions).
                raw_tracks.append(_trim_trailing_predictions(active.frames))
                if cands:
                    # This frame itself has a candidate (just failed to
                    # gate against the now-stale prediction) -- use it to
                    # seed a brand new track immediately.
                    active = _ActiveTrack(_frame_record(best_or_first(cands)))
                else:
                    # No candidate at all this frame -- wait for the next
                    # frame that has one before starting a new track.
                    active = None
                continue

    if active is not None:
        raw_tracks.append(_trim_trailing_predictions(active.frames))

    return [t for t in raw_tracks if len(t) > 0]


def best_or_first(cands: list[dict[str, Any]]) -> dict[str, Any]:
    return max(cands, key=lambda c: c["conf"])


def _trim_trailing_predictions(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop kalman_predicted frames at the very end of a track -- they were
    a bridging attempt that never got confirmed by a real detection, so they
    don't belong to this track (the gap that follows is real)."""
    end = len(frames)
    while end > 0 and frames[end - 1]["source"] == "kalman_predicted":
        end -= 1
    return frames[:end]


def _diam_to_bbox(cx: float, cy: float, diam: float) -> list[float]:
    half = diam / 2.0
    return [cx - half, cy - half, cx + half, cy + half]


def _frame_record(c: dict[str, Any]) -> dict[str, Any]:
    cx, cy, diam = _bbox_center_diam(c["bbox"])
    return {
        "frame": c["frame"],
        "t": c["t"],
        "bbox": list(c["bbox"]),
        "cx": cx,
        "cy": cy,
        "diam_px": diam,
        "conf": c["conf"],
        "source": c["source"],
    }


def _track_velocity_at_end(frames: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Rough (vcx, vcy, vdiam) per second estimated from the last two real
    frames of a track (falls back to the last two available frames)."""
    if len(frames) < 2:
        return 0.0, 0.0, 0.0
    a, b = frames[-2], frames[-1]
    dt = b["t"] - a["t"]
    if dt <= 1e-6:
        return 0.0, 0.0, 0.0
    return (
        (b["cx"] - a["cx"]) / dt,
        (b["cy"] - a["cy"]) / dt,
        (b["diam_px"] - a["diam_px"]) / dt,
    )


def _fit_parabola(
    frames: list[dict[str, Any]], min_real_frames: int
) -> tuple[np.poly1d, np.poly1d] | None:
    """Fit cx(t) as a line and cy(t) as a parabola to a track's REAL
    (non-predicted) frames, with iterative 2.5-sigma outlier rejection --
    ported from the user's own pipeline_common.py: fit_parabola(). A real
    delivery isn't constant-velocity (perspective makes it accelerate
    toward the camera), so this captures the trajectory shape far better
    than extrapolating from just the last 2 points once the gap being
    bridged is more than a couple of frames wide. Returns None if there
    aren't enough real points to trust a fit.
    """
    real = [f for f in frames if f["source"] != "kalman_predicted"]
    if len(real) < min_real_frames:
        return None

    ts = np.array([f["t"] for f in real], dtype=np.float64)
    xs = np.array([f["cx"] for f in real], dtype=np.float64)
    ys = np.array([f["cy"] for f in real], dtype=np.float64)
    mask = np.ones(len(ts), dtype=bool)

    x_coef = np.polyfit(ts, xs, 1)
    y_coef = np.polyfit(ts, ys, 2)
    for _ in range(3):
        if mask.sum() < min_real_frames:
            return None
        x_coef = np.polyfit(ts[mask], xs[mask], 1)
        y_coef = np.polyfit(ts[mask], ys[mask], 2)
        resid = np.hypot(xs - np.polyval(x_coef, ts), ys - np.polyval(y_coef, ts))
        std = resid[mask].std()
        if std < 1e-6:
            break
        new_mask = resid <= 2.5 * std
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    return np.poly1d(x_coef), np.poly1d(y_coef)


def _passes_velocity_merge_check(
    prev: list[dict[str, Any]], nxt: list[dict[str, Any]], gap_s: float, cfg_tracking: dict[str, Any]
) -> bool:
    """Constant-velocity extrapolation from the last 2 real-ish points --
    good for short gaps where the ball's motion is locally linear."""
    vcx, vcy, vdiam = _track_velocity_at_end(prev)
    last = prev[-1]
    pred_cx = last["cx"] + vcx * gap_s
    pred_cy = last["cy"] + vcy * gap_s
    pred_diam = max(last["diam_px"] + vdiam * gap_s, 1e-3)

    start = nxt[0]
    dist = math.hypot(start["cx"] - pred_cx, start["cy"] - pred_cy)
    if dist > cfg_tracking["max_center_jump_px"]:
        return False

    size_ratio = start["diam_px"] / pred_diam
    if size_ratio < 0.4 or size_ratio > 2.5:
        return False

    # The ball is approaching the camera, so diameter should not be
    # shrinking across the merge boundary (a real new delivery restarting
    # small further away is the thing delivery_boundary_gap_s guards
    # against; here we just sanity-check monotonic growth continuity).
    if start["diam_px"] + 1e-6 < last["diam_px"] * 0.5:
        return False

    return True


def _passes_parabola_merge_check(
    prev: list[dict[str, Any]], nxt: list[dict[str, Any]], cfg_tracking: dict[str, Any]
) -> bool:
    """Fit a parabola to `prev`'s trajectory and project it forward to
    `nxt`'s start time -- tolerates the large, ACCELERATING position jumps
    a real delivery makes as it nears the camera, which the constant-
    velocity check correctly refuses to bridge (see config.yaml:
    tracking.parabola_*)."""
    fit = _fit_parabola(prev, cfg_tracking["parabola_min_real_frames"])
    if fit is None:
        return False
    x_poly, y_poly = fit

    start = nxt[0]
    pred_cx = float(x_poly(start["t"]))
    pred_cy = float(y_poly(start["t"]))
    dist = math.hypot(start["cx"] - pred_cx, start["cy"] - pred_cy)
    if dist > cfg_tracking["parabola_max_jump_px"]:
        return False

    last = prev[-1]
    if start["diam_px"] + 1e-6 < last["diam_px"] * 0.5:
        return False

    return True


def _should_merge(
    prev: list[dict[str, Any]],
    nxt: list[dict[str, Any]],
    cfg_tracking: dict[str, Any],
) -> bool:
    gap_s = nxt[0]["t"] - prev[-1]["t"]
    if gap_s <= 0 or gap_s >= cfg_tracking["delivery_boundary_gap_s"]:
        return False

    if gap_s <= cfg_tracking["delivery_gap_merge_s"] and _passes_velocity_merge_check(
        prev, nxt, gap_s, cfg_tracking
    ):
        return True

    if gap_s <= cfg_tracking["parabola_gap_merge_s"] and _passes_parabola_merge_check(prev, nxt, cfg_tracking):
        return True

    return False


def _bridge_gap_frames(prev_last: dict[str, Any], next_first: dict[str, Any]) -> list[dict[str, Any]]:
    """Synthesize kalman_predicted frames for the missing frame numbers
    between two merged tracks, linearly interpolating cx/cy/diam/t between
    the two known real endpoints (this is a plain interpolation rather than
    a forward-only Kalman prediction because, at merge time, we already know
    both boundary detections -- interpolating between them is strictly more
    accurate for display purposes than only extrapolating from the first
    track). Purely for the delivery's `frames` array continuity/visualization
    (dashed rendering per SCHEMA.md); not used by the speed calculation.
    """
    f0, f1 = prev_last["frame"], next_first["frame"]
    n_missing = f1 - f0 - 1
    if n_missing <= 0:
        return []
    out = []
    for k in range(1, n_missing + 1):
        frac = k / (n_missing + 1)
        cx = prev_last["cx"] + (next_first["cx"] - prev_last["cx"]) * frac
        cy = prev_last["cy"] + (next_first["cy"] - prev_last["cy"]) * frac
        diam = prev_last["diam_px"] + (next_first["diam_px"] - prev_last["diam_px"]) * frac
        t = prev_last["t"] + (next_first["t"] - prev_last["t"]) * frac
        out.append(
            {
                "frame": f0 + k,
                "t": t,
                "bbox": _diam_to_bbox(cx, cy, diam),
                "cx": cx,
                "cy": cy,
                "diam_px": diam,
                "conf": 0.0,
                "source": "kalman_predicted",
            }
        )
    return out


def _merge_tracks(tracks: list[list[dict[str, Any]]], cfg_tracking: dict[str, Any]) -> list[list[dict[str, Any]]]:
    if not tracks:
        return []
    merged: list[list[dict[str, Any]]] = [list(tracks[0])]
    for track in tracks[1:]:
        if _should_merge(merged[-1], track, cfg_tracking):
            merged[-1].extend(_bridge_gap_frames(merged[-1][-1], track[0]))
            merged[-1].extend(track)
        else:
            merged.append(list(track))
    return merged


def _thirds_split(real: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    k = max(1, len(real) // 3)
    return real[:k], real[-k:]


def _diam_growth_ratio(track: list[dict[str, Any]]) -> float:
    """(median diam_px of the last third of REAL frames) / (median of the
    first third) -- see config.yaml: tracking.min_diam_growth_ratio /
    max_diam_shrink_ratio for why this (in either direction, depending on
    camera placement) discriminates a real delivery from a static/
    jittering noise blob."""
    real = [f for f in track if f["source"] != "kalman_predicted"]
    if len(real) < 2:
        return 1.0
    first, last = _thirds_split(real)
    start_diam = _median([f["diam_px"] for f in first])
    end_diam = _median([f["diam_px"] for f in last])
    if start_diam <= 1e-6:
        return float("inf")
    return end_diam / start_diam



def segment_deliveries(candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a flat list of per-frame candidate detections into a list of
    delivery dicts (see SCHEMA.md `deliveries[]`, without speed fields).

    `candidates` should be sorted by frame (this function sorts defensively
    if not). Each output delivery dict has: start_frame, end_frame, start_t,
    end_t, frames (list of per-frame dicts with frame/t/bbox/cx/cy/diam_px/
    conf/source).
    """
    cfg_tracking = config["tracking"]

    candidates = sorted(candidates, key=lambda c: (c["frame"], -c["conf"]))

    raw_tracks = _build_raw_tracks(candidates, cfg_tracking)

    min_len = cfg_tracking["min_track_len_frames"]
    raw_tracks = [t for t in raw_tracks if len(t) >= min_len]

    merged_tracks = _merge_tracks(raw_tracks, cfg_tracking)
    # Re-filter after merge in case (shouldn't shrink, but be safe).
    merged_tracks = [t for t in merged_tracks if len(t) >= min_len]

    # min_diam_growth_ratio used to hard-reject a track here instead of
    # flagging it. Changed deliberately: a delivery captured only
    # partway through its flight (e.g. a short clip that starts mid-flight,
    # or a detector that only picks the ball up once it's already close)
    # can show real, coherent motion without much size growth, and a hard
    # reject hid that track entirely -- there was no way to see it even
    # existed. Now every merged track becomes a delivery; low_diam_growth
    # in quality_flags (pipeline/speed.py) is what downgrades confidence.
    deliveries: list[dict[str, Any]] = []
    for track in merged_tracks:
        deliveries.append(
            {
                "start_frame": track[0]["frame"],
                "end_frame": track[-1]["frame"],
                "start_t": track[0]["t"],
                "end_t": track[-1]["t"],
                "frames": track,
                "diam_growth_ratio": _diam_growth_ratio(track),
            }
        )
    return deliveries
