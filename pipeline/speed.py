"""Speed estimation for a single delivery produced by `pipeline/track.py`.

PRIMARY estimate: time-of-flight over the known crease-to-crease distance
(`config["geometry"]["flight_distance_m"]`), using the delivery's real
start/end timestamps (container timestamps, not frame_index/fps).

SECONDARY (optional, approximate): a size-based instantaneous speed curve
for visualization, back-solving a rough focal length from the apparent
ball diameter at the two ends of the delivery.
"""

from __future__ import annotations

from typing import Any


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


# Frames at/below this confidence are treated as "weak" for the purposes of
# the sub-frame start/end refinement below. Chosen as a low bar (well under
# typical yolo_min_conf_trust=0.45) so it only kicks in for genuinely marginal
# edge detections, not ordinary lower-confidence-but-fine ones.
_WEAK_CONF_THRESHOLD = 0.30

# Ratio of real (non predicted) frames to total frames below which a
# delivery is flagged "short_track" -- i.e. it leaned heavily on
# Kalman-bridged frames rather than actual detections.
_SHORT_TRACK_REAL_FRAME_RATIO = 0.6


def _frame_dts(frames: list[dict[str, Any]]) -> list[float]:
    return [b["t"] - a["t"] for a, b in zip(frames, frames[1:]) if b["t"] > a["t"]]


def _refine_endpoints(frames: list[dict[str, Any]]) -> tuple[float, float]:
    """Sub-frame refinement of the delivery's start/end timestamps.

    By construction (see track.py: tracks always start/end on a real
    detection, trailing/leading Kalman-predicted frames are trimmed), the
    first and last frame here are essentially always real detections -- but
    we still defensively check `source` in case that invariant ever changes.

    Judgment call: if the boundary frame is a weak/low-confidence detection
    (or, defensively, a predicted one), we nudge that endpoint by half the
    track's median inter-frame interval, outward (start earlier / end
    later). Rationale: a weak edge detection is more likely to be a slightly
    late "first sighting" / slightly early "last sighting" of the ball than
    a precise boundary, so true flight duration is more likely
    underestimated than overestimated at a noisy edge; nudging outward by
    half a frame is a small, conservative correction (a few percent of a
    ~0.3-1s flight) rather than a large speculative one.
    """
    start_t = frames[0]["t"]
    end_t = frames[-1]["t"]

    dts = _frame_dts(frames)
    half_dt = _median(dts) / 2.0 if dts else 0.0
    if half_dt <= 0:
        return start_t, end_t

    first, last = frames[0], frames[-1]
    if first["source"] == "kalman_predicted" or first["conf"] <= _WEAK_CONF_THRESHOLD:
        start_t -= half_dt
    if last["source"] == "kalman_predicted" or last["conf"] <= _WEAK_CONF_THRESHOLD:
        end_t += half_dt

    return start_t, end_t


def _quality_flags(delivery: dict[str, Any], config: dict[str, Any], speed_kmh: float) -> list[str]:
    frames = delivery["frames"]
    total = len(frames)
    real = sum(1 for f in frames if f["source"] != "kalman_predicted")

    flags: list[str] = []

    # A gap either shows up as explicit kalman_predicted frames, or (for a
    # merged-across-occlusion delivery) as a jump in consecutive frame
    # numbers with no entry at all -- check both defensively.
    has_predicted = real < total
    has_frame_number_gap = any(b["frame"] - a["frame"] > 1 for a, b in zip(frames, frames[1:]))
    if has_predicted or has_frame_number_gap:
        flags.append("occlusion_gap")

    if total > 0 and (real / total) < _SHORT_TRACK_REAL_FRAME_RATIO:
        flags.append("short_track")

    # A separate, coarser signal than short_track: the delivery is simply
    # made of very few samples overall (close to min_track_len_frames),
    # regardless of real/predicted mix.
    min_len = config["tracking"]["min_track_len_frames"]
    if total < min_len * 2:
        flags.append("few_frames")

    speed_cfg = config["speed"]
    if not (speed_cfg["min_plausible_kmh"] <= speed_kmh <= speed_cfg["max_plausible_kmh"]):
        flags.append("speed_out_of_range")

    # See track.py: segment_deliveries -- this used to be a hard reject
    # instead of a flag, which hid tracks that show real coherent motion
    # without much apparent-size change (e.g. a delivery only captured
    # partway through its flight). Now it just downgrades confidence.
    #
    # A real delivery's apparent diameter can either grow (camera at the
    # batting end, ball approaching) or shrink (camera at the bowling end,
    # ball receding -- confirmed camera placement for this project's
    # delivery_*.mp4 nets footage, see config.yaml: tracking.
    # max_diam_shrink_ratio). Accept either as evidence of a genuine
    # flight; only flag a track whose size barely changed either way.
    ratio = delivery.get("diam_growth_ratio", 1.0)
    min_growth = config["tracking"]["min_diam_growth_ratio"]
    max_shrink = config["tracking"]["max_diam_shrink_ratio"]
    if ratio < min_growth and ratio > max_shrink:
        flags.append("low_diam_change")

    return flags


def _confidence_from_flags(flags: list[str]) -> str:
    if "speed_out_of_range" in flags:
        return "low"
    if len(flags) >= 2:
        return "low"
    if len(flags) == 1:
        return "medium"
    return "high"


def _size_speed_curve(frames: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, float]]:
    """Secondary, approximate size-based instantaneous speed curve.

    Back-solves one rough focal length f_px from two (pixel_diameter,
    assumed_distance) pairs -- the delivery's start (assumed at
    flight_distance_m) and end (assumed at ~1.5m, near the camera) -- then
    derives z(t) = f_px * ball_diameter_m / diam_px(t) and speed = -dz/dt.

    For visualization only; returns [] on any degenerate input rather than
    raising, per SCHEMA.md (an empty size_speed_curve is fine, viewer hides
    the chart).
    """
    if len(frames) < 2:
        return []

    ball_diam_m = config["ball"]["diameter_m"]
    flight_distance_m = config["geometry"]["flight_distance_m"]
    near_distance_m = 1.5  # assumed distance at the last frame, near the camera

    diam_start = frames[0]["diam_px"]
    diam_end = frames[-1]["diam_px"]
    eps = 1e-3
    if diam_start <= eps or diam_end <= eps:
        return []

    f_px_start = diam_start * flight_distance_m / ball_diam_m
    f_px_end = diam_end * near_distance_m / ball_diam_m
    f_px = (f_px_start + f_px_end) / 2.0
    if f_px <= 0:
        return []

    # z(t) for every frame, skipping degenerate (near-zero) diameters.
    zs: list[tuple[float, float]] = []  # (t, z)
    for f in frames:
        d = f["diam_px"]
        if d <= eps:
            continue
        zs.append((f["t"], f_px * ball_diam_m / d))

    raw_points: list[tuple[float, float]] = []  # (t_mid, speed_kmh)
    for (t0, z0), (t1, z1) in zip(zs, zs[1:]):
        dt = t1 - t0
        if dt <= 1e-6:
            continue
        speed_mps = -(z1 - z0) / dt
        raw_points.append(((t0 + t1) / 2.0, speed_mps * 3.6))

    if not raw_points:
        return []

    window = max(int(config["speed"]["size_curve_smoothing_window"]), 1)
    speeds = [p[1] for p in raw_points]
    smoothed: list[dict[str, float]] = []
    n = len(speeds)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        avg = sum(speeds[lo:hi]) / (hi - lo)
        smoothed.append({"t": raw_points[i][0], "speed_kmh": avg})

    return smoothed


def compute_delivery_speed(delivery: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Augment a delivery dict (from `track.segment_deliveries`) with speed
    fields, per SCHEMA.md `deliveries[]`: `speed_kmh`, `speed_confidence`,
    `quality_flags`, `size_speed_curve`.

    Returns a new dict (does not mutate the input's top level, though the
    `frames` list is shared by reference for efficiency).
    """
    frames = delivery["frames"]

    start_t, end_t = _refine_endpoints(frames)
    duration_s = end_t - start_t

    flight_distance_m = config["geometry"]["flight_distance_m"]
    if duration_s <= 0:
        # Degenerate (shouldn't happen for a real track with >=2 distinct
        # timestamps) -- avoid a divide-by-zero / negative speed.
        speed_kmh = 0.0
    else:
        speed_kmh = 3.6 * flight_distance_m / duration_s

    flags = _quality_flags(delivery, config, speed_kmh)
    confidence = _confidence_from_flags(flags)
    curve = _size_speed_curve(frames, config)

    out = dict(delivery)
    out["speed_kmh"] = speed_kmh
    out["speed_confidence"] = confidence
    out["quality_flags"] = flags
    out["size_speed_curve"] = curve
    return out
