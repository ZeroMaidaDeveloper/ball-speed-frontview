"""Speed estimation for a single delivery produced by `pipeline/track.py`.

PRIMARY estimate, when `<video_stem>_calib.json` exists (see
pipeline/calibration.py): fit the ball's real-world distance-from-camera
z(t) -- derived every frame from its own apparent diameter via the
pinhole model, using that video's calibrated focal length -- over the
track's real frames, trimmed to the physically-consistent monotonic-
recession prefix (see `_trim_and_fit_distance`). This works correctly for
a PARTIAL track (doesn't need to assume the tracked segment spans the
whole release-to-arrival flight), unlike the fallback below.

SECOND, weaker option, when no `_calib.json` exists but a
`<video_stem>_wickets_calib.json` does (see pipeline/calibration.py:
pixels_per_meter_from_wickets): the tracked ball's total pixel path
length converted straight to metres via a single fixed pixels-per-meter
scale, divided by the elapsed time (`_planar_pixel_speed`). Unlike the
primary estimate this has no depth/perspective model at all -- it's only
strictly accurate for motion at the calibration plane's own distance from
the camera -- so it's used only as a fallback when the video has no
stump-height-pair calibration to derive a real focal length from (e.g.
no far-end wicket is set up in frame at all).

FALLBACK, when no calibration file exists for the video: time-of-flight
over the assumed crease-to-crease distance
(`config["geometry"]["flight_distance_m"]`), using the delivery's real
start/end timestamps (container timestamps, not frame_index/fps). Known
to be inaccurate whenever the tracked segment is a fragment of the real
flight rather than the whole thing -- see config.yaml: geometry.
flight_distance_m.

SECONDARY (optional, approximate): a size-based instantaneous speed curve
for visualization, back-solving a rough focal length from the apparent
ball diameter at the two ends of the delivery.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from calibration import focal_length_px
from track import _diam_growth_ratio


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


def _trim_and_fit_distance(
    ts: list[float], zs: list[float], cfg_calib: dict[str, Any]
) -> dict[str, Any] | None:
    """Fit real-world distance-from-camera z(t) to a line, trimmed to the
    physically-consistent monotonic prefix (see config.yaml:
    calibration.reversal_tolerance_m/reversal_confirm_frames).

    A genuine delivery's z(t) moves consistently in ONE direction (further
    for this project's bowling-end camera placement, closer for a batting-
    end one -- see SCHEMA.md) until the ball is caught/stopped. But
    zoom_track_detect.py's track can run on past that point (e.g. its
    motion-diff rescue locking onto a catcher's hands) -- z(t) then stops
    trending and starts wandering. Fitting the whole track naively would
    silently let that tail corrupt the speed estimate, so: walk forward
    smoothing out single-frame noise, track the running extreme (max if
    receding, min if approaching -- direction is read from a first-pass
    fit over just the first half, so a long corrupted tail can't flip it),
    and cut at the last point before a `reversal_confirm_frames`-long run
    back the other way. Returns None if fewer than 2 points remain.
    """
    n = len(ts)
    if n < 2:
        return None
    ts_arr = np.asarray(ts, dtype=np.float64)
    zs_arr = np.asarray(zs, dtype=np.float64)

    window = max(int(cfg_calib["smoothing_window"]), 1)
    half = window // 2
    smoothed = np.array([zs_arr[max(0, i - half) : min(n, i + half + 1)].mean() for i in range(n)])

    # Direction (is z trending up or down?) from only the FIRST half of the
    # track, not the whole thing -- a fit over the whole series would let a
    # long corrupted tail (see module docstring) drag the sign the wrong
    # way, which is exactly the failure case this function exists to guard
    # against.
    k = max(2, n // 2)
    overall_slope = np.polyfit(ts_arr[:k], smoothed[:k], 1)[0]
    receding = overall_slope >= 0  # True: z should trend up; False: z should trend down

    tol = cfg_calib["reversal_tolerance_m"]
    confirm = max(int(cfg_calib["reversal_confirm_frames"]), 1)
    extreme = smoothed[0]
    last_good_idx = 0
    reversal_start = None
    for i in range(1, n):
        val = smoothed[i]
        within = (val >= extreme - tol) if receding else (val <= extreme + tol)
        if within:
            last_good_idx = i
            reversal_start = None
            extreme = max(extreme, val) if receding else min(extreme, val)
        else:
            if reversal_start is None:
                reversal_start = i
            if i - reversal_start + 1 >= confirm:
                break
    cutoff = last_good_idx

    ts_used = ts_arr[: cutoff + 1]
    zs_used = zs_arr[: cutoff + 1]
    if len(ts_used) < 2:
        return None

    slope, intercept = np.polyfit(ts_used, zs_used, 1)
    pred = slope * ts_used + intercept
    ss_res = float(np.sum((zs_used - pred) ** 2))
    ss_tot = float(np.sum((zs_used - zs_used.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0

    return {"speed_kmh": abs(float(slope)) * 3.6, "r2": r2, "n_used": len(ts_used), "n_total": n}


def _planar_pixel_speed(real: list[dict[str, Any]], pixels_per_meter: float) -> dict[str, Any] | None:
    """Average speed over the track's real frames, converting total pixel
    path length directly to metres via a fixed single-plane
    pixels_per_meter scale (see pipeline/calibration.py: `_wickets_calib.
    json` / pixels_per_meter_from_wickets) instead of a depth-aware pinhole
    fit. Sums consecutive-frame (cx, cy) Euclidean displacement rather than
    endpoint-to-endpoint distance, since the ball's path isn't a straight
    line in image space (parabolic flight arc) -- endpoint distance would
    undercount it. Returns None if there aren't at least 2 real frames or
    the elapsed time is non-positive."""
    if len(real) < 2:
        return None
    path_px = sum(
        ((b["cx"] - a["cx"]) ** 2 + (b["cy"] - a["cy"]) ** 2) ** 0.5 for a, b in zip(real, real[1:])
    )
    duration_s = real[-1]["t"] - real[0]["t"]
    if duration_s <= 0:
        return None
    speed_mps = (path_px / pixels_per_meter) / duration_s
    return {"speed_kmh": speed_mps * 3.6, "n_used": len(real)}


def _quality_flags(
    delivery: dict[str, Any],
    config: dict[str, Any],
    speed_kmh: float,
    speed_method: str,
    fit: dict[str, Any] | None,
    diam_growth_ratio: float,
) -> list[str]:
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
    # `diam_growth_ratio` here is computed over whichever frames the
    # caller trusts (the calibrated-fit's trimmed prefix when available,
    # else the whole track) -- see compute_delivery_speed.
    min_growth = config["tracking"]["min_diam_growth_ratio"]
    max_shrink = config["tracking"]["max_diam_shrink_ratio"]
    if diam_growth_ratio < min_growth and diam_growth_ratio > max_shrink:
        flags.append("low_diam_change")

    if speed_method == "flight_distance_fallback":
        # No <video_stem>_calib.json for this video -- see
        # pipeline/calibration.py. Falling back to dividing a fixed assumed
        # flight distance by the tracked duration, which is only accurate
        # if the track happens to span the whole flight.
        flags.append("uncalibrated_speed_estimate")
    elif speed_method == "planar_pixel_speed":
        # No <video_stem>_calib.json (no depth model), only a weaker
        # single-plane pixels-per-meter scale -- see pipeline/calibration.py
        # and speed.py: _planar_pixel_speed.
        flags.append("planar_calibration_estimate")
        if real < config["calibration"]["min_frames_for_fit"]:
            flags.append("few_calibrated_frames")
    elif fit is not None:
        cfg_calib = config["calibration"]
        if fit["n_used"] < cfg_calib["min_frames_for_fit"]:
            flags.append("few_calibrated_frames")
        if fit["n_used"] < fit["n_total"]:
            # See _trim_and_fit_distance -- part of the track didn't fit
            # the established monotonic trend and was excluded from the
            # speed fit (but is still present in `frames` for display).
            flags.append("trimmed_non_monotonic_tail")
        if fit["r2"] < cfg_calib["min_fit_r2"]:
            flags.append("noisy_size_fit")

    return flags


def _confidence_from_flags(flags: list[str]) -> str:
    if "speed_out_of_range" in flags:
        return "low"
    if len(flags) >= 2:
        return "low"
    if len(flags) == 1:
        return "medium"
    return "high"


def _size_speed_curve(
    frames: list[dict[str, Any]], config: dict[str, Any], f_px: float | None
) -> list[dict[str, float]]:
    """Secondary, approximate size-based instantaneous speed curve:
    z(t) = f_px * ball_diameter_m / diam_px(t), speed = -dz/dt.

    Uses the video's calibrated focal length (`f_px`, see
    pipeline/calibration.py) when available -- the same one
    `compute_delivery_speed`'s primary estimate fits against, so this
    curve is consistent with it rather than a separately-eyeballed number.
    Falls back to back-solving a rough f_px from two (pixel_diameter,
    assumed_distance) pairs at the delivery's start/end (assuming
    flight_distance_m/~1.5m respectively) only when no calibration file
    exists for the video.

    For visualization only; returns [] on any degenerate input rather than
    raising, per SCHEMA.md (an empty size_speed_curve is fine, viewer hides
    the chart).
    """
    if len(frames) < 2:
        return []

    ball_diam_m = config["ball"]["diameter_m"]
    eps = 1e-3

    if f_px is None:
        flight_distance_m = config["geometry"]["flight_distance_m"]
        near_distance_m = 1.5  # assumed distance at the last frame, near the camera
        diam_start = frames[0]["diam_px"]
        diam_end = frames[-1]["diam_px"]
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
        # abs() rather than a signed -(z1-z0)/dt -- z can trend either way
        # depending on camera placement (see SCHEMA.md), and this curve
        # should always read as a positive speed regardless of direction.
        speed_mps = abs(z1 - z0) / dt
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


def compute_delivery_speed(
    delivery: dict[str, Any],
    config: dict[str, Any],
    f_px: float | None = None,
    pixels_per_meter: float | None = None,
) -> dict[str, Any]:
    """Augment a delivery dict (from `track.segment_deliveries`) with speed
    fields, per SCHEMA.md `deliveries[]`: `speed_kmh`, `speed_confidence`,
    `quality_flags`, `speed_method`, `size_speed_curve`.

    `f_px` is the video's calibrated focal length in pixels (see
    pipeline/calibration.py: focal_length_px), or None if no
    `<video_stem>_calib.json` exists for this video. `pixels_per_meter` is
    the weaker single-plane fallback (see pipeline/calibration.py:
    pixels_per_meter_from_wickets), used only when `f_px` is None. The
    caller loads and computes both once per video, not per delivery.

    Returns a new dict (does not mutate the input's top level, though the
    `frames` list is shared by reference for efficiency).
    """
    frames = delivery["frames"]
    start_t, end_t = _refine_endpoints(frames)

    ball_diam_m = config["ball"]["diameter_m"]
    real = [f for f in frames if f["source"] != "kalman_predicted" and f["diam_px"] > 1e-3]

    fit = None
    if f_px is not None and len(real) >= 2:
        ts = [f["t"] for f in real]
        zs = [f_px * ball_diam_m / f["diam_px"] for f in real]
        fit = _trim_and_fit_distance(ts, zs, config["calibration"])

    planar = None
    if fit is None and f_px is None and pixels_per_meter is not None:
        planar = _planar_pixel_speed(real, pixels_per_meter)

    if fit is not None:
        speed_kmh = fit["speed_kmh"]
        speed_method = "calibrated_size"
        # Score diam-change plausibility over the SAME trimmed prefix the
        # speed fit trusts, not the whole (possibly drift-contaminated)
        # track -- otherwise a clean, well-fit delivery could still get
        # flagged low_diam_change purely because of its own already-
        # excluded corrupted tail.
        diam_ratio = _diam_growth_ratio(real[: fit["n_used"]])
    elif planar is not None:
        speed_kmh = planar["speed_kmh"]
        speed_method = "planar_pixel_speed"
        diam_ratio = _diam_growth_ratio(real)
    else:
        # Fallback: no calibration file for this video, or too few real
        # frames to fit -- see module docstring for why this is known to
        # be inaccurate whenever the track doesn't span the whole flight.
        duration_s = end_t - start_t
        flight_distance_m = config["geometry"]["flight_distance_m"]
        speed_kmh = 3.6 * flight_distance_m / duration_s if duration_s > 0 else 0.0
        speed_method = "flight_distance_fallback"
        diam_ratio = delivery.get("diam_growth_ratio", 1.0)

    flags = _quality_flags(delivery, config, speed_kmh, speed_method, fit, diam_ratio)
    confidence = _confidence_from_flags(flags)
    curve = _size_speed_curve(frames, config, f_px)

    out = dict(delivery)
    out["speed_kmh"] = speed_kmh
    out["speed_confidence"] = confidence
    out["quality_flags"] = flags
    out["speed_method"] = speed_method
    if fit is not None:
        out["speed_fit_r2"] = round(fit["r2"], 3)
        out["speed_fit_frames_used"] = fit["n_used"]
    out["size_speed_curve"] = curve
    return out
