"""Self-contained synthetic tests for pipeline/track.py + pipeline/speed.py.

No real video/detector involved -- candidates are generated here to match
the "Per-frame candidate detection" contract in SCHEMA.md. Plain
asserts/prints, run directly with `python3 pipeline/test_track_speed.py`.
"""

from __future__ import annotations

import copy
import random

import yaml

from speed import compute_delivery_speed
from track import segment_deliveries

CONFIG_PATH = __file__.rsplit("/", 1)[0] + "/../config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def simulate_flight(
    *,
    frame_start: int,
    t_start: float,
    n_frames: int,
    duration_s: float,
    z_start_m: float,
    z_end_m: float,
    f_px: float,
    ball_diam_m: float,
    cx0: float = 960.0,
    cy0: float = 540.0,
    drift_px: float = 20.0,
    center_noise_std: float = 2.0,
    diam_noise_frac: float = 0.03,
    rng: random.Random | None = None,
    drop_local_indices: set[int] = frozenset(),
    false_positive_local_indices: set[int] = frozenset(),
) -> list[dict]:
    """Simulate one delivery: ball approaching the camera head-on, apparent
    diameter growing per a decelerating (concave) closing-distance profile,
    plus noise / drops / spurious extra candidates.

    Average speed of this flight (distance / duration) is exactly
    3.6 * (z_start_m - z_end_m) / duration_s km/h -- used as ground truth to
    check against compute_delivery_speed's time-of-flight estimate.
    """
    rng = rng or random.Random(0)
    candidates: list[dict] = []
    distance = z_start_m - z_end_m

    # Evenly spaced *nominal* timestamps, with small per-frame jitter to
    # simulate real (non-constant) phone frame timing -- endpoints are kept
    # exact so ground-truth flight duration is unambiguous.
    for i in range(n_frames):
        u_nominal = i / (n_frames - 1)
        t = t_start + u_nominal * duration_s
        if 0 < i < n_frames - 1:
            t += rng.uniform(-0.001, 0.001)
        u = (t - t_start) / duration_s  # actual fraction of flight elapsed

        # Concave distance-covered profile s(u) = 2u - u^2: velocity
        # decreases linearly from 2*(distance/duration) to 0, i.e. a
        # decelerating approach (plausible drag deceleration), while the
        # *average* speed over the whole flight is exactly distance/duration.
        s = 2 * u - u * u
        z = z_start_m - distance * s

        diam_px = f_px * ball_diam_m / max(z, 1e-3)
        diam_px *= 1.0 + rng.uniform(-diam_noise_frac, diam_noise_frac)

        cx = cx0 + drift_px * u + rng.gauss(0, center_noise_std)
        cy = cy0 + rng.gauss(0, center_noise_std)

        frame_idx = frame_start + i

        if i not in drop_local_indices:
            half = diam_px / 2.0
            conf = max(0.55, min(0.97, rng.gauss(0.8, 0.08)))
            source = "yolo" if rng.random() < 0.5 else "classical"
            candidates.append(
                {
                    "frame": frame_idx,
                    "t": t,
                    "bbox": [cx - half, cy - half, cx + half, cy + half],
                    "conf": conf,
                    "source": source,
                }
            )

        if i in false_positive_local_indices:
            # A spurious false-positive candidate far from the true ball
            # (simulating classical-detector background clutter), on the
            # SAME frame as a real candidate -- tracker must gate it out.
            fx = cx + rng.uniform(300, 500) * rng.choice([-1, 1])
            fy = cy + rng.uniform(200, 400) * rng.choice([-1, 1])
            fd = rng.uniform(8, 30)
            candidates.append(
                {
                    "frame": frame_idx,
                    "t": t,
                    "bbox": [fx - fd / 2, fy - fd / 2, fx + fd / 2, fy + fd / 2],
                    "conf": rng.uniform(0.25, 0.45),
                    "source": "classical",
                }
            )

    return candidates


def check(label: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    assert cond, f"{label}: {detail}"


def scenario_clean_flight(config: dict) -> None:
    print("\n=== Scenario 1: clean flight (single delivery, some FP noise) ===")
    ball_diam_m = config["ball"]["diameter_m"]
    target_kmh = 120.0
    flight_distance_m = config["geometry"]["flight_distance_m"]
    duration_s = 3.6 * flight_distance_m / target_kmh

    cands = simulate_flight(
        frame_start=0,
        t_start=1.000,
        n_frames=18,
        duration_s=duration_s,
        z_start_m=1.5 + flight_distance_m,
        z_end_m=1.5,
        f_px=1600.0,
        ball_diam_m=ball_diam_m,
        rng=random.Random(42),
        false_positive_local_indices={3, 10, 15},
    )

    deliveries = segment_deliveries(cands, config)
    check("exactly one delivery", len(deliveries) == 1, f"got {len(deliveries)}")

    d = compute_delivery_speed(deliveries[0], config)
    err_pct = abs(d["speed_kmh"] - target_kmh) / target_kmh * 100
    check(
        "speed close to ground truth",
        err_pct < 8.0,
        f"estimated={d['speed_kmh']:.1f} km/h target={target_kmh:.1f} km/h err={err_pct:.1f}%",
    )
    check("no occlusion_gap flag", "occlusion_gap" not in d["quality_flags"], str(d["quality_flags"]))
    check("high confidence", d["speed_confidence"] == "high", d["speed_confidence"])
    check(
        "false positives excluded from track",
        all(f["conf"] >= 0.5 for f in d["frames"]),
        "a spurious low-conf FP leaked into the track",
    )
    check("size_speed_curve non-empty", len(d["size_speed_curve"]) > 0)
    print(f"    -> speed_kmh={d['speed_kmh']:.2f} confidence={d['speed_confidence']} flags={d['quality_flags']}")


def scenario_occlusion_merge(config: dict) -> None:
    print("\n=== Scenario 2: mid-flight occlusion gap that SHOULD merge ===")
    ball_diam_m = config["ball"]["diameter_m"]
    target_kmh = 105.0
    flight_distance_m = config["geometry"]["flight_distance_m"]
    duration_s = 3.6 * flight_distance_m / target_kmh

    n_frames = 22
    # Gap bigger than max_missed_frames_bridge (6) so the raw tracker splits
    # into two tracks, but short enough in real time (< delivery_gap_merge_s)
    # that track.py's merge logic should stitch them back together.
    drop = set(range(7, 15))  # 8 consecutive missing frames, mid-flight
    assert len(drop) > config["tracking"]["max_missed_frames_bridge"]

    cands = simulate_flight(
        frame_start=100,
        t_start=5.000,
        n_frames=n_frames,
        duration_s=duration_s,
        z_start_m=1.5 + flight_distance_m,
        z_end_m=1.5,
        f_px=1600.0,
        ball_diam_m=ball_diam_m,
        rng=random.Random(7),
        drop_local_indices=drop,
    )

    deliveries = segment_deliveries(cands, config)
    check("exactly one delivery (merged)", len(deliveries) == 1, f"got {len(deliveries)}")

    d = compute_delivery_speed(deliveries[0], config)
    err_pct = abs(d["speed_kmh"] - target_kmh) / target_kmh * 100
    check(
        "speed close to ground truth despite gap",
        err_pct < 10.0,
        f"estimated={d['speed_kmh']:.1f} km/h target={target_kmh:.1f} km/h err={err_pct:.1f}%",
    )
    check("occlusion_gap flag present", "occlusion_gap" in d["quality_flags"], str(d["quality_flags"]))
    check(
        "delivery spans full frame range (100..121)",
        d["start_frame"] == 100 and d["end_frame"] == 100 + n_frames - 1,
        f"start={d['start_frame']} end={d['end_frame']}",
    )
    n_predicted = sum(1 for f in d["frames"] if f["source"] == "kalman_predicted")
    check("some kalman_predicted bridge frames present", n_predicted > 0, f"n_predicted={n_predicted}")
    print(
        f"    -> speed_kmh={d['speed_kmh']:.2f} confidence={d['speed_confidence']} "
        f"flags={d['quality_flags']} predicted_frames={n_predicted}"
    )


def scenario_two_deliveries_no_merge(config: dict) -> None:
    print("\n=== Scenario 3: two separate deliveries with a long gap (should NOT merge) ===")
    ball_diam_m = config["ball"]["diameter_m"]
    flight_distance_m = config["geometry"]["flight_distance_m"]

    target_kmh_1 = 130.0
    duration_1 = 3.6 * flight_distance_m / target_kmh_1
    cands_1 = simulate_flight(
        frame_start=0,
        t_start=2.000,
        n_frames=16,
        duration_s=duration_1,
        z_start_m=1.5 + flight_distance_m,
        z_end_m=1.5,
        f_px=1550.0,
        ball_diam_m=ball_diam_m,
        rng=random.Random(11),
    )
    gap_s = config["tracking"]["delivery_boundary_gap_s"] + 3.0  # comfortably beyond the boundary threshold

    t_start_2 = cands_1[-1]["t"] + gap_s
    target_kmh_2 = 95.0
    duration_2 = 3.6 * flight_distance_m / target_kmh_2
    frame_start_2 = cands_1[-1]["frame"] + int(round(gap_s * 30)) + 1
    cands_2 = simulate_flight(
        frame_start=frame_start_2,
        t_start=t_start_2,
        n_frames=16,
        duration_s=duration_2,
        z_start_m=1.5 + flight_distance_m,
        z_end_m=1.5,
        f_px=1650.0,
        ball_diam_m=ball_diam_m,
        rng=random.Random(23),
    )

    deliveries = segment_deliveries(cands_1 + cands_2, config)
    check("exactly two deliveries (not merged)", len(deliveries) == 2, f"got {len(deliveries)}")

    d1 = compute_delivery_speed(deliveries[0], config)
    d2 = compute_delivery_speed(deliveries[1], config)
    err1 = abs(d1["speed_kmh"] - target_kmh_1) / target_kmh_1 * 100
    err2 = abs(d2["speed_kmh"] - target_kmh_2) / target_kmh_2 * 100
    check("delivery 1 speed close to target", err1 < 8.0, f"est={d1['speed_kmh']:.1f} target={target_kmh_1} err={err1:.1f}%")
    check("delivery 2 speed close to target", err2 < 8.0, f"est={d2['speed_kmh']:.1f} target={target_kmh_2} err={err2:.1f}%")
    print(f"    -> delivery1 speed_kmh={d1['speed_kmh']:.2f}, delivery2 speed_kmh={d2['speed_kmh']:.2f}")


def scenario_speed_out_of_range(config: dict) -> None:
    print("\n=== Scenario 4 (bonus): implausible speed gets flagged, not silently reported ===")
    ball_diam_m = config["ball"]["diameter_m"]
    flight_distance_m = config["geometry"]["flight_distance_m"]
    # Absurdly slow "delivery" (e.g. a rolled ball) -- below min_plausible_kmh.
    target_kmh = 10.0
    duration_s = 3.6 * flight_distance_m / target_kmh

    cands = simulate_flight(
        frame_start=0,
        t_start=0.0,
        n_frames=12,
        duration_s=duration_s,
        z_start_m=1.5 + flight_distance_m,
        z_end_m=1.5,
        f_px=1600.0,
        ball_diam_m=ball_diam_m,
        rng=random.Random(3),
    )
    deliveries = segment_deliveries(cands, config)
    check("one delivery found", len(deliveries) == 1, f"got {len(deliveries)}")
    d = compute_delivery_speed(deliveries[0], config)
    check("speed_out_of_range flagged", "speed_out_of_range" in d["quality_flags"], str(d["quality_flags"]))
    check("confidence downgraded to low", d["speed_confidence"] == "low", d["speed_confidence"])
    check("speed number still reported (not suppressed)", d["speed_kmh"] > 0)
    print(f"    -> speed_kmh={d['speed_kmh']:.2f} confidence={d['speed_confidence']} flags={d['quality_flags']}")


def main() -> None:
    config = load_config()
    scenario_clean_flight(copy.deepcopy(config))
    scenario_occlusion_merge(copy.deepcopy(config))
    scenario_two_deliveries_no_merge(copy.deepcopy(config))
    scenario_speed_out_of_range(copy.deepcopy(config))
    print("\nAll scenarios passed.")


if __name__ == "__main__":
    main()
