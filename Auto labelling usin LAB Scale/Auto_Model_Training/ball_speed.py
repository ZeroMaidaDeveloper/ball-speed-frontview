#!/usr/bin/env python3
"""
ball_tracker.py  -  MOG2+Lab ball detection + optional YOLO person masks
                   + zoom/pan + manual speed measurement (B) + per‑frame speed.
                   + Two‑point wicket calibration (distance = 20.12 m).
                   + SKIP first 5 detected balls before measuring.
                   + Press 'W' to reset wicket calibration (re‑select stumps).
"""

import cv2
import numpy as np
import math
import json
import os
import argparse
from collections import deque
from ultralytics import YOLO

# ---- EDIT THIS ----
INPUT_VIDEO = "/Users/takneekmacmini/Documents/Auto labelling usin LAB Scale/video/delivery_13.mp4"
ROI_FILE = os.path.splitext(INPUT_VIDEO)[0] + "_roi.json"
FORCE_ROI_RESELECT = False

# ---- DISPLAY / ZOOM ----
ZOOM_STEP = 0.25
ZOOM_MIN = 1.0
ZOOM_MAX = 6.0

# ---- SPEED ESTIMATION ----
D_REL2BAT = 20.12         # metres (fallback if no calibration)
SPEED_MEASURE_FRAMES = 14  # frames to record for manual speed
WICKET_DIST = 20.12        # metres between wickets
SKIP_FRAMES = 5            # number of initial detections to ignore

# ---- CANDIDATE (from MOTION) ----
MIN_AREA   = 3
MAX_AREA   = 1200
MIN_FILL   = 0.35
MAX_ASPECT = 3.0
RED_MEAN_MIN = 135
USE_REDNESS  = True
W_RED = 1.0
W_ROUND = 1.0
W_DIST = 1.5
GATE_RADIUS_PX = 120
MAX_MISSED     = 20
MAX_COAST      = 6
TRAIL_LEN      = 40

# ---- YOLO PERSON FILTER (optional) ----
USE_PERSON_MASK = False
YOLO_MODEL = "yolov8n-seg.pt"
CONF_PERSON = 0.5
PERSON_MIN_HEIGHT_RATIO = 0.30
PERSON_MIN_ABS_HEIGHT = 50

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ROI_COLOR = (255, 120, 0)
CALIB_COLOR = (0, 255, 255)


# =====================================================================
# DETECTION PIPELINE (unchanged)
# =====================================================================

def preprocess(frame, mode):
    if mode == 1:
        return _clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0])
    if mode == 2:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1]
    if mode == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 2]
    if mode == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def clean_motion(raw):
    m = cv2.medianBlur(raw, 3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel, iterations=1)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def render_lab(frame):
    a = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1]
    disp = np.clip((a.astype(np.int16) - 110) * 3, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(disp, cv2.COLORMAP_JET)


def find_candidates(motion_mask, a_channel, min_fill, red_mean_min, use_redness, person_masks):
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > MAX_ASPECT:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        circle_area = math.pi * r * r
        fill = area / circle_area if circle_area > 0 else 0.0
        if fill < min_fill:
            continue
        cmask = np.zeros((h, w), np.uint8)
        cv2.drawContours(cmask, [c], -1, 255, -1, offset=(-x, -y))
        vals = a_channel[y:y + h, x:x + w][cmask > 0]
        mean_a = float(vals.mean()) if vals.size else 128.0
        if use_redness and mean_a < red_mean_min:
            continue
        center = (int(cx), int(cy))
        inside_person = False
        for mask in person_masks:
            if mask[center[1], center[0]] > 0:
                inside_person = True
                break
        if inside_person:
            continue
        cands.append({"center": center, "radius": r, "area": area,
                      "fill": fill, "aspect": aspect, "mean_a": mean_a,
                      "bbox": (x, y, w, h)})
    return cands


def choose(cands, predicted, use_gate):
    best, best_score = None, -1e9
    for c in cands:
        redness = (c["mean_a"] - 128.0) / 40.0
        score = W_RED * redness + W_ROUND * c["fill"]
        if predicted is not None:
            d = math.hypot(c["center"][0] - predicted[0], c["center"][1] - predicted[1])
            if use_gate and d > GATE_RADIUS_PX:
                continue
            score -= W_DIST * (d / GATE_RADIUS_PX)
        if score > best_score:
            best_score, best = score, c
    return best


# =====================================================================
# ROI (unchanged)
# =====================================================================

def select_roi(frame):
    pts = []
    win = "Draw ROI  |  click points  z=undo  c=clear  ENTER=confirm  ESC=whole frame"

    def redraw():
        d = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(d, p, 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(d, pts[i - 1], p, (0, 255, 255), 2)
        if len(pts) >= 3:
            cv2.line(d, pts[-1], pts[0], (0, 255, 255), 1)
            ov = d.copy(); cv2.fillPoly(ov, [np.array(pts, np.int32)], ROI_COLOR)
            d = cv2.addWeighted(ov, 0.20, d, 0.80, 0)
        cv2.putText(d, f"{len(pts)} point(s) - ENTER to confirm",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, d)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y)); redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    result = None
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 13 and len(pts) >= 3:
            result = pts[:]; break
        if k == 27:
            break
        if k == ord('z') and pts:
            pts.pop(); redraw()
        if k == ord('c'):
            pts.clear(); redraw()
    cv2.destroyWindow(win)
    return result


def build_roi_mask(polygon, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if polygon and len(polygon) >= 3:
        cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
    else:
        mask[:] = 255
    return mask


def save_roi(polygon, frame):
    try:
        with open(ROI_FILE, "w") as f:
            json.dump({"polygon": polygon, "frame_size": [frame.shape[1], frame.shape[0]]}, f)
        print(f"Saved ROI to {ROI_FILE}")
    except OSError as e:
        print(f"Could not save ROI: {e}")


def load_roi(frame):
    if FORCE_ROI_RESELECT or not os.path.exists(ROI_FILE):
        return None
    try:
        with open(ROI_FILE) as f:
            data = json.load(f)
        size = data.get("frame_size")
        if size and (size[0] != frame.shape[1] or size[1] != frame.shape[0]):
            print("Saved ROI was for a different frame size; ignoring.")
            return None
        poly = [tuple(p) for p in data.get("polygon", [])]
        if len(poly) >= 3:
            print(f"Loaded ROI ({len(poly)} points). Press 'r' to redraw.")
            return poly
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        print(f"Could not read ROI file: {e}")
    return None


# =====================================================================
# WICKET CALIBRATION (two points)
# =====================================================================

CALIB_FILE = os.path.splitext(INPUT_VIDEO)[0] + "_wickets_calib.json"

def select_wickets(frame):
    """Click on two wicket points (e.g., the stumps). Returns (p1, p2)."""
    pts = []
    win = "Click on two wickets  |  z=undo  ENTER=confirm  ESC=skip"

    def redraw():
        d = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(d, p, 6, CALIB_COLOR, -1)
            cv2.putText(d, str(i+1), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CALIB_COLOR, 2)
        cv2.putText(d, f"{len(pts)}/2 points - ENTER to confirm", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(win, d)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y)); redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    result = None
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 13 and len(pts) == 2:
            result = pts[:]; break
        if k == 27:
            break
        if k == ord('z') and pts:
            pts.pop(); redraw()
    cv2.destroyWindow(win)
    return result


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        return None
    try:
        with open(CALIB_FILE) as f:
            data = json.load(f)
        return data.get("pixels_per_meter"), data.get("points")
    except:
        return None


def save_calibration(p1, p2, W, H):
    pixel_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if pixel_dist == 0:
        print("Wicket points coincide – calibration failed.")
        return None
    ppm = pixel_dist / WICKET_DIST   # pixels per metre
    data = {
        "points": [p1, p2],
        "pixel_distance": pixel_dist,
        "pixels_per_meter": ppm,
        "wicket_distance_m": WICKET_DIST,
        "frame_size": [W, H]
    }
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved: {pixel_dist:.1f} px over {WICKET_DIST} m -> {ppm:.2f} px/m")
    return ppm


# =====================================================================
# IN-FRAME ZOOM helper (unchanged)
# =====================================================================

def zoom_panel(img, x0, y0, Z, W, H):
    if Z <= 1.0:
        return img
    vw = int(round(W / Z)); vh = int(round(H / Z))
    ix0 = int(round(x0)); iy0 = int(round(y0))
    ix0 = max(0, min(ix0, W - vw)); iy0 = max(0, min(iy0, H - vh))
    crop = img[iy0:iy0 + vh, ix0:ix0 + vw]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)


# =====================================================================
# MAIN (with skip feature and wicket reset)
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-video", type=str, default=None,
                        help="path to save output video with overlays")
    args = parser.parse_args()

    # Load YOLO only if needed
    model = None
    if USE_PERSON_MASK:
        print("Loading YOLO segmentation model for person masks...")
        model = YOLO(YOLO_MODEL)
        print("YOLO loaded.")
    else:
        print("YOLO person masking disabled (USE_PERSON_MASK=False).")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("Could not open video:", INPUT_VIDEO); return
    ok, frame0 = cap.read()
    if not ok:
        print("Could not read first frame."); return
    H, W = frame0.shape[0], frame0.shape[1]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ---- Video writer ----
    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (2*W, H))
        print(f"Writing output video to {args.output_video}")

    # ---- ROI ----
    roi_poly = load_roi(frame0)
    if roi_poly is None:
        print("Draw an ROI now.")
        roi_poly = select_roi(frame0)
        if roi_poly:
            save_roi(roi_poly, frame0)
    roi_mask = build_roi_mask(roi_poly, frame0.shape)
    roi_np = np.array(roi_poly, np.int32) if roi_poly else None

    # ---- Wicket calibration ----
    calib = load_calibration()
    if calib is None:
        print("Click on the two wickets (stumps) for calibration.")
        wickets = select_wickets(frame0)
        if wickets and len(wickets) == 2:
            ppm = save_calibration(wickets[0], wickets[1], W, H)
            if ppm is not None:
                calib = ppm
            else:
                print("Calibration failed – using fixed distance fallback.")
        else:
            print("Calibration skipped – using fixed distance fallback.")
    else:
        ppm, pts = calib
        print(f"Loaded calibration: {ppm:.2f} px/m, wickets at {pts}")
        calib = ppm

    pixels_per_meter = calib if isinstance(calib, (int, float)) else None

    # ---- MOG2 ----
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    mode = 1
    idx = 0
    last_idx = -999
    playing = False

    red_mean_min = RED_MEAN_MIN
    min_fill = MIN_FILL
    use_redness = USE_REDNESS
    use_gate = True

    ftrack = deque(maxlen=TRAIL_LEN)
    missed = 0

    # ---- Speed measurement state ----
    speed_measure_active = False
    speed_measure_points = []
    manual_speed_kmh = None
    skip_countdown = 0          # remaining frames to skip
    recording_started = False   # true after skip done

    # ---- viewport ----
    view = {"Z": 1.0, "cx": W / 2.0, "cy": H / 2.0, "x0": 0.0, "y0": 0.0,
            "mouse": (0, 0), "panning": False, "pan_start": None, "center_start": None}

    win = "BALL TRACKER  right-drag=pan  i/o zoom  0 reset | B=measure (toggle)  C=cancel | W=wickets reset | space play/pause"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def panel_x_of(mx):
        return mx if mx < W else mx - W

    def on_mouse(event, x, y, flags, param):
        view["mouse"] = (x, y)
        Z = view["Z"]
        if event == cv2.EVENT_RBUTTONDOWN:
            view["panning"] = True
            view["pan_start"] = (panel_x_of(x), y)
            view["center_start"] = (view["cx"], view["cy"])
        elif event == cv2.EVENT_RBUTTONUP:
            view["panning"] = False
        elif event == cv2.EVENT_MOUSEMOVE and view["panning"]:
            sx, sy = view["pan_start"]; ccx, ccy = view["center_start"]
            view["cx"] = ccx - (panel_x_of(x) - sx) / Z
            view["cy"] = ccy - (y - sy) / Z

    cv2.setMouseCallback(win, on_mouse)

    def zoom_to_cursor(new_Z):
        mx, my = view["mouse"]
        px = panel_x_of(mx)
        Zold = view["Z"]
        world_x = view["x0"] + px / Zold
        world_y = view["y0"] + my / Zold
        vw, vh = W / new_Z, H / new_Z
        x0n = world_x - px / new_Z
        y0n = world_y - my / new_Z
        view["cx"] = x0n + vw / 2.0
        view["cy"] = y0n + vh / 2.0
        view["Z"] = new_Z

    def predicted_point():
        if len(ftrack) >= 2:
            (x1, y1) = ftrack[-2]["pt"]; (x2, y2) = ftrack[-1]["pt"]
            return (2 * x2 - x1, 2 * y2 - y1)
        return None

    def warm(upto):
        nonlocal bg
        bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
        for w in range(max(0, upto - 20), upto):
            cap.set(cv2.CAP_PROP_POS_FRAMES, w)
            ok_w, wf = cap.read()
            if ok_w:
                bg.apply(preprocess(wf, mode))

    warm(idx)

    # ---- YOLO frame counter ----
    yolo_counter = 0
    YOLO_INTERVAL = 8 if USE_PERSON_MASK else 999

    while True:
        if abs(idx - last_idx) != 1:
            warm(idx); ftrack.clear(); missed = 0
            # Do NOT cancel speed measurement on seek (commented out)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 1); playing = False; continue
        last_idx = idx

        a_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1]
        motion = clean_motion(bg.apply(preprocess(frame, mode)))
        motion_roi = cv2.bitwise_and(motion, roi_mask)

        # YOLO person masks
        person_masks = []
        if USE_PERSON_MASK and (yolo_counter % YOLO_INTERVAL == 0 or yolo_counter == 0):
            yolo_counter = 0
            results = model(frame, classes=[0], conf=CONF_PERSON, verbose=False)
            if results and len(results) > 0 and results[0].masks is not None:
                all_boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    all_boxes.append((x1, y1, x2, y2, y2 - y1))
                if all_boxes:
                    max_h = max(b[4] for b in all_boxes)
                    min_height = max(PERSON_MIN_ABS_HEIGHT, PERSON_MIN_HEIGHT_RATIO * max_h)
                    masks = results[0].masks.data.cpu().numpy()
                    for i, (x1, y1, x2, y2, h) in enumerate(all_boxes):
                        if h >= min_height:
                            mk = (masks[i] * 255).astype(np.uint8)
                            if mk.shape != (H, W):
                                mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
                            person_masks.append(mk)
        yolo_counter += 1

        # Detection
        ball = None
        cands = find_candidates(motion_roi, a_channel, min_fill, red_mean_min,
                                use_redness, person_masks)
        ball = choose(cands, predicted_point(), use_gate)
        if ball is not None:
            ftrack.append({"pt": ball["center"], "pred": False}); missed = 0
        else:
            missed += 1
            if missed > MAX_MISSED:
                ftrack.clear()
                if speed_measure_active:
                    speed_measure_active = False
                    speed_measure_points = []
                    manual_speed_kmh = None
                    skip_countdown = 0
                    recording_started = False
                    print("Speed measurement cancelled (ball lost).")

        # -------- Manual speed measurement with skip --------
        if speed_measure_active:
            if ball is not None:
                if skip_countdown > 0:
                    # Still in skip phase
                    skip_countdown -= 1
                    print(f"Skipping frame {idx} (remaining skip: {skip_countdown})")
                else:
                    # Skip done – start recording
                    if not recording_started:
                        recording_started = True
                        print(f"Recording started at frame {idx}")
                    # Add point
                    speed_measure_points.append((idx, ball["center"][0], ball["center"][1]))
                    n_pts = len(speed_measure_points)
                    if n_pts >= 2:
                        # Compute cumulative speed from first recorded point to current
                        f0, x0, y0 = speed_measure_points[0]
                        fi, xi, yi = speed_measure_points[-1]
                        dt = (fi - f0) / fps
                        if dt > 0:
                            if pixels_per_meter is not None and pixels_per_meter > 0:
                                pixel_dist = math.hypot(xi - x0, yi - y0)
                                dist_m = pixel_dist / pixels_per_meter
                            else:
                                dist_m = D_REL2BAT
                            speed_ms = dist_m / dt
                            speed_kmh = speed_ms * 3.6
                            print(f"Frame {fi}: speed = {speed_kmh:.1f} km/h (over {n_pts} frames)")
                        else:
                            print(f"Frame {fi}: speed = N/A (zero time)")
                    if n_pts >= SPEED_MEASURE_FRAMES:
                        # Finalise
                        f0, x0, y0 = speed_measure_points[0]
                        f_last, x_last, y_last = speed_measure_points[-1]
                        dt = (f_last - f0) / fps
                        if dt > 0:
                            if pixels_per_meter is not None and pixels_per_meter > 0:
                                pixel_dist = math.hypot(x_last - x0, y_last - y0)
                                dist_m = pixel_dist / pixels_per_meter
                            else:
                                dist_m = D_REL2BAT
                            speed_ms = dist_m / dt
                            manual_speed_kmh = speed_ms * 3.6
                            print(f"Average speed over {n_pts} frames: {manual_speed_kmh:.1f} km/h")
                        else:
                            manual_speed_kmh = None
                            print("Speed measurement failed (zero time).")
                        speed_measure_active = False
                        speed_measure_points = []
                        skip_countdown = 0
                        recording_started = False
            else:
                # Ball not detected – do nothing (we wait for it to reappear)
                pass

        # Build display
        render = render_lab(frame)
        vis = frame.copy()
        if roi_np is not None:
            cv2.polylines(vis, [roi_np], True, ROI_COLOR, 2)
        for mask in person_masks:
            col = np.zeros_like(vis); col[:, :, 2] = mask
            vis = cv2.addWeighted(vis, 1.0, col, 0.2, 0)
        rvis = render.copy()
        if ball is not None:
            bbox = ball["bbox"]
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            cv2.rectangle(rvis, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

        # Speed HUD
        calib_text = " (calibrated)" if pixels_per_meter is not None else " (fixed distance)"
        if speed_measure_active:
            if skip_countdown > 0:
                speed_text = f"Skipping... ({skip_countdown} left)"
            else:
                speed_text = f"Recording... ({len(speed_measure_points)}/{SPEED_MEASURE_FRAMES})"
        elif manual_speed_kmh is not None:
            speed_text = f"Speed: {manual_speed_kmh:.1f} km/h{calib_text}"
        else:
            speed_text = "Speed: --"
        cv2.putText(vis, speed_text, (W - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(rvis, speed_text, (W - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Zoom
        Z = view["Z"]
        vw, vh = W / Z, H / Z
        cx = min(max(view["cx"], vw / 2.0), W - vw / 2.0)
        cy = min(max(view["cy"], vh / 2.0), H - vh / 2.0)
        view["cx"], view["cy"] = cx, cy
        view["x0"], view["y0"] = cx - vw / 2.0, cy - vh / 2.0
        zvis = zoom_panel(vis, view["x0"], view["y0"], Z, W, H)
        zrvis = zoom_panel(rvis, view["x0"], view["y0"], Z, W, H)

        cv2.putText(zvis, f"frame {idx}/{total-1}   zoom {Z:.2f}x", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(zvis, "right-drag=pan  i/o zoom  0 reset  B=measure  C=cancel  W=wickets", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)
        cv2.putText(zrvis, "LAB render", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        combined = np.hstack([zvis, zrvis])
        cv2.imshow(win, combined)
        cv2.waitKey(1)

        if video_writer is not None:
            video_writer.write(combined)

        # Keyboard
        key = cv2.waitKey(30 if playing else 0) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('d'):
            idx = min(total - 1, idx + 1)
        elif key == ord('a'):
            idx = max(0, idx - 1)
        elif key == ord(']'):
            idx = min(total - 1, idx + 10)
        elif key == ord('['):
            idx = max(0, idx - 10)
        elif key == ord('i'):
            zoom_to_cursor(min(ZOOM_MAX, view["Z"] + ZOOM_STEP))
        elif key == ord('o'):
            zoom_to_cursor(max(ZOOM_MIN, view["Z"] - ZOOM_STEP))
        elif key == ord('0'):
            view["Z"] = 1.0; view["cx"], view["cy"] = W / 2.0, H / 2.0
        elif key == ord('b'):
            if not speed_measure_active:
                if ball is not None:
                    speed_measure_active = True
                    speed_measure_points = []
                    manual_speed_kmh = None
                    skip_countdown = SKIP_FRAMES
                    recording_started = False
                    print(f"Speed measurement started at frame {idx}. Skipping first {SKIP_FRAMES} detections.")
                else:
                    print("No ball detected – cannot start measurement.")
            else:
                speed_measure_active = False
                speed_measure_points = []
                manual_speed_kmh = None
                skip_countdown = 0
                recording_started = False
                print("Speed measurement cancelled by user.")
        elif key == ord('c'):
            if speed_measure_active:
                speed_measure_active = False
                speed_measure_points = []
                manual_speed_kmh = None
                skip_countdown = 0
                recording_started = False
                print("Speed measurement cancelled by user (C).")
        # ---- Wicket calibration reset ----
        elif key == ord('w'):
            print("Resetting wicket calibration. Click on the two wickets again.")
            wickets = select_wickets(frame)
            if wickets and len(wickets) == 2:
                ppm = save_calibration(wickets[0], wickets[1], W, H)
                if ppm is not None:
                    pixels_per_meter = ppm
                    print(f"Calibration updated: {ppm:.2f} px/m")
                else:
                    print("Calibration failed – keeping previous calibration.")
            else:
                print("Calibration cancelled – keeping previous calibration.")
        # ---- Tuning keys ----
        elif key in (ord('+'), ord('=')):
            red_mean_min = min(200, red_mean_min + 1)
        elif key == ord('-'):
            red_mean_min = max(128, red_mean_min - 1)
        elif key == ord('.'):
            min_fill = min(0.95, min_fill + 0.05)
        elif key == ord(','):
            min_fill = max(0.05, min_fill - 0.05)
        elif key == ord('k'):
            use_redness = not use_redness
        elif key == ord('g'):
            use_gate = not use_gate
        elif key == ord('r'):
            new_poly = select_roi(frame)
            if new_poly:
                roi_poly = new_poly
                roi_mask = build_roi_mask(roi_poly, frame.shape)
                roi_np = np.array(roi_poly, np.int32)
                save_roi(roi_poly, frame)
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(win, on_mouse)
        elif ord('1') <= key <= ord('5'):
            mode = key - ord('1')
        elif playing:
            idx = min(total - 1, idx + 1)

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {args.output_video}")
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()