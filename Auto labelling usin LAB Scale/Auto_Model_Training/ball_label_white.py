"""
ball_label_white.py  -  MOG2+Lab WHITE-ball detection + YOLO person-mask filter + manual click
                      + IN-FRAME zoom (window size never changes).
                      + Contour‑based manual bounding box.

Same UI/workflow as ball_label.py. The only change is the colour filter: instead of
looking for "redness" (high Lab a-channel), it looks for "whiteness" - high Lab
L-channel (bright) AND low chroma (a,b close to the neutral 128,128), i.e. bright
and colourless, which is what a white ball looks like against grass/pitch.
"""

import cv2
import numpy as np
import math
import json
import os
import glob
from collections import deque
from ultralytics import YOLO

# ---- EDIT THIS ----
INPUT_VIDEO = "/Users/takneekmacmini/Documents/Auto labelling usin LAB Scale/video/faster.mp4"
ROI_FILE = os.path.splitext(INPUT_VIDEO)[0] + "_roi.json"
FORCE_ROI_RESELECT = False

# ---- DATASET OUTPUT ----
DATASET_DIR = "dataset_white"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
SAVE_IMAGE = "white"
CLASS_ID = 0
MIN_BOX = 20

# ---- MANUAL ANNOTATION ----
MANUAL_BOX_SIZE = 25
SEARCH_RADIUS = 40          # how far around click to search for contour

# ---- ZOOM ----
ZOOM_STEP = 0.25
ZOOM_MIN = 1.0
ZOOM_MAX = 6.0

# ---- DUPLICATE-SAVE PREVENTION ----
COOLDOWN_FRAMES = 8
IOU_DUP = 0.5
CENTROID_DUP = 15

# ---- CANDIDATE (from MOTION) ----
MIN_AREA   = 3
MAX_AREA   = 1200
MIN_FILL   = 0.35
MAX_ASPECT = 3.0
WHITE_L_MIN      = 190   # min Lab-L (lightness) to count as "bright enough to be white"
WHITE_CHROMA_MAX = 25    # max distance from neutral (a=128,b=128) - keeps it achromatic
USE_WHITENESS = True
W_WHITE = 1.0
W_ROUND = 1.0
W_DIST = 1.5
GATE_RADIUS_PX = 120
MAX_MISSED     = 10
MAX_COAST      = 6
TRAIL_LEN      = 40

# ---- YOLO PERSON FILTER (Segmentation) ----
YOLO_MODEL = "yolov8n-seg.pt"
CONF_PERSON = 0.5
PERSON_MIN_HEIGHT_RATIO = 0.30
PERSON_MIN_ABS_HEIGHT = 50

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ROI_COLOR = (255, 120, 0)


# =====================================================================
# DETECTION PIPELINE
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


def render_white(frame):
    """Heatmap that lights up where pixels are bright AND achromatic (white-ish)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)
    whiteness = l - chroma
    disp = np.clip((whiteness - 150) * 3, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(disp, cv2.COLORMAP_JET)


def find_candidates(motion_mask, l_channel, a_channel, b_channel, min_fill,
                    white_l_min, white_chroma_max, use_whiteness, person_masks):
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
        l_vals = l_channel[y:y + h, x:x + w][cmask > 0]
        a_vals = a_channel[y:y + h, x:x + w][cmask > 0]
        b_vals = b_channel[y:y + h, x:x + w][cmask > 0]
        mean_l = float(l_vals.mean()) if l_vals.size else 128.0
        mean_a = float(a_vals.mean()) if a_vals.size else 128.0
        mean_b = float(b_vals.mean()) if b_vals.size else 128.0
        chroma = math.hypot(mean_a - 128.0, mean_b - 128.0)
        if use_whiteness and (mean_l < white_l_min or chroma > white_chroma_max):
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
                      "fill": fill, "aspect": aspect, "mean_l": mean_l, "chroma": chroma,
                      "bbox": (x, y, w, h)})
    return cands


def choose(cands, predicted, use_gate):
    best, best_score = None, -1e9
    for c in cands:
        whiteness = (c["mean_l"] - 128.0) / 40.0 - c["chroma"] / 40.0
        score = W_WHITE * whiteness + W_ROUND * c["fill"]
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
# LABELING HELPERS (unchanged)
# =====================================================================

def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)


def generate_filename():
    existing = glob.glob(os.path.join(IMAGES_DIR, "ball_*.jpg"))
    max_idx = 0
    for p in existing:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            max_idx = max(max_idx, int(stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max_idx + 1


def make_label_box(center, contour_bbox, W, H):
    cx, cy = center
    _, _, cw, ch = contour_bbox
    bw = max(MIN_BOX, cw); bh = max(MIN_BOX, ch)
    x = int(round(cx - bw / 2)); y = int(round(cy - bh / 2))
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    bw = min(bw, W - x); bh = min(bh, H - y)
    return (x, y, bw, bh)


def make_manual_box(center, W, H):
    cx, cy = center
    half = MANUAL_BOX_SIZE // 2
    x = max(0, min(cx - half, W - 1)); y = max(0, min(cy - half, H - 1))
    bw = min(MANUAL_BOX_SIZE, W - x); bh = min(MANUAL_BOX_SIZE, H - y)
    return (x, y, bw, bh)


def convert_bbox_to_yolo(bbox, W, H):
    x, y, w, h = bbox
    return (x + w / 2.0) / W, (y + h / 2.0) / H, w / W, h / H


def save_image(image, index):
    path = os.path.join(IMAGES_DIR, f"ball_{index:06d}.jpg")
    cv2.imwrite(path, image)
    return path


def save_yolo_label(yolo_bbox, index):
    path = os.path.join(LABELS_DIR, f"ball_{index:06d}.txt")
    cx, cy, w, h = yolo_bbox
    with open(path, "w") as f:
        f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return path


def draw_candidate_box(img, bbox, color, text=None):
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if text:
        cv2.putText(img, text, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def is_duplicate_detection(bbox, frame_idx, last):
    if last is None:
        return False
    if frame_idx - last["frame"] > COOLDOWN_FRAMES:
        return False
    cx = bbox[0] + bbox[2] / 2; cy = bbox[1] + bbox[3] / 2
    lx = last["bbox"][0] + last["bbox"][2] / 2; ly = last["bbox"][1] + last["bbox"][3] / 2
    centroid = math.hypot(cx - lx, cy - ly)
    return _iou(bbox, last["bbox"]) > IOU_DUP or centroid < CENTROID_DUP


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
# find contour from manual click
# =====================================================================

def find_contour_bbox_from_click(frame, point, motion_mask, l_channel,
                                 search_radius=40, min_area=5, max_area=800):
    """
    Given a click point, search the motion mask and LAB L‑channel in a small
    neighbourhood to find the ball contour. Returns a tight bounding box
    (x, y, w, h) or None if no suitable contour found.
    """
    cx, cy = point
    H, W = frame.shape[:2]

    # ROI boundaries
    x1 = max(0, cx - search_radius)
    y1 = max(0, cy - search_radius)
    x2 = min(W, cx + search_radius)
    y2 = min(H, cy + search_radius)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    # Crop motion mask and L-channel
    roi_motion = motion_mask[y1:y2, x1:x2]
    roi_l = l_channel[y1:y2, x1:x2]

    # Find contours in the motion ROI
    contours, _ = cv2.findContours(roi_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter contours: must contain the click point (or be very close)
    best_bbox = None
    best_dist = float('inf')
    for cnt in contours:
        cnt_full = cnt + np.array([x1, y1])  # shift back
        dist = cv2.pointPolygonTest(cnt_full, (cx, cy), True)
        if dist >= 0:  # inside or on edge
            x, y, w, h = cv2.boundingRect(cnt_full)
            area = w * h
            if area < min_area or area > max_area:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 3.0:
                continue
            if area < best_dist:  # using area as proxy for distance
                best_dist = area
                best_bbox = (x, y, w, h)
        else:
            d = abs(dist)
            if d < 10 and d < best_dist:
                x, y, w, h = cv2.boundingRect(cnt_full)
                area = w * h
                if area < min_area or area > max_area:
                    continue
                aspect = max(w, h) / max(1, min(w, h))
                if aspect > 3.0:
                    continue
                best_dist = d
                best_bbox = (x, y, w, h)

    if best_bbox is not None:
        return best_bbox

    # Fallback: threshold the L-channel to find a bright/white blob
    _, thresh = cv2.threshold(roi_l, WHITE_L_MIN, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, _kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_full = cnt + np.array([x1, y1])
        dist = cv2.pointPolygonTest(cnt_full, (cx, cy), True)
        if dist >= 0:
            x, y, w, h = cv2.boundingRect(cnt_full)
            area = w * h
            if area < min_area or area > max_area:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 3.0:
                continue
            return (x, y, w, h)

    return None


# =====================================================================
# MAIN
# =====================================================================

def main():
    ensure_dirs()
    print("Loading YOLO segmentation model for person masks...")
    model = YOLO(YOLO_MODEL)
    print("YOLO loaded.")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("Could not open video:", INPUT_VIDEO); return
    ok, frame0 = cap.read()
    if not ok:
        print("Could not read first frame."); return
    H, W = frame0.shape[0], frame0.shape[1]

    roi_poly = load_roi(frame0)
    if roi_poly is None:
        print("Draw an ROI now.")
        roi_poly = select_roi(frame0)
        if roi_poly:
            save_roi(roi_poly, frame0)
    roi_mask = build_roi_mask(roi_poly, frame0.shape)
    roi_np = np.array(roi_poly, np.int32) if roi_poly else None

    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    mode = 1
    idx = 0
    last_idx = -999
    playing = False

    white_l_min = WHITE_L_MIN
    white_chroma_max = WHITE_CHROMA_MAX
    min_fill = MIN_FILL
    use_whiteness = USE_WHITENESS
    use_gate = True

    ftrack = deque(maxlen=TRAIL_LEN)
    missed = 0

    next_index = generate_filename()
    saved_count = 0
    last_saved = None
    manual_pt = None          # (x, y) where user clicked
    manual_bbox = None        # computed tight bounding box from contour

    # ---- viewport ----
    view = {"Z": 1.0, "cx": W / 2.0, "cy": H / 2.0, "x0": 0.0, "y0": 0.0,
            "mouse": (0, 0), "panning": False, "pan_start": None, "center_start": None}

    win = "AUTO-LABEL (WHITE BALL)  S save N skip Q quit | click=label  right-drag=pan  i/o zoom  0 reset"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def panel_x_of(mx):
        return mx if mx < W else mx - W

    def canvas_to_orig(mx, my):
        Z = view["Z"]
        ox = view["x0"] + panel_x_of(mx) / Z
        oy = view["y0"] + my / Z
        return max(0, min(int(round(ox)), W - 1)), max(0, min(int(round(oy)), H - 1))

    def on_mouse(event, x, y, flags, param):
        nonlocal manual_pt, manual_bbox
        view["mouse"] = (x, y)
        Z = view["Z"]
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x, orig_y = canvas_to_orig(x, y)
            manual_pt = (orig_x, orig_y)
            manual_bbox = None          # will be recomputed in main loop
            print(f"Manual point set at {manual_pt}")
        elif event == cv2.EVENT_RBUTTONDOWN:
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
        for w in range(max(0, upto - 60), upto):
            cap.set(cv2.CAP_PROP_POS_FRAMES, w)
            ok_w, wf = cap.read()
            if ok_w:
                bg.apply(preprocess(wf, mode))

    warm(idx)

    # ---- frame counter for YOLO (run every N frames) ----
    yolo_counter = 0
    YOLO_INTERVAL = 2   # run YOLO every 2 frames to speed up UI

    while True:
        # -------- Frame seeking and reading --------
        if abs(idx - last_idx) != 1:
            warm(idx); ftrack.clear(); missed = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 1); playing = False; continue
        last_idx = idx

        # -------- Motion and LAB (needed for both detection and manual contour) --------
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = lab_frame[:, :, 0], lab_frame[:, :, 1], lab_frame[:, :, 2]
        motion = clean_motion(bg.apply(preprocess(frame, mode)))
        motion_roi = cv2.bitwise_and(motion, roi_mask)

        # -------- YOLO person masks (only if not in manual mode and every YOLO_INTERVAL frames) --------
        person_masks = []
        if manual_pt is None and (yolo_counter % YOLO_INTERVAL == 0 or yolo_counter == 0):
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

        # -------- Automatic detection (or manual override) --------
        ball = None
        if manual_pt is not None:
            # If manual_bbox is not yet computed, try to find contour now
            if manual_bbox is None:
                bbox = find_contour_bbox_from_click(frame, manual_pt, motion, l_channel,
                                                    search_radius=SEARCH_RADIUS)
                if bbox is not None:
                    manual_bbox = bbox
                    print(f"Contour found: {manual_bbox}")
                else:
                    # fallback to fixed box
                    manual_bbox = make_manual_box(manual_pt, W, H)
                    print("No contour found, using fixed box.")
            # Use the manual bbox
            ball = {"center": manual_pt, "radius": MANUAL_BOX_SIZE / 2,
                    "area": manual_bbox[2] * manual_bbox[3], "fill": 1.0, "aspect": 1.0,
                    "mean_l": 255.0, "chroma": 0.0, "bbox": manual_bbox}
        else:
            # Automatic detection
            cands = find_candidates(motion_roi, l_channel, a_channel, b_channel, min_fill,
                                    white_l_min, white_chroma_max, use_whiteness, person_masks)
            ball = choose(cands, predicted_point(), use_gate)
            if ball is not None:
                ftrack.append({"pt": ball["center"], "pred": False}); missed = 0
            else:
                missed += 1
                if missed > MAX_MISSED:
                    ftrack.clear()

        render = render_white(frame) if SAVE_IMAGE == "white" else frame
        label_box = None
        if ball is not None:
            if manual_pt is not None:
                label_box = ball["bbox"]   # already computed
            else:
                label_box = make_label_box(ball["center"], ball["bbox"], W, H)

        # -------- Full-size annotated panels --------
        vis = frame.copy()
        if roi_np is not None:
            cv2.polylines(vis, [roi_np], True, ROI_COLOR, 2)
        for mask in person_masks:
            col = np.zeros_like(vis); col[:, :, 2] = mask
            vis = cv2.addWeighted(vis, 1.0, col, 0.2, 0)
        rvis = render.copy()
        if label_box is not None:
            c = (0, 165, 255) if manual_pt is not None else (0, 255, 0)
            draw_candidate_box(vis, label_box, c, "Manual" if manual_pt is not None else "Candidate")
            draw_candidate_box(rvis, label_box, c)
        if manual_pt is not None:
            cv2.circle(vis, manual_pt, 4, (0, 165, 255), -1)
            cv2.circle(rvis, manual_pt, 4, (0, 165, 255), -1)

        # -------- Zoom panels --------
        Z = view["Z"]
        vw, vh = W / Z, H / Z
        cx = min(max(view["cx"], vw / 2.0), W - vw / 2.0)
        cy = min(max(view["cy"], vh / 2.0), H - vh / 2.0)
        view["cx"], view["cy"] = cx, cy
        view["x0"], view["y0"] = cx - vw / 2.0, cy - vh / 2.0
        zvis = zoom_panel(vis, view["x0"], view["y0"], Z, W, H)
        zrvis = zoom_panel(rvis, view["x0"], view["y0"], Z, W, H)

        # -------- HUD (drawn after zoom) --------
        cand_txt = "Manual" if manual_pt is not None else ("Candidate Detected" if ball is not None else "no candidate")
        cand_col = (0, 165, 255) if manual_pt is not None else ((0, 255, 0) if ball is not None else (0, 0, 255))
        cv2.putText(zvis, f"frame {idx}/{total-1}   zoom {Z:.2f}x", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(zvis, cand_txt, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cand_col, 2)
        cv2.putText(zvis, "S save  N skip  click=label  right-drag=pan  i/o zoom  0 reset",
                    (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)
        cv2.putText(zvis, f"Saved: {saved_count}   next id: ball_{next_index:06d}",
                    (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(zrvis, f"SAVES THIS ({SAVE_IMAGE})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(win, np.hstack([zvis, zrvis]))

        # -------- Keyboard handling (UI freeze fix: use longer waitKey) --------
        key = cv2.waitKey(20 if playing else 0) & 0xFF   # longer wait for GUI events

        if key in (27, ord('q')):
            break
        elif key == ord('s'):
            if label_box is None:
                print(f"frame {idx}: nothing to save.")
            elif manual_pt is None and is_duplicate_detection(label_box, idx, last_saved):
                print(f"frame {idx}: skipped (duplicate).")
            else:
                save_image(render, next_index)
                save_yolo_label(convert_bbox_to_yolo(label_box, W, H), next_index)
                last_saved = {"bbox": label_box, "frame": idx}
                saved_count += 1
                print(f"saved ball_{next_index:06d}.jpg + .txt")
                next_index += 1
            manual_pt = None
            manual_bbox = None
            if playing:
                idx = min(total - 1, idx + 1)
        elif key == ord('n'):
            manual_pt = None
            manual_bbox = None
            idx = min(total - 1, idx + 1)
        elif key == ord(' '):
            playing = not playing
        elif key == ord('d'):
            manual_pt = None
            manual_bbox = None
            idx = min(total - 1, idx + 1)
        elif key == ord('a'):
            manual_pt = None
            manual_bbox = None
            idx = max(0, idx - 1)
        elif key == ord(']'):
            manual_pt = None
            manual_bbox = None
            idx = min(total - 1, idx + 10)
        elif key == ord('['):
            manual_pt = None
            manual_bbox = None
            idx = max(0, idx - 10)
        elif key == ord('i'):
            zoom_to_cursor(min(ZOOM_MAX, view["Z"] + ZOOM_STEP))
        elif key == ord('o'):
            zoom_to_cursor(max(ZOOM_MIN, view["Z"] - ZOOM_STEP))
        elif key == ord('0'):
            view["Z"] = 1.0; view["cx"], view["cy"] = W / 2.0, H / 2.0
        elif key in (ord('+'), ord('=')):
            white_l_min = min(255, white_l_min + 1)
        elif key == ord('-'):
            white_l_min = max(100, white_l_min - 1)
        elif key == ord('.'):
            min_fill = min(0.95, min_fill + 0.05)
        elif key == ord(','):
            min_fill = max(0.05, min_fill - 0.05)
        elif key == ord('k'):
            use_whiteness = not use_whiteness
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
    cv2.destroyAllWindows()
    print(f"\nDone. Saved {saved_count} labeled images this session.")
    print(f"Images: {IMAGES_DIR}   Labels: {LABELS_DIR}")


if __name__ == "__main__":
    main()
