"""
Shared building blocks for the Lab + MOG2 autolabel / train / track pipeline.

Import this from autolabel.py and track_kalman.py so the RENDER used at label time
is byte-for-byte identical to the render used at inference time (critical for YOLO).
"""

import cv2
import numpy as np
import math
import json
import os

# ---- detector params (from backup.py) ----
MIN_AREA = 3
MAX_AREA = 1500
MIN_FILL = 0.35
MAX_ASPECT = 3.0
RED_MEAN_MIN = 135
W_RED, W_ROUND, W_DIST = 1.0, 1.0, 1.5
GATE_RADIUS_PX = 120
MAX_MISSED = 10
MAX_COAST = 6

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ROI_COLOR = (255, 120, 0)


# ===================== renders (train == inference) =====================

def lab_a(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1]


def preprocess_L(frame):
    return _clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0])


def render_lab(frame):
    """Lab-a JET heatmap (the 'warm=red' image you see the ball in)."""
    a = lab_a(frame)
    disp = np.clip((a.astype(np.int16) - 110) * 3, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(disp, cv2.COLORMAP_JET)


def render_fused(frame, motion):
    """3-channel fusion: B=gray, G=MOG2 motion, R=Lab-a. Puts colour + motion in one image."""
    a = lab_a(frame)
    a_disp = np.clip((a.astype(np.int16) - 110) * 3, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m = motion if motion is not None else np.zeros_like(gray)
    return cv2.merge([gray, m, a_disp])


def get_render(frame, motion, mode="lab"):
    return render_fused(frame, motion) if mode == "fused" else render_lab(frame)


# ===================== MOG2 =====================

def make_bg():
    return cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)


def warm_bg(cap, upto, span=60):
    """Rebuild MOG2 and warm it on frames [upto-span, upto)."""
    bg = make_bg()
    for w in range(max(0, upto - span), upto):
        cap.set(cv2.CAP_PROP_POS_FRAMES, w)
        ok, f = cap.read()
        if ok:
            bg.apply(preprocess_L(f))
    return bg


def clean_motion(raw):
    m = cv2.medianBlur(raw, 3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel, iterations=1)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


# ===================== candidate detection =====================

def find_candidates(motion_mask, a_channel, min_fill=MIN_FILL,
                    red_mean_min=RED_MEAN_MIN, use_redness=True):
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
        cands.append({"center": (int(cx), int(cy)), "radius": float(r), "area": area,
                      "fill": fill, "aspect": aspect, "mean_a": mean_a})
    return cands


def choose(cands, predicted, use_gate=True):
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


# ===================== ROI =====================

def select_roi(frame):
    pts = []
    win = "Draw ROI | click points  z=undo  c=clear  ENTER=confirm  ESC=whole frame"

    def redraw():
        d = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(d, p, 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(d, pts[i - 1], p, (0, 255, 255), 2)
        if len(pts) >= 3:
            cv2.line(d, pts[-1], pts[0], (0, 255, 255), 1)
            ov = d.copy(); cv2.fillPoly(ov, [np.array(pts, np.int32)], ROI_COLOR)
            d = cv2.addWeighted(ov, 0.2, d, 0.8, 0)
        cv2.putText(d, f"{len(pts)} pts - ENTER", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, d)

    def on_mouse(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y)); redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    res = None
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 13 and len(pts) >= 3:
            res = pts[:]; break
        if k == 27:
            break
        if k == ord('z') and pts:
            pts.pop(); redraw()
        if k == ord('c'):
            pts.clear(); redraw()
    cv2.destroyWindow(win)
    return res


def build_roi_mask(polygon, shape):
    mask = np.zeros(shape[:2], np.uint8)
    if polygon and len(polygon) >= 3:
        cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
    else:
        mask[:] = 255
    return mask


def load_roi(path, frame):
    if not path or not os.path.exists(path):
        return None
    try:
        data = json.load(open(path))
        size = data.get("frame_size")
        if size and (size[0] != frame.shape[1] or size[1] != frame.shape[0]):
            return None
        poly = [tuple(p) for p in data.get("polygon", [])]
        return poly if len(poly) >= 3 else None
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None


def save_roi(path, polygon, frame):
    json.dump({"polygon": polygon, "frame_size": [frame.shape[1], frame.shape[0]]}, open(path, "w"))


# ===================== Kalman + parabola =====================

class BallKalman:
    """Constant-velocity 2D Kalman filter for smoothing / short prediction."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.ready = False

    def init(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.ready = True

    def predict(self):
        p = self.kf.predict()
        return float(p[0]), float(p[1])

    def correct(self, x, y):
        if not self.ready:
            self.init(x, y)
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))


def fit_parabola(frames, xs, ys, iters=3, k=2.5):
    """x ~ linear in t, y ~ quadratic in t, with iterative outlier rejection.
    Returns (cx, cy, keep_mask) or (None, None, None) if too few points."""
    t = np.asarray(frames, float)
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    if len(t) < 4:
        return None, None, None
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        if keep.sum() < 4:
            break
        cy = np.polyfit(t[keep], y[keep], 2)
        resid = y - np.polyval(cy, t)
        s = resid[keep].std() or 1.0
        nk = np.abs(resid) < k * s
        if (nk == keep).all():
            keep = nk; break
        keep = nk
    if keep.sum() < 4:
        return None, None, None
    cy = np.polyfit(t[keep], y[keep], 2)
    cx = np.polyfit(t[keep], x[keep], 1)
    return cx, cy, keep


def eval_parabola(cx, cy, frames):
    t = np.asarray(frames, float)
    return np.polyval(cx, t), np.polyval(cy, t)