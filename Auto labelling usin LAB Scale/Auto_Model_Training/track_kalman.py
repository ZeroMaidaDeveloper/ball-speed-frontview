"""
Clean YOLO ball detector -> RECTANGLE around the ball. No Kalman, no parabola.

Each frame: build the render (same as training), run YOLO, take the highest-conf
box inside the ROI, draw it as a rectangle with its confidence, and write the box
corners to CSV. That's it.

  python track_boxes.py --video v.mp4 --weights best.pt --render lab --conf 0.25 \
         --out boxes.mp4 --csv boxes.csv

Tip: box GREEN = conf >= 0.25 (trusted), ORANGE = weak guess below 0.25.
If every box is orange / jumps around, the model is undertrained - that's a data
problem, not a drawing problem (see the printed 'best conf overall').
"""

import argparse
import csv
import os
import cv2
import numpy as np
import pipeline_common as pc
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--render", choices=["lab", "fused"], default="lab")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="draw boxes at/above this. Lower (e.g. 0.01) only to inspect a weak model.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default="boxes.mp4")
    ap.add_argument("--csv", default="boxes.csv")
    ap.add_argument("--all", action="store_true", help="also draw non-best boxes faintly")
    args = ap.parse_args()

    roi_path = os.path.splitext(args.video)[0] + "_roi.json"
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, frame0 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    poly = pc.load_roi(roi_path, frame0)
    roi_mask = pc.build_roi_mask(poly, frame0.shape)
    roi_np = np.array(poly, np.int32) if poly else None

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    bg = pc.make_bg()      # only needed for fused render
    rows = []
    fi = 0
    n_hit = 0
    best_overall = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        motion = pc.clean_motion(bg.apply(pc.preprocess_L(frame))) if args.render == "fused" else None
        render = pc.get_render(frame, motion, args.render)

        res = model.predict(render, imgsz=args.imgsz, conf=args.conf,
                            classes=[0], verbose=False)[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cf = float(b.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if roi_mask[min(max(cy, 0), H - 1), min(max(cx, 0), W - 1)] > 0:
                boxes.append((x1, y1, x2, y2, cf))
        boxes.sort(key=lambda b: b[4], reverse=True)

        vis = frame.copy()
        if roi_np is not None:
            cv2.polylines(vis, [roi_np], True, pc.ROI_COLOR, 2)

        if boxes:
            if args.all:
                for (x1, y1, x2, y2, cf) in boxes[1:]:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 1)
            x1, y1, x2, y2, cf = boxes[0]
            best_overall = max(best_overall, cf)
            n_hit += 1
            col = (0, 255, 0) if cf >= 0.25 else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"ball {cf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            rows.append([fi, x1, y1, x2, y2, round(cf, 3)])
        else:
            rows.append([fi, "", "", "", "", 0.0])

        writer.write(vis)
        fi += 1

    cap.release()
    writer.release()

    with open(args.csv, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["frame", "x1", "y1", "x2", "y2", "conf"])
        wtr.writerows(rows)

    print(f"boxes drawn: {n_hit}/{fi} frames   best conf overall: {best_overall:.3f}")
    if best_overall < 0.25:
        print(">> No confident detections. The model is undertrained (data problem), "
              "not a drawing problem. Label more frames and retrain.")
    print(f"Saved video -> {args.out}")
    print(f"Saved boxes -> {args.csv}")


if __name__ == "__main__":
    main()