"""
Debug: SEE exactly what YOLO is fed and what it returns.

Right panel = the render passed to model.predict() with EVERY YOLO box + confidence
drawn on it. Left panel = the normal frame. If the right panel is the blue Lab
heatmap, the model is running on Lab images (it is). The big number is the best
confidence this frame - if it stays near 0, the model didn't learn (need more data).

  python debug_infer.py --video v.mp4 --weights best.pt --render lab

Controls: space play/pause  d/a step  ]/[ jump  q quit
"""

import argparse
import cv2
import numpy as np
import pipeline_common as pc
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--render", choices=["lab", "fused"], default="lab")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="low so you can SEE the model's best guess even if weak")
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg = pc.make_bg()
    idx, playing, last = 0, False, -999

    win = "DEBUG  left=normal frame   right=WHAT YOLO SEES  | space d/a ]/[ q"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if idx != last + 1:
            bg = pc.warm_bg(cap, idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 1); playing = False; continue
        last = idx

        motion = pc.clean_motion(bg.apply(pc.preprocess_L(frame)))
        render = pc.get_render(frame, motion, args.render)   # exact model input

        res = model.predict(render, imgsz=args.imgsz, conf=args.conf,
                            classes=[0], verbose=False)[0]

        model_in = render.copy()
        best = 0.0
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cf = float(b.conf[0])
            best = max(best, cf)
            col = (0, 255, 0) if cf >= 0.25 else (0, 165, 255)
            cv2.rectangle(model_in, (x1, y1), (x2, y2), col, 2)
            cv2.putText(model_in, f"{cf:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        cv2.putText(model_in, f"render={args.render}  boxes={len(res.boxes)}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(model_in, f"best conf: {best:.3f}", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if best >= 0.25 else (0, 0, 255), 2)

        left = frame.copy()
        cv2.putText(left, f"frame {idx}/{total-1}  (normal frame - display only)",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(win, np.hstack([left, model_in]))
        key = cv2.waitKey(20 if playing else 0) & 0xFF
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
        elif playing:
            idx = min(total - 1, idx + 1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()