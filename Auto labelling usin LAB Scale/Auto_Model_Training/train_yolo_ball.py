"""
Train YOLOv8s to detect the ball.

  python train_yolo_ball.py --data ball_ds/data.yaml --epochs 100 --imgsz 1280

Small-ball settings that matter:
  * imgsz 1280 (or higher) so a few-pixel ball survives; small objects are YOLO's
    weak spot, and resolution is the biggest lever.
  * start from yolov8s.pt (COCO weights) - transfers general object features.
  * mosaic/scale augmentation helps the ball appear at varied sizes.
Best weights land in runs/detect/<name>/weights/best.pt -> use that for tracking.
"""

import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--name", default="ball_yolov8s")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=30,
        scale=0.5,          # size augmentation
        mosaic=1.0,
        close_mosaic=10,
        cos_lr=True,
    )
    print("Best weights: runs/detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()