"""
Fix an empty (or lopsided) train/val split in a YOLO dataset.

Pools every image already under <ds>/images/{train,val}, then redistributes with a
GUARANTEED non-empty val set (at least max(1, round(val*N)) images). Labels move with
their images; missing labels are created empty (negatives).

Run after an autolabel session and before training:
  python split_dataset.py --ds ball_ds --val 0.2
"""

import argparse
import glob
import os
import random
import shutil

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="dataset dir (has images/ and labels/)")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img_root = os.path.join(args.ds, "images")
    lbl_root = os.path.join(args.ds, "labels")

    # collect every image currently in train/ or val/
    items = {}   # stem -> (current_sub, img_path, ext)
    for sub in ("train", "val"):
        for p in glob.glob(os.path.join(img_root, sub, "*")):
            ext = os.path.splitext(p)[1].lower()
            if ext in IMG_EXTS:
                stem = os.path.splitext(os.path.basename(p))[0]
                items[stem] = (sub, p, ext)

    n = len(items)
    if n == 0:
        print("No images found - nothing to split.")
        return

    stems = list(items)
    random.Random(args.seed).shuffle(stems)
    n_val = 0 if n < 2 else max(1, round(n * args.val))
    val_stems = set(stems[:n_val])

    for sub in ("train", "val"):
        os.makedirs(os.path.join(img_root, sub), exist_ok=True)
        os.makedirs(os.path.join(lbl_root, sub), exist_ok=True)

    moved = 0
    for stem, (cur_sub, img_path, ext) in items.items():
        tgt = "val" if stem in val_stems else "train"
        # image
        dst_img = os.path.join(img_root, tgt, stem + ext)
        if os.path.abspath(dst_img) != os.path.abspath(img_path):
            shutil.move(img_path, dst_img)
            moved += 1
        # label (create empty if the source has none)
        src_lbl = os.path.join(lbl_root, cur_sub, stem + ".txt")
        dst_lbl = os.path.join(lbl_root, tgt, stem + ".txt")
        if os.path.exists(src_lbl):
            if os.path.abspath(src_lbl) != os.path.abspath(dst_lbl):
                shutil.move(src_lbl, dst_lbl)
        elif not os.path.exists(dst_lbl):
            open(dst_lbl, "w").close()

    with open(os.path.join(args.ds, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.ds)}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n  0: ball\n")

    n_train = n - n_val
    print(f"Split {n} images -> train={n_train}, val={n_val} (moved {moved}).")
    if n < 50:
        print(f"WARNING: only {n} images total. This will TRAIN but massively overfit "
              f"and won't detect reliably. Aim for a few hundred across several deliveries "
              f"(use bulk auto-save 'A' in autolabel.py to gather them fast).")


if __name__ == "__main__":
    main()