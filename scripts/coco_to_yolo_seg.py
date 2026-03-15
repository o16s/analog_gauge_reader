"""Convert COCO instance segmentation (with RLE masks) to YOLOv8 segmentation format.

Usage:
    python scripts/coco_to_yolo_seg.py \
        --coco "/path/to/analog gauges.coco/train/_annotations.coco.json" \
        --out  data/segmentation

Produces:
    data/segmentation/
        train/images/  (80%)
        train/labels/
        val/images/    (20%)
        val/labels/
        data.yaml
"""
import argparse
import json
import os
import random
import shutil

import numpy as np


def rle_to_mask(rle, h, w):
    """Decode COCO compressed RLE to a binary mask."""
    counts = rle["counts"]
    if isinstance(counts, str):
        # Decode compressed RLE string
        decoded = []
        i = 0
        while i < len(counts):
            x = 0
            shift = 0
            more = True
            while more:
                c = ord(counts[i]) - 48
                i += 1
                x |= (c & 0x1F) << shift
                more = c >= 0x20
                shift += 5
            if x & 1:
                x = -(x >> 1)
            else:
                x = x >> 1
            decoded.append(x)
        counts = decoded

    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        mask[pos : pos + c] = val
        pos += c
        val = 1 - val
    return mask.reshape((h, w), order="F")


def mask_to_polygon(mask):
    """Extract the largest contour from a binary mask as a polygon."""
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return contour.squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="Path to _annotations.coco.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.coco) as f:
        coco = json.load(f)

    images_dir = os.path.dirname(args.coco)
    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Determine train/val split
    image_ids = sorted(anns_by_img.keys())
    random.seed(args.seed)
    random.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * args.val_split))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    # Create output dirs
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, split, "labels"), exist_ok=True)

    converted = 0
    skipped = 0

    for img_id, anns in anns_by_img.items():
        img_info = img_map[img_id]
        w, h = img_info["width"], img_info["height"]
        fname = img_info["file_name"]
        stem = os.path.splitext(fname)[0]
        split = "val" if img_id in val_ids else "train"

        # Copy image
        src = os.path.join(images_dir, fname)
        if not os.path.isfile(src):
            skipped += 1
            continue
        shutil.copy2(src, os.path.join(args.out, split, "images", fname))

        # Convert annotations to YOLO format
        lines = []
        for ann in anns:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                # RLE → mask → polygon
                mask = rle_to_mask(seg, h, w)
                poly = mask_to_polygon(mask)
                if poly is None:
                    continue
                # Normalize to 0-1
                coords = []
                for x, y in poly:
                    coords.append(f"{x / w:.6f}")
                    coords.append(f"{y / h:.6f}")
            elif isinstance(seg, list) and seg:
                # Already polygon format
                flat = seg[0]
                coords = []
                for i in range(0, len(flat), 2):
                    coords.append(f"{flat[i] / w:.6f}")
                    coords.append(f"{flat[i + 1] / h:.6f}")
            else:
                continue

            # Class 0 (single class: needle)
            lines.append("0 " + " ".join(coords))

        label_path = os.path.join(args.out, split, "labels", stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        converted += 1

    # Write data.yaml
    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.abspath(os.path.join(args.out, 'train', 'images'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(args.out, 'val', 'images'))}\n")
        f.write("nc: 1\n")
        f.write("names: ['Gauge Needle']\n")

    n_train = len(train_ids)
    n_val = len(val_ids)
    print(f"Converted {converted} images ({n_train} train, {n_val} val), skipped {skipped}")
    print(f"Output: {os.path.abspath(args.out)}")
    print(f"Config: {os.path.abspath(yaml_path)}")


if __name__ == "__main__":
    main()
