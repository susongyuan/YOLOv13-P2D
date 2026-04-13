"""
Estimate red-apple proportion from labeled apple boxes.

Input:
- images directory
- YOLO label directory (supports flat txt folder or train/val subfolders)

Output:
- per-box CSV
- dataset summary JSON (red/green/mixed ratio)

Example:
python estimate_red_apple_ratio.py \
  --images-dir MinneApple/detection/train/images \
  --labels-dir MinneApple/yolo/labels
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BoxColorStat:
    image: str
    box_id: int
    red_ratio: float
    green_ratio: float
    class_name: str


def yolo_txt_for_image(labels_dir: Path, stem: str) -> Path | None:
    direct = labels_dir / f"{stem}.txt"
    if direct.exists():
        return direct
    train = labels_dir / "train" / f"{stem}.txt"
    if train.exists():
        return train
    val = labels_dir / "val" / f"{stem}.txt"
    if val.exists():
        return val
    return None


def parse_yolo_txt(txt_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        s = line.strip().split()
        if len(s) < 5:
            continue
        _, cx, cy, bw, bh = s[:5]
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def classify_box_color(crop_bgr: np.ndarray, red_thr: float, green_thr: float) -> tuple[float, float, str]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    valid = (s > 30) & (v > 30)
    if int(valid.sum()) == 0:
        return 0.0, 0.0, "mixed"

    red_mask = ((h <= 12) | (h >= 165)) & valid
    green_mask = ((h >= 25) & (h <= 95)) & valid

    total = float(valid.sum())
    red_ratio = float(red_mask.sum()) / total
    green_ratio = float(green_mask.sum()) / total

    if red_ratio >= red_thr and red_ratio > green_ratio:
        cls = "red"
    elif green_ratio >= green_thr and green_ratio > red_ratio:
        cls = "green"
    else:
        cls = "mixed"
    return red_ratio, green_ratio, cls


def run(args: argparse.Namespace) -> None:
    images_dir = Path(args.images_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    stats: list[BoxColorStat] = []

    for i, img_path in enumerate(image_paths, start=1):
        txt = yolo_txt_for_image(labels_dir, img_path.stem)
        if txt is None:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = parse_yolo_txt(txt, w, h)

        for bidx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            crop = img[y1 : y2 + 1, x1 : x2 + 1]
            if crop.size == 0:
                continue
            red_ratio, green_ratio, cls = classify_box_color(
                crop,
                red_thr=args.red_threshold,
                green_thr=args.green_threshold,
            )
            stats.append(
                BoxColorStat(
                    image=img_path.name,
                    box_id=bidx,
                    red_ratio=red_ratio,
                    green_ratio=green_ratio,
                    class_name=cls,
                )
            )

        if i % 100 == 0 or i == len(image_paths):
            print(f"[{i}/{len(image_paths)}] processed")

    if not stats:
        raise SystemExit("No labeled boxes found to analyze.")

    csv_path = out_dir / "box_color_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "box_id", "red_ratio", "green_ratio", "class_name"],
        )
        writer.writeheader()
        for s in stats:
            writer.writerow(
                {
                    "image": s.image,
                    "box_id": s.box_id,
                    "red_ratio": f"{s.red_ratio:.6f}",
                    "green_ratio": f"{s.green_ratio:.6f}",
                    "class_name": s.class_name,
                }
            )

    total = len(stats)
    red_n = sum(1 for s in stats if s.class_name == "red")
    green_n = sum(1 for s in stats if s.class_name == "green")
    mixed_n = total - red_n - green_n

    summary = {
        "total_boxes": total,
        "red_boxes": red_n,
        "green_boxes": green_n,
        "mixed_boxes": mixed_n,
        "red_ratio": red_n / total,
        "green_ratio": green_n / total,
        "mixed_ratio": mixed_n / total,
        "red_threshold": args.red_threshold,
        "green_threshold": args.green_threshold,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "per_box_csv": str(csv_path),
    }

    summary_path = out_dir / "red_ratio_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate red-vs-green proportion from apple boxes.")
    p.add_argument(
        "--images-dir",
        type=str,
        default="MinneApple/detection/train/images",
        help="Directory containing orchard images.",
    )
    p.add_argument(
        "--labels-dir",
        type=str,
        default="MinneApple/yolo/labels",
        help="YOLO labels directory (flat or with train/val).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="MinneApple/color_ratio_results",
        help="Output directory.",
    )
    p.add_argument("--red-threshold", type=float, default=0.35, help="Red-pixel ratio threshold.")
    p.add_argument("--green-threshold", type=float, default=0.35, help="Green-pixel ratio threshold.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
