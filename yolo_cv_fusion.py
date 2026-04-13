"""
Pure YOLO counting/evaluation for MinneApple.

This script supports two YOLO sources:
1) Real model inference via Ultralytics (--yolo-model).
2) Reading existing YOLO txt labels as a stand-in (--yolo-labels-dir).
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
class YoloDetection:
    bbox: tuple[int, int, int, int]
    conf: float


def gt_count_from_mask(mask_path: Path) -> int:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    obj_ids = np.unique(mask)
    return int(np.sum(obj_ids != 0))


def read_yolo_txt_dets(txt_path: Path, img_w: int, img_h: int) -> list[YoloDetection]:
    if not txt_path.exists():
        return []
    dets: list[YoloDetection] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = parts[:5]
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        x1 = int((cx - bw / 2.0) * img_w)
        y1 = int((cy - bh / 2.0) * img_h)
        x2 = int((cx + bw / 2.0) * img_w)
        y2 = int((cy + bh / 2.0) * img_h)
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append(YoloDetection((x1, y1, x2, y2), 1.0))
    return dets


def infer_yolo_ultralytics(
    model_path: str, image_path: Path, conf: float = 0.25
) -> list[YoloDetection]:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_path)
    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    dets: list[YoloDetection] = []
    if result.boxes is None:
        return dets
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for box, c in zip(xyxy, confs):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        dets.append(YoloDetection((x1, y1, x2, y2), float(c)))
    return dets


def draw_overlay(img_bgr: np.ndarray, yolo_dets: list[YoloDetection], title: str) -> np.ndarray:
    out = img_bgr.copy()
    for det in yolo_dets:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return out


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    images_dir = root / "detection" / "train" / "images"
    masks_dir = root / "detection" / "train" / "masks"
    yolo_labels_dir = Path(args.yolo_labels_dir).resolve() if args.yolo_labels_dir else None

    out_dir = root / args.out_dir
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"No images found under {images_dir}")

    use_model = bool(args.yolo_model)
    if not use_model and yolo_labels_dir is None:
        raise SystemExit("Provide either --yolo-model or --yolo-labels-dir")

    rows: list[dict[str, str | int | float]] = []
    err_yolo: list[float] = []
    sq_yolo: list[float] = []

    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        if use_model:
            yolo_dets = infer_yolo_ultralytics(
                model_path=args.yolo_model,
                image_path=img_path,
                conf=args.yolo_conf,
            )
        else:
            label_txt = yolo_labels_dir / f"{img_path.stem}.txt"
            if not label_txt.exists():
                train_txt = yolo_labels_dir / "train" / f"{img_path.stem}.txt"
                val_txt = yolo_labels_dir / "val" / f"{img_path.stem}.txt"
                if train_txt.exists():
                    label_txt = train_txt
                elif val_txt.exists():
                    label_txt = val_txt
            yolo_dets = read_yolo_txt_dets(label_txt, w, h)

        gt = gt_count_from_mask(masks_dir / img_path.name)

        yolo_count = len(yolo_dets)
        ey = abs(yolo_count - gt)
        err_yolo.append(float(ey))
        sq_yolo.append(float((yolo_count - gt) ** 2))

        rows.append(
            {
                "image": img_path.name,
                "gt_count": gt,
                "yolo_count": yolo_count,
                "abs_err_yolo": ey,
            }
        )

        if args.save_vis and idx <= args.max_vis:
            title = f"GT={gt} YOLO={yolo_count}"
            vis = draw_overlay(img, yolo_dets, title)
            cv2.imwrite(str(vis_dir / img_path.name), vis)

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] processed")

    def _mae(v: list[float]) -> float:
        return float(np.mean(v)) if v else -1.0

    def _rmse(v: list[float]) -> float:
        return float(np.sqrt(np.mean(v))) if v else -1.0

    summary = {
        "images": len(rows),
        "mae_yolo": _mae(err_yolo),
        "rmse_yolo": _rmse(sq_yolo),
    }

    csv_path = out_dir / "yolo_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "gt_count",
                "yolo_count",
                "abs_err_yolo",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "yolo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure YOLO counting/evaluation for MinneApple.")
    p.add_argument(
        "--root",
        type=str,
        default="MinneApple",
        help="Dataset root directory (default: ./MinneApple).",
    )
    p.add_argument(
        "--yolo-model",
        type=str,
        default="",
        help="Path to YOLO model weights (.pt). Optional if --yolo-labels-dir is set.",
    )
    p.add_argument(
        "--yolo-labels-dir",
        type=str,
        default="MinneApple/yolo/labels",
        help="Directory with YOLO txt labels. Useful as stand-in if no model inference.",
    )
    p.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    p.add_argument("--out-dir", type=str, default="yolo_results", help="Output directory")
    p.add_argument("--save-vis", action="store_true", help="Save YOLO overlay images.")
    p.add_argument("--max-vis", type=int, default=60, help="Max overlay images to save.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
