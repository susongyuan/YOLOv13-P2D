from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import yaml


@dataclass
class HardRule:
    small_area_thr: float = 0.0015
    low_light_thr: float = 65.0
    backlight_thr: float = 190.0
    dense_count_thr: int = 12
    overlap_iou_thr: float = 0.15
    overlap_pairs_thr: int = 2


def _xywhn_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _read_yolo_label(label_path: Path) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x, y, w, h = parts[:5]
            boxes.append((float(x), float(y), float(w), float(h)))
    return boxes


def _iter_val_images(dataset_root: Path, val_spec: str) -> Iterable[Path]:
    val_path = Path(val_spec)
    if not val_path.is_absolute():
        val_path = dataset_root / val_path

    if val_path.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in sorted(val_path.glob(ext)):
                yield p.resolve()
        return

    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        with val_path.open("r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                pp = Path(p)
                if not pp.is_absolute():
                    pp = (dataset_root / pp).resolve()
                yield pp
        return

    raise FileNotFoundError(f"Unsupported val specification: {val_spec}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard validation subset for orchard apple detection.")
    parser.add_argument(
        "--data",
        type=str,
        default=r"C:\Users\Administrator\Desktop\project\3\MinneApple\yolo\data.yaml",
        help="Path to dataset yaml.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"C:\Users\Administrator\Desktop\project\3\MinneApple\yolo\hard_subset",
        help="Output folder for hard subset metadata.",
    )
    args = parser.parse_args()

    rules = HardRule()
    data_path = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = Path(data_cfg["path"]).resolve()
    val_spec = data_cfg["val"]

    hard_list_file = out_dir / "hard_val.txt"
    stats_file = out_dir / "hard_subset_stats.csv"
    hard_yaml_file = out_dir / "data_hard.yaml"

    rows = []
    hard_images: list[str] = []

    for img_path in _iter_val_images(dataset_root, val_spec):
        rel = img_path.relative_to(dataset_root).as_posix()
        label_path = dataset_root / "labels" / "val" / f"{img_path.stem}.txt"
        boxes = _read_yolo_label(label_path)

        obj_count = len(boxes)
        areas = [w * h for _, _, w, h in boxes]
        small_ratio = (sum(a < rules.small_area_thr for a in areas) / obj_count) if obj_count else 0.0

        xyxy = [_xywhn_to_xyxy(b) for b in boxes]
        overlap_pairs = 0
        for i in range(len(xyxy)):
            for j in range(i + 1, len(xyxy)):
                if _iou(xyxy[i], xyxy[j]) > rules.overlap_iou_thr:
                    overlap_pairs += 1

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        brightness = float(img.mean()) if img is not None else 127.5

        cond_small = small_ratio >= 0.35
        cond_dense = obj_count >= rules.dense_count_thr
        cond_overlap = overlap_pairs >= rules.overlap_pairs_thr
        cond_light = brightness <= rules.low_light_thr or brightness >= rules.backlight_thr
        # Combined difficulty score for ranking fallback.
        score = 0.0
        score += min(1.0, small_ratio / 0.5) * 1.2
        score += min(1.0, obj_count / 20.0) * 1.1
        score += min(1.0, overlap_pairs / 8.0) * 1.2
        score += (1.0 if brightness <= rules.low_light_thr or brightness >= rules.backlight_thr else 0.0) * 0.9
        is_hard = cond_small or cond_dense or cond_overlap or cond_light

        rows.append(
            {
                "abs_image": str(img_path.as_posix()),
                "image": rel,
                "objects": obj_count,
                "small_ratio": round(small_ratio, 4),
                "overlap_pairs": overlap_pairs,
                "brightness": round(brightness, 2),
                "difficulty_score": round(score, 4),
                "hard_small": int(cond_small),
                "hard_dense": int(cond_dense),
                "hard_overlap": int(cond_overlap),
                "hard_light": int(cond_light),
                "is_hard": int(is_hard),
            }
        )

    prelim_hard = [r for r in rows if r["is_hard"] == 1]
    if rows and (len(prelim_hard) / len(rows)) > 0.7:
        # If hard subset is too large, keep the top 40% most difficult samples for better discrimination.
        topk = max(20, int(len(rows) * 0.4))
        ranked = sorted(rows, key=lambda x: float(x["difficulty_score"]), reverse=True)
        hard_set = {r["abs_image"] for r in ranked[:topk]}
        for r in rows:
            r["is_hard"] = int(r["abs_image"] in hard_set)

    hard_images = [r["abs_image"] for r in rows if r["is_hard"] == 1]

    with hard_list_file.open("w", encoding="utf-8") as f:
        for p in hard_images:
            f.write(p + "\n")

    with stats_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else ["abs_image", "image", "difficulty_score", "is_hard"],
        )
        writer.writeheader()
        writer.writerows(rows)

    hard_cfg = dict(data_cfg)
    hard_cfg["val"] = str(hard_list_file.as_posix())
    with hard_yaml_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(hard_cfg, f, sort_keys=False, allow_unicode=False)

    print(f"Saved hard subset list: {hard_list_file}")
    print(f"Saved hard subset stats: {stats_file}")
    print(f"Saved hard subset yaml:  {hard_yaml_file}")
    print(f"Hard images: {len(hard_images)} / {len(rows)}")


if __name__ == "__main__":
    main()
