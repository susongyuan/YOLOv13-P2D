from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ULTRA_ROOT = ROOT / "yolov13-main"
if ULTRA_ROOT.exists() and str(ULTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRA_ROOT))

from ultralytics import YOLO


def _safe_get(d: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    return float(default)


def _extract_metrics(res) -> dict:
    d = {}
    if hasattr(res, "results_dict") and isinstance(res.results_dict, dict):
        d.update(res.results_dict)
    if hasattr(res, "speed") and isinstance(res.speed, dict):
        d["_speed_infer_ms"] = float(res.speed.get("inference", 0.0))
    return {
        "precision": _safe_get(d, "metrics/precision(B)", "metrics/precision"),
        "recall": _safe_get(d, "metrics/recall(B)", "metrics/recall"),
        "map50": _safe_get(d, "metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": _safe_get(d, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "inference_ms": _safe_get(d, "_speed_infer_ms"),
    }


def _run_val(model_path: str, data_yaml: str, nms_type: str) -> dict:
    model = YOLO(model_path)
    res = model.val(data=data_yaml, plots=False, verbose=False, nms_type=nms_type)
    return _extract_metrics(res)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict], title: str) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")


def _plot_map(path: Path, rows: list[dict], key: str, title: str) -> None:
    names = [r["model_name"] for r in rows]
    values = [float(r[key]) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(key)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare apple detection models.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model paths (.pt or .yaml), e.g. best.pt yolov13-apple-p2.yaml",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Optional model names aligned with --models.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=r"C:\Users\Administrator\Desktop\project\3\MinneApple\yolo\data.yaml",
        help="Full validation dataset yaml.",
    )
    parser.add_argument(
        "--hard-data",
        type=str,
        default=r"C:\Users\Administrator\Desktop\project\3\MinneApple\yolo\hard_subset\data_hard.yaml",
        help="Hard subset dataset yaml.",
    )
    parser.add_argument(
        "--nms-type",
        type=str,
        default="diou",
        choices=["iou", "diou"],
        help="NMS type used during validation.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"C:\Users\Administrator\Desktop\project\3\comparison_results",
        help="Output directory for tables and plots.",
    )
    args = parser.parse_args()

    model_names = args.names if args.names and len(args.names) == len(args.models) else [Path(m).stem for m in args.models]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_rows: list[dict] = []
    hard_rows: list[dict] = []
    hard_data_exists = Path(args.hard_data).exists()

    for model_name, model_path in zip(model_names, args.models):
        full = _run_val(model_path, args.data, args.nms_type)
        full_rows.append(
            {
                "model_name": model_name,
                "model_path": model_path,
                "precision": round(full["precision"], 4),
                "recall": round(full["recall"], 4),
                "map50": round(full["map50"], 4),
                "map50_95": round(full["map50_95"], 4),
                "inference_ms": round(full["inference_ms"], 3),
            }
        )

        if hard_data_exists:
            hard = _run_val(model_path, args.hard_data, args.nms_type)
            hard_rows.append(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "precision": round(hard["precision"], 4),
                    "recall": round(hard["recall"], 4),
                    "map50": round(hard["map50"], 4),
                    "map50_95": round(hard["map50_95"], 4),
                    "inference_ms": round(hard["inference_ms"], 3),
                }
            )

    _write_csv(out_dir / "comparison_full.csv", full_rows)
    _write_markdown(out_dir / "comparison_full.md", full_rows, "Apple Model Comparison (Full Validation)")
    _plot_map(out_dir / "comparison_full_map50_95.png", full_rows, "map50_95", "mAP50-95 (Full Validation)")

    if hard_rows:
        _write_csv(out_dir / "comparison_hard.csv", hard_rows)
        _write_markdown(out_dir / "comparison_hard.md", hard_rows, "Apple Model Comparison (Hard Subset)")
        _plot_map(out_dir / "comparison_hard_map50_95.png", hard_rows, "map50_95", "mAP50-95 (Hard Subset)")

    print(f"Saved comparison outputs to: {out_dir}")
    print(f"Full validation models: {len(full_rows)}")
    print(f"Hard subset enabled: {bool(hard_rows)}")


if __name__ == "__main__":
    main()
