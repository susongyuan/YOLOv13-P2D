from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent
ULTRA_ROOT = ROOT / "yolov13-main"
if ULTRA_ROOT.exists() and str(ULTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRA_ROOT))

from ultralytics import YOLO

from run_ablation_suite import _apply_common_train_final


def _extract(res) -> dict:
    rd = getattr(res, "results_dict", {}) or {}
    speed = getattr(res, "speed", {}) or {}
    inference_ms = float(speed.get("inference", 0.0))
    return {
        "precision": float(rd.get("metrics/precision(B)", rd.get("metrics/precision", 0.0))),
        "recall": float(rd.get("metrics/recall(B)", rd.get("metrics/recall", 0.0))),
        "map50": float(rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", 0.0))),
        "map50_95": float(rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", 0.0))),
        "inference_ms": inference_ms,
        "fps": (1000.0 / inference_ms) if inference_ms > 0 else 0.0,
    }


def _find_latest_weights(project_dir: Path, model_name: str) -> tuple[Path, Path]:
    candidates = sorted(
        [p for p in project_dir.glob(f"{model_name}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in candidates:
        best = d / "weights" / "best.pt"
        last = d / "weights" / "last.pt"
        if best.exists() or last.exists():
            return best, last
    return project_dir / model_name / "weights" / "best.pt", project_dir / model_name / "weights" / "last.pt"


def _read_summary_cache(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r.get(key, ""): r for r in rows if r.get(key)}


def _evaluate_yolo_baseline(common: dict, item: dict, best_weight: Path, last_weight: Path, eval_device: str | None = None) -> dict:
    eval_weight = best_weight if best_weight.exists() else last_weight
    if not eval_weight.exists():
        raise FileNotFoundError(f"No available checkpoint for baseline eval: {best_weight} / {last_weight}")
    model = YOLO(str(eval_weight))
    kwargs = {"data": common["data"]}
    if eval_device is not None:
        kwargs["device"] = eval_device
    val = model.val(**kwargs)
    m = _extract(val)
    m["best_weight"] = str(best_weight) if best_weight.exists() else ""
    m["last_weight"] = str(last_weight) if last_weight.exists() else ""
    m.update({"model_name": item["name"], "standard_repro": bool(item.get("standard", False))})
    return m


def run_yolo_baseline(
    common: dict,
    item: dict,
    batch_override: int | None = None,
    imgsz_override: int | None = None,
    workers_override: int | None = None,
    resume: bool = False,
    skip_completed: bool = False,
    existing_summary: dict[str, dict] | None = None,
) -> dict:
    project_dir = Path(common["project"]).resolve()
    best_weight, last_weight = _find_latest_weights(project_dir, item["name"])

    if skip_completed and best_weight.exists():
        print(f"[SKIP] {item['name']}: existing best checkpoint found, evaluating only.")
        if existing_summary and item["name"] in existing_summary:
            row = dict(existing_summary[item["name"]])
            row["model_name"] = item["name"]
            row["best_weight"] = str(best_weight) if best_weight.exists() else ""
            row["last_weight"] = str(last_weight) if last_weight.exists() else ""
            return row
        return _evaluate_yolo_baseline(common, item, best_weight, last_weight, eval_device="cpu")

    if resume and last_weight.exists():
        print(f"[RESUME] {item['name']}: resuming from {last_weight}")
        model = YOLO(str(last_weight))
        model.train(resume=True)
        best_weight, last_weight = _find_latest_weights(project_dir, item["name"])
        return _evaluate_yolo_baseline(common, item, best_weight, last_weight)

    model = YOLO(item["model"])
    train_args = dict(
        data=common["data"],
        epochs=int(common["epochs"]),
        imgsz=int(imgsz_override if imgsz_override is not None else common["imgsz"]),
        batch=int(batch_override if batch_override is not None else common["batch"]),
        project=common["project"],
        name=item["name"],
        device=str(common["device"]),
        save=True,
    )
    if workers_override is not None:
        train_args["workers"] = int(workers_override)
    elif "workers" in common:
        train_args["workers"] = int(common["workers"])

    _apply_common_train_final(train_args, common, batch_override, imgsz_override, workers_override)

    model.train(**train_args)
    best_weight, last_weight = _find_latest_weights(project_dir, item["name"])
    return _evaluate_yolo_baseline(common, item, best_weight, last_weight)


def run_fasterrcnn_baseline(common: dict, item: dict, batch_override: int | None = None, skip_completed: bool = False) -> dict:
    out_dir = Path(common["project"]).resolve() / item["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    weight_path = out_dir / "fasterrcnn_resnet50_final.pth"
    if skip_completed and metrics_path.exists() and weight_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["best_weight"] = str(weight_path)
        metrics["last_weight"] = str(weight_path)
        metrics.update({"model_name": item["name"], "standard_repro": bool(item.get("standard", False))})
        print(f"[SKIP] {item['name']}: existing metrics/weights found.")
        return metrics

    cmd = [
        "python",
        str(Path(__file__).resolve().parent / "train_fasterrcnn_resnet50.py"),
        "--data",
        str(common["data"]),
        "--epochs",
        str(common["epochs"]),
        "--batch",
        str(max(1, int((batch_override if batch_override is not None else common["batch"])) // 2)),
        "--device",
        str(common["device"]),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["best_weight"] = str(out_dir / "fasterrcnn_resnet50_final.pth")
    metrics["last_weight"] = metrics["best_weight"]
    metrics.update({"model_name": item["name"], "standard_repro": bool(item.get("standard", False))})
    return metrics


def run_hard_eval_for_yolo(model_name: str, project_dir: Path, hard_data: str) -> dict | None:
    weights = project_dir / model_name / "weights" / "best.pt"
    if not weights.exists():
        return None
    model = YOLO(str(weights))
    res = model.val(data=hard_data)
    m = _extract(res)
    m["model_name"] = model_name
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four strong baseline models with standard settings.")
    parser.add_argument(
        "--plan",
        type=str,
        default=r"C:\Users\Administrator\Desktop\project\3\experiments\baseline_plan.yaml",
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size for low-memory runs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size for low-memory runs.")
    parser.add_argument("--workers", type=int, default=None, help="Override dataloader workers for YOLO baselines.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. '0' or 'cpu'.")
    parser.add_argument("--resume", action="store_true", help="Resume YOLO baselines from existing last.pt when available.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip baselines that already have completed artifacts.")
    args = parser.parse_args()

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    common = plan["common"]
    if args.device is not None:
        common["device"] = str(args.device)
    baselines = plan["baselines"]
    out_dir = Path(common["project"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    hard_rows = []
    existing_summary = _read_summary_cache(out_dir / "baseline_summary.csv", "model_name")
    for b in baselines:
        if b["kind"] == "yolo":
            rows.append(
                run_yolo_baseline(
                    common,
                    b,
                    batch_override=args.batch,
                    imgsz_override=args.imgsz,
                    workers_override=args.workers,
                    resume=args.resume,
                    skip_completed=args.skip_completed,
                    existing_summary=existing_summary,
                )
            )
            if Path(common["hard_data"]).exists():
                hr = run_hard_eval_for_yolo(b["name"], out_dir, common["hard_data"])
                if hr:
                    hard_rows.append(hr)
        elif b["kind"] == "fasterrcnn":
            rows.append(run_fasterrcnn_baseline(common, b, batch_override=args.batch, skip_completed=args.skip_completed))

    csv_path = out_dir / "baseline_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if hard_rows:
        hcsv = out_dir / "baseline_hard_summary.csv"
        with hcsv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(hard_rows[0].keys()))
            writer.writeheader()
            writer.writerows(hard_rows)

    print(f"Saved baseline summary: {csv_path}")


if __name__ == "__main__":
    main()
