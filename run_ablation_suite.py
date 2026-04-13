from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent
ULTRA_ROOT = ROOT / "yolov13-main"
if ULTRA_ROOT.exists() and str(ULTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRA_ROOT))

from ultralytics import YOLO


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


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Applied after hyp yaml merge so ablation_plan `common` wins (patience, cos_lr, batch, etc.).
_COMMON_FINAL_KEYS = (
    "patience",
    "cos_lr",
    "lrf",
    "lr0",
    "warmup_epochs",
    "epochs",
    "cache",
    "fraction",
    "optimizer",
)


def _apply_common_train_final(
    train_args: dict,
    common: dict,
    batch_override: int | None,
    imgsz_override: int | None,
    workers_override: int | None,
) -> None:
    for key in _COMMON_FINAL_KEYS:
        if key in common and common[key] is not None:
            train_args[key] = common[key]
    train_args["batch"] = int(batch_override if batch_override is not None else common["batch"])
    train_args["imgsz"] = int(imgsz_override if imgsz_override is not None else common["imgsz"])
    if workers_override is not None:
        train_args["workers"] = int(workers_override)
    elif "workers" in common:
        train_args["workers"] = int(common["workers"])


def _normalize_none_like_values(train_args: dict) -> None:
    """Normalize string None/null values loaded from YAML to real None."""
    for k, v in list(train_args.items()):
        if isinstance(v, str) and v.strip().lower() in {"none", "null"}:
            train_args[k] = None


def _find_latest_weights(project_dir: Path, variant_name: str) -> tuple[Path, Path]:
    candidates = sorted(
        [p for p in project_dir.glob(f"{variant_name}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in candidates:
        best = d / "weights" / "best.pt"
        last = d / "weights" / "last.pt"
        if best.exists() or last.exists():
            return best, last
    return project_dir / variant_name / "weights" / "best.pt", project_dir / variant_name / "weights" / "last.pt"


def _read_summary_cache(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r.get(key, ""): r for r in rows if r.get(key)}


def _safe_float(v: str | None, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _summary_from_results_csv(run_dir: Path, variant_name: str, nms_type: str, best_weight: Path, last_weight: Path) -> dict | None:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best_row = max(rows, key=lambda r: _safe_float(r.get("metrics/mAP50-95(B)")))
    return {
        "precision": _safe_float(best_row.get("metrics/precision(B)")),
        "recall": _safe_float(best_row.get("metrics/recall(B)")),
        "map50": _safe_float(best_row.get("metrics/mAP50(B)")),
        "map50_95": _safe_float(best_row.get("metrics/mAP50-95(B)")),
        "inference_ms": 0.0,
        "fps": 0.0,
        "variant": variant_name,
        "nms_type": nms_type,
        "best_weight": str(best_weight) if best_weight.exists() else "",
        "last_weight": str(last_weight) if last_weight.exists() else "",
    }


def _evaluate_variant(
    common: dict,
    v: dict,
    best_weight: Path,
    last_weight: Path,
    eval_device: str | None = None,
) -> tuple[dict, dict | None]:
    nms_type = v.get("nms_type", "iou")
    eval_weight = best_weight if best_weight.exists() else last_weight
    if not eval_weight.exists():
        raise FileNotFoundError(f"No available checkpoint for evaluation: {best_weight} / {last_weight}")

    model = YOLO(str(eval_weight))
    val_kwargs = {"data": common["data"], "nms_type": nms_type}
    if eval_device is not None:
        val_kwargs["device"] = eval_device
    val_res = model.val(**val_kwargs)
    metrics = _extract(val_res)
    metrics["variant"] = v["name"]
    metrics["nms_type"] = nms_type
    metrics["best_weight"] = str(best_weight) if best_weight.exists() else ""
    metrics["last_weight"] = str(last_weight) if last_weight.exists() else ""

    hard_metrics = None
    hard_data = common.get("hard_data")
    if hard_data and Path(hard_data).exists():
        hard_model = YOLO(str(eval_weight))
        hard_kwargs = {"data": hard_data, "nms_type": nms_type}
        if eval_device is not None:
            hard_kwargs["device"] = eval_device
        hard_val = hard_model.val(**hard_kwargs)
        hard_metrics = _extract(hard_val)
        hard_metrics["variant"] = v["name"]
        hard_metrics["nms_type"] = nms_type
        hard_metrics["weight_used"] = str(eval_weight)
    return metrics, hard_metrics


def _train_one(
    common: dict,
    v: dict,
    batch_override: int | None = None,
    imgsz_override: int | None = None,
    workers_override: int | None = None,
    resume: bool = False,
    skip_completed: bool = False,
    existing_summary: dict[str, dict] | None = None,
    existing_hard_summary: dict[str, dict] | None = None,
) -> tuple[dict, dict | None]:
    name = v["name"]
    project_dir = Path(common["project"]).resolve()
    best_weight, last_weight = _find_latest_weights(project_dir, name)

    if skip_completed and best_weight.exists():
        print(f"[SKIP] {name}: existing best checkpoint found, evaluating only.")
        if existing_summary and name in existing_summary:
            row = dict(existing_summary[name])
            row["variant"] = name
            row["nms_type"] = v.get("nms_type", "iou")
            row["best_weight"] = str(best_weight) if best_weight.exists() else ""
            row["last_weight"] = str(last_weight) if last_weight.exists() else ""
            hard_row = dict(existing_hard_summary[name]) if (existing_hard_summary and name in existing_hard_summary) else None
            return row, hard_row

        run_dir = best_weight.parent.parent if best_weight.exists() else last_weight.parent.parent
        fallback_row = _summary_from_results_csv(run_dir, name, v.get("nms_type", "iou"), best_weight, last_weight)
        if fallback_row is not None:
            return fallback_row, None
        return _evaluate_variant(common, v, best_weight, last_weight, eval_device="cpu")

    if resume and last_weight.exists():
        print(f"[RESUME] {name}: resuming from {last_weight}")
        model = YOLO(str(last_weight))
        model.train(resume=True)
        best_weight, last_weight = _find_latest_weights(project_dir, name)
        return _evaluate_variant(common, v, best_weight, last_weight)

    if "model_config" in v:
        model = YOLO(v["model_config"])
        pretrained = Path(v["pretrained"])
        if pretrained.exists():
            model = model.load(str(pretrained))
    else:
        model = YOLO(v["model"])

    train_args = {
        "data": common["data"],
        "epochs": int(common["epochs"]),
        "imgsz": int(imgsz_override if imgsz_override is not None else common["imgsz"]),
        "batch": int(batch_override if batch_override is not None else common["batch"]),
        "project": common["project"],
        "name": name,
        "device": str(common["device"]),
        "save": True,
    }
    if workers_override is not None:
        train_args["workers"] = int(workers_override)
    elif "workers" in common:
        train_args["workers"] = int(common["workers"])

    if not v.get("use_default_hyp", False):
        hyp_path = v.get("hyp_config")
        if hyp_path:
            hyp = _load_yaml(hyp_path)
            # Remove inference-only keys that should not leak into training
            for _inf_key in ("nms_type", "iou", "max_det"):
                hyp.pop(_inf_key, None)
            train_args.update(hyp)
            train_args.update(
                {
                    "data": common["data"],
                    "project": common["project"],
                    "name": name,
                    "device": str(common["device"]),
                }
            )

    _apply_common_train_final(train_args, common, batch_override, imgsz_override, workers_override)
    _normalize_none_like_values(train_args)

    model.train(**train_args)
    best_weight, last_weight = _find_latest_weights(project_dir, name)
    return _evaluate_variant(common, v, best_weight, last_weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full YOLOv13 ablation suite.")
    parser.add_argument(
        "--plan",
        type=str,
        default=r"C:\Users\Administrator\Desktop\project\3\experiments\ablation_plan.yaml",
        help="Ablation plan yaml.",
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size for low-memory runs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size for low-memory runs.")
    parser.add_argument("--workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. '0' or 'cpu'.")
    parser.add_argument("--resume", action="store_true", help="Resume current variant from existing last.pt when available.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip training for variants that already have best.pt and only evaluate.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated variant names to run, e.g. 'a5_full' or 'a4_p2_aug,a5_full'.")
    args = parser.parse_args()

    plan = _load_yaml(args.plan)
    common = plan["common"]
    if args.device is not None:
        common["device"] = str(args.device)
    variants = plan["variants"]
    if args.only:
        only_set = {s.strip() for s in args.only.split(",")}
        variants = [v for v in variants if v["name"] in only_set]
        if not variants:
            print(f"[ERROR] No variants matched --only={args.only}")
            return
    out_dir = Path(common["project"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    hard_rows = []
    existing_summary = _read_summary_cache(out_dir / "ablation_summary.csv", "variant")
    existing_hard_summary = _read_summary_cache(out_dir / "ablation_hard_summary.csv", "variant")
    for v in variants:
        row, hard_row = _train_one(
            common,
            v,
            batch_override=args.batch,
            imgsz_override=args.imgsz,
            workers_override=args.workers,
            resume=args.resume,
            skip_completed=args.skip_completed,
            existing_summary=existing_summary,
            existing_hard_summary=existing_hard_summary,
        )
        rows.append(row)
        if hard_row is not None:
            hard_rows.append(hard_row)

    csv_path = out_dir / "ablation_summary.csv"
    json_path = out_dir / "ablation_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    if hard_rows:
        hard_csv_path = out_dir / "ablation_hard_summary.csv"
        hard_json_path = out_dir / "ablation_hard_summary.json"
        with hard_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(hard_rows[0].keys()))
            writer.writeheader()
            writer.writerows(hard_rows)
        hard_json_path.write_text(json.dumps(hard_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {hard_csv_path}")
        print(f"Saved: {hard_json_path}")


if __name__ == "__main__":
    main()
