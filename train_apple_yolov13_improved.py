from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent
ULTRA_ROOT = ROOT / "yolov13-main"
if ULTRA_ROOT.exists() and str(ULTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRA_ROOT))

from ultralytics import YOLO


def build_default_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    model_yaml = root / "yolov13-main" / "ultralytics" / "cfg" / "models" / "v13" / "yolov13s-apple-p2.yaml"
    train_cfg = root / "yolov13-main" / "ultralytics" / "cfg" / "experiments" / "apple_orchard_improved.yaml"
    data_yaml = root / "MinneApple" / "yolo" / "data.yaml"
    pretrained_s = root / "yolov13-main" / "yolov13s.pt"
    return model_yaml, train_cfg, data_yaml, pretrained_s


def main() -> None:
    root = Path(__file__).resolve().parent
    model_yaml, train_cfg, data_yaml, pretrained_s = build_default_paths(root)

    parser = argparse.ArgumentParser(description="Train improved YOLOv13 for apple detection.")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Direct model path (.pt/.yaml). If set, overrides --model-config and --pretrained.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=str(model_yaml),
        help="Model YAML used for improved architecture.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=str(pretrained_s),
        help="Pretrained weights used to initialize --model-config (default: yolov13s.pt).",
    )
    parser.add_argument("--data", type=str, default=str(data_yaml), help="Dataset config yaml.")
    parser.add_argument("--cfg", type=str, default=str(train_cfg), help="Training hyperparameter yaml.")
    parser.add_argument("--project", type=str, default=str(root / "runs_apple"), help="Training output directory.")
    parser.add_argument("--name", type=str, default="yolov13_apple_improved", help="Experiment name.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device string, e.g. '0', '0,1', or 'cpu'.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Override workers if needed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs if needed.")
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f) or {}

    # Allow CLI overrides for commonly tuned fields.
    if args.workers is not None:
        train_args["workers"] = args.workers
    if args.epochs is not None:
        train_args["epochs"] = args.epochs

    train_args.update(
        {
            "data": args.data,
            "project": args.project,
            "name": args.name,
            "device": args.device,
        }
    )

    if args.model:
        model = YOLO(args.model)
    else:
        model = YOLO(args.model_config)
        pretrained_path = Path(args.pretrained)
        if pretrained_path.exists():
            model = model.load(str(pretrained_path))
        else:
            print(f"[WARN] Pretrained weights not found: {pretrained_path}. Training starts from random initialization.")
    model.train(**train_args)
    model.val(data=args.data, nms_type=train_args.get("nms_type", "iou"))


if __name__ == "__main__":
    main()
