# YOLOv13 Apple Reproduction and Improvements

This workspace now includes the official YOLOv13 source at:

- `yolov13-main`

Based on your proposal constraints, the following improvements are implemented (without knowledge distillation and without ONNX deployment):

- `P2` small-object branch model: `yolov13-main/ultralytics/cfg/models/v13/yolov13-apple-p2.yaml`
- Orchard-focused augmentation/training config: `yolov13-main/ultralytics/cfg/experiments/apple_orchard_improved.yaml`
- DIoU-NMS support:
  - `yolov13-main/ultralytics/utils/ops.py`
  - `yolov13-main/ultralytics/models/yolo/detect/predict.py`
  - `yolov13-main/ultralytics/models/yolo/detect/val.py`
  - `yolov13-main/ultralytics/cfg/default.yaml`

## Run training

```bash
pip install -r yolov13-main/requirements.txt
pip install -e yolov13-main
# place official YOLOv13-S weights at yolov13-main/yolov13s.pt (recommended)
python train_apple_yolov13_improved.py --device 0
```

Default dataset path:

- `MinneApple/yolo/data.yaml`

If needed, override:

```bash
python train_apple_yolov13_improved.py --data "path/to/data.yaml" --epochs 200 --device cpu
```

Use YOLOv13-S weight explicitly:

```bash
python train_apple_yolov13_improved.py ^
  --model-config "yolov13-main/ultralytics/cfg/models/v13/yolov13-apple-p2.yaml" ^
  --pretrained "yolov13-main/yolov13s.pt" ^
  --device 0
```

## Build hard subset

```bash
python build_hard_subset.py
```

Generated files:

- `MinneApple/yolo/hard_subset/hard_val.txt`
- `MinneApple/yolo/hard_subset/data_hard.yaml`
- `MinneApple/yolo/hard_subset/hard_subset_stats.csv`

## Compare multiple models (full set + hard subset)

```bash
python eval_compare_models.py ^
  --models "runs_apple/yolov13_apple_improved/weights/best.pt" "yolov13-main/yolov13s.pt" ^
  --names "improved_yolov13" "yolov13s_baseline" ^
  --data "MinneApple/yolo/data.yaml" ^
  --hard-data "MinneApple/yolo/hard_subset/data_hard.yaml" ^
  --nms-type diou
```

Outputs are saved under:

- `comparison_results/comparison_full.csv`
- `comparison_results/comparison_hard.csv` (if hard subset exists)
- `comparison_results/*.md`
- `comparison_results/*map50_95.png`

## Full ablation suite (thesis)

```bash
python run_ablation_suite.py --plan "experiments/ablation_plan.yaml"
```

Ablation variants:

- `a0_official_yolov13s` (official YOLOv13-S baseline)
- `a1_p2_only`
- `a2_scene_aug_only` (scene augmentation only)
- `a3_diou_only` (DIoU-NMS only)
- `a4_p2_scene_aug`
- `a5_full_p2_aug_diou`

## Four strong baselines (standard reproduction)

```bash
python run_baseline_suite.py --plan "experiments/baseline_plan.yaml"
```

Configured baselines:

- `YOLOv8s` (official)
- `RT-DETR-L` (official)
- `YOLO11s` (official)
- `Faster R-CNN ResNet50-FPN` (standard torchvision implementation)

## Run ablation + baseline together (sequential, all on GPU)

```bash
python run_all_experiments.py --batch 8 --imgsz 960 --workers 8 --ablation-device 0 --baseline-device 0
```

Notes:

- This command runs **sequentially** by default (do not add `--parallel`)
- For all-on-GPU full training, keep both devices as `0`
- If you want background logs, use `--parallel` and check `runs_logs/*.log`

## Build final thesis tables

```bash
python make_final_comparison.py
```

Generated outputs include:

- `final_tables/combined_comparison.csv`
- `final_tables/combined_comparison.xlsx` (Excel with `ablation`, `baseline`, `combined` sheets)
- `final_tables/ablation_table.md`
- `final_tables/baseline_table.md`
- `final_tables/ablation_hard_table.md` (if hard subset exists)
- `final_tables/baseline_hard_table.md` (if hard subset exists)
- `final_tables/ablation_metrics.tiff`
- `final_tables/baseline_metrics.tiff`
- `final_tables/all_models_metrics.tiff`
- `final_tables/all_models_speed.tiff`
