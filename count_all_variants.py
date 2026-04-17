"""Compute per-image counting MAE/RMSE/Within-N for all ablation variants A0-A5."""
import sys, os, json, csv
import numpy as np
from pathlib import Path

sys.path.insert(0, 'C:/Users/Administrator/Desktop/project/3/yolov13-main')

VAL_IMG_DIR = Path('C:/Users/Administrator/Desktop/project/3/MinneApple/yolo/images/val')
VAL_LBL_DIR = Path('C:/Users/Administrator/Desktop/project/3/MinneApple/yolo/labels/val')

VARIANTS = [
    ('A0', 'a0_official_yolov13s'),
    ('A1', 'a1_p2_only'),
    ('A2', 'a2_scene_aug_only'),
    ('A3', 'a3_diou_only'),
    ('A4', 'a4_p2_scene_aug'),
    ('A5', 'a5_full_p2_aug_diou'),
]

BASE = Path('C:/Users/Administrator/Desktop/project/3/runs_ablation')

def get_gt_counts():
    counts = {}
    for lbl in VAL_LBL_DIR.glob('*.txt'):
        with open(lbl) as f:
            lines = [l.strip() for l in f if l.strip()]
        counts[lbl.stem] = len(lines)
    return counts

def run_variant(name, run_dir):
    from ultralytics import YOLO
    weight = BASE / run_dir / 'weights' / 'best.pt'
    model = YOLO(str(weight))
    imgs = sorted(VAL_IMG_DIR.glob('*.png')) + sorted(VAL_IMG_DIR.glob('*.jpg'))
    results_map = {}
    print(f"\n=== {name}: running inference on {len(imgs)} images ===")
    for img in imgs:
        res = model(str(img), verbose=False, workers=0)[0]
        results_map[img.stem] = int(res.boxes.shape[0]) if res.boxes is not None else 0
    return results_map

def compute_stats(pred_map, gt_map):
    keys = list(gt_map.keys())
    errors = []
    for k in keys:
        gt = gt_map.get(k, 0)
        pred = pred_map.get(k, 0)
        errors.append(abs(pred - gt))
    errors = np.array(errors)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    w3 = int(np.sum(errors <= 3))
    w5 = int(np.sum(errors <= 5))
    n = len(errors)
    return mae, rmse, w3, w5, n

def main():
    gt_map = get_gt_counts()
    print(f"GT images: {len(gt_map)}, total objects: {sum(gt_map.values())}")

    all_results = {}
    for name, run_dir in VARIANTS:
        try:
            pred_map = run_variant(name, run_dir)
            mae, rmse, w3, w5, n = compute_stats(pred_map, gt_map)
            all_results[name] = dict(mae=mae, rmse=rmse, w3=w3, w5=w5, n=n)
            print(f"{name}: MAE={mae:.2f} RMSE={rmse:.2f} W3={w3}/{n} W5={w5}/{n}")
        except Exception as e:
            print(f"ERROR {name}: {e}")
            all_results[name] = None

    # Save results
    out = Path('C:/Users/Administrator/Desktop/project/3/runs_ablation/count_all_variants.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")

    # Print summary table
    print("\n=== SUMMARY ===")
    print(f"{'Variant':<6} {'MAE':>6} {'RMSE':>6} {'W3':>8} {'W5':>8}")
    for name, _ in VARIANTS:
        r = all_results.get(name)
        if r:
            print(f"{name:<6} {r['mae']:>6.2f} {r['rmse']:>6.2f} {r['w3']:>4}/{r['n']:<3} {r['w5']:>4}/{r['n']:<3}")
        else:
            print(f"{name:<6} ERROR")

if __name__ == '__main__':
    main()
