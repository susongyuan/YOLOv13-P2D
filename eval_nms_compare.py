"""
Compare IoU-NMS vs DIoU-NMS on the validation set using a0's best.pt.
Runs val() twice with different NMS settings and prints a side-by-side table.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov13-main'))

from ultralytics import YOLO

MODEL   = r'C:/Users/Administrator/Desktop/project/3/runs_ablation/a0_official_yolov13s/weights/best.pt'
DATA    = r'C:/Users/Administrator/Desktop/project/3/MinneApple/yolo/data.yaml'
IMGSZ   = 960
DEVICE  = '0'

CONFIGS = [
    dict(name='IoU-NMS  (baseline)',  iou=0.65, max_det=300),
    dict(name='DIoU-NMS (ours)',      iou=0.65, max_det=300, nms_type='diou'),
]

results = []
for cfg in CONFIGS:
    name = cfg.pop('name')
    print(f'\n{"="*60}')
    print(f'Running: {name}')
    print(f'{"="*60}')
    model = YOLO(MODEL)
    metrics = model.val(
        data=DATA,
        imgsz=IMGSZ,
        device=DEVICE,
        workers=0,
        verbose=False,
        **cfg,
    )
    r = {
        'name':     name,
        'mAP50':    metrics.box.map50,
        'mAP75':    metrics.box.map75,
        'mAP5095':  metrics.box.map,
        'P':        metrics.box.mp,
        'R':        metrics.box.mr,
    }
    r['F1'] = 2 * r['P'] * r['R'] / (r['P'] + r['R'] + 1e-9)
    results.append(r)
    print(f"  mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP5095']:.4f}  P={r['P']:.4f}  R={r['R']:.4f}  F1={r['F1']:.4f}")

print('\n' + '='*70)
print('NMS Comparison on MinneApple val set (a0 best.pt, imgsz=960)')
print('='*70)
print(f"{'Method':<26} {'mAP50':>7} {'mAP75':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'F1':>7}")
print('-'*70)
base = results[0]
for r in results:
    dm = r['mAP50'] - base['mAP50']
    sign = '+' if dm >= 0 else ''
    delta = f"({sign}{dm:.4f})"
    print(f"{r['name']:<26} {r['mAP50']:>7.4f}{delta:<10} {r['mAP75']:>7.4f} {r['mAP5095']:>9.4f} {r['P']:>7.4f} {r['R']:>7.4f} {r['F1']:>7.4f}")
print('='*70)
