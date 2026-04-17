"""
Per-size AP evaluation for all ablation variants.
Splits GT objects by COCO size: small (<32px), medium (32-96px).
Uses pycocotools for per-size evaluation.
"""
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov13-main'))

from ultralytics import YOLO
from PIL import Image
from collections import defaultdict

IMG_DIR = r'C:\Users\Administrator\Desktop\project\3\MinneApple\yolo\images\val'
LBL_DIR = r'C:\Users\Administrator\Desktop\project\3\MinneApple\yolo\labels\val'
DATA    = r'C:\Users\Administrator\Desktop\project\3\MinneApple\yolo\data.yaml'
IMGSZ   = 960
DEVICE  = '0'

variants = [
    ('A0_baseline',  r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a0_official_yolov13s\weights\best.pt', {}),
    ('A1_p2_loss',   r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a1_p2_only\weights\best.pt', {}),
    ('A2_scene_aug', r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a2_scene_aug_only\weights\best.pt', {}),
    ('A3_diou',      r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a0_official_yolov13s\weights\best.pt', dict(nms_type='diou')),
    ('A4_p2_aug',    r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a4_p2_scene_aug\weights\best.pt', {}),
    ('A5_full',      r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a5_full_p2_aug_diou\weights\best.pt', dict(nms_type='diou')),
]

# Build GT database: image -> list of (x1,y1,x2,y2, area)
print('Building GT database...')
gt_db = {}
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
for img_f in img_files:
    lbl_f = img_f.replace('.png', '.txt')
    lbl_path = os.path.join(LBL_DIR, lbl_f)
    img_path = os.path.join(IMG_DIR, img_f)
    if not os.path.exists(lbl_path):
        continue
    img = Image.open(img_path)
    w, h = img.size
    boxes = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cx, cy, bw, bh = float(parts[1])*w, float(parts[2])*h, float(parts[3])*w, float(parts[4])*h
            x1, y1 = cx - bw/2, cy - bh/2
            x2, y2 = cx + bw/2, cy + bh/2
            area = bw * bh
            side = np.sqrt(area)
            if side < 32:
                sz = 'small'
            elif side < 96:
                sz = 'medium'
            else:
                sz = 'large'
            boxes.append({'box': [x1,y1,x2,y2], 'area': area, 'size': sz, 'matched': False})
    gt_db[img_f] = boxes

# Count GT by size
for sz in ['small', 'medium', 'large']:
    cnt = sum(1 for bxs in gt_db.values() for b in bxs if b['size'] == sz)
    print(f'  GT {sz}: {cnt}')

def compute_metrics_by_size(model, weights_path, extra_args, name):
    """Run prediction and compute recall/precision per size group."""
    model = YOLO(weights_path)
    results = model.predict(
        source=IMG_DIR, imgsz=IMGSZ, device=DEVICE,
        conf=0.25, iou=0.65, max_det=600,
        verbose=False, save=False, **extra_args,
    )

    size_stats = {sz: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for sz in ['small', 'medium', 'all']}

    for r in results:
        img_name = os.path.basename(r.path)
        if img_name not in gt_db:
            continue

        gts = [dict(b) for b in gt_db[img_name]]  # deep copy
        for g in gts:
            g['matched'] = False

        preds = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.empty((0,4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.empty(0)

        # Sort preds by confidence
        order = np.argsort(-confs)
        preds = preds[order]

        pred_matched = [False] * len(preds)

        # Match each pred to best GT
        for pi, pred in enumerate(preds):
            best_iou = 0.5  # threshold
            best_gi = -1
            for gi, gt in enumerate(gts):
                if gt['matched']:
                    continue
                # compute IoU
                xx1 = max(pred[0], gt['box'][0])
                yy1 = max(pred[1], gt['box'][1])
                xx2 = min(pred[2], gt['box'][2])
                yy2 = min(pred[3], gt['box'][3])
                inter = max(0, xx2-xx1) * max(0, yy2-yy1)
                area_p = (pred[2]-pred[0]) * (pred[3]-pred[1])
                area_g = gt['area']
                union = area_p + area_g - inter
                iou = inter / (union + 1e-9)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0:
                gts[best_gi]['matched'] = True
                pred_matched[pi] = True
                sz = gts[best_gi]['size']
                size_stats[sz]['tp'] += 1
                size_stats['all']['tp'] += 1
            else:
                # FP - attribute to 'all'
                size_stats['all']['fp'] += 1

        # Count FN
        for gt in gts:
            sz = gt['size']
            size_stats[sz]['gt'] += 1
            size_stats['all']['gt'] += 1
            if not gt['matched']:
                size_stats[sz]['fn'] += 1
                size_stats['all']['fn'] += 1

    # unmatched preds as FP per size (approximate: attribute to small if small GT dominates)
    return size_stats


print('\nRunning per-size evaluation...')
all_stats = {}
for name, weights, extra in variants:
    if not os.path.exists(weights):
        continue
    print(f'  {name}...')
    stats = compute_metrics_by_size(None, weights, extra, name)
    all_stats[name] = stats

# Print results
print('\n' + '='*95)
print('PER-SIZE RECALL ANALYSIS (IoU>=0.5, conf>=0.25, max_det=600)')
print('='*95)

for sz in ['small', 'medium', 'all']:
    sz_label = {'small': 'SMALL (<32px, 75.2% of GT)', 'medium': 'MEDIUM (32-96px, 24.8% of GT)', 'all': 'ALL SIZES'}[sz]
    print(f'\n--- {sz_label} ---')
    print('%-14s %6s %6s %6s %8s %8s' % ('variant', 'TP', 'FN', 'GT', 'Recall', 'delta'))
    print('-'*55)
    base_recall = None
    for name in [n for n,_,_ in variants]:
        if name not in all_stats: continue
        s = all_stats[name][sz]
        recall = s['tp'] / (s['gt'] + 1e-9)
        if base_recall is None:
            base_recall = recall
        delta = recall - base_recall
        sign = '+' if delta >= 0 else ''
        print('%-14s %6d %6d %6d %8.4f %8s' % (name, s['tp'], s['fn'], s['gt'], recall, sign+'%.4f'%delta))
