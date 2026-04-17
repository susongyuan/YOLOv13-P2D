"""
Per-size recall evaluation for YOLOv13n ablation (a0n vs a5n).
Splits GT objects by side length: small (<32px), medium (32-96px), large (>=96px).
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov13-main'))

from ultralytics import YOLO
from PIL import Image

IMG_DIR = r'C:\Users\Administrator\Desktop\project\3\MinneApple\yolo\images\val'
LBL_DIR = r'C:\Users\Administrator\Desktop\project\3\MinneApple\yolo\labels\val'
IMGSZ   = 960
DEVICE  = '0'

variants = [
    ('a0n_baseline', r'C:\Users\Administrator\Desktop\project\3\runs_nano\a0n_yolov13n_baseline5\weights\best.pt', {}),
    ('a5n_full',     r'C:\Users\Administrator\Desktop\project\3\runs_nano\a5n_yolov13n_full_b8\weights\best.pt',   dict(nms_type='diou')),
]

# Build GT database
print('Building GT database...')
gt_db = {}
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
for img_f in img_files:
    lbl_f = img_f.replace('.png', '.txt')
    lbl_path = os.path.join(LBL_DIR, lbl_f)
    img_path  = os.path.join(IMG_DIR, img_f)
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
            sz = 'small' if side < 32 else ('medium' if side < 96 else 'large')
            boxes.append({'box': [x1,y1,x2,y2], 'area': area, 'size': sz, 'matched': False})
    gt_db[img_f] = boxes

for sz in ['small', 'medium', 'large']:
    cnt = sum(1 for bxs in gt_db.values() for b in bxs if b['size'] == sz)
    print(f'  GT {sz}: {cnt}')

def evaluate(weights_path, extra_args):
    model = YOLO(weights_path)
    results = model.predict(
        source=IMG_DIR, imgsz=IMGSZ, device=DEVICE,
        conf=0.25, iou=0.65, max_det=600,
        verbose=False, save=False, **extra_args,
    )
    stats = {sz: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for sz in ['small', 'medium', 'large', 'all']}

    for r in results:
        img_name = os.path.basename(r.path)
        if img_name not in gt_db:
            continue
        gts = [dict(b) for b in gt_db[img_name]]
        for g in gts: g['matched'] = False

        preds = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.empty((0,4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.empty(0)
        preds = preds[np.argsort(-confs)]

        for pred in preds:
            best_iou, best_gi = 0.5, -1
            for gi, gt in enumerate(gts):
                if gt['matched']: continue
                xx1, yy1 = max(pred[0], gt['box'][0]), max(pred[1], gt['box'][1])
                xx2, yy2 = min(pred[2], gt['box'][2]), min(pred[3], gt['box'][3])
                inter = max(0, xx2-xx1) * max(0, yy2-yy1)
                union = (pred[2]-pred[0])*(pred[3]-pred[1]) + gt['area'] - inter
                iou = inter / (union + 1e-9)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_gi >= 0:
                gts[best_gi]['matched'] = True
                sz = gts[best_gi]['size']
                stats[sz]['tp'] += 1
                stats['all']['tp'] += 1
            else:
                stats['all']['fp'] += 1

        for gt in gts:
            sz = gt['size']
            stats[sz]['gt'] += 1
            stats['all']['gt'] += 1
            if not gt['matched']:
                stats[sz]['fn'] += 1
                stats['all']['fn'] += 1

    return stats

print('\nRunning evaluation...')
all_stats = {}
for name, weights, extra in variants:
    print(f'  {name}...')
    all_stats[name] = evaluate(weights, extra)

# Print results
print('\n' + '='*70)
print('PER-SIZE RECALL  (IoU>=0.5, conf>=0.25, max_det=600)')
print('='*70)
for sz in ['small', 'medium', 'large', 'all']:
    labels = {'small': 'SMALL  (<32px side)',
              'medium':'MEDIUM (32-96px)',
              'large': 'LARGE  (>=96px)',
              'all':   'ALL'}
    print(f'\n--- {labels[sz]} ---')
    print(f'{"variant":<16} {"TP":>6} {"FN":>6} {"GT":>6} {"Recall":>8} {"Precision":>10} {"delta":>8}')
    print('-'*60)
    base_recall = None
    for name, _, _ in variants:
        if name not in all_stats: continue
        s = all_stats[name][sz]
        recall = s['tp'] / (s['gt'] + 1e-9)
        prec = s['tp'] / (s['tp'] + s['fp'] + 1e-9) if sz == 'all' else float('nan')
        if base_recall is None: base_recall = recall
        delta = recall - base_recall
        prec_str = f'{prec:.4f}' if not np.isnan(prec) else '  n/a  '
        sign = '+' if delta >= 0 else ''
        print(f'{name:<16} {s["tp"]:>6} {s["fn"]:>6} {s["gt"]:>6} {recall:>8.4f} {prec_str:>10} {sign+f"{delta:.4f}":>8}')
