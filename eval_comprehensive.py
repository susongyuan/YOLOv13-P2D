"""
Comprehensive evaluation for YOLOv13s ablation (A0-A5) and YOLOv13n (a0n, a5n).
Three analyses:
  1. Per-size recall on full val set
  2. Per-size recall on hard subset (40 dense/occluded images)
  3. max_det=300 vs max_det=600 comparison (full val, all sizes)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov13-main'))

from ultralytics import YOLO
from PIL import Image

BASE   = r'C:\Users\Administrator\Desktop\project\3'
IMG_DIR = os.path.join(BASE, r'MinneApple\yolo\images\val')
LBL_DIR = os.path.join(BASE, r'MinneApple\yolo\labels\val')
HARD_TXT = os.path.join(BASE, r'MinneApple\yolo\hard_subset\hard_val.txt')
IMGSZ   = 960
DEVICE  = '0'

s_variants = [
    ('A0_baseline', os.path.join(BASE, r'runs_ablation\a0_official_yolov13s\weights\best.pt'),        {}),
    ('A1_p2',       os.path.join(BASE, r'runs_ablation\a1_p2_only\weights\best.pt'),                  {}),
    ('A2_aug',      os.path.join(BASE, r'runs_ablation\a2_scene_aug_only\weights\best.pt'),            {}),
    ('A3_diou',     os.path.join(BASE, r'runs_ablation\a0_official_yolov13s\weights\best.pt'),         dict(nms_type='diou')),
    ('A4_p2_aug',   os.path.join(BASE, r'runs_ablation\a4_p2_scene_aug\weights\best.pt'),              {}),
    ('A5_full',     os.path.join(BASE, r'runs_ablation\a5_full_p2_aug_diou\weights\best.pt'),          dict(nms_type='diou')),
]

n_variants = [
    ('a0n_baseline', os.path.join(BASE, r'runs_nano\a0n_yolov13n_baseline5\weights\best.pt'), {}),
    ('a5n_full',     os.path.join(BASE, r'runs_nano\a5n_yolov13n_full_b8\weights\best.pt'),   dict(nms_type='diou')),
]

# ── GT database ────────────────────────────────────────────────────────────────
def build_gt_db(img_files):
    db = {}
    for img_f in img_files:
        name = os.path.basename(img_f)
        lbl_path = os.path.join(LBL_DIR, name.replace('.png', '.txt'))
        img_path  = os.path.join(IMG_DIR, name)
        if not os.path.exists(lbl_path) or not os.path.exists(img_path):
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
        db[name] = boxes
    return db

all_imgs  = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
with open(HARD_TXT) as f:
    hard_imgs = [os.path.basename(l.strip()) for l in f if l.strip()]

print(f'Full val: {len(all_imgs)} images | Hard subset: {len(hard_imgs)} images')
gt_full = build_gt_db(all_imgs)
gt_hard = build_gt_db(hard_imgs)

for label, db in [('full', gt_full), ('hard', gt_hard)]:
    for sz in ['small', 'medium']:
        cnt = sum(1 for bxs in db.values() for b in bxs if b['size'] == sz)
        print(f'  GT {label} {sz}: {cnt}')

# ── Evaluation core ────────────────────────────────────────────────────────────
def run_eval(weights, extra, gt_db, source_imgs, max_det=600):
    model = YOLO(weights)
    source = [os.path.join(IMG_DIR, f) for f in source_imgs if f in gt_db]
    results = model.predict(
        source=source, imgsz=IMGSZ, device=DEVICE,
        conf=0.25, iou=0.65, max_det=max_det,
        verbose=False, save=False, **extra,
    )
    stats = {sz: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for sz in ['small', 'medium', 'all']}

    for r in results:
        img_name = os.path.basename(r.path)
        if img_name not in gt_db: continue
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

# ── Print helpers ──────────────────────────────────────────────────────────────
def print_table(title, variants_results, sizes=('small', 'medium', 'all')):
    sz_labels = {'small': 'SMALL (<32px)', 'medium': 'MEDIUM (32-96px)', 'all': 'ALL'}
    print(f'\n{"="*70}')
    print(title)
    print('='*70)
    for sz in sizes:
        print(f'\n  --- {sz_labels[sz]} ---')
        print(f'  {"variant":<16} {"TP":>5} {"FN":>5} {"GT":>5} {"Recall":>8} {"delta":>8}')
        print('  ' + '-'*50)
        base = None
        for name, stats in variants_results:
            s = stats[sz]
            rec = s['tp'] / (s['gt'] + 1e-9)
            if base is None: base = rec
            d = rec - base
            sign = '+' if d >= 0 else ''
            print(f'  {name:<16} {s["tp"]:>5} {s["fn"]:>5} {s["gt"]:>5} {rec:>8.4f} {sign+f"{d:.4f}":>8}')

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 & 2: Full val + Hard subset, per size
# ══════════════════════════════════════════════════════════════════════════════
for family, variants in [('YOLOv13s', s_variants), ('YOLOv13n', n_variants)]:
    print(f'\n\n{"#"*70}')
    print(f'# {family}')
    print(f'{"#"*70}')

    full_results, hard_results = [], []
    for name, weights, extra in variants:
        if not os.path.exists(weights):
            print(f'  SKIP {name} (weights not found)')
            continue
        print(f'  Evaluating {name}...')
        full_results.append((name, run_eval(weights, extra, gt_full, all_imgs, max_det=600)))
        hard_results.append((name, run_eval(weights, extra, gt_hard, hard_imgs, max_det=600)))

    print_table(f'[{family}] ANALYSIS 1: Full val (100 imgs), max_det=600', full_results)
    print_table(f'[{family}] ANALYSIS 2: Hard subset (40 imgs), max_det=600', hard_results)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: max_det 300 vs 600 (full val)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n\n{"="*70}')
print('ANALYSIS 3: max_det=300 vs max_det=600 (full val, ALL sizes)')
print('='*70)

for family, variants in [('YOLOv13s', s_variants), ('YOLOv13n', n_variants)]:
    print(f'\n  [{family}]')
    print(f'  {"variant":<16} {"R@300":>8} {"R@600":>8} {"gain":>8} {"TP@300":>7} {"TP@600":>7} {"+TP":>5}')
    print('  ' + '-'*62)
    for name, weights, extra in variants:
        if not os.path.exists(weights): continue
        s300 = run_eval(weights, extra, gt_full, all_imgs, max_det=300)['all']
        s600 = run_eval(weights, extra, gt_full, all_imgs, max_det=600)['all']
        r300 = s300['tp'] / (s300['gt'] + 1e-9)
        r600 = s600['tp'] / (s600['gt'] + 1e-9)
        gain = r600 - r300
        sign = '+' if gain >= 0 else ''
        tp_gain = s600['tp'] - s300['tp']
        print(f'  {name:<16} {r300:>8.4f} {r600:>8.4f} {sign+f"{gain:.4f}":>8} {s300["tp"]:>7} {s600["tp"]:>7} {"+"+str(tp_gain):>5}')

print('\nDone.')
