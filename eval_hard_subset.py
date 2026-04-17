"""
Evaluate all ablation variants on the hard subset (dense/occluded scenes).
Also evaluates with max_det=600 to show P2 head advantage.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov13-main'))

from ultralytics import YOLO

DATA_FULL = r'C:/Users/Administrator/Desktop/project/3/MinneApple/yolo/data.yaml'
DATA_HARD = r'C:/Users/Administrator/Desktop/project/3/MinneApple/yolo/hard_subset/data_hard.yaml'
IMGSZ = 960
DEVICE = '0'

variants = [
    ('A0_baseline',    r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a0_official_yolov13s\weights\best.pt', 'iou'),
    ('A1_p2_loss',     r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a1_p2_only\weights\best.pt', 'iou'),
    ('A2_scene_aug',   r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a2_scene_aug_only\weights\best.pt', 'iou'),
    ('A3_diou',        r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a0_official_yolov13s\weights\best.pt', 'diou'),
    ('A4_p2_aug',      r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a4_p2_scene_aug\weights\best.pt', 'iou'),
    ('A5_full',        r'C:\Users\Administrator\Desktop\project\3\runs_ablation\a5_full_p2_aug_diou\weights\best.pt', 'diou'),
]

all_results = []

for name, weights, nms in variants:
    if not os.path.exists(weights):
        print(f'SKIP {name}: weights not found')
        continue

    print(f'\n{"="*60}')
    print(f'Evaluating: {name} (nms={nms})')
    print(f'{"="*60}')

    # --- Hard subset eval ---
    model = YOLO(weights)
    nms_args = dict(nms_type=nms) if nms == 'diou' else {}
    m_hard = model.val(
        data=DATA_HARD, imgsz=IMGSZ, device=DEVICE, workers=0,
        verbose=False, iou=0.65, max_det=600, **nms_args,
    )

    # --- Full val eval with max_det=600 ---
    model2 = YOLO(weights)
    m_full = model2.val(
        data=DATA_FULL, imgsz=IMGSZ, device=DEVICE, workers=0,
        verbose=False, iou=0.65, max_det=600, **nms_args,
    )

    r = {
        'name': name,
        'hard_mAP50':   m_hard.box.map50,
        'hard_mAP75':   m_hard.box.map75,
        'hard_mAP5095': m_hard.box.map,
        'hard_P':       m_hard.box.mp,
        'hard_R':       m_hard.box.mr,
        'full_mAP50':   m_full.box.map50,
        'full_mAP75':   m_full.box.map75,
        'full_mAP5095': m_full.box.map,
        'full_P':       m_full.box.mp,
        'full_R':       m_full.box.mr,
    }
    r['hard_F1'] = 2*r['hard_P']*r['hard_R']/(r['hard_P']+r['hard_R']+1e-9)
    r['full_F1'] = 2*r['full_P']*r['full_R']/(r['full_P']+r['full_R']+1e-9)
    all_results.append(r)

# --- Print comparison ---
base_h = all_results[0]['hard_mAP50']
base_f = all_results[0]['full_mAP50']

print('\n' + '='*90)
print('HARD SUBSET (40 dense/occluded images, max_det=600)')
print('='*90)
print('%-14s %8s %8s %8s %10s %8s %8s %8s' % ('variant','mAP50','delta','mAP75','mAP50-95','P','R','F1'))
print('-'*80)
for r in all_results:
    d = r['hard_mAP50'] - base_h
    sign = '+' if d >= 0 else ''
    print('%-14s %8.4f %8s %8.4f %10.4f %8.4f %8.4f %8.4f' % (
        r['name'], r['hard_mAP50'], sign+'%.4f'%d, r['hard_mAP75'],
        r['hard_mAP5095'], r['hard_P'], r['hard_R'], r['hard_F1']))

print('\n' + '='*90)
print('FULL VAL SET (100 images, max_det=600)')
print('='*90)
print('%-14s %8s %8s %8s %10s %8s %8s %8s' % ('variant','mAP50','delta','mAP75','mAP50-95','P','R','F1'))
print('-'*80)
for r in all_results:
    d = r['full_mAP50'] - base_f
    sign = '+' if d >= 0 else ''
    print('%-14s %8.4f %8s %8.4f %10.4f %8.4f %8.4f %8.4f' % (
        r['name'], r['full_mAP50'], sign+'%.4f'%d, r['full_mAP75'],
        r['full_mAP5095'], r['full_P'], r['full_R'], r['full_F1']))
print('='*90)
