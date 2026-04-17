#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重绘 YOLOv13 改进架构图
新增：P2 高分辨率检测头（红色）+ DIoU-NMS 标注（橙色）
"""
import matplotlib
matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

fig, ax = plt.subplots(figsize=(22, 15))
ax.set_xlim(0, 22)
ax.set_ylim(0, 15)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ─── Color scheme ──────────────────────────────────────────────────────────────
C_BB    = '#B8D4F0'   # backbone
C_NECK  = '#B4E0B4'   # neck blocks
C_CONV  = '#C8D8EC'   # conv
C_DS    = '#DCCBF0'   # DS-C3k2 / DSConv
C_UPS   = '#C8EEC8'   # upsample
C_CAT   = '#F5D87A'   # concat / 拼接
C_DET   = '#87CEEB'   # detection (existing)
C_HAC   = '#E0C8F0'   # HyperACE
C_P2    = '#FF8080'   # NEW P2 — red
C_NMS   = '#FF9944'   # DIoU-NMS — orange
C_ARROW_NEW = '#CC2222'
C_ARROW     = '#444444'

# ─── Helpers ───────────────────────────────────────────────────────────────────
def box(cx, cy, w=1.4, h=0.52, text='', fc='white', ec='#444', fs=8.5,
        fw='normal', lw=1.4, zorder=3, ec_color=None):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle='round,pad=0.06',
                       facecolor=fc, edgecolor=ec_color or ec,
                       linewidth=lw, zorder=zorder)
    ax.add_patch(r)
    if text:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fs, fontweight=fw, zorder=zorder+1, color='black')

def arr(x1, y1, x2, y2, color=C_ARROW, lw=1.3, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->, head_width=0.14, head_length=0.09',
                    color=color, lw=lw,
                    connectionstyle=f'arc3,rad={rad}'),
                zorder=2)

def line(x1, y1, x2, y2, color=C_ARROW, lw=1.3, ls='-', zorder=1):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=zorder)

def cat_sym(cx, cy, r=0.19):
    c = plt.Circle((cx, cy), r, fc='white', ec='#555', lw=1.3, zorder=5)
    ax.add_patch(c)
    ax.text(cx, cy, '⊕', ha='center', va='center', fontsize=11,
            fontweight='bold', zorder=6)

def label(cx, cy, text, fs=8, color='#444', fw='normal', zorder=6, ha='center'):
    ax.text(cx, cy, text, ha=ha, va='center', fontsize=fs,
            color=color, fontweight=fw, zorder=zorder)

# ─── Section background shading ────────────────────────────────────────────────
def region(x, y, w, h, fc, ec, alpha=0.35):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                facecolor=fc, edgecolor=ec,
                                linewidth=1.6, zorder=0, alpha=alpha))

region(0.1,  0.3, 3.0, 14.4, '#D8EEFF', '#6699CC')   # Backbone
region(3.3,  0.3, 8.1, 14.4, '#D8F0D8', '#66AA66')   # Neck
region(11.6, 0.3, 6.2, 14.4, '#FFF8E0', '#CCAA33')   # Head

label(1.65, 14.5, '骨干网络', fs=11, fw='bold', color='#225588')
label(7.4,  14.5, 'Neck', fs=11, fw='bold', color='#226622')
label(14.7, 14.5, 'Head', fs=11, fw='bold', color='#775511')

# FullPAD Tunnel dashed regions
fp1_patch = FancyBboxPatch((3.5, 1.0), 3.5, 12.5, boxstyle='round,pad=0.1',
                            fc='none', ec='#9966CC', lw=1.5, ls='--', zorder=1)
ax.add_patch(fp1_patch)
label(5.25, 13.3, 'FullPAD Tunnel.el', fs=8.5, color='#7755AA', fw='bold')

fp2_patch = FancyBboxPatch((7.2, 1.0), 4.2, 12.5, boxstyle='round,pad=0.1',
                            fc='none', ec='#9966CC', lw=1.5, ls='--', zorder=1)
ax.add_patch(fp2_patch)
label(9.3, 13.3, 'FullPAD Tunnel', fs=8.5, color='#7755AA', fw='bold')

# ─── Y levels (center) ─────────────────────────────────────────────────────────
YP2 = 12.5   # NEW P2
YH3 = 9.8
YH4 = 6.8
YH5 = 3.6
YHAC = 1.1

# ─── X reference ───────────────────────────────────────────────────────────────
XBB = 1.65        # backbone column

# Neck1 (FullPAD Tunnel.el)
XN1L = 4.2        # neck1 left blocks
XN1R = 6.3        # neck1 right (cat symbol area)

# Neck2 (FullPAD Tunnel)
XN2L = 8.0        # neck2 left
XN2R = 10.5       # neck2 right

# Head
XHD  = 13.1       # head processing blocks
XDET = 16.2       # detection boxes
XNMS = 19.0       # DIoU-NMS boxes

BW = 1.4  # normal box width
BH = 0.52

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE
# ══════════════════════════════════════════════════════════════════════════════
box(XBB, 14.0, text='输入', fc=C_BB, w=1.3, h=0.45)
arr(XBB, 14.0-0.23, XBB, 13.55)

box(XBB, 13.3, text='Conv', fc=C_CONV, w=1.3)
arr(XBB, 13.3-BH/2, XBB, 12.85)

box(XBB, 12.6, text='Conv', fc=C_CONV, w=1.3)

# B2 exits → NEW P2 path
arr(XBB, 12.6-BH/2, XBB, 12.1)
box(XBB, 11.85, text='DS-C3k2', fc=C_DS, w=1.35)

# ← NEW: B2 exits right (red arrow)
line(XBB+1.35/2, YP2, 3.4, YP2, color=C_ARROW_NEW, lw=2.0)
arr(3.38, YP2, 3.55, YP2, color=C_ARROW_NEW, lw=2.0)
label(3.0, YP2+0.28, 'B₂', fs=8.5, color=C_ARROW_NEW, fw='bold')

# Continue down to B3
arr(XBB, 11.85-BH/2, XBB, 11.1)
box(XBB, 10.85, text='Conv', fc=C_CONV, w=1.3)
arr(XBB, 10.85-BH/2, XBB, 10.3)
box(XBB, 10.05, text='DS-C3k2', fc=C_DS, w=1.35)

# B3 exits right
arr(XBB+1.35/2, YH3, XN1L-BW/2, YH3, color=C_ARROW)
label(2.9, YH3+0.28, 'B₃', fs=8.5, color='#444', fw='bold')

# B4
arr(XBB, 10.05-BH/2, XBB, 8.5)
box(XBB, 8.25, text='DS积块', fc=C_BB, w=1.3)
arr(XBB, 8.25-BH/2, XBB, 7.65)
box(XBB, 7.4, text='A2C2f', fc=C_BB, w=1.3)

arr(XBB+1.35/2, 7.4, XN1L-BW/2, YH4, color=C_ARROW)
label(2.9, YH4+0.28, 'B₄', fs=8.5, color='#444', fw='bold')

# B5
arr(XBB, 7.4-BH/2, XBB, 6.0)
box(XBB, 5.75, text='DS积块', fc=C_BB, w=1.3)
arr(XBB, 5.75-BH/2, XBB, 5.1)
box(XBB, 4.85, text='A2C2f', fc=C_BB, w=1.3)

arr(XBB+1.35/2, 4.85, XN1L-BW/2, YH5, color=C_ARROW)
label(2.9, YH5+0.28, 'B₅', fs=8.5, color='#444', fw='bold')

# ══════════════════════════════════════════════════════════════════════════════
# NECK — Tunnel 1 (left side)
# ══════════════════════════════════════════════════════════════════════════════
# H3 level
box(XN1L, YH3, text='DS-C3k2', fc=C_DS)
arr(XN1L+BW/2, YH3, XN1R-0.2, YH3)

# Cat sym H3
cat_sym(XN1R, YH3)

# H4 level
box(XN1L, YH4, text='DS-C3k2', fc=C_DS)
# upsample: H4→H3
box(XN1R-0.1, (YH3+YH4)/2+0.15, w=1.15, h=0.45, text='上采样', fc=C_UPS)
arr(XN1L+BW/2, YH4, XN1R-0.7, YH4)
arr(XN1R-0.1, YH4+0.23, XN1R-0.1, (YH3+YH4)/2+0.39)
arr(XN1R-0.1, (YH3+YH4)/2-0.09, XN1R, YH3-0.19)

# H5 level
box(XN1L, YH5, text='DS-C3k2', fc=C_DS)
# Cat sym H4 with upsample from H5
cat_sym(XN1R, YH4)
arr(XN1L+BW/2, YH4, XN1R-0.19, YH4)

# upsample H5→H4
box(XN1R-0.1, (YH4+YH5)/2+0.15, w=1.15, h=0.45, text='上采样', fc=C_UPS)
arr(XN1L+BW/2, YH5, XN1R-0.7, YH5)
arr(XN1R-0.1, YH5+0.23, XN1R-0.1, (YH4+YH5)/2+0.39)
arr(XN1R-0.1, (YH4+YH5)/2-0.09, XN1R, YH4-0.19)

# H5 from B5 down cat
cat_sym(XN1R, YH5)
arr(XN1L+BW/2, YH5, XN1R-0.19, YH5)

# ═══ NEW: P2 branch in Neck Tunnel 1 ══════════════════════════════════════════
# B2 → concat with (upsample from H3)
box(XN1L, YP2, text='DS-C3k2', fc=C_P2, ec_color=C_ARROW_NEW, lw=2.0)
arr(3.55, YP2, XN1L-BW/2, YP2, color=C_ARROW_NEW, lw=2.0)

cat_sym(XN1R, YP2)
arr(XN1L+BW/2, YP2, XN1R-0.19, YP2, color=C_ARROW_NEW, lw=1.8)

# upsample H3→P2
box(XN1R-0.1, (YP2+YH3)/2+0.1, w=1.15, h=0.45, text='上采样', fc='#FFBBBB',
    ec_color=C_ARROW_NEW, lw=1.8)
arr(XN1R, YH3+0.19, XN1R-0.1, (YP2+YH3)/2-0.12, color=C_ARROW_NEW, lw=1.5)
arr(XN1R-0.1, (YP2+YH3)/2+0.32, XN1R-0.1, YP2+0.19, color=C_ARROW_NEW, lw=1.5)
arr(XN1R-0.1, YP2, XN1R-0.19, YP2, color=C_ARROW_NEW, lw=1.5)

# H3 cat → gets feed from N1R up
arr(XN1R, YH3-0.19, XN1R, YH3-0.8)  # down from H3 cat
label(XN1R+0.08, YH3-0.5, 'H₃', fs=8, color='#444', fw='bold', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# NECK — Tunnel 2 (right side)
# ══════════════════════════════════════════════════════════════════════════════
# Connect from N1 cat symbols → N2L blocks
arr(XN1R+0.19, YH3, XN2L-BW/2, YH3)
arr(XN1R+0.19, YH4, XN2L-BW/2, YH4)
arr(XN1R+0.19, YH5, XN2L-BW/2, YH5)
arr(XN1R+0.19, YP2, XN2L-BW/2, YP2, color=C_ARROW_NEW, lw=1.8)

# N2 P2 block (new)
box(XN2L, YP2, text='DS-C3k2', fc=C_P2, ec_color=C_ARROW_NEW, lw=2.0)
arr(XN2L+BW/2, YP2, XN2R-0.19, YP2, color=C_ARROW_NEW, lw=1.8)

# N2 H3
box(XN2L, YH3, text='DS-C3k2', fc=C_DS)
arr(XN2L+BW/2, YH3, XN2R-0.19, YH3)
cat_sym(XN2R, YH3)

# Conv at H3 from below (from N2L-H4 path)
box((XN2L+XN2R)/2, (YP2+YH3)/2, text='Conv', fc=C_CONV, w=1.1, h=0.45)
# Connect P2 cat → Conv → H3 cat
arr(XN2R, YP2-0.19, XN2R, (YP2+YH3)/2+0.23, color=C_ARROW_NEW, lw=1.5)
arr(XN2R, (YP2+YH3)/2-0.23, XN2R, YH3+0.19, color=C_ARROW_NEW, lw=1.5)

# N2 H4
box(XN2L, YH4, text='DS-C3k2', fc=C_DS)
arr(XN2L+BW/2, YH4, XN2R-0.19, YH4)
cat_sym(XN2R, YH4)

# Conv between H3 and H4
box((XN2L+XN2R)/2, (YH3+YH4)/2, text='Conv', fc=C_CONV, w=1.1, h=0.45)
arr(XN2R, YH3-0.19, XN2R, (YH3+YH4)/2+0.23)
arr(XN2R, (YH3+YH4)/2-0.23, XN2R, YH4+0.19)

# N2 H5
box(XN2L, YH5, text='DS-C3k2', fc=C_DS)
arr(XN2L+BW/2, YH5, XN2R-0.19, YH5)
cat_sym(XN2R, YH5)

box((XN2L+XN2R)/2, (YH4+YH5)/2, text='DS-C3k2', fc=C_DS, w=1.3)
arr(XN2R, YH4-0.19, XN2R, (YH4+YH5)/2+0.23)
arr(XN2R, (YH4+YH5)/2-0.23, XN2R, YH5+0.19)

# B3/B4/B5 also connect to N2 (skip connections from backbone)
line(XN2L-BW/2-0.0, YH3, XN2L-BW/2, YH3)  # already done by arr above

# ══════════════════════════════════════════════════════════════════════════════
# HEAD + DETECTION + DiOU-NMS
# ══════════════════════════════════════════════════════════════════════════════
# P2 detection (NEW)
arr(XN2R+0.19, YP2, XHD-BW/2, YP2, color=C_ARROW_NEW, lw=1.8)
box(XHD, YP2, text='检测P2', fc=C_P2, ec_color=C_ARROW_NEW, lw=2.2, fs=9, fw='bold')
label(XHD, YP2+0.55, '★ 新增', fs=8, color=C_ARROW_NEW, fw='bold')

arr(XHD+BW/2, YP2, XDET-0.9, YP2, color=C_ARROW_NEW, lw=1.8)
box(XDET, YP2, w=1.55, h=0.52, text='DIoU-NMS', fc=C_NMS, ec_color='#CC6600',
    lw=2.2, fs=9, fw='bold')
label(XDET, YP2+0.55, '★ 新增', fs=8, color='#CC5500', fw='bold')

arr(XDET+1.55/2, YP2, XNMS, YP2, color=C_ARROW_NEW, lw=1.5)
label(XNMS+0.3, YP2, '输出P2', fs=8.5, color=C_ARROW_NEW, fw='bold', ha='left')

# H3 detection
arr(XN2R+0.19, YH3, XHD-BW/2, YH3)
box(XHD, YH3, text='检测H₃', fc=C_DET, fs=9)
arr(XHD+BW/2, YH3, XDET-0.9, YH3)
box(XDET, YH3, w=1.55, h=0.52, text='DIoU-NMS', fc=C_NMS, ec_color='#CC6600',
    lw=2.2, fs=9, fw='bold')
label(XDET, YH3+0.55, '★ 替换', fs=8, color='#CC5500', fw='bold')
arr(XDET+1.55/2, YH3, XNMS, YH3)
label(XNMS+0.3, YH3, '输出H₃', fs=8.5, ha='left')

# H4 detection
arr(XN2R+0.19, YH4, XHD-BW/2, YH4)
box(XHD, YH4, text='检测H₄', fc=C_DET, fs=9)
arr(XHD+BW/2, YH4, XDET-0.9, YH4)
box(XDET, YH4, w=1.55, h=0.52, text='DIoU-NMS', fc=C_NMS, ec_color='#CC6600',
    lw=2.2, fs=9, fw='bold')
label(XDET, YH4+0.55, '★ 替换', fs=8, color='#CC5500', fw='bold')
arr(XDET+1.55/2, YH4, XNMS, YH4)
label(XNMS+0.3, YH4, '输出H₄', fs=8.5, ha='left')

# H5 detection
arr(XN2R+0.19, YH5, XHD-BW/2, YH5)
box(XHD, YH5, text='检测H₅', fc=C_DET, fs=9)
arr(XHD+BW/2, YH5, XDET-0.9, YH5)
box(XDET, YH5, w=1.55, h=0.52, text='DIoU-NMS', fc=C_NMS, ec_color='#CC6600',
    lw=2.2, fs=9, fw='bold')
label(XDET, YH5+0.55, '★ 替换', fs=8, color='#CC5500', fw='bold')
arr(XDET+1.55/2, YH5, XNMS, YH5)
label(XNMS+0.3, YH5, '输出H₅', fs=8.5, ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# HyperACE
# ══════════════════════════════════════════════════════════════════════════════
hac_patch = FancyBboxPatch((3.3, 0.35), 8.0, 0.9, boxstyle='round,pad=0.08',
                            fc=C_HAC, ec='#8855AA', lw=1.5, zorder=3)
ax.add_patch(hac_patch)
ax.text(7.3, 0.80, 'HyperACE', ha='center', va='center', fontsize=10,
        fontweight='bold', color='#55228A', zorder=4)

# HyperACE connects to H3/H4/H5
for xv, yv, lbl in [(XN1R, YH3, 'H₃'), (XN1R, YH4, 'H₄'), (XN1R, YH5, 'H₅')]:
    line(xv, yv-0.6, xv, 1.25, color='#9966CC', lw=1.2, ls='--')
    label(xv+0.1, 1.55, lbl, fs=7.5, color='#7744AA', ha='left')

# HyperACE feature arrows (→)
for x_off in [4.5, 6.0, 7.5]:
    arr(x_off, 0.80, x_off+0.9, 0.80, color='#8855AA', lw=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
lx, ly = 0.2, 4.5
legend_items = [
    (C_P2,   C_ARROW_NEW, '① 新增 P2 检测头（stride=4，小目标增强）'),
    (C_NMS,  '#CC6600',   '② DIoU-NMS 替换原 IoU-NMS（密集框抑制优化）'),
    ('#E0F0FF', '#4488AA', '③ 场景化数据增强（训练前处理，不修改网络）'),
]
ax.add_patch(FancyBboxPatch((lx-0.1, ly-0.4), 3.1, len(legend_items)*0.75+0.6,
                            boxstyle='round,pad=0.1', fc='#FFFFF0',
                            ec='#AAAAAA', lw=1.2, zorder=5))
ax.text(lx+1.4, ly+len(legend_items)*0.75+0.1, '改进说明', ha='center',
        fontsize=9.5, fontweight='bold', color='#333', zorder=6)
for i, (fc, ec, txt) in enumerate(legend_items):
    yy = ly + (len(legend_items)-1-i)*0.75
    sq = FancyBboxPatch((lx, yy-0.18), 0.36, 0.36, boxstyle='round,pad=0.03',
                        fc=fc, ec=ec, lw=1.5, zorder=6)
    ax.add_patch(sq)
    ax.text(lx+0.5, yy, txt, ha='left', va='center', fontsize=8.0,
            color='#333', zorder=6)

# Title
ax.text(11, 0.0, '改进后 YOLOv13 架构图（含 P2 检测头 + DIoU-NMS）',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        color='#222244', zorder=6)

out = 'C:/Users/Administrator/Desktop/project/3/thesis_output/arch_improved.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')
