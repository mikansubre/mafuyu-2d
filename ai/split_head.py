"""Split the head into Live2D-style parts and paint what each part hides (LaMa).

Input : web/assets/head.png (+ parts.json placement), 4_4.png
Output: parts/head/*.png  full-canvas RGBA layers
  hair_back, face, brow_L, brow_R, side_L, side_R, bangs, ribbon_L, ribbon_R, ahoge
Run with ai/venv python, after tools/split_layers.py.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, 'web', 'assets')
OUT = os.path.join(ROOT, 'parts', 'head')
os.makedirs(OUT, exist_ok=True)
DEBUG = os.environ.get('SPLIT_DEBUG')

src = np.array(Image.open(os.path.join(ROOT, '4_4.png')).convert('RGBA')).astype(np.float32)
H, W = src.shape[:2]
meta = json.load(open(os.path.join(ASSETS, 'parts.json'), encoding='utf-8'))
L = meta['layers']['head']
head = np.zeros((H, W, 4), np.float32)
head[L['y']:L['y'] + L['h'], L['x']:L['x'] + L['w']] = np.array(Image.open(os.path.join(ASSETS, L['file'])).convert('RGBA'))
# the eyes / mouth were removed from head.png (they are separate parts); that is what we want for the face
ha = head[..., 3] > 20
rgb = head[..., :3]
r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
lum = rgb.mean(-1)
yy, xx = np.mgrid[0:H, 0:W]
mx, mn = rgb.max(-1), rgb.min(-1)
sat = (mx - mn) / np.maximum(mx, 1)


def poly(points):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(points, np.int32)], 1)
    return m > 0


def dil(m, k):
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


# ---------------------------------------------------------------- face
skin = ha & (r > b + 4) & (lum > 200) & (yy > 560) & (yy < 960) & (xx > 790) & (xx < 1250)
lab, n = ndimage.label(skin)
sizes = ndimage.sum(skin, lab, range(1, n + 1))
skin = np.isin(lab, np.where(sizes > 300)[0] + 1)
hull = cv2.convexHull(np.column_stack(np.where(skin)[::-1]).astype(np.int32))
face_hull = np.zeros((H, W), np.uint8)
cv2.fillPoly(face_hull, [hull], 1)
face_hull = face_hull > 0
# skull: forehead continues up under the bangs (needed so the face can move behind them)
skull = ((xx - 1020) / 215.0) ** 2 + ((yy - 650) / 345.0) ** 2 <= 1
face_region = (face_hull | (skull & (yy < 760))) & ~((yy < 650) & ~skull)
# face outline (dark line along the cheek/jaw) belongs to the face
ring = dil(face_hull, 15) & ~(cv2.erode(face_hull.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
face_line = ring & ha & (lum < 90) & (yy > 700)
face_vis = (skin | face_line | (face_hull & ha & (lum > 150) & (r > b))) & ha

# ---------------------------------------------------------------- brows (thin arcs drawn across the bangs)
# traced by hand; the brow = dark pixels within a few px of the trace
BROW_TRACE = {'L': [(822, 613), (850, 601), (875, 595), (912, 590)],
              'R': [(1106, 591), (1150, 590), (1197, 599), (1221, 609)]}
brows, brow_band = {}, {}
for k, pts in BROW_TRACE.items():
    band = np.zeros((H, W), np.uint8)
    cv2.polylines(band, [np.array(pts, np.int32)], False, 1, thickness=9)
    band = band > 0
    brow_band[k] = band
    brows[k] = band & ha & (lum < 165)

# ---------------------------------------------------------------- ribbons + gems, ahoge
rib_boxes = {'L': [(690, 395), (790, 395), (790, 1000), (620, 1000), (620, 560)],
             'R': [(1250, 395), (1350, 395), (1420, 560), (1420, 1000), (1255, 1000)]}
ribbon = {}
for k, pts in rib_boxes.items():
    box = poly(pts)
    gem = box & (yy < 520) & ((sat > 0.35) | ((lum > 150) & (sat < 0.08) & (yy < 500)))
    dark = box & ha & (lum < 125)
    teal = box & ha & (g > r + 18) & (b > r + 10) & (sat > 0.18)
    # thick cloth only: open away thin dark lines (hair strand outlines), then grow back to the ribbon outline
    core = cv2.morphologyEx((dark | teal).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
    m = gem | (dil(core, 7) & (dark | teal | (ha & (lum < 170))))
    m = ndimage.binary_closing(m, iterations=2)
    holes = ndimage.binary_fill_holes(m) & ~m
    lab_h, nh = ndimage.label(holes)
    small = np.isin(lab_h, np.where(ndimage.sum(holes, lab_h, range(1, nh + 1)) < 400)[0] + 1)
    m = (m | small) & ha
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    ribbon[k] = np.isin(lab, np.where(sizes > 1500)[0] + 1) | (gem & ha)
ahoge = ha & (yy < 292) & (xx > 990)

# ---------------------------------------------------------------- side locks (hair framing the face, in front)
side_zone = {'L': poly([(785, 560), (815, 560), (900, 760), (1000, 1080), (760, 1080), (770, 700)]),
             'R': poly([(1225, 560), (1255, 560), (1290, 700), (1290, 1080), (1050, 1080), (1150, 760)])}
hair = ha & ~face_vis & ~ribbon['L'] & ~ribbon['R'] & ~ahoge & ~brows['L'] & ~brows['R']
side = {k: hair & z & ~face_hull for k, z in side_zone.items()}
# lock tips hanging over the collar (below the jaw, between the twin tails) are in front too
low = hair & (yy > 930) & (xx > 760) & (xx < 1280) & ~side['L'] & ~side['R'] & ((lum > 140) | dil(lum > 170, 5))
side['L'] |= low & (xx < 1020)
side['R'] |= low & (xx >= 1020)

for k in side:   # drop specks (stray shading lines picked up near the neck)
    lab, n = ndimage.label(side[k])
    side[k] = np.isin(lab, np.where(ndimage.sum(side[k], lab, range(1, n + 1)) >= 150)[0] + 1)
# ---------------------------------------------------------------- bangs: hair in front of the face / forehead
bangs = hair & face_region & ~side['L'] & ~side['R']
bangs |= hair & skull                                  # crown down to the fringe (cut follows the head curve)
bangs &= ~(side['L'] | side['R'])
bangs &= yy < 880                                       # nothing of the fringe reaches the mouth
# ---------------------------------------------------------------- back hair: everything else that is hair
hair_back = hair & ~bangs & ~side['L'] & ~side['R']

parts = {'hair_back': hair_back, 'face': face_vis, 'brow_L': brows['L'], 'brow_R': brows['R'],
         'side_L': side['L'], 'side_R': side['R'], 'bangs': bangs,
         'ribbon_L': ribbon['L'], 'ribbon_R': ribbon['R'], 'ahoge': ahoge}

if DEBUG:
    col = {'hair_back': (120, 120, 220), 'face': (250, 200, 180), 'brow_L': (255, 0, 0), 'brow_R': (255, 0, 0),
           'side_L': (60, 200, 60), 'side_R': (60, 160, 200), 'bangs': (240, 200, 60),
           'ribbon_L': (60, 60, 60), 'ribbon_R': (60, 60, 60), 'ahoge': (255, 0, 255)}
    vis = np.full((H, W, 3), 255, np.uint8)
    for k, m in parts.items():
        vis[m] = col[k]
    left = ha & ~np.logical_or.reduce(list(parts.values()))
    vis[left] = (255, 0, 0)
    Image.fromarray(vis[150:1100, 560:1480]).resize((460, 475)).save(os.path.join(DEBUG, 'head_split.png'))
    print('unassigned px:', int(left.sum()))

np.savez_compressed(os.path.join(OUT, '_masks.npz'), **parts, face_region=face_region, skull=skull,
                    brow_band=brow_band['L'] | brow_band['R'])
print('masks ok')
