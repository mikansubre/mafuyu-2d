"""Repaint hidden areas with LaMa (big-lama) instead of flat colour fills.

  python lama_refine.py base   after build_parts.py : eyes / mouth removed from base.png
  python lama_refine.py layers after split_layers.py: collar under the hair (body.png),
                                                      tails behind the sleeves (back.png)
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'web', 'assets')
MASKS = os.path.join(OUT, '_masks')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from regions import EYE_OPEN, EYE_PATCH, MOUTH_BOX  # noqa: E402
from simple_lama_inpainting import SimpleLama  # noqa: E402

lama = SimpleLama()


def inpaint(rgba, mask, box, matte):
    """LaMa on rgba[box] where mask[box] > 0; transparent pixels are shown to LaMa over `matte`."""
    x0, y0, x1, y1 = box
    c = rgba[y0:y1, x0:x1].astype(np.float32)
    a = c[..., 3:4] / 255
    rgb = c[..., :3] * a + np.array(matte, np.float32) * (1 - a)
    m = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    res = np.array(lama(Image.fromarray(rgb.astype(np.uint8)), Image.fromarray(m)))[:y1 - y0, :x1 - x0]
    # blend back with a soft edge so the seam is invisible
    soft = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.5)
    soft = np.maximum(soft, m / 255)[..., None]
    out = rgba[y0:y1, x0:x1].astype(np.float32)
    out[..., :3] = out[..., :3] * (1 - soft) + res * soft
    rgba[y0:y1, x0:x1] = out.astype(np.uint8)


def stage_base():
    path = os.path.join(OUT, 'base.png')
    base = np.array(Image.open(path).convert('RGBA'))
    src = np.array(Image.open(os.path.join(ROOT, '4_4.png')).convert('RGBA'))
    H, W = base.shape[:2]
    m = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(m)
    for k in 'LR':
        d.polygon(EYE_PATCH[k], fill=255)
        d.polygon(EYE_OPEN[k], fill=255)
    mx0, my0, mx1, my1 = MOUTH_BOX
    d.rectangle((mx0 + 8, my0 + 6, mx1 - 8, my1 - 6), fill=255)
    m = cv2.dilate(np.array(m), np.ones((7, 7), np.uint8))
    # start from the original art so LaMa sees the real surroundings, not the old fill
    work = src.copy()
    inpaint(work, m, (760, 560, 1296, 944), (255, 255, 255))
    sel = m > 0
    soft = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.5)
    region = np.maximum(soft, sel)[..., None]
    base = (base * (1 - region) + work * region).astype(np.uint8)
    Image.fromarray(base).save(path)
    print('base: eyes / mouth repainted')


def load_layer(name, parts):
    l = parts['layers'][name]
    return l, np.array(Image.open(os.path.join(OUT, l['file'])).convert('RGBA'))


def stage_layers():
    parts = json.load(open(os.path.join(OUT, 'parts.json'), encoding='utf-8'))
    # body: LaMa smears the dark collar into the neck skin, so the collar keeps the plain fill from split_layers
    for name, matte in (('back', (225, 228, 238)),):
        mpath = os.path.join(MASKS, f'{name}_fill.png')
        if not os.path.exists(mpath):
            continue
        l, img = load_layer(name, parts)
        full = np.array(Image.open(mpath).convert('L'))
        mask = full[l['y']:l['y'] + l['h'], l['x']:l['x'] + l['w']]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        pad = 48
        box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(l['w'], xs.max() + pad), min(l['h'], ys.max() + pad))
        inpaint(img, mask, box, matte)
        Image.fromarray(img).save(os.path.join(OUT, l['file']))
        print(f'{name}: hidden area repainted')


if __name__ == '__main__':
    {'base': stage_base, 'layers': stage_layers}[sys.argv[1]]()
