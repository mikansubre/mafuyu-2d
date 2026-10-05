"""Split the flat illustration into rig parts for the web Live2D-style viewer.

Outputs into ../web/assets:
  base.png               full body with eyes / mouth filled with skin
  eye{L,R}_white.png     sclera (iris painted out)
  eye{L,R}_iris.png      iris only (moves for eye tracking)
  eye{L,R}_mask.png      eye opening mask (clips the iris)
  eye{L,R}_line.png      lashes / lids around the opening
  eye{L,R}_closed.png    drawn closed-eye line (blink)
  eye{L,R}_happy.png     ^ ^ eyes taken from IMG_2344
  mouth_smile.png        original mouth line
  mouth_open.png         open mouth taken from IMG_2344
  weights_a.png          R=head  G=twin tails  B=ahoge          (1/4 res)
  weights_b.png          R=upper body  G=skirt  B=front side hair (1/4 res)
  parts.json             placement of every part
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from regions import EYE_OPEN, EYE_PATCH, IRIS, MOUTH_BOX, NOSE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'web', 'assets')
os.makedirs(OUT, exist_ok=True)

src = np.array(Image.open(os.path.join(ROOT, '4_4.png')).convert('RGBA')).astype(np.float32)
H, W = src.shape[:2]
ref = np.array(Image.open(os.path.join(ROOT, 'IMG_2344.JPG')).convert('RGB')).astype(np.float32)

parts = {'size': [W, H], 'layers': {}}
# keep the baked THA3 flow from a previous AI build (ai/export_flow.py) so --no-ai rebuilds still use it
_prev = os.path.join(OUT, 'parts.json')
if os.path.exists(_prev) and os.path.exists(os.path.join(OUT, 'flow.bin')):
    _flow = json.load(open(_prev, encoding='utf-8')).get('flow')
    if _flow:
        parts['flow'] = _flow


def poly_mask(points, shape=(H, W), feather=0.0):
    m = Image.new('L', (shape[1], shape[0]), 0)
    ImageDraw.Draw(m).polygon(points, fill=255)
    m = np.array(m).astype(np.float32) / 255
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m


def bbox_of(points, pad):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(min(xs)) - pad, int(min(ys)) - pad, int(max(xs)) + pad, int(max(ys)) + pad)


def save_rgba(arr, name, box):
    x0, y0, x1, y1 = box
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(OUT, name))
    parts['layers'][name.rsplit('.', 1)[0]] = {'file': name, 'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}


def skin_fill(img, mask, sigma=14):
    """Fill `mask` with a smooth skin gradient estimated from nearby skin-like pixels."""
    rgb = img[..., :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = rgb.mean(-1)
    skin = (lum > 228) & (r >= b + 2) & (img[..., 3] > 250) & (mask < 0.01)
    skin = skin.astype(np.float32)
    out = img.copy()
    num = np.stack([cv2.GaussianBlur(rgb[..., c] * skin, (0, 0), sigma) for c in range(3)], -1)
    den = cv2.GaussianBlur(skin, (0, 0), sigma)[..., None]
    fill = num / np.maximum(den, 1e-4)
    # second pass with a wider kernel where the first one had no support
    num2 = np.stack([cv2.GaussianBlur(rgb[..., c] * skin, (0, 0), sigma * 3) for c in range(3)], -1)
    den2 = cv2.GaussianBlur(skin, (0, 0), sigma * 3)[..., None]
    fill2 = num2 / np.maximum(den2, 1e-4)
    t = np.clip(den / 0.25, 0, 1)
    fill = fill * t + fill2 * (1 - t)
    m = mask[..., None]
    out[..., :3] = rgb * (1 - m) + fill * m
    out[..., 3] = np.maximum(img[..., 3], mask * 255)
    return out


base = src.copy()

# ---------------------------------------------------------------- eyes
for side in 'LR':
    box = bbox_of(EYE_PATCH[side], 10)
    x0, y0, x1, y1 = box
    crop = src[y0:y1, x0:x1].copy()
    loc = lambda pts: [(px - x0, py - y0) for px, py in pts]  # noqa: E731
    shp = crop.shape[:2]
    patch_m = poly_mask(loc(EYE_PATCH[side]), shp, 2.0)
    open_hard = poly_mask(loc(EYE_OPEN[side]), shp)
    open_m = cv2.GaussianBlur(open_hard, (0, 0), 0.8)

    (cx, cy), (rx, ry) = IRIS[side]
    yy, xx = np.mgrid[0:shp[0], 0:shp[1]]
    ell = (((xx - (cx - x0)) / rx) ** 2 + ((yy - (cy - y0)) / ry) ** 2) <= 1.0
    rgb = crop[..., :3]
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1)
    irisish = ell & ((sat > 0.18) | (mn < 150))
    irisish = ndimage.binary_closing(irisish, iterations=2)
    irisish = ndimage.binary_fill_holes(irisish)
    iris_m = cv2.GaussianBlur(irisish.astype(np.float32), (0, 0), 0.7) * np.clip(open_m * 1.5, 0, 1)

    # sclera: opening area with the iris painted out
    inp_mask = (cv2.dilate(irisish.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).astype(np.uint8) * 255
    white_rgb = cv2.inpaint(np.ascontiguousarray(rgb.astype(np.uint8)), inp_mask, 6, cv2.INPAINT_TELEA).astype(np.float32)
    white = np.dstack([white_rgb, open_m * 255])
    iris = np.dstack([rgb, iris_m * 255])
    line_a = np.clip(patch_m - open_hard * 0.0, 0, 1) * (1 - open_m) * crop[..., 3] / 255
    line = np.dstack([rgb, line_a * 255])
    mask_img = np.dstack([np.full_like(rgb, 255), open_m * 255])

    save_rgba(white, f'eye{side}_white.png', box)
    save_rgba(iris, f'eye{side}_iris.png', box)
    save_rgba(line, f'eye{side}_line.png', box)
    save_rgba(mask_img, f'eye{side}_mask.png', box)

    # blank the eye on the base layer
    full_m = np.zeros((H, W), np.float32)
    full_m[y0:y1, x0:x1] = np.maximum(patch_m, open_m)
    base = skin_fill(base, full_m)

    # closed-eye line: drawn at 4x then downsampled for anti-aliasing
    S = 4
    cl = Image.new('RGBA', ((x1 - x0) * S, (y1 - y0) * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(cl)
    op = EYE_OPEN[side]
    left = min(op, key=lambda p: p[0])
    right = max(op, key=lambda p: p[0])
    ly = max(p[1] for p in op) - 12
    pts = []
    for i in range(41):
        t = i / 40
        px = left[0] - 4 + (right[0] - left[0] + 8) * t
        py = ly + 10 * np.sin(np.pi * t) - 4 * (1 - t if side == 'L' else t)
        pts.append(((px - x0) * S, (py - y0) * S))
    col = (46, 46, 66, 255)
    for i in range(40):
        t = i / 40
        wdt = (3.0 + 4.5 * np.sin(np.pi * t)) * S
        d.line([pts[i], pts[i + 1]], fill=col, width=int(wdt))
    # small lash flick on the outer corner
    outer = pts[0] if side == 'L' else pts[-1]
    dx = -1 if side == 'L' else 1
    d.line([outer, (outer[0] + dx * 10 * S, outer[1] - 7 * S)], fill=col, width=4 * S)
    cl = cl.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    cl.save(os.path.join(OUT, f'eye{side}_closed.png'))
    parts['layers'][f'eye{side}_closed'] = {'file': f'eye{side}_closed.png', 'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}
    parts['layers'][f'eye{side}_white']['pivotY'] = max(p[1] for p in EYE_OPEN[side]) - 6
    parts['layers'][f'eye{side}_white']['iris'] = [cx, cy]

# ---------------------------------------------------------------- mouth (original smile)
mx0, my0, mx1, my1 = MOUTH_BOX
mcrop = src[my0:my1, mx0:mx1].copy()
lum = mcrop[..., :3].mean(-1)
dark = np.clip((235 - lum) / 80, 0, 1)
dark = cv2.GaussianBlur(cv2.dilate(dark, np.ones((3, 3), np.float32)), (0, 0), 0.6)
save_rgba(np.dstack([mcrop[..., :3], dark * 255]), 'mouth_smile.png', MOUTH_BOX)
mm = np.zeros((H, W), np.float32)
mm[my0:my1, mx0:mx1] = cv2.GaussianBlur((dark > 0.05).astype(np.float32), (0, 0), 3) * 1.0
mm = np.clip(cv2.dilate(mm, np.ones((7, 7), np.float32)) * 1.5, 0, 1)
base = skin_fill(base, mm, sigma=8)

# ---------------------------------------------------------------- parts from IMG_2344
REF_NOSE = (771, 570)
SCALE = 1.2


def ref_part(box, name, kind):
    rx0, ry0, rx1, ry1 = box
    c = ref[ry0:ry1, rx0:rx1].copy()
    lum = c.mean(-1)
    if kind == 'line':
        a = np.clip((225 - lum) / 120, 0, 1)
        rgb = np.zeros_like(c)
        rgb[...] = (52, 50, 72)
        out = np.dstack([rgb, a * 255])
    else:  # filled mouth
        skin_ref = np.median(np.concatenate([c[0], c[-1]]), axis=0)
        dist = np.linalg.norm(c - skin_ref, axis=-1)
        m = dist > 28
        m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            m = lab == (np.argmax(sizes) + 1)
        a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
        out = np.dstack([c, a * 255])
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGBA')
    nw, nh = int(round((rx1 - rx0) * SCALE)), int(round((ry1 - ry0) * SCALE))
    img = img.resize((nw, nh), Image.LANCZOS)
    tx = NOSE[0] + (rx0 - REF_NOSE[0]) * SCALE
    ty = NOSE[1] + (ry0 - REF_NOSE[1]) * SCALE
    img.save(os.path.join(OUT, name + '.png'))
    parts['layers'][name] = {'file': name + '.png', 'x': int(round(tx)), 'y': int(round(ty)), 'w': nw, 'h': nh}


ref_part((624, 482, 731, 530), 'eyeL_happy', 'line')
ref_part((811, 486, 919, 530), 'eyeR_happy', 'line')
ref_part((732, 592, 808, 642), 'mouth_open', 'fill')

Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(OUT, 'base.png'))
parts['layers']['base'] = {'file': 'base.png', 'x': 0, 'y': 0, 'w': W, 'h': H}

with open(os.path.join(OUT, 'parts.json'), 'w') as f:
    json.dump(parts, f, indent=1)
print('ok', list(parts['layers']))

# ---------------------------------------------------------------- deformation weight maps (1/4 res)
Q = 4
small = cv2.resize(src, (W // Q, H // Q), interpolation=cv2.INTER_AREA)
sh, sw = small.shape[:2]
alpha = small[..., 3] > 20
lum = small[..., :3].mean(-1)
yy, xx = np.mgrid[0:sh, 0:sw] * Q

# body = biggest blob of large dark areas (uniform + sleeves + skirt)
dark = (lum < 120) & alpha
dark_big = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
lab, n = ndimage.label(dark_big)
sizes = ndimage.sum(dark_big, lab, range(1, n + 1))
body = lab == (np.argmax(sizes) + 1)
body = ndimage.binary_fill_holes(body)
skin = alpha & (lum > 200) & (small[..., 0] > small[..., 2] + 3)
body_or_skin = body | (skin & (yy > 1000))
dist = ndimage.distance_transform_edt(~body_or_skin) * Q  # px (full res) to nearest body pixel

side = (xx < 790) | (xx > 1250)
tail = alpha & side & (yy > 640) & (yy < 1700) & ~body_or_skin
# fill the gaps between strands so neighbouring vertices get similar weights
tail = cv2.morphologyEx(tail.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
tail = cv2.dilate(tail.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
tail &= side & ~body_or_skin
w_tail = cv2.GaussianBlur(tail.astype(np.float32), (0, 0), 3.0)
w_tail = np.clip(w_tail * 1.6, 0, 1) * np.clip((dist - 6) / 60, 0, 1)
w_tail *= np.clip((yy - 700) / 260, 0, 1)
w_tail[body_or_skin] = 0

skirt_poly = poly_mask([(p[0] // Q, p[1] // Q) for p in [(720, 1690), (1320, 1690), (1430, 2020), (1490, 2290), (1600, 2470), (490, 2470), (560, 2290), (610, 2020)]], (sh, sw))
skirt = (skirt_poly > 0.5) & alpha & ~skin
skirt = cv2.GaussianBlur(skirt.astype(np.float32), (0, 0), 1.0)
skirt *= np.clip((yy - 1760) / 650, 0, 1) ** 1.5
# never move skin (thighs, hands): the blur above would otherwise spill a few px onto them
skirt[cv2.dilate(skin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0

wa = np.dstack([w_tail, skirt, np.zeros_like(w_tail)]) * 255
Image.fromarray(np.clip(wa, 0, 255).astype(np.uint8), 'RGB').save(os.path.join(OUT, 'weights.png'))
parts['weights'] = {'file': 'weights.png', 'scale': Q, 'w': sw, 'h': sh}
ys, xs = np.where(alpha)
parts['bbox'] = [int(xs.min() * Q), int(ys.min() * Q), int(xs.max() * Q + Q), int(ys.max() * Q + Q)]
with open(os.path.join(OUT, 'parts.json'), 'w') as f:
    json.dump(parts, f, indent=1)
print('weights ok')
