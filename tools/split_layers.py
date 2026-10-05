"""Split base.png into back (twin tails) / body / head layers and paint what each one hides.

Run after build_parts.py. Adds back.png, body.png, head.png to web/assets and parts.json.

  head : hair, face, ribbons, side locks (drawn on top, slides over the neck)
  body : uniform, neck (extended up behind the jaw), arms, skirt, legs
  back : lower twin tails (drawn behind the body, extended behind the sleeves)
"""
import json
import os

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'web', 'assets')
DEBUG = os.environ.get('SPLIT_DEBUG')

base = np.array(Image.open(os.path.join(OUT, 'base.png')).convert('RGBA')).astype(np.float32)
H, W = base.shape[:2]
rgb, alpha = base[..., :3], base[..., 3]
r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
lum = rgb.mean(-1)
yy, xx = np.mgrid[0:H, 0:W]
opaque = alpha > 20

CX = 1020
CUT_Y = 1000            # head / back split for the twin tails, and default head bottom
NECK_X = (944, 1094)    # neck outline columns
CHIN_Y = 952


def jaw_y(x):
    """y of the jaw outline (bottom of the face) for x inside the neck strip."""
    # piecewise linear through measured points: (945,920) (1020,952) (1095,920)
    return np.where(x < CX, 920 + (x - 945) / (CX - 945) * 32, 952 - (x - CX) / (1095 - CX) * 32)


# ---- body silhouette: big dark uniform blob + skin of neck/hands/legs
dark = ((lum < 120) & opaque).astype(np.uint8)
dark_big = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
lab, n = ndimage.label(dark_big)
sizes = ndimage.sum(dark_big, lab, range(1, n + 1))
uniform = ndimage.binary_fill_holes(lab == (np.argmax(sizes) + 1))
uniform = cv2.dilate(uniform.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # include the outline

skinish = opaque & (r > b + 4) & (lum > 170)
# +9 keeps the whole jaw outline (about 5px thick) on the head layer
neck_strip = (xx >= NECK_X[0] - 4) & (xx <= NECK_X[1] + 4) & (yy > jaw_y(xx) + 9) & (yy < 1130)

# ---- hair hanging below the jaw (side locks): light, bluish, connected to hair above
hair_light = opaque & (lum > 150) & (b >= r - 2) & ~skinish
band = (yy > 900) & (yy < 1130) & (xx > 760) & (xx < 1280)
lab, n = ndimage.label(hair_light & band)
keep = np.zeros(n + 1, bool)
top_labels = np.unique(lab[(yy < 960) & (lab > 0)])
keep[top_labels] = True
locks = keep[lab]
locks = cv2.dilate(locks.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
locks &= opaque

# ---- head mask
# centre: stop above the inner collar pieces (side locks are added back via `locks`)
above = yy < np.where((xx > 800) & (xx < 1240), 966, CUT_Y)
above &= ~(neck_strip)
above &= ~(uniform & (yy > 960))
sil = (uniform | dark_big.astype(bool) | neck_strip | (skinish & (yy > 940))) & opaque
# close the gaps left where hair strands cross the collar edge, so they get repainted instead of cut out
sil = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
sil = ndimage.binary_fill_holes(sil) & opaque
central = (xx > 800) & (xx < 1240) & (yy >= 940) & (yy < 1100)
loose_hair = central & opaque & ~sil                                  # strands over the background
stray = central & (yy < 1060) & cv2.dilate(hair_light.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & opaque
head = (above | locks | loose_hair | stray) & opaque & (yy > 100)
# the tops of the neck outlines (just under the jaw) belong to the neck, not the head
neck_cols = (np.abs(xx - NECK_X[0]) <= 6) | (np.abs(xx - NECK_X[1]) <= 6)
head &= ~(neck_cols & (yy > jaw_y(xx) + 4) & (yy < 1000))
head &= ~(dark_big.astype(bool) & (yy >= 940) & (xx > 800) & (xx < 1240))  # collar pieces stay on the body
head = ndimage.binary_closing(head, iterations=1) & opaque

# ---- back (twin tails below the cut)
side = (xx < 800) | (xx > 1240)
back = opaque & side & (yy >= CUT_Y - 6) & (yy < 1800) & ~uniform & ~(skinish & (yy > 1500))
back &= ~head | (yy >= CUT_Y - 6)
lab, n = ndimage.label(back)
sizes = ndimage.sum(back, lab, range(1, n + 1))
back = np.isin(lab, np.where(sizes > 400)[0] + 1)

# ---- legs: thighs down to the shoes, drawn behind the skirt so the skirt can swing without bending them
skirt_area = np.zeros((H, W), np.uint8)
cv2.fillPoly(skirt_area, [np.array([(720, 1690), (1320, 1690), (1430, 2020), (1490, 2290), (1600, 2470),
                                    (490, 2470), (560, 2290), (610, 2020)], np.int32)], 1)
skirt_area = (skirt_area > 0) & opaque & ~skinish
legs = opaque & (yy > 2330) & (xx > 480) & (xx < 1580) & ~skirt_area
# skirt lining (large dark area) stays with the skirt; thin dark lines (thigh outlines) stay on the legs
lining = cv2.morphologyEx(((lum < 120) & (yy < 2600)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
legs &= ~cv2.dilate(lining.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
lab, n = ndimage.label(legs)
legs = np.isin(lab, np.unique(lab[(yy > 3200) & legs]))      # keep what connects down to the feet (drops the hands)

# ---- body
body_m = opaque & ~head & ~(back & (yy >= CUT_Y)) & ~legs


def feather(m, s=0.7):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), s)


def crop_save(img, mask_a, name, pad=24):
    ys, xs = np.where(mask_a > 0.01)
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(W, xs.max() + pad), min(H, ys.max() + pad)
    out = img[y0:y1, x0:x1].copy()
    out[..., 3] = np.clip(mask_a[y0:y1, x0:x1], 0, 1) * 255
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(OUT, name + '.png'))
    return {'file': name + '.png', 'x': int(x0), 'y': int(y0), 'w': int(x1 - x0), 'h': int(y1 - y0)}


# head layer: plain cut (overlap 8px into the tails so there is no seam)
head_a = np.minimum(feather(head), alpha / 255)
head_a[(yy >= CUT_Y) & (yy < CUT_Y + 8) & side & back] = 1.0 * (alpha[(yy >= CUT_Y) & (yy < CUT_Y + 8) & side & back] / 255)

# body layer: repaint what the head and hair covered
body = base.copy()
# 2) collar / neck skin under the side locks: inpaint
under = ((locks | head) & sil & (yy >= 940)) | (central & sil & ~skinish & (lum > 120))
under = cv2.dilate(under.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
# fill from nearby collar / skin pixels only (never from hair), small kernel first, wider as fallback
y0i, y1i, x0i, x1i = 900, 1180, 700, 1340
hair_any = cv2.dilate(hair_light.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) | locks | head
valid = (sil & ~under & ~hair_any)[y0i:y1i, x0i:x1i].astype(np.float32)
src_rgb = body[y0i:y1i, x0i:x1i, :3]
acc = np.zeros_like(src_rgb)
acc_t = np.zeros(valid.shape + (1,), np.float32)
for sigma in (4, 10, 30):
    num = np.stack([cv2.GaussianBlur(src_rgb[..., k] * valid, (0, 0), sigma) for k in range(3)], -1)
    den = cv2.GaussianBlur(valid, (0, 0), sigma)[..., None]
    t = np.clip(den / 0.15, 0, 1)
    acc += num / np.maximum(den, 1e-5) * (1 - acc_t) * t
    acc_t += (1 - acc_t) * t
fill = acc / np.maximum(acc_t, 1e-5)
m = under[y0i:y1i, x0i:x1i]
src_rgb[m] = fill[m]
# 1) neck continues up behind the jaw: repeat a row taken just under the chin shadow
SRC_ROW = 972
x0n, x1n = NECK_X[0] - 6, NECK_X[1] + 6
row = base[SRC_ROW, x0n:x1n].copy()
neck_alpha = np.zeros((H, W), np.float32)
for y in range(780, SRC_ROW):
    body[y, x0n:x1n] = row
    neck_alpha[y, x0n:x1n] = row[:, 3] / 255
body_alpha = feather(body_m) * alpha / 255
body_alpha = np.maximum(body_alpha, under * 1.0)
body_alpha = np.maximum(body_alpha, neck_alpha)
body_alpha[(yy < 780)] = 0

# back layer: extend tails ~50px in behind the sleeves (nearest tail colour)
back_a = feather(back) * alpha / 255
behind = (cv2.dilate(back.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))) > 0) & uniform & ~back
dist, (iy, ix) = ndimage.distance_transform_edt(~back, return_indices=True)
back_img = base.copy()
back_img[behind] = base[iy[behind], ix[behind]]
back_a = np.maximum(back_a, behind * 1.0)

# sway weight for the face-framing side locks (B channel of weights.png, sampled by the head layer).
# Only hair pixels: the face (skin + its outline) is excluded so the cheek line never wobbles.
face = ndimage.binary_fill_holes(skinish & (yy > 600) & (yy < 965) & (xx > 800) & (xx < 1240))
face = cv2.dilate(face.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0
band = ((xx > 770) & (xx < 1000)) | ((xx > 1040) & (xx < 1270))
sway = head & (hair_light | cv2.dilate(hair_light.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)) & ~face & band
sway = cv2.GaussianBlur(sway.astype(np.float32), (0, 0), 6)
sway = np.clip(sway * 1.8, 0, 1) * np.clip((yy - 740) / 340, 0, 1) ** 1.2
sway[face] = 0
wpath = os.path.join(OUT, 'weights.png')
wimg = np.array(Image.open(wpath).convert('RGB'))
wimg[..., 2] = (cv2.resize(sway, (wimg.shape[1], wimg.shape[0]), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
Image.fromarray(wimg).save(wpath)

# masks of repainted areas, for ai/lama_refine.py (optional AI repaint)
os.makedirs(os.path.join(OUT, '_masks'), exist_ok=True)
Image.fromarray((under & (body_alpha > 0.5)).astype(np.uint8) * 255).save(os.path.join(OUT, '_masks', 'body_fill.png'))
Image.fromarray(behind.astype(np.uint8) * 255).save(os.path.join(OUT, '_masks', 'back_fill.png'))

parts = json.load(open(os.path.join(OUT, 'parts.json'), encoding='utf-8'))
# legs layer: extend every thigh column ~150px up under the skirt by repeating the pixel just below its top
legs_img = base.copy()
legs_a = feather(legs) * alpha / 255
for x in range(480, 1580):
    col = np.where(legs[2330:2560, x])[0]
    if len(col) == 0:
        continue
    top = 2330 + col[0]
    srcy = min(top + 8, H - 1)
    if not legs[srcy, x]:
        continue
    legs_img[2220:top + 8, x] = base[srcy, x]
    legs_a[2220:top + 8, x] = alpha[srcy, x] / 255

for name, img, a in [('back', back_img, back_a), ('legs', legs_img, legs_a), ('body', body, body_alpha), ('head', base, head_a)]:
    info = crop_save(img, a, name)
    info['mesh'] = name
    parts['layers'][name] = info
parts['cut'] = {'y': CUT_Y, 'chin': CHIN_Y}
parts['layers'].pop('base', None)  # replaced by the three layers
json.dump(parts, open(os.path.join(OUT, 'parts.json'), 'w', encoding='utf-8'), indent=1)

if DEBUG:
    vis = np.zeros((H, W, 3), np.uint8) + 255
    vis[body_m] = (90, 160, 90)
    vis[back] = (90, 90, 200)
    vis[head] = (220, 120, 120)
    Image.fromarray(vis).resize((W // 4, H // 4)).save(os.path.join(DEBUG, 'split_vis.png'))
print('split ok', {k: parts['layers'][k]['w'] for k in ('back', 'body', 'head')})
