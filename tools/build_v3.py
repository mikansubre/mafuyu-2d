"""Build the Cubism FREE PSD (v3) straight from the artist's original 立ち絵差分 PSD.

Run with ai/venv python:  ai/venv/Scripts/python.exe tools/build_v3.py [scale]
Input:  original/七瀬真冬さん立ち絵差分.psd   (artist's file, 2000x4156, untouched)
Output: v3/dist/mafuyu_v3_free.psd             (scaled, fits Cubism FREE's single 2048 texture)
        v3/parts/*.png                         (full-size previews of each layer)

v3 is a "2D" model: no head turning, so the artist's merged body layer is used as is.
Changes made on the way:
- winter uniform + black socks only (summer / white socks are left for a later model)
- the body is cut in three (upper body / skirt / legs) so each fits in the 2048 atlas
- the twin tails are split into L / R (screen side) for separate physics
- stray specks far from the drawing are dropped so the layer bounds stay tight
- each expression is split into eyes (+ brows, blush, nose) and mouth, so blinking and lip sync
  work independently; only the EYES / MOUTHS listed below are kept
- spread-out effect marks (hearts, sparkles...) are split into one layer per cluster
"""
import os
import sys

import cv2
import numpy as np
import rectpack
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer
from psd_tools.constants import BlendMode, SectionDivider, Tag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'original', '七瀬真冬さん立ち絵差分.psd')
V3 = os.path.join(ROOT, 'v3')
OUT = os.path.join(V3, 'dist', 'mafuyu_v3_free.psd')
PARTS = os.path.join(V3, 'parts')
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
ATLAS = 2048
BODY_CUT = 2000      # y where the merged body is cut into upper body / skirt (through the opaque skirt)
LEGS_CUT = 2480      # skirt hem; below it only the (much narrower) legs
OVERLAP = 6          # rows both halves keep, so no seam shows
CLUSTER = 40         # px; marks closer than this belong to the same piece
MIN_PIECE = 40       # px of alpha; smaller isolated pieces are strays
MOUTH_BOX = (965, 852, 1080, 930)  # x0, y0, x1, y1: every expression's mouth sits inside this
# expression numbers whose eyes (+ brows, blush, nose) are kept; the rest look alike or don't fit the atlas
EYES = {1: '通常', 2: '笑顔', 5: '閉じ', 7: '白丸', 8: 'ウィンク', 10: 'ぎゅっ', 12: 'ジト目', 13: '黒目',
        14: 'ぐるぐる', 18: '泣き', 24: 'ハート', 25: 'キラキラ'}
# expression numbers whose mouth is kept (8 = 1, 11 = 10 ≈ 2, 14 = 7, 25 = 24 are duplicates)
MOUTHS = [1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


def find(group, *path):
    for name in path:
        group = next(l for l in group if l.name == name)
    return group


def full(layer, W, H):
    img = np.zeros((H, W, 4), np.uint8)
    pil = layer.topil()  # composite() renders hidden layers empty
    if pil is not None:
        x0, y0 = max(layer.left, 0), max(layer.top, 0)
        a = np.array(pil.convert('RGBA'))[y0 - layer.top:, x0 - layer.left:]
        a = a[:H - y0, :W - x0]
        img[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    return img


def pieces(img):
    """Clusters of the drawing (strays dropped), as a list of masks."""
    a = (img[..., 3] > 0).astype(np.uint8)
    near = cv2.dilate(a, np.ones((CLUSTER, CLUSTER), np.uint8))
    n, lab = cv2.connectedComponents(near, connectivity=8)
    out = []
    for i in range(1, n):
        m = (lab == i) & (a > 0)
        if m.sum() >= MIN_PIECE:
            out.append(m)
    return out


def clean(img):
    keep = np.zeros(img.shape[:2], bool)
    for m in pieces(img):
        keep |= m
    out = img.copy()
    out[~keep] = 0
    return out


def split(img):
    res = []
    for m in sorted(pieces(img), key=lambda m: np.where(m)[1].min()):
        p = np.zeros_like(img)
        p[m] = img[m]
        res.append(p)
    return res


def eyes_and_mouth(img):
    a = (img[..., 3] > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(a, connectivity=8)
    x0, y0, x1, y1 = MOUTH_BOX
    m = np.zeros(a.shape, bool)
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if x >= x0 and y >= y0 and x + w <= x1 and y + h <= y1:
            m |= lab == i
    eyes, mouth = img.copy(), np.zeros_like(img)
    eyes[m] = 0
    mouth[m] = img[m]
    return eyes, mouth


def scaled(img, sw, sh):
    # resize premultiplied so edges don't pick up the colour of transparent pixels
    img = img.astype(np.float32)
    a = img[..., 3:] / 255
    pre = cv2.resize(np.dstack([img[..., :3] * a, img[..., 3]]), (sw, sh), interpolation=cv2.INTER_AREA)
    na = pre[..., 3:] / 255
    out = np.dstack([np.where(na > 0, pre[..., :3] / np.maximum(na, 1e-6), 0), pre[..., 3]])
    return np.clip(out, 0, 255).astype(np.uint8)


def plan(src):
    """[(group name, group ascii, [(layer name, ascii, image, visible)])], bottom to top."""
    W, H = src.size
    lay = lambda *p: full(find(src, *p), W, H)

    tails = split(clean(lay('ツインテール')))
    assert len(tails) == 2, len(tails)

    body = clean(lay('冬服', '靴下黒'))
    upper, skirt, legs = body.copy(), body.copy(), body.copy()
    upper[BODY_CUT + OVERLAP:] = 0
    skirt[:BODY_CUT] = 0
    skirt[LEGS_CUT + OVERLAP:] = 0
    legs[:LEGS_CUT] = 0

    arm_r = [('下ろし', 'down', True), ('手を振る', 'wave', False)]
    arm_l = [('スクバ持ち', 'bag', True), ('下ろし', 'down', False), ('ピース', 'peace', False),
             ('手を口元', 'mouth', False)]

    faces = {int(l.name): eyes_and_mouth(clean(full(l, W, H))) for l in find(src, '表情')}
    eyes = [(f'目{n:02d} {name}', f'eyes{n:02d}', faces[n][0], n == 1) for n, name in EYES.items()]
    mouths = [(f'口{n:02d}', f'mouth{n:02d}', faces[n][1], n == 1) for n in MOUTHS]

    marks = []
    for l in find(src, 'アクセサリー'):
        ps = split(clean(full(l, W, H)))
        n = int(l.name)
        if len(ps) > 1 and n in (13, 14):  # widely spread hearts / sparkles
            marks += [(f'漫符{n:02d}{"abc"[i]}', f'fx{n:02d}{"abc"[i]}', p, False) for i, p in enumerate(ps)]
        else:
            marks.append((f'漫符{n:02d}', f'fx{n:02d}', clean(full(l, W, H)), False))

    return [
        ('後ろ', 'back', [('ツインテール L', 'tail_L', tails[0], True),
                          ('ツインテール R', 'tail_R', tails[1], True)]),
        ('体', 'body', [('脚', 'legs', legs, True),
                        ('スカート', 'skirt', skirt, True),
                        ('上半身', 'body_upper', upper, True),
                        ('校章', 'badge', clean(lay('冬服', '校章')), True)]),
        ('腕 R', 'arm_R', [(f'腕R {n}', f'arm_R_{a}', clean(lay('腕→', '冬服', n)), v) for n, a, v in arm_r]),
        ('腕 L', 'arm_L', [(f'腕L {n}', f'arm_L_{a}', clean(lay('腕←', '冬服', n)), v) for n, a, v in arm_l]),
        ('目', 'eyes', eyes),
        ('口', 'mouth', mouths),
        ('漫符', 'marks', marks),
        ('アホ毛', 'ahoge', [(f'アホ毛{l.name}', f'ahoge{l.name}', clean(full(l, W, H)), l.name == '0')
                            for l in find(src, 'アホ毛')]),
        ('前', 'front', [('前髪透け', 'bangs_over_eyes', clean(lay('前髪透け')), True)]),
    ]


def fits(rects, size, pad=8):
    """Whether the layers fit one atlas (MaxRects, close to what Cubism's auto layout does)."""
    packer = rectpack.newPacker(rotation=False)
    for i, (w, h) in enumerate(rects):
        packer.add_rect(w + pad, h + pad, i)
    packer.add_bin(size, size)
    packer.pack()
    return len(packer.rect_list()) == len(rects)


def main():
    src = PSDImage.open(SRC)
    W, H = src.size
    sw, sh = round(W * SCALE), round(H * SCALE)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    # RGBA so alpha is stored as layer transparency (in RGB mode psd-tools writes a user mask)
    out = PSDImage.new('RGBA', (sw, sh), color=(255, 255, 255, 0))
    rects = []
    for gname, gascii, layers in plan(src):
        grp = Group.new(out, name='group')
        grp.name, grp._record.name = gname, 'group_' + gascii
        # pass-through folder; psd-tools' default made Cubism merge each group into one ArtMesh
        grp._record.blend_mode = BlendMode.PASS_THROUGH
        grp._record.tagged_blocks.set_data(Tag.SECTION_DIVIDER_SETTING, kind=SectionDivider.OPEN_FOLDER,
                                           signature=b'8BIM', blend_mode=BlendMode.PASS_THROUGH)
        for name, ascii_, img, visible in layers:
            Image.fromarray(img).save(os.path.join(PARTS, f'{ascii_}.png'))
            small = scaled(img, sw, sh)
            ys, xs = np.where(small[..., 3] > 0)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            rects.append((x1 - x0, y1 - y0))
            pl = PixelLayer.frompil(Image.fromarray(small[y0:y1, x0:x1], 'RGBA'), grp, name='layer',
                                    top=int(y0), left=int(x0))
            pl.name, pl._record.name = name, ascii_
            # every layer is saved visible: Cubism imports hidden layers as hidden ArtMeshes, which opacity
            # keys can't bring back. `visible` is the default state, set later with parameter keys
            pl.visible = True
    out.save(OUT)
    area = sum(w * h for w, h in rects)
    print('saved', OUT, f'{sw}x{sh}', len(rects), 'layers', round(os.path.getsize(OUT) / 1e6, 1), 'MB')
    print(f'atlas use {area / ATLAS ** 2:.0%} of {ATLAS}x{ATLAS}, fits: {fits(rects, ATLAS)}')


if __name__ == '__main__':
    main()
