"""Build a Live2D (Cubism Editor) ready PSD from the separated parts.

Run with ai/venv python after split_layers.py, lama_refine.py and split_head.py.
Output: dist/mafuyu_live2d_parts.psd  (+ parts/*.png previews)

Conventions: L / R = left / right as seen on screen. Layers that are alternates
(closed eyes, open mouth...) are included but hidden.
"""
import json
import os

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer
from psd_tools.constants import BlendMode, SectionDivider, Tag
from scipy import ndimage
from simple_lama_inpainting import SimpleLama

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, 'web', 'assets')
PARTS = os.path.join(ROOT, 'parts')
os.makedirs(PARTS, exist_ok=True)

meta = json.load(open(os.path.join(ASSETS, 'parts.json'), encoding='utf-8'))
W, H = meta['size']
M = dict(np.load(os.path.join(PARTS, 'head', '_masks.npz')))
lama = SimpleLama()
yy, xx = np.mgrid[0:H, 0:W]


def full_layer(name):
    """web asset layer placed on the full canvas, float32 RGBA"""
    l = meta['layers'][name]
    out = np.zeros((H, W, 4), np.float32)
    out[l['y']:l['y'] + l['h'], l['x']:l['x'] + l['w']] = np.array(Image.open(os.path.join(ASSETS, l['file'])).convert('RGBA'))
    return out


def dil(m, k):
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def masked(img, m):
    out = img.copy()
    out[..., 3] = img[..., 3] * m
    return out


def lama_fill(img, region, matte=(255, 255, 255), pad=40):
    """repaint `region` of img (full canvas) with LaMa, returns new RGB for the whole canvas"""
    ys, xs = np.where(region)
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(W, xs.max() + pad), min(H, ys.max() + pad)
    c = img[y0:y1, x0:x1]
    a = c[..., 3:4] / 255
    rgb = c[..., :3] * a + np.array(matte, np.float32) * (1 - a)
    m = region[y0:y1, x0:x1].astype(np.uint8) * 255
    res = np.array(lama(Image.fromarray(rgb.astype(np.uint8)), Image.fromarray(m)))[:y1 - y0, :x1 - x0]
    out = img[..., :3].copy()
    out[y0:y1, x0:x1][region[y0:y1, x0:x1]] = res[region[y0:y1, x0:x1]]
    return out


def smooth_fill(img, region, sample, sigma=12):
    """fill region with a smooth colour estimated from `sample` pixels (flat anime shading)"""
    rgb = img[..., :3]
    s = sample.astype(np.float32)
    acc = np.zeros_like(rgb)
    acc_t = np.zeros((H, W, 1), np.float32)
    for sg in (sigma, sigma * 3, sigma * 10):
        num = np.stack([cv2.GaussianBlur(rgb[..., k] * s, (0, 0), sg) for k in range(3)], -1)
        den = cv2.GaussianBlur(s, (0, 0), sg)[..., None]
        t = np.clip(den / 0.15, 0, 1)
        acc += num / np.maximum(den, 1e-5) * (1 - acc_t) * t
        acc_t += (1 - acc_t) * t
    fill = acc / np.maximum(acc_t, 1e-5)
    out = img.copy()
    out[..., :3][region] = fill[region]
    out[..., 3][region] = 255
    return out


layers = {}   # name -> full canvas float RGBA

# ================================================================ head
head = full_layer('head')
ha = head[..., 3] > 20
lum = head[..., :3].mean(-1)

# face: visible skin + outline, forehead / cheeks under hair filled with skin colour
face_area = (M['face_region'] | dil(M['face'], 3)) & ~(yy > 958)
# above the eyes the face is plain skin (keeps no ghost of the see-through bangs); below, the original
# cheeks / blush / nose; the outline everywhere
skin_sample = M['face'] & (lum > 200) & (yy > 790)
face = smooth_fill(head, face_area, skin_sample | (M['face'] & (lum > 200)), sigma=10)
orig_keep = M['face'] & ((yy > 790) | (lum < 90))
face[orig_keep] = head[orig_keep]
face[..., 3] = np.where(face_area | M['face'], 255, 0) * np.where(orig_keep, head[..., 3] / 255, 1)
layers['face'] = face

# bangs: remove the brows and repaint the hair strands they crossed
brow_px = M['brow_L'] | M['brow_R']
band = dil(brow_px, 5) & M['brow_band']
bang_rgb = lama_fill(head, band, (255, 255, 255))
bangs = head.copy()
bangs[..., :3] = bang_rgb
hairish = (bang_rgb[..., 0] <= bang_rgb[..., 2] + 4) | (bang_rgb.mean(-1) < 140)
bangs[..., 3] = head[..., 3] * (M['bangs'] | (band & dil(M['bangs'], 7) & hairish))
# see-through bangs over the eyes (as in the original art): the eye parts already contain the strands,
# so the bangs are faint there; they stay continuous when the eyes close
eye_fp = np.zeros((H, W), np.float32)
for side in 'LR':
    for n in ('white', 'iris', 'line'):
        eye_fp = np.maximum(eye_fp, full_layer(f'eye{side}_{n}')[..., 3] / 255)
eye_fp = cv2.GaussianBlur(eye_fp, (0, 0), 2)
bangs[..., 3] *= 1 - 0.65 * np.clip(eye_fp * 1.5, 0, 1)
layers['bangs'] = bangs
for k in ('brow_L', 'brow_R', 'side_L', 'side_R', 'ribbon_L', 'ribbon_R', 'ahoge'):
    layers[k] = masked(head, M[k])

# back hair: fill the whole head silhouette behind face / bangs / side locks / ribbons with hair
sil = ndimage.binary_fill_holes(ha & (yy < 1000)) & (yy < 1000) & ~M['ahoge']
covered = sil & ~M['hair_back']
back_hair = masked(head, M['hair_back'])
# flat hair colour, a bit darker than the outer hair (inside of the head reads as shadow)
hair_sample = M['hair_back'] & (lum > 170)
filled = smooth_fill(head, covered, hair_sample, sigma=25)
back_hair[..., :3][covered] = filled[..., :3][covered] * 0.93
back_hair[..., 3][covered] = 255
layers['hair_back'] = back_hair

# ================================================================ body side
back = full_layer('back')
layers['tail_L'] = masked(back, xx < 1020)
layers['tail_R'] = masked(back, xx >= 1020)
layers['legs'] = full_layer('legs')

body = full_layer('body')
ba = body[..., 3] > 20
br, bg_, bb = body[..., 0], body[..., 1], body[..., 2]
blum = body[..., :3].mean(-1)
neck_zone = np.zeros((H, W), np.uint8)
cv2.fillPoly(neck_zone, [np.array([(930, 760), (1110, 760), (1110, 1000), (1150, 1090), (1020, 1160), (890, 1090), (930, 1000)], np.int32)], 1)
neck_skin = (neck_zone > 0) & ba & (br > bb + 4) & (blum > 170)
neck_skin = ndimage.binary_fill_holes(ndimage.binary_closing(neck_skin, iterations=2))
neck_line = dil(neck_skin, 7) & ~neck_skin & ba & (blum < 90) & (yy < 1000) & (neck_zone > 0)
neck_m = neck_skin | neck_line | ((neck_zone > 0) & ba & (yy < 972))   # incl. the extension painted behind the jaw
neck = masked(body, neck_m)
# continue the neck down behind the collar
neck_hidden = (neck_zone > 0) & ~neck_m & (yy > 980) & (np.abs(xx - 1020) < 85)
layers['neck'] = smooth_fill(neck, neck_hidden, neck_skin & (yy > 1000), sigma=8)

# ---- arms / skirt / torso
def poly(points):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(points, np.int32)], 1)
    return m > 0


# sleeve outlines traced on the art (inner edge = where the plaid of the skirt starts)
ARM = {'L': [(380, 1060), (600, 1060), (640, 1180), (675, 1300), (694, 1390), (697, 1740), (697, 1930),
             (630, 2117), (574, 2228), (380, 2228)],
       'R': [(1660, 1060), (1440, 1060), (1400, 1180), (1365, 1300), (1342, 1390), (1338, 1740), (1330, 1840),
             (1360, 1920), (1395, 1980), (1420, 2040), (1450, 2120), (1480, 2200), (1505, 2280), (1660, 2280)]}
HAND = {'L': [(380, 2220), (578, 2220), (535, 2317), (497, 2440), (380, 2440)],
        'R': [(1495, 2250), (1660, 2250), (1660, 2440), (1575, 2440), (1555, 2383), (1505, 2280)]}
hem_y = 1690 + 0.00045 * (xx - 1020) ** 2          # bottom edge of the shirt (curved)
HEM_Y = 1740
rest = ba & ~neck_m
arms = {}
for k in 'LR':
    arms[k] = rest & (poly(ARM[k]) | poly(HAND[k]))
arm_any = arms['L'] | arms['R']
skirt_m = rest & ~arm_any & (yy >= hem_y)
# only the main skirt body; stray bits (cuff / finger tips outside the traced arm outline) go to that arm
lab, n = ndimage.label(skirt_m)
main = np.argmax(ndimage.sum(skirt_m, lab, range(1, n + 1))) + 1
big = np.isin(lab, np.where(ndimage.sum(skirt_m, lab, range(1, n + 1)) > 1000)[0] + 1)
stray = skirt_m & big & (lab != main)
skirt_m &= ~stray
arms['L'] |= stray & (xx < 1020)
arms['R'] |= stray & (xx >= 1020)
arm_any = arms['L'] | arms['R']
torso_m = rest & ~arm_any & ~skirt_m

torso = masked(body, torso_m)
behind_arm = dil(torso_m, 81) & arm_any & (yy > 1150) & (yy < HEM_Y)
torso = smooth_fill(torso, behind_arm, torso_m & (blum < 110) & dil(arm_any, 61), sigma=10)
layers['torso'] = torso

skirt = masked(body, skirt_m)
skirt_hidden = (dil(skirt_m, 81) & arm_any & (yy >= hem_y - 10)) |                ((yy >= hem_y - 50) & (yy < hem_y + 3) & (xx > 705) & (xx < 1335) & ~skirt_m)
lama_in = body.copy()
lama_in[..., 3] = np.where(skirt_m, body[..., 3], 0)
skirt_rgb = lama_fill(lama_in, skirt_hidden, (60, 64, 90), pad=60)
skirt[..., :3][skirt_hidden] = skirt_rgb[skirt_hidden]
skirt[..., 3][skirt_hidden] = 255
layers['skirt'] = skirt
for k in 'LR':
    layers[f'arm_{k}'] = masked(body, arms[k])

# ================================================================ eyes / mouth (already separate in web assets)
for side in 'LR':
    line = full_layer(f'eye{side}_line')
    cy = meta['layers'][f'eye{side}_white']['iris'][1]
    layers[f'lash_up_{side}'] = masked(line, yy < cy)
    layers[f'lash_low_{side}'] = masked(line, yy >= cy)
    for n in ('white', 'iris', 'closed', 'happy'):
        layers[f'eye_{n}_{side}'] = full_layer(f'eye{side}_{n}')
layers['mouth_close'] = full_layer('mouth_smile')
layers['mouth_open'] = full_layer('mouth_open')

# ================================================================ PSD
NAMES = {
    'tail_L': 'ツインテール L', 'tail_R': 'ツインテール R', 'hair_back': '後ろ髪',
    'legs': '脚', 'neck': '首', 'skirt': 'スカート', 'torso': '胴体', 'arm_L': '腕 L', 'arm_R': '腕 R',
    'face': '顔', 'mouth_close': '口 閉じ', 'mouth_open': '口 開き',
    'eye_white_L': '白目 L', 'eye_iris_L': '黒目 L', 'lash_low_L': '下まつげ L', 'lash_up_L': '上まつげ L',
    'eye_closed_L': '閉じ目 L', 'eye_happy_L': '笑い目 L',
    'eye_white_R': '白目 R', 'eye_iris_R': '黒目 R', 'lash_low_R': '下まつげ R', 'lash_up_R': '上まつげ R',
    'eye_closed_R': '閉じ目 R', 'eye_happy_R': '笑い目 R',
    'brow_L': '眉 L', 'brow_R': '眉 R',
    'side_L': '横髪 L', 'side_R': '横髪 R', 'bangs': '前髪',
    'ribbon_L': 'リボン L', 'ribbon_R': 'リボン R', 'ahoge': 'アホ毛',
}
HIDDEN = {'mouth_open', 'eye_closed_L', 'eye_closed_R', 'eye_happy_L', 'eye_happy_R'}
STRUCTURE = [  # bottom -> top
    ('後ろ', ['tail_L', 'tail_R', 'hair_back']),
    ('体', ['legs', 'skirt', 'neck', 'torso', 'arm_L', 'arm_R']),
    ('顔', ['face', 'mouth_close', 'mouth_open',
            'eye_white_L', 'eye_iris_L', 'lash_low_L', 'lash_up_L', 'eye_closed_L', 'eye_happy_L',
            'eye_white_R', 'eye_iris_R', 'lash_low_R', 'lash_up_R', 'eye_closed_R', 'eye_happy_R',
            'brow_L', 'brow_R']),
    ('前', ['side_L', 'side_R', 'bangs', 'ribbon_L', 'ribbon_R', 'ahoge']),
]



def set_name(layer, name, ascii_name):
    # Japanese goes into the unicode name block; the legacy (MacRoman) name stays readable ASCII
    layer.name = name
    layer._record.name = ascii_name


def save_psd(out, scale=1.0, previews=False, flat=False):
    sw, sh = round(W * scale), round(H * scale)
    # RGBA so psd-tools stores alpha as layer transparency; in RGB mode it becomes a user mask,
    # which Cubism doesn't read (each group came in as one flattened ArtMesh)
    psd = PSDImage.new('RGBA', (sw, sh), color=(255, 255, 255, 0))
    for gname, members in STRUCTURE:
        if flat:  # no folders at all; parts are then made inside Cubism
            grp = psd
        else:
            grp = Group.new(psd, name='group')
            set_name(grp, gname, 'group')
        # Photoshop-style pass-through folder; psd-tools' default (Normal, no blend key) made Cubism
        # import each group as a single merged ArtMesh
        if not flat:
            grp._record.blend_mode = BlendMode.PASS_THROUGH
            grp._record.tagged_blocks.set_data(Tag.SECTION_DIVIDER_SETTING, kind=SectionDivider.OPEN_FOLDER,
                                               signature=b'8BIM', blend_mode=BlendMode.PASS_THROUGH)
        for key in members:
            img = np.clip(layers[key], 0, 255)
            if scale != 1.0:
                # resize premultiplied so edges don't pick up the colour of transparent pixels
                a = img[..., 3:] / 255
                pre = cv2.resize(np.dstack([img[..., :3] * a, img[..., 3]]).astype(np.float32), (sw, sh),
                                 interpolation=cv2.INTER_AREA)
                na = pre[..., 3:] / 255
                img = np.dstack([np.where(na > 0, pre[..., :3] / np.maximum(na, 1e-6), 0), pre[..., 3]])
            img = np.clip(img, 0, 255).astype(np.uint8)
            a = img[..., 3]
            ys, xs = np.where(a > 0)
            if len(xs) == 0:
                print('empty layer', key)
                continue
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            pil = Image.fromarray(img[y0:y1, x0:x1], 'RGBA')
            if previews:
                pil.save(os.path.join(PARTS, f'{key}.png'))
            lay = PixelLayer.frompil(pil, grp, name=key, top=int(y0), left=int(x0))
            set_name(lay, NAMES[key], key)
            if key in HIDDEN:
                lay.visible = False
    psd.save(out)
    print('saved', out, f'{sw}x{sh}', round(os.path.getsize(out) / 1e6, 1), 'MB')


save_psd(os.path.join(ROOT, 'dist', 'mafuyu_live2d_parts.psd'), previews=True)
# Cubism FREE allows a single 2048x2048 texture; at 0.65 all parts fill ~65% of it
save_psd(os.path.join(ROOT, 'dist', 'mafuyu_live2d_parts_free.psd'), scale=0.65)
# fallback if Cubism still merges folders: same layers, no groups
save_psd(os.path.join(ROOT, 'dist', 'mafuyu_live2d_parts_free_flat.psd'), scale=0.65, flat=True)
