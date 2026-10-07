"""Build the Cubism FREE PSD from the hand-made v2 parts.

Run with ai/venv python:  ai/venv/Scripts/python.exe tools/build_v2.py
Input:  v2/mafuyu_live2d_parts_v2.psd   (hand-cut master, full size 2000x4156)
Output: v2/dist/mafuyu_v2_free.psd      (0.65x, fits Cubism FREE's single 2048 texture)
        v2/parts/*.png                  (full-size previews of each layer)

The reference layer 元絵（参照用） is dropped. Folder structure, order and hidden flags
are taken over from the master as they are.
"""
import os
import re

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer
from psd_tools.constants import BlendMode, SectionDivider, Tag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, 'v2')
SRC = os.path.join(V2, 'mafuyu_live2d_parts_v2.psd')
OUT = os.path.join(V2, 'dist', 'mafuyu_v2_free.psd')
PARTS = os.path.join(V2, 'parts')
SCALE = 0.65
SKIP = {'元絵（参照用）'}

# legacy (MacRoman) layer names; Cubism shows the unicode name, these are what other tools see
ASCII = {
    '後ろ': 'back', '体': 'body', '顔': 'face', '前': 'front',
    'ツインテール': 'tail', '後ろ髪（仮）': 'hair_back', '脚': 'legs', 'スカート': 'skirt',
    '襟の裏': 'collar_back', '首': 'neck', '胴体': 'torso', 'スカーフ': 'scarf', '腕': 'arm',
    '口 開き': 'mouth_open', '白目': 'eye_white', '黒目': 'iris', '下まつげ': 'lash_lower',
    '上まつげ': 'lash_upper', '閉じ目': 'eye_closed', '笑い目': 'eye_smile', '横髪': 'side',
    '前髪': 'bangs', 'リボンの垂れ': 'ribbon_tail', 'リボン': 'ribbon', 'アホ毛': 'ahoge',
}


def ascii_name(name, is_group):
    m = re.fullmatch(r'(.+?) ([LR])', name)
    base, side = (m.group(1), '_' + m.group(2)) if m else (name, '')
    a = ASCII.get(base, 'layer')
    return ('group_' + a if is_group and a != 'layer' else a) + side


def set_name(layer, name, is_group=False):
    layer.name = name
    layer._record.name = ascii_name(name, is_group)


def scaled(img, sw, sh):
    # resize premultiplied so edges don't pick up the colour of transparent pixels
    img = img.astype(np.float32)
    a = img[..., 3:] / 255
    pre = cv2.resize(np.dstack([img[..., :3] * a, img[..., 3]]), (sw, sh), interpolation=cv2.INTER_AREA)
    na = pre[..., 3:] / 255
    out = np.dstack([np.where(na > 0, pre[..., :3] / np.maximum(na, 1e-6), 0), pre[..., 3]])
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    src = PSDImage.open(SRC)
    W, H = src.size
    sw, sh = round(W * SCALE), round(H * SCALE)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    # RGBA so alpha is stored as layer transparency (in RGB mode psd-tools writes a user mask,
    # which Cubism doesn't read)
    out = PSDImage.new('RGBA', (sw, sh), color=(255, 255, 255, 0))

    def copy(layers, parent):
        for l in layers:
            if l.name in SKIP:
                continue
            if l.is_group():
                grp = Group.new(parent, name='group')
                set_name(grp, l.name, True)
                # pass-through folder; psd-tools' default made Cubism merge each group into one ArtMesh
                grp._record.blend_mode = BlendMode.PASS_THROUGH
                grp._record.tagged_blocks.set_data(Tag.SECTION_DIVIDER_SETTING, kind=SectionDivider.OPEN_FOLDER,
                                                   signature=b'8BIM', blend_mode=BlendMode.PASS_THROUGH)
                grp.visible = l.visible
                copy(l, grp)
                continue
            # topil() rather than composite(): composite renders hidden layers (closed eyes...) empty
            full = np.zeros((H, W, 4), np.uint8)
            pil = l.topil()
            if pil is not None:
                full[l.top:l.bottom, l.left:l.right] = np.array(pil.convert('RGBA'))
            if not full[..., 3].any():
                print('empty layer', l.name)
                continue
            key = ascii_name(l.name, False)
            Image.fromarray(full).save(os.path.join(PARTS, f'{key}.png'))
            img = scaled(full, sw, sh)
            ys, xs = np.where(img[..., 3] > 0)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            lay = PixelLayer.frompil(Image.fromarray(img[y0:y1, x0:x1], 'RGBA'), parent, name='layer',
                                     top=int(y0), left=int(x0))
            set_name(lay, l.name)
            lay.visible = l.visible

    copy(src, out)
    out.save(OUT)
    print('saved', OUT, f'{sw}x{sh}', round(os.path.getsize(OUT) / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
