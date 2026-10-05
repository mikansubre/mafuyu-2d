"""Bake Talking Head Anime 3 head-rotation warps into forward displacement fields for the web rig.

THA3 runs on a 512px downscale of the art; we only keep its warp field (where each pixel moves),
invert it to a forward displacement in source-image pixels, and store a grid of poses:
  yaw (THA head_y) x pitch (THA head_x)
Output: web/assets/flow.bin (int16, 0.1px units) + parts.json["flow"].
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'web', 'assets')
sys.path.insert(0, os.path.join(HERE, 'tha3'))
os.chdir(os.path.join(HERE, 'tha3'))  # model paths are relative to the repo
from tha3.nn.editor.editor_07 import Editor07  # noqa: E402
from tha3.poser.modes.standard_float import create_poser  # noqa: E402
from tha3.util import extract_pytorch_image_from_PIL_image  # noqa: E402

YAWS = [-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1]
PITCHES = [-1, -0.5, 0, 0.5, 1]
# field region (source px) and spacing
X0, Y0, X1, Y1, STEP = 280, 80, 1760, 1720, 16
UNIT = 0.1

dev = torch.device('cuda')
poser = create_poser(dev)
names = [n for g in poser.get_pose_parameter_groups() for n in g.get_parameter_names()]
idx = {n: i for i, n in enumerate(names)}

# head (hair top 290 .. chin 952) must fill the 128px box at (192..320, 64..192) of the 512 canvas
S = 128 / 662
OX, OY = 256 - 1020 * S, 64 - 290 * S
src = Image.open(os.path.join(ROOT, '4_4.png')).convert('RGBA')
small = src.resize((round(src.width * S), round(src.height * S)), Image.LANCZOS)
canvas = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
canvas.paste(small, (round(OX), round(OY)), small)
a = np.array(canvas)
a[a[..., 3] < 8] = 0
inp = extract_pytorch_image_from_PIL_image(Image.fromarray(a)).to(dev).unsqueeze(0)

nx = (X1 - X0) // STEP + 1
ny = (Y1 - Y0) // STEP + 1
ys, xs = torch.meshgrid(torch.arange(ny, device=dev) * STEP + Y0, torch.arange(nx, device=dev) * STEP + X0, indexing='ij')
P = torch.stack([xs, ys], 0).float()  # source px of each field node


def backward_px(grid_change, q):
    """Backward map at source points q (2,ny,nx): returns where output pixel q samples from (source px)."""
    cx, cy = q[0] * S + OX, q[1] * S + OY
    g = torch.stack([cx / 256 - 1, cy / 256 - 1], -1)[None]
    ch = Fn.grid_sample(grid_change, g, mode='bilinear', align_corners=False, padding_mode='zeros')[0]
    return torch.stack([(cx + ch[0] * 256 - OX) / S, (cy + ch[1] * 256 - OY) / S], 0)


raw = np.zeros((len(PITCHES), len(YAWS), 2, ny, nx), np.float32)
for j, pitch in enumerate(PITCHES):
    for i, yaw in enumerate(YAWS):
        pose = torch.zeros(1, len(names), device=dev)
        pose[0, idx['head_y']] = yaw
        pose[0, idx['head_x']] = pitch
        with torch.no_grad():
            grid = poser.get_posing_outputs(inp, pose)[Editor07.GRID_CHANGE_INDEX]
            # forward displacement d(p): find q with B(q) = p  ->  q = p + d.  Fixed-point iteration.
            q = P.clone()
            for _ in range(12):
                q = q + (P - backward_px(grid, q)) * 0.8
            d = (q - P).cpu().numpy()
        raw[j, i] = d

# the neutral pose is not exactly identity (network noise); make it so
raw -= raw[PITCHES.index(0), YAWS.index(0)]
for j, pitch in enumerate(PITCHES):
    for i, yaw in enumerate(YAWS):
        print(f'pitch {pitch:+.2f} yaw {yaw:+.2f}  max |d| = {np.abs(raw[j, i]).max():.1f}px')
fields = np.clip(np.round(raw / UNIT), -32767, 32767).astype(np.int16)

fields.tofile(os.path.join(OUT, 'flow.bin'))
meta_path = os.path.join(OUT, 'parts.json')
parts = json.load(open(meta_path, encoding='utf-8'))
parts['flow'] = {'file': 'flow.bin', 'x0': X0, 'y0': Y0, 'step': STEP, 'nx': nx, 'ny': ny,
                 'yaws': YAWS, 'pitches': PITCHES, 'unit': UNIT,
                 'credit': 'Talking Head(?) Anime from a Single Image 3 by Pramook Khungurn (CC BY 4.0)'}
json.dump(parts, open(meta_path, 'w', encoding='utf-8'), indent=1)
print('saved', fields.shape)
