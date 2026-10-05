"""Use THA3 only as a motion estimator: take its warp field and apply it to the full-res art."""
import os, sys, torch, numpy as np
import torch.nn.functional as Fn
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'tha3')); os.chdir(os.path.join(HERE, 'tha3'))
from tha3.poser.modes.standard_float import create_poser
from tha3.nn.editor.editor_07 import Editor07
from tha3.util import extract_pytorch_image_from_PIL_image

dev = torch.device('cuda')
poser = create_poser(dev)
names = [n for g in poser.get_pose_parameter_groups() for n in g.get_parameter_names()]
idx = {n: i for i, n in enumerate(names)}
S = 128 / 662                     # source px -> THA canvas px
OX, OY = 256 - 1020 * S, 64 - 290 * S

src = Image.open(os.path.join(HERE, '..', '4_4.png')).convert('RGBA')
small = src.resize((round(src.width * S), round(src.height * S)), Image.LANCZOS)
canvas = Image.new('RGBA', (512, 512), (0, 0, 0, 0)); canvas.paste(small, (round(OX), round(OY)), small)
a = np.array(canvas); a[a[..., 3] < 8] = 0; canvas = Image.fromarray(a)
inp = extract_pytorch_image_from_PIL_image(canvas).to(dev).unsqueeze(0)

def outputs(**kw):
    pose = torch.zeros(1, len(names), device=dev)
    for k, v in kw.items(): pose[0, idx[k]] = v
    with torch.no_grad():
        return poser.get_posing_outputs(inp, pose)

def hires(**kw):
    o = outputs(**kw)
    grid = o[Editor07.GRID_CHANGE_INDEX]                # (1,2,512,512) backward offsets in [-1,1] units
    alpha = o[Editor07.COLOR_CHANGE_ALPHA_INDEX]        # where the editor repaints instead of warping
    color = o[Editor07.COLOR_CHANGE_IMAGE_INDEX]
    # region of the source covered by the canvas, rendered at 4x canvas resolution around the head
    x0, y0, x1, y1 = 600, 120, 1440, 1140
    W, H = x1 - x0, y1 - y0
    ys, xs = torch.meshgrid(torch.arange(H, device=dev) + y0 + 0.5, torch.arange(W, device=dev) + x0 + 0.5, indexing='ij')
    cx, cy = xs * S + OX, ys * S + OY                   # canvas px
    gx, gy = cx / 256 - 1, cy / 256 - 1                 # canvas grid coords
    g = torch.stack([gx, gy], -1)[None]
    ch = Fn.grid_sample(grid, g, mode='bilinear', align_corners=False)[0]   # (2,H,W)
    sx = (cx + ch[0] * 256 - OX) / S; sy = (cy + ch[1] * 256 - OY) / S      # source px to sample
    srcT = torch.from_numpy(np.array(src)).float().to(dev).permute(2, 0, 1)[None] / 255
    sg = torch.stack([sx / src.width * 2 - 1, sy / src.height * 2 - 1], -1)[None]
    warped = Fn.grid_sample(srcT, sg, mode='bilinear', align_corners=False)[0]
    al = Fn.grid_sample(alpha, g, mode='bilinear', align_corners=False)[0]
    col = Fn.grid_sample(color, g, mode='bilinear', align_corners=False)[0]   # [-1,1] rgba
    col = (col + 1) / 2
    out = warped * (1 - al) + col * al
    return Image.fromarray((out.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)), \
           Image.fromarray((al[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

if __name__ == '__main__':
    tiles = []
    for p in [dict(), dict(head_y=-1), dict(head_y=1), dict(head_x=1), dict(head_x=-1)]:
        im, al = hires(**p); tiles.append(im)
        if p == dict(head_y=1): al.save(os.path.join(HERE, 'flow_alpha.png'))
    W = Image.new('RGBA', (420 * 5, 510), (230, 235, 245, 255))
    for i, t in enumerate(tiles):
        W.alpha_composite(t.resize((420, 510), Image.LANCZOS), (i * 420, 0))
    W.save(os.path.join(HERE, 'flow_grid.png'))
