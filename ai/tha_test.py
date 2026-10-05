import sys, torch, numpy as np
from PIL import Image
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'tha3'))
os.chdir(os.path.join(HERE, 'tha3'))  # model paths are relative to the repo
from tha3.poser.modes.standard_float import create_poser
from tha3.util import extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy

dev = torch.device('cuda')
poser = create_poser(dev)
names = [n for g in poser.get_pose_parameter_groups() for n in g.get_parameter_names()]
idx = {n: i for i, n in enumerate(names)}

def prep(scale_mul=1.0):
    src = Image.open(os.path.join(HERE, '..', '4_4.png')).convert('RGBA')
    s = 128 / 662 * scale_mul
    big = src.resize((round(src.width * s), round(src.height * s)), Image.LANCZOS)
    out = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    out.paste(big, (round(256 - 1020 * s), round(64 - 290 * s)), big)
    a = np.array(out); a[a[..., 3] < 8] = 0
    return Image.fromarray(a)

def run(img, **kw):
    t = extract_pytorch_image_from_PIL_image(img).to(dev)
    pose = torch.zeros(1, len(names), device=dev)
    for k, v in kw.items(): pose[0, idx[k]] = v
    with torch.no_grad():
        o = poser.pose(t.unsqueeze(0), pose)[0]
    return Image.fromarray(np.uint8(np.rint(convert_output_image_from_torch_to_numpy(o.cpu()) * 255)))

if __name__ == '__main__':
    print(names)
    img = prep(); img.save(os.path.join(HERE, 'tha_in.png'))
    poses = [dict(), dict(head_y=-1), dict(head_y=1), dict(head_x=-1), dict(head_x=1), dict(neck_z=1), dict(head_y=-1, body_y=-1), dict(eye_wink_left=1, eye_wink_right=1, mouth_aaa=1)]
    tiles = [run(img, **p) for p in poses]
    W = Image.new('RGBA', (256 * 4, 256 * 2), (230, 235, 245, 255))
    for i, t in enumerate(tiles):
        W.alpha_composite(t.crop((128, 0, 384, 256)), ((i % 4) * 256, (i // 4) * 256))
    W.save(os.path.join(HERE, 'tha_grid.png'))
