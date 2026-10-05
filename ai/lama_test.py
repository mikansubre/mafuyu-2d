import os, sys, numpy as np, cv2
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from regions import EYE_PATCH, EYE_OPEN, MOUTH_BOX
from simple_lama_inpainting import SimpleLama
src = Image.open(os.path.join(ROOT, '4_4.png')).convert('RGBA')
box = (760, 560, 1290, 940)
crop = src.crop(box)
rgb = Image.new('RGB', crop.size, (255, 255, 255)); rgb.paste(crop, (0, 0), crop)
m = Image.new('L', crop.size, 0); d = ImageDraw.Draw(m)
for k in 'LR':
    d.polygon([(x - box[0], y - box[1]) for x, y in EYE_PATCH[k]], fill=255)
    d.polygon([(x - box[0], y - box[1]) for x, y in EYE_OPEN[k]], fill=255)
mx0, my0, mx1, my1 = MOUTH_BOX
d.rectangle((mx0 - box[0] + 8, my0 - box[1] + 6, mx1 - box[0] - 8, my1 - box[1] - 6), fill=255)
m = Image.fromarray(cv2.dilate(np.array(m), np.ones((7, 7), np.uint8)))
lama = SimpleLama()
out = lama(rgb, m).crop((0, 0) + crop.size)
W = Image.new('RGB', (crop.width * 2, crop.height)); W.paste(rgb, (0, 0)); W.paste(out, (crop.width, 0))
W.save(os.path.join(HERE, 'lama_eyes.png'))
