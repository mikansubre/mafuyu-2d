"""Draw where each rotation deformer's centre goes -> v3/rig/回転の中心_v3.png"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
S = 0.6
# name, (x, y) in the original 2000x4156 canvas
PIVOTS = [('1 Rot_Body 体の揺れ（足元）', (1000, 4080)), ('2 Rot_Breath 呼吸（足元・1と同じ）', (1000, 4080)),
          ('3 Rot_TailL ツインテールL', (740, 445)), ('4 Rot_TailR ツインテールR', (1290, 445)),
          ('5 Rot_ArmL 腕L（肩）', (690, 1110)), ('6 Rot_ArmR 腕R（肩）', (1330, 1110)),
          ('7 Rot_Ahoge アホ毛（根元）', (1028, 286))]


def main():
    img = Image.new('RGBA', (1200, 2494), 'white')
    for n in ['tail_L', 'tail_R', 'legs', 'skirt', 'body_upper', 'badge', 'arm_R_down', 'arm_L_bag', 'eyes01',
              'mouth01', 'ahoge0', 'bangs_over_eyes']:
        p = Image.open(os.path.join(ROOT, 'v3', 'parts', n + '.png')).resize((1200, 2494), Image.LANCZOS)
        img.alpha_composite(p)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc', 26)
    for i, (label, (x, y)) in enumerate(PIVOTS):
        x, y = x * S, y * S
        d.ellipse((x - 14, y - 14, x + 14, y + 14), outline='red', width=5)
        d.line((x - 22, y, x + 22, y), fill='red', width=2)
        d.line((x, y - 22, x, y + 22), fill='red', width=2)
        tx = x + 30 if x < 800 else x - 30 - d.textlength(label, font=font)
        ty = y - 15 - (34 if i == 1 else 0) - (0 if i != 0 else -10)
        d.text((tx, ty), label, fill='red', font=font, stroke_width=4, stroke_fill='white')
    out = os.path.join(HERE, '回転の中心_v3.png')
    img.convert('RGB').save(out)
    print(out, [(n, round(x * S), round(y * S)) for n, (x, y) in PIVOTS])


if __name__ == '__main__':
    main()
