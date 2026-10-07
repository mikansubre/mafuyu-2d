"""Generate the cubism-mcp edit batches for v3's switches (params + opacity keys).

python v3/rig/gen_keys.py  ->  v3/rig/batch_*.json  (each one is passed to cubism_edit_batch)
ArtMesh IDs are the ones Cubism gives on a fresh import of v3/dist/mafuyu_v3_free.psd.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PARAMS_DELETE = ['ParamBrowLY', 'ParamBrowRY', 'ParamBrowLX', 'ParamBrowRX', 'ParamBrowLAngle', 'ParamBrowRAngle',
                 'ParamBrowLForm', 'ParamBrowRForm', 'ParamEyeBallX', 'ParamEyeBallY', 'ParamHairSide',
                 'ParamBodyAngleY']
PARAMS_ADD = [('ParamEyeType', '目の種類', 0, 0, 10), ('ParamMouthType', '口の種類', 0, 0, 16),
              ('ParamMarks', '漫符', 0, 0, 13), ('ParamAhogeType', 'アホ毛の形', 0, 0, 7),
              ('ParamArmL', '左腕（画面左）', 0, 0, 3), ('ParamArmR', '右腕（画面右）', 0, 0, 1),
              ('ParamArmSway', '腕揺れ', -1, 0, 1)]

EYE = {'01': 59, '02': 58, '05': 57, '07': 56, '08': 55, '10': 54, '12': 53, '13': 52, '14': 51, '18': 50,
       '24': 49, '25': 48}
EYE_TYPES = ['07', '08', '10', '12', '13', '14', '18', '24', '25', '02']  # ParamEyeType 1..10
MOUTH = {'01': 47, '02': 46, '03': 45, '04': 44, '05': 43, '06': 42, '07': 41, '09': 40, '12': 39, '13': 38,
         '15': 37, '16': 36, '17': 35, '18': 34, '19': 33, '20': 32, '21': 31, '22': 30, '23': 29, '24': 28}
MOUTH_TYPES = ['03', '04', '05', '07', '09', '12', '13', '15', '16', '17', '18', '19', '20', '21', '23', '24']
# lip sync (ParamMouthType 0): mouth shown over a ParamMouthOpenY range, crossfading in the gaps
LIPSYNC = {'01': (0, .1), '22': (.2, .4), '06': (.5, .7), '02': (.8, 1)}
OPEN_KEYS = [0, .1, .2, .4, .5, .7, .8, 1]
MARKS = [[27], [26], [24], [23], [22], [21], [20], [19], [18], [17], [16], [15, 14, 13], [12, 11, 10]]  # 1..13
CHEEK = 25          # 漫符03 照れ
AHOGE = [9, 8, 7, 6, 5, 4, 3, 2]  # アホ毛0..7
ARM_L = [63, 62, 61, 60]          # スクバ持ち, 下ろし, ピース, 手を口元
ARM_R = [65, 64]                  # 下ろし, 手を振る


TAIL_L, TAIL_R = 71, 70
STATIC = [1, 66, 67, 68, 69]  # 前髪透け, 校章, 上半身, スカート, 脚 (+ every switch mesh not on a limb)
# rotation deformer: (param, keys, angle at each key)
SWING = {'Rot_TailL': ('ParamHairBack', [-1, 0, 1], [-6, 0, 6]),
         'Rot_TailR': ('ParamHairBack', [-1, 0, 1], [-6, 0, 6]),
         'Rot_ArmL': ('ParamArmSway', [-1, 0, 1], [-3, 0, 3]),
         'Rot_ArmR': ('ParamArmSway', [-1, 0, 1], [-3, 0, 3]),
         'Rot_Ahoge': ('ParamHairFront', [-1, 0, 1], [-10, 0, 10])}
BODY_Z, BODY_X = 2, 1   # Rot_Body degrees at ParamAngleZ / ParamAngleX = ±30 (pivot at the feet)
BREATH_SCALE = 100.8    # Rot_Breath scale at ParamBreath = 1


def mesh(n):
    return 'ArtMesh' if n == 1 else f'ArtMesh{n}'


def keys(obj, param, values):
    return [{'action': 'AddParameterKey', 'params': {'ObjectId': obj, 'ParameterId': param, 'KeyValue': v}}
            for v in values]


def opacity(obj, value, **at):
    return {'action': 'EditArtMesh', 'params': {'Id': obj, 'Opacity': value,
                                                'Parameters': [{'Id': p, 'Value': v} for p, v in at.items()]}}


def switch(n, param, index, top):
    """Mesh n is shown only at param == index (0..top)."""
    obj = mesh(n)
    ks = sorted({0, top, *(k for k in (index - 1, index, index + 1) if 0 <= k <= top)})
    return keys(obj, param, ks) + [opacity(obj, 100 if k == index else 0, **{param: k}) for k in ks]


def grid(n, pa, ka, pb, kb, shown):
    """Mesh n keyed on two params; shown(a, b) -> bool."""
    obj = mesh(n)
    acts = keys(obj, pa, ka) + keys(obj, pb, kb)
    acts += [opacity(obj, 100 if shown(a, b) else 0, **{pa: a, pb: b}) for a in ka for b in kb]
    return acts


def main():
    batches = {}
    batches['1_params'] = ([{'action': 'DeleteParameter', 'params': {'Id': p}} for p in PARAMS_DELETE] +
                           [{'action': 'AddParameter', 'params': {'Id': i, 'Name': n, 'Min': lo, 'Default': d,
                                                                  'Max': hi}} for i, n, lo, d, hi in PARAMS_ADD])
    eyes = grid(EYE['01'], 'ParamEyeLOpen', [0, .35, .42, 1], 'ParamEyeType', [0, 1, 10],
                lambda o, t: t == 0 and o >= .42)
    eyes += grid(EYE['05'], 'ParamEyeLOpen', [0, .35, .42, 1], 'ParamEyeType', [0, 1, 10],
                 lambda o, t: t == 0 and o <= .35)
    for i, e in enumerate(EYE_TYPES, 1):
        eyes += switch(EYE[e], 'ParamEyeType', i, 10)
    batches['2_eyes'] = eyes
    mouth = []
    for m, (lo, hi) in LIPSYNC.items():
        mouth += grid(MOUTH[m], 'ParamMouthOpenY', OPEN_KEYS, 'ParamMouthType', [0, 1, 16],
                      lambda o, t, lo=lo, hi=hi: t == 0 and lo <= o <= hi)
    batches['3_mouth_lipsync'] = mouth
    batches['4_mouth_fixed'] = [a for i, m in enumerate(MOUTH_TYPES, 1) for a in switch(MOUTH[m], 'ParamMouthType', i, 16)]
    marks = [a for i, ms in enumerate(MARKS, 1) for n in ms for a in switch(n, 'ParamMarks', i, 13)]
    marks += switch(CHEEK, 'ParamCheek', 1, 1)
    batches['5_marks'] = marks
    other = [a for i, n in enumerate(AHOGE) for a in switch(n, 'ParamAhogeType', i, 7)]
    other += [a for i, n in enumerate(ARM_L) for a in switch(n, 'ParamArmL', i, 3)]
    other += [a for i, n in enumerate(ARM_R) for a in switch(n, 'ParamArmR', i, 1)]
    batches['6_ahoge_arms'] = other
    # hierarchy: Rot_Body > Rot_Breath > (tails / arms / ahoge rotations, everything else)
    tree = [{'action': 'EditRotationDeformer', 'params': {'Id': 'Rot_Breath', 'ParentDeformerId': 'Rot_Body'}}]
    tree += [{'action': 'EditRotationDeformer', 'params': {'Id': d, 'ParentDeformerId': 'Rot_Breath'}} for d in SWING]
    limb = {TAIL_L: 'Rot_TailL', TAIL_R: 'Rot_TailR', **{n: 'Rot_ArmL' for n in ARM_L},
            **{n: 'Rot_ArmR' for n in ARM_R}, **{n: 'Rot_Ahoge' for n in AHOGE}}
    tree += [{'action': 'EditArtMesh', 'params': {'Id': mesh(n), 'ParentDeformerId': limb.get(n, 'Rot_Breath')}}
             for n in range(1, 72)]
    batches['7_tree'] = tree
    rot = lambda d, angle=None, scale=None, **at: {'action': 'EditRotationDeformer', 'params': {
        'Id': d, **({'Angle': angle} if angle is not None else {}), **({'Scale': scale} if scale is not None else {}),
        'Parameters': [{'Id': p, 'Value': v} for p, v in at.items()]}}
    motion = keys('Rot_Body', 'ParamAngleZ', [-30, 0, 30]) + keys('Rot_Body', 'ParamAngleX', [-30, 0, 30])
    motion += [rot('Rot_Body', BODY_Z * z / 30 - BODY_X * x / 30, ParamAngleZ=z, ParamAngleX=x)
               for z in (-30, 0, 30) for x in (-30, 0, 30)]
    motion += keys('Rot_Breath', 'ParamBreath', [0, 1])
    motion += [rot('Rot_Breath', scale=100, ParamBreath=0), rot('Rot_Breath', scale=BREATH_SCALE, ParamBreath=1)]
    for d, (p, ks, angles) in SWING.items():
        motion += keys(d, p, ks) + [rot(d, a, **{p: k}) for k, a in zip(ks, angles)]
    batches['8_motion'] = motion
    # fixes after the first VTS test: eyes crossfaded around 0.35-0.42 (both half visible) -> switch lower and faster
    batches['9_fix_eyes'] = [{'action': 'MoveParameterKey', 'params': {
        'ObjectId': mesh(EYE[e]), 'ParameterId': 'ParamEyeLOpen', 'FromValue': a, 'ToValue': b}}
        for e in ('01', '05') for a, b in ((.35, .22), (.42, .25))]
    # neck bend: one warp over everything above the skirt; its grid is shaped by hand at ParamAngleZ = ±30
    upper = [mesh(n) for n in range(1, 68) if n not in ARM_L + ARM_R + AHOGE] + list(SWING)
    batches['10_neck'] = [{'action': 'AddWarpDeformer', 'params': {
        'Id': 'Warp_Neck', 'Name': '首の曲げ', 'TargetObjectIds': upper, 'Mode': 'AsParent',
        'BezierDivH': 3, 'BezierDivV': 6, 'WarpDivH': 6, 'WarpDivV': 12}}]
    batches['10_neck'] += keys('Warp_Neck', 'ParamAngleZ', [-30, 0, 30])
    # upper-body bend: a warp over the whole figure (neck warp + skirt + legs); shaped by hand at ParamBodyAngleZ = ±10
    batches['11_body_bend'] = [{'action': 'AddWarpDeformer', 'params': {
        'Id': 'Warp_Bend', 'Name': '上半身の曲げ', 'TargetObjectIds': ['Warp_Neck', mesh(68), mesh(69)],
        'Mode': 'AsParent', 'BezierDivH': 3, 'BezierDivV': 8, 'WarpDivH': 6, 'WarpDivV': 16}}]
    batches['11_body_bend'] += keys('Warp_Bend', 'ParamBodyAngleZ', [-10, 0, 10])
    for name, acts in batches.items():
        with open(os.path.join(HERE, f'batch_{name}.json'), 'w', encoding='utf-8') as f:
            json.dump(acts, f, ensure_ascii=False, separators=(',', ':'))
        print(name, len(acts))


if __name__ == '__main__':
    main()
