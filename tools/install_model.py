"""Cubismで moc3 を書き出した後に実行する。

  python tools/install_model.py        v2（正式版）: v2/export/ → VTSの mafuyu_v2 フォルダ
  python tools/install_model.py --v1   v1（前のバージョン）: export/ → VTSの mafuyu フォルダ
  python tools/install_model.py --v3   v3（原稿から作った2Dモデル）: v3/export/ → VTSの mafuyu_v3 フォルダ

1. 書き出し先に別名（mafuyu_live2d_parts_free.* など）で書き出されたファイルを mafuyu.* に揃える
2. mafuyu.physics3.json を生成し、model3.json に Physics の参照を足す
   v3 では表情（expressions/*.exp3.json）も作って model3.json に登録する（VTSのホットキー用）
3. VTube Studio のモデルフォルダへコピー（mafuyu.vtube.json などVTS側のファイルは残す）
"""
import json, shutil, pathlib, sys

V1 = "--v1" in sys.argv
V3 = "--v3" in sys.argv
ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPORT = ROOT / "export" if V1 else ROOT / ("v3" if V3 else "v2") / "export"
MODELS = pathlib.Path(r"D:\SteamLibrary\steamapps\common\VTube Studio\VTube Studio_Data\StreamingAssets\Live2DModels")
VTS = MODELS / ("mafuyu" if V1 else "mafuyu_v3" if V3 else "mafuyu_v2")
NAME = "mafuyu.physics3.json"

def inp(pid, weight, typ):
    return {"Source": {"Target": "Parameter", "Id": pid}, "Weight": weight, "Type": typ, "Reflect": False}

HEAD = [inp("ParamAngleX", 60, "X"), inp("ParamAngleZ", 60, "Angle"),
        inp("ParamBodyAngleX", 40, "X"), inp("ParamBodyAngleZ", 40, "Angle")]
BODY = [inp("ParamBodyAngleX", 100, "X"), inp("ParamBodyAngleZ", 100, "Angle")]

def chain(lengths, mobility, delay, accel):
    """振り子の頂点列。根元は固定、先に行くほど柔らかく遅れる。"""
    v = [{"Position": {"X": 0, "Y": 0}, "Mobility": 1, "Delay": 1, "Acceleration": 1, "Radius": 0}]
    y = 0
    for i, l in enumerate(lengths):
        y += l
        v.append({"Position": {"X": 0, "Y": y}, "Mobility": mobility, "Delay": delay,
                  "Acceleration": accel, "Radius": l})
    return v

# (名前, 入力, 出力パラメータ, 出力倍率, 振り子)
if V1:
    SETTINGS = [
        ("前髪",         HEAD, "ParamHairFront", 3.0, chain([3], 0.95, 0.9, 1.5)),
        ("横髪",         HEAD, "ParamHairSide",  3.5, chain([4], 0.95, 0.85, 1.5)),
    ]
elif V3:
    # v3 は髪・顔・体が1枚なので、揺らすのは回転デフォーマを付けたツインテール・アホ毛・腕だけ
    SETTINGS = [
        ("アホ毛",       HEAD, "ParamHairFront", 1.5, chain([10], 0.9, 0.9, 1.0)),
        ("ツインテール", HEAD, "ParamHairBack",  2.0, chain([9, 9], 0.88, 0.8, 1.0)),
        ("腕",           BODY, "ParamArmSway",   1.2, chain([12], 0.85, 0.9, 1.0)),
    ]
else:
    # v2 は手で切り抜き直したので、ツインテール・スカート・腕も揺らす。
    # 振り子が短いと小刻みにバネのように揺れるので、長くして揺れやすさ・振りも抑える
    SETTINGS = [
        ("前髪",         HEAD, "ParamHairFront", 1.5, chain([10], 0.9, 0.9, 1.0)),
        ("横髪",         HEAD, "ParamHairSide",  2.0, chain([12], 0.9, 0.85, 1.0)),
        ("後ろ髪・ツインテール", HEAD, "ParamHairBack", 2.0, chain([9, 9], 0.88, 0.8, 1.0)),
        ("スカート",     BODY, "ParamSkirt",     1.5, chain([10], 0.85, 0.8, 1.0)),
        ("腕",           BODY, "ParamArmSway",   1.2, chain([12], 0.85, 0.9, 1.0)),
    ]

# v3 の表情（VTSでホットキーに割り当てる）。値は v3/rig/gen_keys.py の切り替えパラメータの番号
# 目の種類: 1 白丸 2 ウィンク 3 ぎゅっ 4 ジト目 5 黒目 6 ぐるぐる 7 泣き 8 ハート 9 キラキラ 10 笑顔
# 漫符: 1 怒り 2 はてな 3 汗（あご） 4 青ざめ 5 もやもや 6 音符 7 汗 8 考え中 9 汗（頭） 10 照れ汗 11 汗 12 ハート 13 キラキラ
EXPRESSIONS = {
    "笑顔": {"ParamEyeType": 10},
    "ウィンク": {"ParamEyeType": 2},
    "ぎゅっ": {"ParamEyeType": 3},
    "ジト目": {"ParamEyeType": 4},
    "びっくり": {"ParamEyeType": 1, "ParamMarks": 2},
    "こわい": {"ParamEyeType": 5, "ParamMarks": 4},
    "ぐるぐる": {"ParamEyeType": 6, "ParamMarks": 5},
    "泣き": {"ParamEyeType": 7},
    "ハート": {"ParamEyeType": 8, "ParamMarks": 12},
    "キラキラ": {"ParamEyeType": 9, "ParamMarks": 13},
    "照れ": {"ParamCheek": 1, "ParamMarks": 10},
    "怒り": {"ParamMarks": 1},
    "はてな": {"ParamMarks": 2},
    "音符": {"ParamMarks": 6},
    "考え中": {"ParamMarks": 8},
    "汗": {"ParamMarks": 9},
    "手を振る": {"ParamArmR": 1},
    "ピース": {"ParamArmL": 2},
    "手を口元": {"ParamArmL": 3},
    "腕を下ろす": {"ParamArmL": 1},
}

def write_expressions(refs):
    d = EXPORT / "expressions"
    d.mkdir(exist_ok=True)
    refs["Expressions"] = []
    for name, params in EXPRESSIONS.items():
        exp = {"Type": "Live2D Expression", "FadeInTime": 0.1, "FadeOutTime": 0.1,
               "Parameters": [{"Id": k, "Value": v, "Blend": "Overwrite"} for k, v in params.items()]}
        (d / f"{name}.exp3.json").write_text(json.dumps(exp, ensure_ascii=False, indent="\t"), encoding="utf-8")
        refs["Expressions"].append({"Name": name, "File": f"expressions/{name}.exp3.json"})

def build():
    settings, dictionary = [], []
    for i, (name, inputs, out, scale, verts) in enumerate(SETTINGS, 1):
        sid = f"PhysicsSetting{i}"
        dictionary.append({"Id": sid, "Name": name})
        settings.append({
            "Id": sid,
            "Input": inputs,
            "Output": [{"Destination": {"Target": "Parameter", "Id": out},
                        "VertexIndex": len(verts) - 1, "Scale": scale, "Weight": 100,
                        "Type": "Angle", "Reflect": False}],
            "Vertices": verts,
            "Normalization": {"Position": {"Minimum": -10, "Default": 0, "Maximum": 10},
                              "Angle": {"Minimum": -10, "Default": 0, "Maximum": 10}},
        })
    return {
        "Version": 3,
        "Meta": {
            "PhysicsSettingCount": len(settings),
            "TotalInputCount": sum(len(s["Input"]) for s in settings),
            "TotalOutputCount": sum(len(s["Output"]) for s in settings),
            "VertexCount": sum(len(s["Vertices"]) for s in settings),
            "EffectiveForces": {"Gravity": {"X": 0, "Y": -1}, "Wind": {"X": 0, "Y": 0}},
            "PhysicsDictionary": dictionary,
        },
        "PhysicsSettings": settings,
    }

def normalize_export():
    if not EXPORT.exists():
        sys.exit(f"書き出しフォルダがありません: {EXPORT}")
    if not list(EXPORT.glob("*.model3.json")):
        sys.exit(f"model3.json がありません。Cubismで {EXPORT} に書き出してください")
    """別名の書き出し（Cubismの既定名など）があれば mafuyu.* に置き換える。新しい方を優先。"""
    for m in EXPORT.glob("*.model3.json"):
        stem = m.name[:-len(".model3.json")]
        if stem == "mafuyu":
            continue
        print("rename:", stem, "-> mafuyu")
        for f in list(EXPORT.iterdir()):
            if f.name.startswith(stem + "."):
                dst = EXPORT / ("mafuyu" + f.name[len(stem):])
                if dst.is_dir():
                    shutil.rmtree(dst)
                elif dst.exists():
                    dst.unlink()
                f.rename(dst)

def write_model3():
    """model3.json の参照を mafuyu.* に揃え、物理演算を足す。"""
    (EXPORT / NAME).write_text(json.dumps(build(), ensure_ascii=False, indent="\t"), encoding="utf-8")
    m = EXPORT / "mafuyu.model3.json"
    model = json.loads(m.read_text(encoding="utf-8-sig"))
    refs = model["FileReferences"]
    refs["Moc"] = "mafuyu.moc3"
    refs["Textures"] = ["mafuyu." + t.split(".", 1)[1].split("/", 1)[0] + "/" + t.split("/", 1)[1]
                        for t in refs["Textures"]]
    if "DisplayInfo" in refs:
        refs["DisplayInfo"] = "mafuyu.cdi3.json"
    refs["Physics"] = NAME
    if V3:
        write_expressions(refs)
    m.write_text(json.dumps(model, ensure_ascii=False, indent="\t"), encoding="utf-8")

def copy_to_vts():
    """VTS側の mafuyu.vtube.json などは残し、モデル本体だけ上書きする。"""
    if not MODELS.exists():
        print("skip (VTSのモデルフォルダなし):", MODELS); return
    VTS.mkdir(exist_ok=True)
    for f in EXPORT.iterdir():
        dst = VTS / f.name
        if f.is_dir():
            shutil.copytree(f, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(f, dst)
    print("copied to VTS:", VTS)

if __name__ == "__main__":
    normalize_export()
    write_model3()
    copy_to_vts()
    print("export:", sorted(p.name for p in EXPORT.iterdir()))
