"""Inline web/ (html + js + assets) into dist/mafuyu_live2d.html so it opens with a double-click."""
import base64
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, 'web')
A = os.path.join(WEB, 'assets')

parts = json.load(open(os.path.join(A, 'parts.json'), encoding='utf-8'))
files = {l['file'] for l in parts['layers'].values()} | {parts['weights']['file']}
if 'flow' in parts:
    files.add(parts['flow']['file'])
MIME = {'.png': 'image/png', '.bin': 'application/octet-stream'}
assets = {f: f'data:{MIME[os.path.splitext(f)[1]]};base64,' + base64.b64encode(open(os.path.join(A, f), 'rb').read()).decode()
          for f in sorted(files)}

rig = open(os.path.join(WEB, 'rig.js'), encoding='utf-8').read()
app = open(os.path.join(WEB, 'app.js'), encoding='utf-8').read()
rig = re.sub(r'^export \{[^}]*\};\s*$', '', rig, flags=re.M).replace('export class Model', 'class Model')
app = re.sub(r"^import .*?;\s*$", '', app, count=1, flags=re.M)

html = open(os.path.join(WEB, 'index.html'), encoding='utf-8').read()
inline = (
    '<script>window.MAFUYU_ASSETS=' + json.dumps(assets) + ';window.MAFUYU_PARTS=' + json.dumps(parts) + ';</script>\n'
    '<script type="module">\n' + rig + '\n' + app + '\n</script>'
)
html = html.replace('<script type="module" src="app.js"></script>', inline)
os.makedirs(os.path.join(ROOT, 'dist'), exist_ok=True)
out = os.path.join(ROOT, 'dist', 'mafuyu_live2d.html')
open(out, 'w', encoding='utf-8').write(html)
print(out, round(os.path.getsize(out) / 1e6, 2), 'MB')
