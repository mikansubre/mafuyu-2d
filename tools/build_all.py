"""Rebuild every asset, the single-file viewer and the Live2D parts PSD.

  python tools/build_all.py            full build (uses the AI tools in ai/venv when present)
  python tools/build_all.py --no-ai    skip LaMa repaint and THA3 flow baking
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_PY = os.path.join(ROOT, 'ai', 'venv', 'Scripts', 'python.exe')
use_ai = '--no-ai' not in sys.argv and os.path.exists(AI_PY)


def run(py, script, *args):
    print('>>', os.path.basename(script), *args, flush=True)
    subprocess.run([py, os.path.join(ROOT, script), *args], check=True)


run(sys.executable, 'tools/build_parts.py')
if use_ai:
    run(AI_PY, 'ai/lama_refine.py', 'base')
run(sys.executable, 'tools/split_layers.py')
if use_ai:
    run(AI_PY, 'ai/lama_refine.py', 'layers')
    run(AI_PY, 'ai/export_flow.py')
    run(AI_PY, 'ai/split_head.py')
    run(AI_PY, 'ai/build_psd.py')
run(sys.executable, 'tools/bundle.py')
