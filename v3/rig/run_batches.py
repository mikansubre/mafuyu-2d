"""Send v3/rig/batch_*.json to the open Cubism model through cubism-mcp's client.

uvx --from cubism-mcp python v3/rig/run_batches.py batch_2_eyes.json [more.json ...]
Uses the token cubism-mcp already saved (~/.cubism-mcp/token.txt), so no new approval is needed.
Each file runs in one EditBegin/EditEnd; any error rolls that file back and stops.
"""
import asyncio
import json
import os
import sys

import cubism_mcp

HERE = os.path.dirname(os.path.abspath(__file__))


async def main(files):
    cubism_mcp.client.start()
    for name in files:
        with open(os.path.join(HERE, name), encoding='utf-8') as f:
            actions = json.load(f)
        out = json.loads(await cubism_mcp.cubism_edit_batch(actions))
        if out.get('cancelled') or 'completed' not in out:
            bad = out.get('results', [{}])[-1] if out.get('results') else out
            print(name, 'FAILED', json.dumps(bad, ensure_ascii=False)[:500])
            sys.exit(1)
        print(name, 'ok', out['completed'], '/', out['total'])


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1:]))
