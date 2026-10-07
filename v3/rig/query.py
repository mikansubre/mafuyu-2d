"""Read objects from the open Cubism model: uvx --from cubism-mcp python v3/rig/query.py ID [ID ...]"""
import asyncio
import json
import sys

import cubism_mcp


async def main(ids):
    cubism_mcp.client.start()
    uid = json.loads(await cubism_mcp.cubism_get_model_uid())
    uid = uid.get('ModelUID') or uid.get('ModelUid') or uid
    for i in ids:
        print(i, await cubism_mcp.cubism_get_object(uid, i))


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1:]))
