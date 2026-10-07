"""Quick state check of the open v3 model (which deformers exist, eye keys, parents)."""
import asyncio
import json

import cubism_mcp


async def main():
    cubism_mcp.client.start()
    uid = json.loads(await cubism_mcp.cubism_get_model_uid())
    uid = uid.get('ModelUID') or uid.get('ModelUid') or uid
    for i in ['Warp_Neck', 'Rot_Body', 'Rot_Breath', 'Rot_TailL', 'Rot_Ahoge']:
        o = json.loads(await cubism_mcp.cubism_get_object(uid, i))
        print(i, 'exists' if o.get('Result') else 'MISSING', (o.get('Data') or {}).get('ParentDeformerId'))
    for i in ['ArtMesh59', 'ArtMesh57', 'ArtMesh', 'ArtMesh71']:
        k = json.loads(await cubism_mcp.cubism_get_parameter_keys(uid, i))
        o = json.loads(await cubism_mcp.cubism_get_object(uid, i))
        print(i, o['Data']['ParentDeformerId'], [(p['Id'], p['KeyValues']) for p in k['Parameters']])
    for i in ['Rot_Body', 'Rot_TailL']:
        k = json.loads(await cubism_mcp.cubism_get_parameter_keys(uid, i))
        print(i, [(p['Id'], p['KeyValues']) for p in k['Parameters']])


asyncio.run(main())
