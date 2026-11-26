import base64
import hashlib
import json
import os
import pathlib
PAYLOAD = {}  # INSERT
storage=pathlib.Path('/25085400120' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014')
phase=next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012',storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
addons=phase/'artifacts/round2_20261014_v1/addons_v2'
manifest=json.loads((addons/'deployment_manifest_fusion_20261014.json').read_text())
for name,row in PAYLOAD.items():
    target=addons/name;data=base64.b64decode(row['base64'])
    assert hashlib.sha256(data).hexdigest()==row['sha256']
    if target.exists():
        with (addons/(name+'.bak_before_cli_bridge')).open('xb') as f:f.write(target.read_bytes())
        temporary=addons/(name+'.cli_new')
        with temporary.open('xb') as f:f.write(data)
        os.replace(temporary,target)
    else:
        with target.open('xb') as f:f.write(data)
    manifest[name]=row['sha256']
assert all(hashlib.sha256((addons/name).read_bytes()).hexdigest()==digest for name,digest in manifest.items())
with (addons/'deployment_manifest_cli_bridge_20261014.json').open('x') as f:json.dump(manifest,f,indent=2)
print(json.dumps({'status':'test_only_predictions_reused','training_source_unchanged':True,'files':list(PAYLOAD)}))
