import base64
import hashlib
import json
import os
import pathlib

PAYLOAD = {}  # INSERT
storage = pathlib.Path('/25085400120' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
addons = phase/'artifacts/round2_20261014_v1/addons_v2'
target = addons/'dump_preds.py'
backup = addons/'dump_preds.py.bak_unfused_20261014'
data = base64.b64decode(PAYLOAD['base64'])
assert hashlib.sha256(data).hexdigest() == PAYLOAD['sha256']
with backup.open('xb') as f:f.write(target.read_bytes())
temporary = addons/'dump_preds.py.fused_new'
with temporary.open('xb') as f:f.write(data)
os.replace(temporary,target)
manifest = json.loads((addons/'deployment_manifest.json').read_text())
manifest['dump_preds.py'] = PAYLOAD['sha256']
assert all(hashlib.sha256((addons/name).read_bytes()).hexdigest()==digest for name,digest in manifest.items())
with (addons/'deployment_manifest_fusion_20261014.json').open('x') as f:json.dump(manifest,f,indent=2)
print(json.dumps({'status':'fusion_path_aligned','backup':str(backup),'new_sha256':PAYLOAD['sha256'],'training_code_unchanged':True}))
