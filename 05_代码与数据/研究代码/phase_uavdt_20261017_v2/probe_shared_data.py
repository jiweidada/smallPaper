import hashlib,json
from pathlib import Path
p=Path('/24085403014/datasets_verified_20261017/UAVDT_official')
manifest=p/'raw/UAVDT_complete_image_manifest_v1.json'
r={'account':CONFIG['account'],'B_native_visible':manifest.is_file()}
if manifest.is_file():
    raw=manifest.read_bytes();items=json.loads(raw)
    r['manifest_sha256']=hashlib.sha256(raw).hexdigest();r['samples']=[]
    for item in [items[0],items[len(items)//2],items[-1]]:
        sample=p/'native_frames_v2'/item['relative']
        r['samples'].append({'relative':item['relative'],'matches':sample.is_file() and hashlib.sha256(sample.read_bytes()).hexdigest()==item['sha256']})
print(json.dumps(r),flush=True)
