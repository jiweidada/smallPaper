"""CPU-only asset receipt; does not import author code or construct a detector."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import zipfile
import torch

root=Path('/root/c10_region_restore_20261029_v1/architecture_probe_v1/dome_assets_v1')
receipt=json.loads((root/'asset_verification.json').read_text(encoding='utf-8'))
assert receipt['source_commit']=='2dde3bc1946a3e9fad9abd0612b59fc39bd6b861'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
source_sha=digest(root/'author_source.zip')
weight_sha=digest(root/'dome-s-visdrone_converted.pth')
assert source_sha==receipt['source_zip_sha256']
assert weight_sha==receipt['weight']['sha256']=='42874302fd97e17c2a23a46a8ec35d2ea4d2ffe479f0d632c1327fafbf910eb7'
prefix='Dome-DETR-'+receipt['source_commit']+'/'
with zipfile.ZipFile(root/'author_source.zip') as z:
    assert z.testzip() is None
    for row in receipt['source_files']:
        content=z.read(prefix+row['path'])
        assert len(content)==row['bytes']
        assert hashlib.sha256(content).hexdigest()==row['sha256']
state=torch.load(root/'dome-s-visdrone_converted.pth',map_location='cpu',weights_only=True)
container='direct_state_dict'
if isinstance(state,dict) and 'model' in state and isinstance(state['model'],dict):
    state=state['model']; container='model'
assert isinstance(state,dict) and state
assert all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items())
output={'utc':datetime.now(timezone.utc).isoformat(),
 'status':'PASS_REMOTE_ASSET_HASH_SOURCE_ARCHIVE_AND_CPU_WEIGHT_ONLY_LOAD',
 'source_files':len(receipt['source_files']),'source_zip_sha256':source_sha,'weight_sha256':weight_sha,
 'weight_bytes':(root/'dome-s-visdrone_converted.pth').stat().st_size,
 'state_container':container,'state_dict_tensors':len(state),
 'tensor_elements_including_buffers':sum(t.numel() for t in state.values()),
 'example_keys':list(state)[:12], 'torch_version':torch.__version__,
 'all_loaded_tensors_cpu':all(v.device.type=='cpu' for v in state.values()),
 'author_modules_imported':False,'detector_constructed':False,'strict_model_load_verified':False,
 'new_full_detector_forwards':0,'new_fits':0,'main_detector_updates':0,
 'scope':'Asset and CPU deserialization compatibility only; no inference, model equivalence, AP or training-memory feasibility.'}
(root/'remote_asset_preflight.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(output,ensure_ascii=False))
