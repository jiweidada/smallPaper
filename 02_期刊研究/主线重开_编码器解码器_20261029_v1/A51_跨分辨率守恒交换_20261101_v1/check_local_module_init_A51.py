from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from conservative_scale_exchange import ScaleExchange
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(20261102);module=ScaleExchange(192,'M')
h=hashlib.sha256()
for key,value in module.state_dict().items():h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
remote=json.loads((ROOT/'M_startup/initialization.json').read_text())['new_module_initialization_sha256']
result={'status':'PASS' if h.hexdigest()==remote else 'NEEDS_EXACT_REMOTE_INIT_ARTIFACT',
 'local_torch_version':torch.__version__,'local_module_initialization_sha256':h.hexdigest(),
 'recorded_remote_module_initialization_sha256':remote,
 'scope':'CPU module construction/hash only; verifies planned checkpoint comparison reference across runtime versions',
 'whole_detector_forwards':0,'module_forwards':0,'optimizer_updates':0}
p=ROOT/'local_module_init_consistency_v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
