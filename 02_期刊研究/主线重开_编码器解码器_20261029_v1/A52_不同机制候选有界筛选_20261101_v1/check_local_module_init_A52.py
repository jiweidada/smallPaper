from pathlib import Path
import sys,hashlib,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from distribution_relation import DistributionRelation
torch.set_num_threads(4)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(20261102);m=DistributionRelation(8,4,300,'M',4)
h=hashlib.sha256()
for k,v in m.state_dict().items():h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
expected=json.loads((ROOT/'M_startup/initialization.json').read_text())['new_module_initialization_sha256']
j={'status':'EXACT_CPU_RECREATION_MATCH' if h.hexdigest()==expected else 'EXACT_CPU_RECREATION_MISMATCH_USE_REMOTE_UNTRAINED_STATE',
 'local_torch_version':str(torch.__version__),'local_num_threads':torch.get_num_threads(),'local_sha256':h.hexdigest(),
 'formal_M_declared_initial_sha256':expected,'training_changed':False,'detector_forwards':0,'optimizer_updates':0,
 'reason_scope':'Cross-version exact tensor bytes differ; specific numerical cause not assumed. Do not modify trained initialization or silently loosen checks.'}
(ROOT/'local_module_init_cross_version_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps(j))

