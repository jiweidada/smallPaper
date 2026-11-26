from pathlib import Path
import hashlib,json,torch
from distribution_relation import DistributionRelation
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
assert hashlib.sha256((ROOT/'distribution_relation.py').read_bytes()).hexdigest()=='24df94d4101fe1afbd42d7bcb9238b37b66dee73b04f880dc62b5ad1853d6fbe'
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(20261102);m=DistributionRelation(8,4,300,'M',4)
state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
h=hashlib.sha256()
for k,v in state.items():h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
expected=json.loads((ROOT/'arm_M/initialization.json').read_text())['new_module_initialization_sha256']
assert h.hexdigest()==expected=='69da2b6c3fd9b4c7fa879d807880542cb3a31243004895a67502d8c418829bd0',(h.hexdigest(),expected)
p=ROOT/'new_module_untrained_initial_state.pth';assert not p.exists()
torch.save({'state_dict':state,'seed':20261102,'tensor_sha256':h.hexdigest(),'torch_version':str(torch.__version__)},p)
j={'status':'PASS','path':str(p),'bytes':p.stat().st_size,'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'tensor_sha256':h.hexdigest(),'matches_declared_formal_M_initial_state':True,
 'detector_model_constructions':0,'detector_forwards':0,'optimizer_updates':0,
 'scope':'Small CPU module recreated in same server environment after formal start; archived state matches the preexisting formal initial hash. Never used to reset running training.'}
(ROOT/'new_module_untrained_initial_state_receipt.json').write_text(json.dumps(j,indent=2))
print(json.dumps(j))

