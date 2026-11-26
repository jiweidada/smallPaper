"""One safe local CPU data load of the official public warm-start checkpoint."""
from pathlib import Path
import hashlib
import json
import sys
import torch

P=Path(__file__).resolve().parent

def main():
    p=P/'assets_local_v1/official_diffdet_coco_res50_300boxes.pth'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='31c31e39f7b44bd9735c3181fdb0ea8bd092aa64229067866e4f56480efa6f49'
    dest=P/'warm_checkpoint_tensor_metadata_v1.json';assert not dest.exists()
    data=torch.load(p,map_location='cpu',weights_only=True)
    state=data['model'] if 'model' in data else data
    assert isinstance(state,dict)
    rows={k:dict(shape=list(v.shape),dtype=str(v.dtype),finite=bool(torch.isfinite(v).all()))
          for k,v in state.items() if torch.is_tensor(v)}
    result=dict(status='OFFICIAL_WARM_CHECKPOINT_CPU_METADATA_LOADED',top_keys=list(data),
        tensors=rows,tensor_entries=len(rows),state_non_tensor_keys=[k for k,v in state.items() if not torch.is_tensor(v)],
        checkpoint_loads=1,model_constructors=0,model_forwards=0,optimizer_updates=0,
        checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'}))
    print(json.dumps(dict(first=list(rows.items())[:8],last=list(rows.items())[-8:])))

if __name__=='__main__':main()
