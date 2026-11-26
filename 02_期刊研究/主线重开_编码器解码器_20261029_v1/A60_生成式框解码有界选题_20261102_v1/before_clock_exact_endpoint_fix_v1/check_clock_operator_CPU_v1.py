"""One bounded CPU check of operator interfaces and the single-step GT boundary."""
from pathlib import Path
import hashlib
import json
import sys
import time

P=Path(__file__).resolve().parent
RUNTIME=P.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'
sys.path.insert(0,str(RUNTIME))
import torch
from box_progress_conditioning_v1 import training_inputs,single_step_inputs,relative_edge_distance,geometric_alpha

def main():
    out=P/'clock_operator_CPU_check_v1.json'
    assert not out.exists()
    start=time.monotonic();torch.set_num_threads(2)
    # Four fixtures, three modes each: 12 calls, no model construction or update.
    g=torch.tensor([[[.5,.5,.00625,.009375],[.3,.7,.05,.08]]],dtype=torch.float64)
    p=torch.tensor([[[.1,.2,.5,.6],[.8,.2,.2,.3]]],dtype=torch.float64)
    fixtures=[(p,g,torch.tensor([0.],dtype=torch.float64)),
              (p,g,torch.tensor([.5],dtype=torch.float64)),
              (p,g,torch.tensor([1.],dtype=torch.float64)),
              (g,g,torch.tensor([.37],dtype=torch.float64))]
    rows=[]
    for prior,gt,u in fixtures:
        outputs={m:training_inputs(m,prior,gt,u) for m in ['B','M','Q']}
        assert torch.equal(outputs['M'][0],outputs['Q'][0])
        assert torch.equal(outputs['M'][2],outputs['Q'][2])
        if u.item()==0:
            for value in outputs.values():
                assert torch.equal(value[0],prior) and torch.equal(value[1],torch.full((1,2),999.,dtype=torch.float64))
        if u.item()==1:
            for value in outputs.values():assert torch.allclose(value[0],gt,atol=1e-15)
        if torch.equal(prior,gt):
            for value in outputs.values():assert torch.allclose(value[2],u[:,None].expand(1,2),atol=1e-15)
        if u.item()==.5:
            assert not torch.equal(outputs['M'][1],outputs['Q'][1])
        rows.append(dict(progress=u.item(),M_Q_noisy_boxes_bitwise_equal=True,
            B_time=outputs['B'][1].tolist(),M_time=outputs['M'][1].tolist(),Q_time=outputs['Q'][1].tolist()))
    infer_boxes,infer_time=single_step_inputs(p)
    assert infer_boxes is p and torch.equal(infer_time,torch.full((1,2),999.,dtype=torch.float64))
    d=relative_edge_distance(p,g)
    a=geometric_alpha(torch.tensor([[.5]],dtype=torch.float64),d)
    actual=relative_edge_distance(p+(g-p)*a[...,None],g)
    assert torch.allclose(torch.log1p(actual),.5*torch.log1p(d),atol=1e-12)
    result=dict(status='PASS_CPU_CLOCK_INTERFACES_AND_GT_FREE_SINGLE_STEP',torch_version=torch.__version__,
        fixture_training_input_calls=12,single_step_input_calls=1,rows=rows,
        module_sha256=hashlib.sha256((P/'box_progress_conditioning_v1.py').read_bytes()).hexdigest(),
        model_constructors=0,model_forwards=0,optimizer_updates=0,backwards=0,
        elapsed_seconds=time.monotonic()-start,
        limits='Standalone box/time operator only. Does not validate a learned decoder, per-query FiLM integration, memory capacity, performance or novelty.')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result))

if __name__=='__main__':main()
