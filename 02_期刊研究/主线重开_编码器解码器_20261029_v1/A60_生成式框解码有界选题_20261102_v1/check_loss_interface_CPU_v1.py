from pathlib import Path
import json
import sys
import time

P=Path(__file__).resolve().parent
if sys.platform=='win32':sys.path.insert(0,str(P.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from runtime_A60_v1 import criterion,loss_targets
from box_progress_conditioning_v1 import xywh_to_xyxy

def main():
    p=P/'loss_interface_CPU_v1.json';assert not p.exists()
    torch.set_num_threads(2);torch.manual_seed(20261102);start=time.monotonic();rows=[]
    c=criterion()
    for n in [2,0]:
        gt=torch.tensor([[.5,.5,.02,.03],[.2,.2,.04,.04]],dtype=torch.float32)[:n]
        target=dict(boxes=gt,labels=torch.tensor([0,4],dtype=torch.long)[:n])
        box=torch.rand(1,16,4);box[...,2:]=.02+.2*box[...,2:]
        bbox=(xywh_to_xyxy(box)*640.).requires_grad_();logit=torch.randn(1,16,10,requires_grad=True)
        base=dict(pred_logits=logit,pred_boxes=bbox)
        output=dict(base,aux_outputs=[base.copy() for _ in range(5)])
        loss=c(output,loss_targets([target],'cpu'))
        total=sum(v*c.weight_dict[k] for k,v in loss.items() if k in c.weight_dict)
        assert torch.isfinite(total);total.backward()
        assert torch.isfinite(logit.grad).all() and torch.isfinite(bbox.grad).all()
        rows.append(dict(GT=n,loss=float(total.detach()),loss_terms=len(loss),finite_gradients=True))
    result=dict(status='PASS_NONEMPTY_AND_EMPTY_GT_LOSS_INTERFACE',rows=rows,
        criterion_module_constructors=1,criterion_calls=2,CPU_loss_backwards=2,
        detector_constructors=0,detector_forwards=0,optimizer_updates=0,elapsed_seconds=time.monotonic()-start)
    with p.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
