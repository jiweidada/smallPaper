from pathlib import Path
import json
import sys
import time
from types import SimpleNamespace

P=Path(__file__).resolve().parent
if sys.platform=='win32':sys.path.insert(0,str(P.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from torchvision.ops import box_iou,generalized_box_iou
from criterion_A60_v1 import HungarianMatcherDynamicK

def main():
    out=P/'fixed_matcher_contract_CPU_v1.json';assert not out.exists()
    torch.set_num_threads(2);rows=[];start=time.monotonic()
    for n in [10,902]:
        torch.manual_seed(20261102+n)
        def corners(count):
            center=torch.rand(count,2);wh=.01+.12*torch.rand(count,2)
            return torch.cat((center-wh/2,center+wh/2),dim=1)
        prior=corners(1024);gt=corners(n)
        iou=box_iou(prior,gt);cost=5*torch.cdist(prior,gt,p=1)+2*(1-generalized_box_iou(prior,gt))
        (selected,index),per_gt=HungarianMatcherDynamicK.dynamic_k_matching(SimpleNamespace(ota_k=5),cost,iou,n)
        assert index.unique().numel()==n and per_gt.unique().numel()==n
        rows.append(dict(gt=n,selected_queries=int(selected.sum()),represented_GT=int(index.unique().numel()),
                         unique_per_GT_queries=int(per_gt.unique().numel())))
    result=dict(status='PASS_FIXED_DENSE_ASSIGNMENT_CONTRACT',rows=rows,matcher_function_calls=2,
        model_constructors=0,model_forwards=0,optimizer_updates=0,elapsed_seconds=time.monotonic()-start,
        limits='Same two synthetic geometry cases as native-source check; no actual detector outcome or claimed research novelty.')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
