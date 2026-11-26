"""Bounded assignment-contract check before using the public matcher on dense UAV GT."""
from pathlib import Path
import ast
import hashlib
import json
import multiprocessing as mp
import sys
from types import SimpleNamespace

P=Path(__file__).resolve().parent
if sys.platform=='win32':sys.path.insert(0,str(P.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))

def worker(gt_count,queue):
    import torch
    from torchvision.ops import box_iou,generalized_box_iou
    torch.set_num_threads(2);torch.manual_seed(20261102+gt_count)
    source=(P/'DiffusionDet_primary_source_v1/diffusiondet/loss.py').read_text()
    tree=ast.parse(source)
    fn=next(f for c in tree.body if isinstance(c,ast.ClassDef) for f in c.body
        if isinstance(f,ast.FunctionDef) and f.name=='dynamic_k_matching')
    module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
    namespace={'torch':torch};exec(compile(module,'original_selected_matcher_function','exec'),namespace)
    def corners(n):
        center=torch.rand(n,2);wh=.01+.12*torch.rand(n,2)
        return torch.cat((center-wh/2,center+wh/2),dim=1)
    q=1024;prior=corners(q);gt=corners(gt_count)
    iou=box_iou(prior,gt)
    cost=5*torch.cdist(prior,gt,p=1)+2*(1-generalized_box_iou(prior,gt))
    (selected,index),per_gt=namespace['dynamic_k_matching'](SimpleNamespace(ota_k=5),cost,iou,gt_count)
    represented=int(index.unique().numel())
    queue.put(dict(gt_count=gt_count,queries=q,selected=int(selected.sum()),
        represented_GT=represented,per_GT_unique_queries=int(per_gt.unique().numel()),
        every_GT_represented_in_returned_training_indices=represented==gt_count,
        model_constructors=0,model_forwards=0,optimizer_updates=0))

def main():
    target=P/'native_matcher_contract_CPU_v1.json';assert not target.exists()
    rows=[]
    for n in [10,902]:
        queue=mp.Queue();process=mp.Process(target=worker,args=(n,queue));process.start();process.join(15)
        if process.is_alive():
            process.terminate();process.join()
            rows.append(dict(gt_count=n,status='BOUNDED_PROCESS_TIMEOUT_NO_RETRY'))
        elif process.exitcode!=0:rows.append(dict(gt_count=n,status='FUNCTION_PROCESS_FAILED',exit=process.exitcode))
        else:rows.append(queue.get(timeout=2))
    result=dict(status='SOURCE_ASSIGNMENT_CONTRACT_CHECK_COMPLETE',rows=rows,
        selected_source_sha256=hashlib.sha256((P/'DiffusionDet_primary_source_v1/diffusiondet/loss.py').read_bytes()).hexdigest(),
        matcher_function_calls_allocated=2,model_constructors=0,model_forwards=0,optimizer_updates=0,
        limits='Synthetic valid box geometry and selected public function only. A failed contract is an implementation issue, not an AP failure or proof of actual prior-model dataset outcomes.')
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
