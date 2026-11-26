"""Explicit-device deterministic synthetic smoke; never reads images/cache/GT.

python synthetic_device_test.py --device cuda:0 --output /new/audit.json
Execute only when the device is idle/authorized. Two AdamW steps use entirely
procedural inputs, one refiner step then one gate step; no real-data training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

# Must precede CUDA initialization. Do not alter CUDA_VISIBLE_DEVICES supplied
# by the controlling runner, which chooses the permitted idle device.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import torch
from refiner import (ROIRefiner,prepare_rois,fixed_bindings,refinement_loss,
                     gain_loss,utility_targets,apply_refinement)


def require(condition,message):
    if not bool(condition):raise AssertionError(message)


def state_sha(model,prefixes=None):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        if prefixes is not None and not k.startswith(tuple(prefixes)):continue
        v=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


def smoke(args,result):
    device=torch.device(args.device)
    require(device.type in ('cpu','cuda'),'only explicit cpu/cuda devices supported')
    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    yy,xx=torch.meshgrid(torch.arange(96,device=device),torch.arange(128,device=device),indexing='ij')
    image=torch.stack(((xx%7)/6.,(yy%11)/10.,((xx+yy)%13)/12.)).float()
    boxes=torch.tensor([[40.,32.,50.,42.]],device=device)
    gt=torch.tensor([[41.,32.,51.,42.]],device=device)
    labels=torch.tensor([0],device=device);scores=torch.tensor([.8],device=device)
    qids=torch.tensor([3],device=device)
    features=torch.linspace(-1.,1.,256,device=device).reshape(1,256).requires_grad_(True)
    rois=prepare_rois(image,boxes,qids,input_hw=(96,128),arm='raw')
    bindings=fixed_bindings(boxes,labels,scores,gt,labels)
    model=ROIRefiner(query_dim=256,stats_dim=0,initialization_seed=123).to(device)
    model.set_stage('refiner')
    initial=state_sha(model);gate_initial=state_sha(model,('gate.',))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    output=model(rois,boxes,labels,features)
    require(torch.equal(output['boxes1'],boxes),'zero-init geometry differs')
    loss=refinement_loss(boxes,output['boxes1'],gt,bindings,rois.eligible)['loss']
    require(torch.isfinite(loss),'nonfinite synthetic geometry loss')
    loss.backward()
    require(features.grad is None,'detector-feature gradient leak')
    gradient=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()  # Procedural synthetic inputs ONLY.
    require(state_sha(model)!=initial,'synthetic refiner update absent')
    require(state_sha(model,('gate.',))==gate_initial,'refiner step modified frozen gate')
    model.zero_grad(set_to_none=True);model.set_stage('gate')
    frozen=state_sha(model,('encoder.','geometry.'));gate_before=state_sha(model,('gate.',))
    gate_optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    output=model(rois,boxes,labels,features)
    measured=utility_targets(boxes,output['boxes1'],labels,gt,labels,bindings,rois.eligible,scores=scores)
    require({'measured_u','new_count_by_threshold','lost_count_by_threshold',
             'target_y','identity_valid','local_fixed_gt_delta'} <= set(measured),
            'full single-item utility labels missing')
    require(all(not t.requires_grad for t in measured.values()),'utility not detached')
    # Deliberately nonzero SYNTHETIC target checks optimizer plumbing; it is
    # not claimed to equal the measured utility of this procedural proposal.
    gate_objective=gain_loss(output['gain'],{'target_y':torch.tensor([.3],device=device)},rois.eligible)
    gate_objective.backward()
    require(all(p.grad is None for p in model.encoder.parameters()),'gate-to-encoder gradient leak')
    require(all(p.grad is None for p in model.geometry.parameters()),'gate-to-geometry gradient leak')
    gate_optimizer.step()  # Procedural synthetic inputs ONLY.
    require(state_sha(model,('encoder.','geometry.'))==frozen,'gate changed frozen model')
    require(state_sha(model,('gate.',))!=gate_before,'synthetic gate update absent')
    with torch.no_grad():
        final=model(rois,boxes,labels,features)
        require(((final['gain']>=-1)&(final['gain']<=1)).all(),'gain not bounded')
        rejected,accepted=apply_refinement(boxes,final['boxes1'],final['gain'],rois.eligible,threshold=1.)
        require(torch.equal(rejected,boxes) and not bool(accepted.any()),'tau1 does not reject all')
    if device.type=='cuda':torch.cuda.synchronize(device)
    result.update(status='PASS',torch=torch.__version__,device=str(device),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                  real_images_read=0,real_optimizer_steps=0,synthetic_optimizer_steps=2,
                  geometry_loss=float(loss.detach()),geometry_grad_norm=float(gradient),
                  gate_loss=float(gate_objective.detach()),initial_state_sha256=initial,
                  final_state_sha256=state_sha(model),frozen_geometry_state_sha256=frozen,
                  gate_isolation=True,tau1_all_reject=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',required=True,help='explicit cpu or authorized idle cuda:N')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result={'status':'RUNNING','scope':'synthetic-only deterministic forward/backward/AdamW smoke',
            'real_images_read':0,'real_optimizer_steps':0,
            'refiner_sha256':hashlib.sha256(Path(__file__).with_name('refiner.py').read_bytes()).hexdigest()}
    started=time.perf_counter()
    try:smoke(args,result)
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({'status':result['status'],'output':str(path),'real_optimizer_steps':0}),flush=True)


if __name__=='__main__':main()
