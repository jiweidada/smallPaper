"""CPU backward of saved leaf logits using unchanged native DEIMCriterion.

No model is instantiated. No weights, data stream, optimizer or GPU are used.
The original saved matching indices and targets are consumed without rematching.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import argparse, datetime, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',required=True);parser.add_argument('--seed',type=int,required=True);a=parser.parse_args()
    torch.set_num_threads(1);started=time.monotonic();prefix=Path(a.prefix)
    root=prefix/'DEIM/experiments/matched_quality_signal_20261010_v1';protocol=read(root/'protocol.json')
    assert sha(Path(__file__))==protocol['code_sha256']['verify_native.py']
    with (root/'one_shot.lock').open('x') as f:f.write(str(os.getpid()))
    output=root/'native_verification';output.mkdir(exist_ok=False)
    save(root/'process.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'seed':a.seed,'started_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(root/'protocol.json')})
    capture=prefix/'DEIM/experiments/native_augmented_supervision_20261010_v1/capture_v1'
    identity=read(capture/'source_identity.json');native=prefix/'DEIM/experiments/fine_residual_screen24_20261009_v2/native'
    before={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')};assert before==identity['native_files_sha256']
    sys.path.insert(0,str(native))
    from engine.deim.deim_criterion import DEIMCriterion
    from engine.deim.box_ops import box_iou, box_cxcywh_to_xyxy
    cfg=dict(identity['config']['DEIMCriterion']);cfg.pop('matcher');criterion=DEIMCriterion(matcher=None,num_classes=10,**cfg);criterion.epoch=4
    assert criterion.gamma==1.5 and criterion.mal_alpha is None and criterion._get_vfl_weight_warmup_factor()==1
    records=[];arrays={};maximum_loss_error=0.;maximum_gradient_error=0.
    for endpoint in (12,132):
     for batch in range(16):
      path=capture/f'endpoint{endpoint}/batch_{batch:03d}';manifest=read(path/'manifest.json')
      for name,h in manifest['files'].items():assert sha(path/name)==h
      with np.load(path/'arrays.npz',allow_pickle=False) as npz:data={k:npz[k] for k in npz.files}
      ev=read(path/'evidence.json');targets=[{'boxes':torch.from_numpy(data[f'target.{j}.boxes']),'labels':torch.from_numpy(data[f'target.{j}.labels'])} for j in range(4)]
      events=[e for e in ev['events'] if e['kind']=='loss_call' and e['loss']=='mal' and not e['is_DN']];assert len(events)==5
      for si,event in enumerate(events):
       stage=event['stage'];logits=torch.from_numpy(data[stage+'.pred_logits']).clone().requires_grad_(True);boxes=torch.from_numpy(data[stage+'.pred_boxes']);indices=[(torch.from_numpy(data[event['prefix']+f'.batch{j}.query_index']),torch.from_numpy(data[event['prefix']+f'.batch{j}.GT_row_index'])) for j in range(4)]
       loss=criterion.loss_labels_mal({'pred_logits':logits,'pred_boxes':boxes},targets,indices,event['num_boxes_normalizer'])['loss_mal'];gradient=torch.autograd.grad(loss,logits)[0]
       idx=criterion._get_src_permutation_idx(indices);matched_boxes=boxes[idx];gt_boxes=torch.cat([targets[j]['boxes'][g] for j,(_,g) in enumerate(indices)]);ious=torch.diag(box_iou(box_cxcywh_to_xyxy(matched_boxes),box_cxcywh_to_xyxy(gt_boxes))[0]).detach()
       classes=torch.cat([targets[j]['labels'][g] for j,(_,g) in enumerate(indices)])
       target=torch.zeros_like(logits);onehot=torch.zeros_like(logits);target[idx[0],idx[1],classes]=ious.pow(criterion.gamma);onehot[idx[0],idx[1],classes]=1
       p=logits.detach().sigmoid();weights=p.pow(criterion.gamma)*(1-onehot)+onehot;expected=weights*(p-target)/event['num_boxes_normalizer']
       tol=protocol['verification'];difference=(gradient-expected).abs();gradient_error=float(difference.max());assert torch.allclose(gradient,expected,atol=tol['native_leaf_gradient_atol'],rtol=tol['native_leaf_gradient_rtol'])
       native_saved=event['native_unweighted_loss_scalars']['loss_mal'];loss_error=abs(float(loss.detach())-native_saved);assert loss_error<=tol['raw_MAL_abs_tolerance']
       maximum_loss_error=max(maximum_loss_error,loss_error);maximum_gradient_error=max(maximum_gradient_error,gradient_error)
       others=logits.detach()[idx].clone();others[torch.arange(len(classes)),classes]=-torch.inf;other_class=others.argmax(-1)
       key=f'seed{a.seed}.endpoint{endpoint}.batch{batch:03d}.stage{si}'
       arrays[key+'.GT_grad']=gradient[idx[0],idx[1],classes].numpy();arrays[key+'.other_grad']=gradient[idx[0],idx[1],other_class].numpy();arrays[key+'.IoU']=ious.numpy();arrays[key+'.GT_row']=torch.cat([g for _,g in indices]).numpy();arrays[key+'.query']=idx[1].numpy();arrays[key+'.max_other_class']=other_class.numpy()
       records.append({'key':key,'stage':stage,'native_CPU_raw_MAL':float(loss.detach()),'archived_CUDA_raw_MAL':native_saved,'abs_loss_error':loss_error,'all_gradient_coordinates_compared':gradient.numel(),'max_gradient_abs_error':gradient_error,'full_gradient_sha256':hashlib.sha256(gradient.contiguous().numpy().tobytes()).hexdigest(),'matched_rows':len(classes)})
       assert time.monotonic()-started<protocol['remote_budget_per_seed']['hard_timeout_seconds']
    np.savez_compressed(output/'matched_native_gradients.npz',**arrays)
    after={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')};assert before==after
    result={'status':'PASS_NATIVE_CPU_MAL_LEAF_AUTOGRAD','seed':a.seed,'calls':records,'leaf_logit_backward_calls':len(records),'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'training_epochs_added':0,'all_gradient_coordinates_compared':sum(r['all_gradient_coordinates_compared'] for r in records),'maximum_CPU_to_archived_CUDA_loss_error':maximum_loss_error,'maximum_native_autograd_to_detached_analytic_gradient_error':maximum_gradient_error,'native_source_sha_unchanged':True,'native_files':len(before),'protocol_sha256':sha(root/'protocol.json'),'code_sha256':sha(Path(__file__)),'torch_version':torch.__version__,'device':'CPU_FP32','elapsed_seconds':time.monotonic()-started,'finished_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output_npz_sha256':sha(output/'matched_native_gradients.npz')}
    save(output/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='calls'}),flush=True)
if __name__=='__main__':main()
