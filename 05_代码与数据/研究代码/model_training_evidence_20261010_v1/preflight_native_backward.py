"""CPU FP32 native forward/loss/backward parity; not an optimizer training run."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from training_evidence import TrainingEvidenceRecorder,sha

def save(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def clone_data(value):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,dict):return {k:clone_data(v) for k,v in value.items()}
    if isinstance(value,list):return [clone_data(v) for v in value]
    if isinstance(value,tuple):return tuple(clone_data(v) for v in value)
    return copy.deepcopy(value)

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def state():return (torch.get_rng_state().clone(),random.getstate(),copy.deepcopy(np.random.get_state()))
def restore(s):torch.set_rng_state(s[0]);random.setstate(s[1]);np.random.set_state(s[2])
def state_equal(a,b):return torch.equal(a[0],b[0]) and a[1]==b[1] and a[2][0]==b[2][0] and np.array_equal(a[2][1],b[2][1]) and a[2][2:]==b[2][2:]

def gradient_receipt(model):
    h=hashlib.sha256();present=absent=0
    for name,p in sorted(model.named_parameters()):
        h.update(name.encode())
        if p.grad is None:h.update(b'None');absent+=1
        else:
            value=p.grad.detach().cpu().contiguous()
            assert torch.isfinite(value).all()
            h.update(str((str(value.dtype),tuple(value.shape))).encode());h.update(value.numpy().tobytes());present+=1
    return dict(sha256=h.hexdigest(),parameters_with_gradient=present,parameters_without_gradient=absent)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native',required=True);ap.add_argument('--args',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    save(out/'start.json',dict(status='RUNNING_NATIVE_BACKWARD_OBSERVER_ENGINEERING_PREFLIGHT',training_epochs_added=0,optimizer_steps=0,device='CPU',dtype='FP32',max_wall_seconds=300))
    recorder=None
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        native=Path(args.native);sys.path.insert(0,str(native))
        sources={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        from engine.core import YAMLConfig
        from engine.backbone.hgnetv2 import HGNetv2
        from engine.deim.hybrid_encoder import HybridEncoder
        from engine.deim.dfine_decoder import DFINETransformer
        from engine.deim.deim import DEIM
        from engine.deim.deim_criterion import DEIMCriterion
        from engine.deim.matcher import HungarianMatcher
        from torchvision.ops import box_convert
        config=json.loads(Path(args.args).read_text())['yaml_cfg']
        assert sha(args.checkpoint)=='30febfa18e6e21dc67b16169bd26875b82340eb3435534719a963904dadf68b5'
        fixture=copy.deepcopy(config);fixture.pop('__include__',None);fixture.pop('config',None)
        fixture['val_dataloader']['num_workers']=0
        fixture['val_dataloader']['dataset']['transforms']['ops'].append(dict(type='ConvertBoxes',fmt='cxcywh',normalize=True))
        save(out/'fixture.yml',fixture)
        cfg=YAMLConfig(str(out/'fixture.yml'))
        dataset=cfg.val_dataloader.dataset
        x,target=dataset[1];x=x.as_subclass(torch.Tensor).unsqueeze(0);targets=[target]
        assert list(x.shape)==[1,3,640,640]
        image_id=int(target['image_id']);meta=dataset.coco.imgs[image_id]
        rows=[r for r in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[image_id])) if not r.get('iscrowd',0)]
        raw=torch.tensor([r['bbox'] for r in rows],dtype=torch.float32).reshape(-1,4)
        raw[:,2:]+=raw[:,:2];raw[:,0::2].clamp_(0,meta['width']);raw[:,1::2].clamp_(0,meta['height'])
        keep=(raw[:,2]>raw[:,0])&(raw[:,3]>raw[:,1]);rows=[r for r,k in zip(rows,keep.tolist()) if k]
        resized=raw[keep]*torch.tensor([640/meta['width'],640/meta['height']]*2)
        expected=box_convert(resized,'xyxy','cxcywh')/640
        target_boxes=target['boxes'].as_subclass(torch.Tensor)
        assert len(rows)==len(target['labels']) and target['labels'].tolist()==[r['category_id'] for r in rows]
        box_error=float((expected-target_boxes).abs().max());assert box_error<1e-6
        source_ids=[[r['id'] for r in rows]]
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);assert payload['last_epoch']==131
        checkpoint_state=payload['ema']['module']
        def build():
            model=DEIM(HGNetv2(**config['HGNetv2']),HybridEncoder(**config['HybridEncoder'],eval_spatial_size=[640,640]),DFINETransformer(**config['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640]),num_classes=10)
            model.load_state_dict(checkpoint_state,strict=True);model.train()
            recipe=copy.deepcopy(config['DEIMCriterion']);match=recipe.pop('matcher');match.pop('type',None)
            criterion=DEIMCriterion(matcher=HungarianMatcher(**match,use_focal_loss=True),num_classes=10,**recipe)
            return model,criterion
        plain,pc=build();observed,oc=build()
        assert equal(plain.state_dict(),observed.state_dict())
        before_parameters={n:p.detach().clone() for n,p in plain.named_parameters()}
        torch.manual_seed(1723);random.seed(1723);np.random.seed(1723);rng_before=state()
        tick=time.monotonic();pt=clone_data(targets);po=plain(x.clone(),targets=pt);pl=pc(po,pt,epoch=0,step=0,global_step=0,epoch_step=1);sum(pl.values()).backward()
        plain_seconds=time.monotonic()-tick;plain_rng_after=state();pg=gradient_receipt(plain)
        restore(rng_before)
        recorder=TrainingEvidenceRecorder(observed,oc)
        ot=clone_data(targets)
        recorder.begin(dict(purpose='engineer recorder equivalence only',image_id=image_id,epoch_one_based=1,attempt_one_based=1,fixture='native validation direct640 plus native train box normalization; no augmentation or optimizer',source_identity_verified=True,source_identity_box_max_error=box_error,AMP=False,clip_max_norm=.1),ot,loss_scale=1.0,source_annotation_ids=source_ids,input_tensor=x)
        tick=time.monotonic();oo=observed(x.clone(),targets=ot);ol=oc(oo,ot,epoch=0,step=0,global_step=0,epoch_step=1);sum(ol.values()).backward();observed_seconds=time.monotonic()-tick
        assert equal(po,oo) and equal(pl,ol) and equal(pt,ot)
        assert state_equal(plain_rng_after,state())
        og=gradient_receipt(observed);assert pg==og
        for (name,p),(other_name,q) in zip(plain.named_parameters(),observed.named_parameters()):
            assert name==other_name and ((p.grad is None and q.grad is None) or (p.grad is not None and q.grad is not None and torch.equal(p.grad,q.grad)))
            assert torch.equal(p,before_parameters[name]) and torch.equal(q,before_parameters[name])
        assert equal(plain.state_dict(),observed.state_dict())
        recorder.parameter_gradients('FP32_after_backward_before_native_clip; scale1_no_unscale_needed')
        pn=torch.nn.utils.clip_grad_norm_(plain.parameters(),.1);on=torch.nn.utils.clip_grad_norm_(observed.parameters(),.1)
        assert torch.equal(pn,on) and gradient_receipt(plain)==gradient_receipt(observed)
        recorder.parameter_gradients('FP32_after_native_clip_max_norm0.1; scale1_no_unscale_needed')
        evidence=recorder.end(out/'sampled_batch');recorder.close();recorder=None
        assert evidence['status']=='CAPTURED' and not evidence['errors']
        assert evidence['actual_optimizer_steps_in_sample']==0 and len(evidence['gradients'])>0
        assert all(e['stage']!='UNRESOLVED' for e in evidence['events'] if e['kind'] in ('matcher','loss_call'))
        assert sources=={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        result=dict(status='PASS_NATIVE_CPU_FP32_FORWARD_LOSS_BACKWARD_AND_CLIP_EQUIVALENCE',image_id=image_id,GT_rows=len(rows),source_annotation_identity_verified=True,source_identity_max_box_error=box_error,model_output_equal=True,weighted_loss_equal=True,gradients_before_clip_equal=True,gradients_after_clip_equal=True,RNG_trajectory_equal=True,model_parameters_unchanged=True,model_buffers_follow_same_native_train_updates=True,native_source_unchanged=True,training_epochs_added=0,optimizer_steps=0,native_forward_calls=2,native_backward_calls=2,gradient_receipt_before_clip=pg,loss_count=len(pl),matcher_calls=sum(e['kind']=='matcher' for e in evidence['events']),loss_calls=sum(e['kind']=='loss_call' for e in evidence['events']),activation_gradient_observations=len(evidence['gradients']),plain_seconds=plain_seconds,observed_seconds=observed_seconds,elapsed_seconds=time.monotonic()-started,AMP_CUDA_parity_verified=False,native_augmented_training_history_recorded=False,scientific_root_cause_identified=False,checkpoint_sha256=sha(args.checkpoint),code_sha256={p.name:sha(p) for p in Path(__file__).resolve().parent.glob('*.py')},source_python_sha256=sources,limits='Single engineering fixture on mature weights in training mode. CPU FP32, no optimizer, no learning history and no augmentation; parameters are unchanged while BN buffers undergo identical native training-mode updates in private models. This is observer parity, not detector efficacy or causal/gradient diagnostic evidence. Source identity only verified for this fixture; native Mosaic/MixUp provenance requires a separate integration. CUDA AMP, optimizer callbacks and multiworker loader adapter not exercised here.')
        save(out/'completion.json',result);save(out/'manifest.json',dict(files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_python_sha256','code_sha256','limits')}),flush=True)
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',error=traceback.format_exc(),training_epochs_added=0,optimizer_steps=0))
        raise
    finally:
        if recorder is not None:recorder.close()

if __name__=='__main__':main()
