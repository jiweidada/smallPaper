"""Frozen EMA train-mode supervision observation, never optimization/history.

32 outcome-independent validation images at BASE12 and BASE132. Native heads,
matcher and criterion execute normally. Separate leaf-box native loss gradients
test a final-assignment-anchored union counterfactual; no model backward/steps.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import copy
import csv
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
from training_evidence import TrainingEvidenceRecorder, sha


CHECKPOINTS = {
    1: {12:'ba3b6351fa8a4f7c769698c616ed6d20acf6a60ba119cf8edc8611851c4d27d4',132:'30febfa18e6e21dc67b16169bd26875b82340eb3435534719a963904dadf68b5'},
    2: {12:'6198ee5348630ce97c322c16aa4499a50a2f7af0d08dea1ec3a4cb632d9d5046',132:'5d3b838e296c866e0472e07d4c5b3dd59f7566029d1fca947ab8cd8b3fc1fb3a'}
}


def save(path, value):
    with Path(path).open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def parameter_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.named_parameters()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def reset_random(image_id):
    torch.manual_seed(1723+image_id);torch.cuda.manual_seed_all(1723+image_id);random.seed(1723+image_id);np.random.seed(1723+image_id)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--seed',type=int,choices=[1,2],required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(out/'start.json',dict(status='RUNNING_FROZEN_ASSIGNMENT_OBSERVATION',device='CUDA_FP32',seed=args.seed,training_epochs_added=0,optimizer_steps=0,automatic_retry=False))
    recorder=None
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        root=Path(args.root);native=root/'DEIM/experiments/fine_residual_screen24_20261009_v2/native';sys.path.insert(0,str(native))
        source_sha={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        from engine.core import YAMLConfig
        from engine.backbone.hgnetv2 import HGNetv2
        from engine.deim.hybrid_encoder import HybridEncoder
        from engine.deim.dfine_decoder import DFINETransformer
        from engine.deim.deim import DEIM
        from engine.deim.deim_criterion import DEIMCriterion
        from engine.deim.matcher import HungarianMatcher
        from torchvision.ops import box_convert,box_iou
        run132=root/f'DEIM/experiments/baseline_official_s_640_20261007_v1/seed{args.seed}/run'
        run12=root/f'DEIM/experiments/fine_residual_screen24_20261009_v2/runs/baseline_seed{args.seed}'
        config=copy.deepcopy(json.loads((run132/'args.json').read_text())['yaml_cfg'])
        early_config=json.loads((run12/'args.json').read_text())['yaml_cfg']
        for key in ('HGNetv2','HybridEncoder','DFINETransformer','DEIMCriterion','num_classes'):
            assert config[key]==early_config[key],key
        assert config['HGNetv2']['pretrained']==False and config['num_classes']==10
        ann_file=Path(config['val_dataloader']['dataset']['ann_file']);assert sha(ann_file)=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
        ann=json.loads(ann_file.read_text());ids=sorted(r['id'] for r in ann['images']);assert len(ids)==548
        positions=[j*(len(ids)-1)//31 for j in range(32)];panel=[ids[j] for j in positions]
        fixture=copy.deepcopy(config);fixture.pop('__include__',None);fixture.pop('config',None);fixture['val_dataloader']['num_workers']=0
        fixture['val_dataloader']['dataset']['transforms']['ops'].append(dict(type='ConvertBoxes',fmt='cxcywh',normalize=True));save(out/'fixture.yml',fixture)
        cfg=YAMLConfig(str(out/'fixture.yml'));dataset=cfg.val_dataloader.dataset
        model=DEIM(HGNetv2(**config['HGNetv2']),HybridEncoder(**config['HybridEncoder'],eval_spatial_size=[640,640]),DFINETransformer(**config['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640]),num_classes=10).cuda().train()
        def build_criterion():
            recipe=copy.deepcopy(config['DEIMCriterion']);match=recipe.pop('matcher');match.pop('type',None)
            return DEIMCriterion(matcher=HungarianMatcher(**match,use_focal_loss=True),num_classes=10,**recipe)
        criterion=build_criterion().cuda();leaf_criterion=build_criterion()
        recorder=TrainingEvidenceRecorder(model,criterion)
        panel_identity=[];conflicts=[];sample_summaries=[];stage_summaries=[];forward_calls=0;leaf_gradient_calls=0
        for endpoint,checkpoint in [(12,run12/'checkpoint0011.pth'),(132,run132/'last.pth')]:
            assert sha(checkpoint)==CHECKPOINTS[args.seed][endpoint]
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False);assert payload['last_epoch']==endpoint-1
            weights=payload['ema']['module'];model.load_state_dict(weights,strict=True);before=parameter_sha(model)
            for i,(position,image_id) in enumerate(zip(positions,panel)):
                model.load_state_dict(weights,strict=True);reset_random(image_id)
                x,target=dataset[position];assert int(target['image_id'])==image_id
                meta=dataset.coco.imgs[image_id];rows=[r for r in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[image_id])) if not r.get('iscrowd',0)]
                raw=torch.tensor([r['bbox'] for r in rows],dtype=torch.float32).reshape(-1,4);raw[:,2:]+=raw[:,:2];raw[:,0::2].clamp_(0,meta['width']);raw[:,1::2].clamp_(0,meta['height'])
                keep=(raw[:,2]>raw[:,0])&(raw[:,3]>raw[:,1]);rows=[r for r,k in zip(rows,keep.tolist()) if k]
                expected=box_convert(raw[keep]*torch.tensor([640/meta['width'],640/meta['height']]*2),'xyxy','cxcywh')/640
                assert target['labels'].tolist()==[r['category_id'] for r in rows]
                error=float((expected-target['boxes'].as_subclass(torch.Tensor)).abs().max()) if len(rows) else 0.;assert error<1e-6
                image_file=Path(config['val_dataloader']['dataset']['img_folder'])/meta['file_name']
                identity=dict(image_id=image_id,source_image_sha256=sha(image_file),annotation_ids=[r['id'] for r in rows],source_identity_max_box_error=error)
                if endpoint==12:panel_identity.append(identity)
                else:assert identity==panel_identity[i]
                x=x.as_subclass(torch.Tensor).unsqueeze(0).cuda();targets=[{k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in target.items()}]
                if i==0:
                    with torch.no_grad():plain=model(x,targets=targets)
                    forward_calls+=1;plain_buffers={n:b.detach().clone() for n,b in model.named_buffers()};plain_rng=[torch.get_rng_state().clone()]+[r.clone() for r in torch.cuda.get_rng_state_all()]
                    model.load_state_dict(weights,strict=True);reset_random(image_id)
                recorder.begin(dict(endpoint=endpoint,seed=args.seed,image_id=image_id,source_identity_verified=True,source_identity_box_max_error=error,purpose='frozen train-mode direct-validation supervision; not historical training',dtype='GPU_FP32',AMP=False,backward_executed=False,source_image_sha256=identity['source_image_sha256']),targets,loss_scale=1.,source_annotation_ids=[identity['annotation_ids']])
                with torch.no_grad():outputs=model(x,targets=targets)
                forward_calls+=1
                if i==0:
                    assert equal(plain,outputs)
                    assert all(torch.equal(b,plain_buffers[n]) for n,b in model.named_buffers())
                    assert all(torch.equal(a,b) for a,b in zip(plain_rng,[torch.get_rng_state()]+torch.cuda.get_rng_state_all()))
                    del plain,plain_buffers
                with torch.no_grad():losses=criterion(outputs,targets,epoch=endpoint-1,step=0,global_step=0,epoch_step=1)
                assert all(torch.isfinite(v) for v in losses.values()) and not recorder.errors
                # Retain compact actual outputs and supervision; discard large
                # corner-distribution/DN tensors and duplicated matched values.
                discarded=len(recorder.data)
                compact={k:v for k,v in recorder.data.items() if k.startswith(('target.','matcher','weighted_loss.')) or (k.startswith('model') and '.dn_' not in k and k.endswith(('.pred_boxes','.pred_logits'))) or (k.startswith('loss_call') and k.endswith(('.query_index','.GT_row_index')))}
                recorder.data=compact;recorder.array_bytes=sum(v.nbytes for v in compact.values());recorder.metadata.update(compact_only=True,discarded_array_count=discarded-len(compact),historical_learning=False)
                sample=out/f'endpoint{endpoint}/image_{image_id:06d}';evidence=recorder.end(sample)
                assert evidence['status']=='CAPTURED' and all(e['stage']!='UNRESOLVED' for e in evidence['events'] if e['kind'] in ('matcher','loss_call'))
                data=compact;normal=[e for e in evidence['events'] if e['kind']=='loss_call' and not e['is_DN']]
                for stage in sorted({e['stage'] for e in normal}):
                    c=next(e for e in normal if e['stage']==stage and e['loss']=='mal')['prefix']+'.batch0'
                    r=next(e for e in normal if e['stage']==stage and e['loss']=='boxes')['prefix']+'.batch0'
                    cm=dict(zip(data[c+'.query_index'].tolist(),data[c+'.GT_row_index'].tolist()));rm=dict(zip(data[r+'.query_index'].tolist(),data[r+'.GT_row_index'].tolist()))
                    qs=[q for q,g in rm.items() if q in cm and cm[q]!=g]
                    stage_summaries.append(dict(endpoint=endpoint,image_id=image_id,stage=stage,classification_pairs=len(cm),regression_pairs=len(rm),GT_conflicts=len(qs),regression_only_queries=sum(q not in cm for q in rm),cross_class_conflicts=sum(rows[cm[q]]['category_id']!=rows[rm[q]]['category_id'] for q in qs)))
                    if stage!='model':continue
                    q=torch.from_numpy(data[r+'.query_index']);g=torch.from_numpy(data[r+'.GT_row_index']);cg=g.clone()
                    for j,query in enumerate(q.tolist()):
                        if query in cm:cg[j]=cm[query]
                    assert q.unique().numel()==q.numel() and q.numel()==cg.numel()
                    base_boxes=torch.from_numpy(data['model.pred_boxes']).clone();target_cpu=[dict(boxes=torch.from_numpy(data['target.0.boxes']),labels=torch.from_numpy(data['target.0.labels']))]
                    normalizer=next(e for e in normal if e['stage']=='model' and e['loss']=='boxes')['num_boxes_normalizer']
                    def gradient(indices):
                        leaf=base_boxes.clone().requires_grad_(True)
                        raw_loss=leaf_criterion.loss_boxes(dict(pred_boxes=leaf),target_cpu,indices,normalizer)
                        value=sum(raw_loss[k]*leaf_criterion.weight_dict[k] for k in raw_loss)
                        grad=torch.autograd.grad(value,leaf)[0];assert torch.isfinite(grad).all()
                        return grad[0]
                    gu=gradient([(q,g)]);ga=gradient([(q,cg)])
                    cq=torch.tensor(list(cm),dtype=torch.int64);gg=torch.tensor(list(cm.values()),dtype=torch.int64);go=gradient([(cq,gg)]);leaf_gradient_calls+=3
                    unchanged=q[g==cg];assert torch.equal(gu[unchanged],ga[unchanged])
                    union_norm=gu.norm(dim=-1);own_norm=go.norm(dim=-1)
                    count_protected=0
                    for query in qs:
                        own=cm[query];other=rm[query];boxes=base_boxes[0,query].unsqueeze(0)
                        own_iou=float(box_iou(box_convert(boxes,'cxcywh','xyxy'),box_convert(target_cpu[0]['boxes'][own].unsqueeze(0),'cxcywh','xyxy'))[0,0])
                        other_iou=float(box_iou(box_convert(boxes,'cxcywh','xyxy'),box_convert(target_cpu[0]['boxes'][other].unsqueeze(0),'cxcywh','xyxy'))[0,0])
                        dot=float(torch.dot(gu[query],go[query]));denom=float(union_norm[query]*own_norm[query]);cosine=dot/denom if denom>0 else None
                        pred_class=int(data['model.pred_logits'][0,query].argmax());class_correct=pred_class==rows[own]['category_id']
                        protected=own_iou>=.75 and class_correct;count_protected+=protected
                        conflicts.append(dict(seed=args.seed,endpoint=endpoint,image_id=image_id,query=query,own_GT_annotation=rows[own]['id'],union_GT_annotation=rows[other]['id'],own_class=rows[own]['category_id'],union_class=rows[other]['category_id'],predicted_class=pred_class,own_IoU=own_iou,union_IoU=other_iou,own_input_short_px=float(target_cpu[0]['boxes'][own,2:].min()*640),own_original_area=float(rows[own]['area']),own_COCO_small=float(rows[own]['area'])<=1024,own_strict_correct=protected,union_vs_own_bbox_gradient_dot=dot,union_vs_own_bbox_gradient_cosine=cosine,first_order_union_descent_increases_own_bbox_loss=dot<0,union_gradient_L2=float(union_norm[query]),own_gradient_L2=float(own_norm[query])))
                    sample_summaries.append(dict(endpoint=endpoint,image_id=image_id,GT_rows=len(rows),final_own_pairs=len(cm),final_union_pairs=len(rm),final_GT_conflicts=len(qs),final_strict_correct_conflicts=count_protected,unchanged_union_rows_gradient_equal=True,source_identity_max_box_error=error))
                print(json.dumps(dict(stage='CAPTURED',seed=args.seed,endpoint=endpoint,image_id=image_id,final_conflicts=sample_summaries[-1]['final_GT_conflicts'])),flush=True)
            assert parameter_sha(model)==before
            model.load_state_dict(weights,strict=True)
            del payload,weights
        recorder.close();recorder=None
        assert source_sha=={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        summaries={}
        for endpoint in (12,132):
            sr=[r for r in sample_summaries if r['endpoint']==endpoint];cr=[r for r in conflicts if r['endpoint']==endpoint]
            summaries[str(endpoint)]=dict(images=len(sr),GT_rows=sum(r['GT_rows'] for r in sr),classification_pairs=sum(r['final_own_pairs'] for r in sr),union_pairs=sum(r['final_union_pairs'] for r in sr),GT_conflicts=len(cr),cross_class_conflicts=sum(r['own_class']!=r['union_class'] for r in cr),COCO_small_conflicts=sum(r['own_COCO_small'] for r in cr),own_strict_correct_conflicts=sum(r['own_strict_correct'] for r in cr),own_strict_correct_small_conflicts=sum(r['own_strict_correct'] and r['own_COCO_small'] for r in cr),gradient_opposed_conflicts=sum(r['first_order_union_descent_increases_own_bbox_loss'] for r in cr),gradient_opposed_strict_correct_small_conflicts=sum(r['first_order_union_descent_increases_own_bbox_loss'] and r['own_strict_correct'] and r['own_COCO_small'] for r in cr))
        save(out/'panel_identity.json',dict(selection='32 evenly spaced positions in sorted548 validation IDs, outcome-independent, fixed before observation',image_ids=panel,images=panel_identity,annotation_sha256=sha(ann_file)))
        save(out/'sample_summaries.json',sample_summaries);save(out/'stage_summaries.json',stage_summaries)
        save(out/'conflicts.json',conflicts)
        with (out/'conflicts.csv').open('x',newline='',encoding='utf-8-sig') as f:
            fields=list(conflicts[0]) if conflicts else ['seed','endpoint','image_id','query'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(conflicts)
        result=dict(status='COMPLETE_FROZEN_ASSIGNMENT_DIAGNOSTIC_NOT_TRAINING',seed=args.seed,summaries=summaries,
            model_forward_calls=forward_calls,model_backward_calls=0,leaf_box_gradient_evaluations=leaf_gradient_calls,training_epochs_added=0,optimizer_steps=0,
            first_image_GPU_observer_output_buffer_RNG_equivalence_each_endpoint=True,model_parameters_unchanged=True,native_source_unchanged=True,
            source_identity_verified=True,historical_learning_recorded=False,AP_evaluated=False,scientific_root_cause_established=False,
            checkpoint_sha256=CHECKPOINTS[args.seed],code_sha256={p.name:sha(p) for p in Path(__file__).resolve().parent.glob('*.py')},source_python_sha256=source_sha,elapsed_seconds=time.monotonic()-start,
            limits='Frozen EMA train-mode direct-validation observations with BN reset to checkpoint before every image. This is neither native eval AP nor historical augmentation/optimizer behavior. Native union versus final-own anchored union keeps query identities, order, count, denominator and loss weights fixed; unchanged query gradients exact. Opposed leaf-box gradients quantify a local directional conflict, not a network parameter update or counterfactual AP improvement. Validation panel previously explored; no fresh holdout or statistical-significance claim. No new detector training configuration.')
        save(out/'completion.json',result);save(out/'manifest.json',dict(files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
        print(json.dumps({k:v for k,v in result.items() if k not in ('code_sha256','source_python_sha256','limits')}),flush=True)
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',error=traceback.format_exc(),training_epochs_added=0,optimizer_steps=0));raise
    finally:
        if recorder is not None:recorder.close()


if __name__=='__main__':main()
