"""CPU-only engineering-cache integration, never optimizes on real images.

Usage: python integration_cpu_test.py --cache .../engineering/val --output .../audit.json
Real cached images: forward/backward only, NO optimizer construction or step.
The isolated AdamW smoke test creates only procedural synthetic pixels/boxes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

# Set before torch import, including DataLoader spawn imports. Never use CUDA.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from data import ROIDataset, single_image_collate, worker_init
from refiner import (ROIRefiner, prepare_rois, fixed_bindings, refinement_loss,
                     utility_targets, gain_loss, apply_refinement)


def require(condition, message):
    if not bool(condition): raise AssertionError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def state_sha(model, prefixes=None):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if prefixes is not None and not name.startswith(tuple(prefixes)): continue
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def require_cpu_model(model):
    require(all(p.device.type=='cpu' for p in model.parameters()),'model left CPU')
    require(not torch.cuda.is_initialized(),'CUDA was initialized')


def exact_native_degraded(dataset, image_id, actual_rois):
    """Independently rebuild the native uint8 PIL resize, then compare ROIs."""
    record=dataset.records[image_id]
    with Image.open(dataset.image_paths[image_id]) as source:
        rgb=source.convert('RGB')
        native_pil=rgb.resize((640,640),resample=Image.Resampling.BILINEAR)
        raw=torch.from_numpy(np.array(rgb,copy=True,dtype=np.uint8)).permute(2,0,1).float()/255.
        native=torch.from_numpy(np.array(native_pil,copy=True,dtype=np.uint8)).permute(2,0,1).float()/255.
    with np.load(dataset.npz_paths[image_id],allow_pickle=False) as cache:
        boxes=torch.from_numpy(cache['boxes'].copy())
        qids=torch.from_numpy(cache['query_ids'].copy())
    expected=prepare_rois(raw,boxes,qids,arm='degraded',input_hw=(640,640),degraded_image=native)
    require(torch.equal(actual_rois.crops,expected.crops),'dataset degraded ROI differs from native PIL uint8 source')
    require(torch.equal(actual_rois.roi_boxes,expected.roi_boxes),'native PIL test physical range differs')
    return {'exact_degraded_ROI_equal':True,'native_input_sha256':hashlib.sha256(native.numpy().tobytes()).hexdigest(),
            'original_dimensions':[record['width'],record['height']]}


def inspect_real_image(model,batch):
    """No optimizer in this function. Parameters must be byte-identical after."""
    before=state_sha(model)
    model.set_stage('refiner');model.zero_grad(set_to_none=True)
    # These deliberate leaf gradients catch accidental feature/GT backprop.
    features=batch['features'].clone().requires_grad_(True)
    gt_boxes=batch['gt_boxes'].clone().requires_grad_(True)
    result=model(batch['rois'],batch['boxes0'],batch['labels'],features)
    require(torch.equal(result['boxes1'],batch['boxes0']),'zero initialization changes a real baseline box')
    require(torch.equal(result['eligible'],batch['rois'].eligible),'forward eligibility changed')
    for policy,threshold in [('all',0.),('gate',0.),('gate',1.),('none',0.)]:
        boxes,accepted=apply_refinement(batch['boxes0'],result['boxes1'],result['gain'],result['eligible'],
                                        policy=policy,threshold=threshold)
        require(torch.equal(boxes,batch['boxes0']),'zero-init output invariant failed')
        require(not bool(accepted.any()),'zero/no-op proposal counted as a changed output')
    losses=refinement_loss(batch['boxes0'],result['boxes1'],gt_boxes,batch['bindings'],result['eligible'])
    require(torch.isfinite(losses['loss']),'nonfinite real-data geometry loss')
    losses['loss'].backward()
    require(features.grad is None,'gradient leaked into cached detector features')
    require(gt_boxes.grad is None,'gradient leaked into GT tensors')
    require(any(p.grad is not None for p in model.encoder.parameters()) or
            any(p.grad is not None for p in model.geometry.parameters()),'refiner backward has no parameter path')
    require(all(p.grad is None for p in model.gate.parameters()),'geometry loss trained gate')
    real_refine_loss=float(losses['loss'].detach())

    model.zero_grad(set_to_none=True);model.set_stage('gate')
    frozen_before=state_sha(model,('encoder.','geometry.'))
    result=model(batch['rois'],batch['boxes0'],batch['labels'],batch['features'])
    targets=utility_targets(batch['boxes0'],result['boxes1'],batch['labels'],batch['gt_boxes'],
                            batch['gt_labels'],batch['bindings'],result['eligible'],scores=batch['scores'])
    require(all(not value.requires_grad for value in targets.values()),'gain labels are not detached')
    require(torch.isfinite(targets['target_y']).all(),'nonfinite rejection/gain labels')
    gate_loss=gain_loss(result['gain'],targets,result['eligible'],balance_groups=True)
    require(torch.isfinite(gate_loss),'nonfinite real-data gate loss')
    gate_loss.backward()
    require(all(p.grad is None for p in model.encoder.parameters()),'gate gradients reached encoder')
    require(all(p.grad is None for p in model.geometry.parameters()),'gate gradients reached geometry')
    require(any(p.grad is not None for p in model.gate.parameters()),'gate backward has no parameter path')
    require(state_sha(model,('encoder.','geometry.'))==frozen_before,'gate backward mutated frozen tensors')
    require(state_sha(model)==before,'real-data backward mutated model parameters/buffers')
    require_cpu_model(model)
    y=targets['target_y'][result['eligible']]
    row={'image_id':int(batch['image_id']),'outputs':len(batch['boxes0']),
         'eligible_outputs':int(result['eligible'].sum()),'unique_ROIs':len(batch['rois'].roi_query_ids),
         'GT_count':len(batch['gt_boxes']),'old_success_bindings':int(batch['bindings'].old_success.sum()),
         'geometry_loss':real_refine_loss,'gain_loss':float(gate_loss.detach()),
         'gate_target_groups':{'positive':int((y>0).sum()),'zero':int((y==0).sum()),'negative':int((y<0).sum())},
         'parameters_unchanged':True,'real_optimizer_steps':0}
    model.zero_grad(set_to_none=True)
    return row,targets


def synthetic_adamw_test():
    """Every pixel, feature, class, and box in this function is procedural."""
    yy,xx=torch.meshgrid(torch.arange(96),torch.arange(128),indexing='ij')
    image=torch.stack(((xx%7)/6.,(yy%11)/10.,((xx+yy)%13)/12.)).float()
    boxes=torch.tensor([[40.,32.,50.,42.]])
    gt=torch.tensor([[41.,32.,51.,42.]])
    labels=torch.tensor([0]);scores=torch.tensor([.8]);qids=torch.tensor([3])
    features=torch.linspace(-1.,1.,256).reshape(1,256)
    rois=prepare_rois(image,boxes,qids,input_hw=(96,128),arm='raw')
    binding=fixed_bindings(boxes,labels,scores,gt,labels)
    model=ROIRefiner(query_dim=256,stats_dim=0,initialization_seed=123).cpu()
    model.set_stage('refiner')
    before_geometry=state_sha(model,('encoder.','geometry.'))
    before_gate=state_sha(model,('gate.',))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    out=model(rois,boxes,labels,features)
    loss=refinement_loss(boxes,out['boxes1'],gt,binding,rois.eligible)['loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()  # SYNTHETIC DATA ONLY, never uses inspect_real_image inputs.
    require(state_sha(model,('encoder.','geometry.'))!=before_geometry,'synthetic AdamW did not update geometry')
    require(state_sha(model,('gate.',))==before_gate,'synthetic refiner step changed frozen gate')
    model.zero_grad(set_to_none=True);model.set_stage('gate')
    geometry_frozen=state_sha(model,('encoder.','geometry.'))
    before_gate=state_sha(model,('gate.',))
    gate_optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    out=model(rois,boxes,labels,features)
    # This handcrafted target is synthetic optimizer plumbing, not measured benefit.
    gl=gain_loss(out['gain'],{'target_y':torch.tensor([.3])},rois.eligible,balance_groups=True)
    gl.backward();gate_optimizer.step()  # SYNTHETIC DATA ONLY.
    require(state_sha(model,('gate.',))!=before_gate,'synthetic AdamW did not update gate')
    require(state_sha(model,('encoder.','geometry.'))==geometry_frozen,'synthetic gate step changed frozen refiner')
    require_cpu_model(model)
    return {'synthetic_optimizer_steps':2,'geometry_update':True,'gate_update':True,
            'gate_refiner_isolation':True,'synthetic_only':True}


def real_utility_stress(raw,degraded):
    """Fixed geometric probe of real cached outputs; no model or optimizer."""
    targets_by_arm={};timings={};changed_counts={}
    for arm,batch in (('raw',raw),('degraded',degraded)):
        proposal=batch['boxes0'].clone()
        eligible=batch['rois'].eligible
        shift=.03*(batch['boxes0'][:,2]-batch['boxes0'][:,0])
        proposal[eligible,0]+=shift[eligible]
        proposal[eligible,2]+=shift[eligible]
        start=time.perf_counter()
        targets=utility_targets(batch['boxes0'],proposal,batch['labels'],batch['gt_boxes'],
                                batch['gt_labels'],batch['bindings'],eligible,scores=batch['scores'])
        timings[arm]=time.perf_counter()-start
        n=len(proposal)
        for key in ('measured_u','target_y','identity_valid','local_fixed_gt_delta'):
            require(targets[key].shape==(n,),'stress utility vector shape: '+key)
        for key in ('new_count_by_threshold','lost_count_by_threshold'):
            value=targets[key]
            require(value.shape==(n,10),'stress threshold count shape: '+key)
            require(torch.isfinite(value).all() and (value>=0).all(),'stress invalid event count: '+key)
            require(not value.is_floating_point() or torch.equal(value,value.round()),
                    'stress noninteger event count: '+key)
        require(targets['identity_valid'].dtype==torch.bool,'stress identity mask not boolean')
        require(torch.isfinite(targets['measured_u']).all(),'stress measured_u not finite')
        require(torch.isfinite(targets['target_y']).all(),'stress target_y not finite')
        require(all(not value.requires_grad for value in targets.values()),'stress labels not detached')
        expected=(targets['new_count_by_threshold']-targets['lost_count_by_threshold']).float().mean(-1)
        torch.testing.assert_close(targets['measured_u'],expected,rtol=0,atol=1e-7,
                                   msg='stress u differs from mean(new-lost), lambda=1')
        valid=eligible & targets['identity_valid']
        require(torch.equal(targets['target_y'][~valid],torch.full_like(targets['target_y'][~valid],-1.)),
                'stress invalid target is not -1')
        torch.testing.assert_close(targets['target_y'][valid],targets['measured_u'][valid],rtol=0,atol=1e-7,
                                   msg='stress valid target differs from measured utility')
        unchanged=(proposal==batch['boxes0']).all(-1)
        require((targets['measured_u'][unchanged]==0).all(),'unchanged stress item has nonzero utility')
        require((targets['new_count_by_threshold'][unchanged]==0).all() and
                (targets['lost_count_by_threshold'][unchanged]==0).all(),'unchanged stress item has events')
        targets_by_arm[arm]=targets;changed_counts[arm]=int((~unchanged).sum())
    require(set(targets_by_arm['raw'])==set(targets_by_arm['degraded']),'stress arm fields differ')
    for key,value in targets_by_arm['raw'].items():
        other=targets_by_arm['degraded'][key]
        if value.is_floating_point():
            torch.testing.assert_close(value,other,rtol=0,atol=0,equal_nan=True,
                                       msg='stress raw/degraded utility mismatch: '+key)
        else:
            require(torch.equal(value,other),'stress raw/degraded utility mismatch: '+key)
    t=targets_by_arm['raw'];u=t['measured_u'];y=t['target_y']
    return {'scope':'Engineering label stress only; fixed proposals are not model results or an AP claim',
            'perturbation_xyxy':'eligible only: +[0.03*w,0,0.03*w,0], no clipping or rematching of bindings',
            'elapsed_seconds':timings,'elapsed_seconds_total':sum(timings.values()),
            'changed_output_items':changed_counts,'same_full_utility_labels':True,
            'formula_mean_new_minus_lost_checked':True,'invalid_target_minus1_checked':True,
            'measured_utility_counts':{'nonzero':int((u!=0).sum()),'gain':int((u>0).sum()),'loss':int((u<0).sum())},
            'event_counts':{'items_with_new_hit':int((t['new_count_by_threshold']>0).any(-1).sum()),
                            'items_with_lost_hit':int((t['lost_count_by_threshold']>0).any(-1).sum())},
            'target_counts_including_invalid_minus1':{'positive':int((y>0).sum()),'zero':int((y==0).sum()),
                                                      'negative':int((y<0).sum())},
            'identity_valid_eligible':int((t['identity_valid'] & raw['rois'].eligible).sum()),
            'real_optimizer_steps':0}


def fixed_pool_equivalence():
    """CPU numerical check for the known 8x8 -> 2x2 nonoverlapping mean."""
    values=torch.linspace(-1.,1.,2*32*8*8).reshape(2,32,8,8)
    a=values.clone().requires_grad_(True);b=values.clone().requires_grad_(True)
    adaptive=torch.nn.AdaptiveAvgPool2d((2,2))(a)
    fixed=torch.nn.AvgPool2d(4)(b)
    torch.testing.assert_close(adaptive,fixed,atol=1e-6,rtol=1e-6)
    upstream=torch.linspace(-.3,.9,adaptive.numel()).reshape_as(adaptive)
    (adaptive*upstream).sum().backward();(fixed*upstream).sum().backward()
    torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-6)
    return {'forward_close':True,'backward_close':True,'input_shape':[2,32,8,8],
            'max_forward_abs_error':float((adaptive-fixed).abs().max()),
            'max_backward_abs_error':float((a.grad-b.grad).abs().max())}


def loader_smoke(dataset,expected):
    """Actual two-worker spawn/pickle check, not just num_workers=0 emulation."""
    loader=DataLoader(dataset,batch_size=1,num_workers=2,shuffle=False,
                      multiprocessing_context='spawn',persistent_workers=False,
                      worker_init_fn=worker_init,collate_fn=single_image_collate)
    seen=[]
    for batch in loader:
        image_id=int(batch['image_id']);seen.append(image_id)
        require(batch['boxes0'].device.type=='cpu','DataLoader worker returned non-CPU tensor')
        require(torch.equal(batch['boxes0'],expected[image_id]['boxes0']),'worker changed baseline output')
        require(torch.equal(batch['rois'].crops,expected[image_id]['crops']),'worker ROI/pickle mismatch')
        require(torch.equal(batch['bindings'].det_to_gt,expected[image_id]['det_to_gt']),'worker binding mismatch')
    require(seen==dataset.image_ids,'workers changed image order/coverage')
    return {'workers':2,'context':'spawn','images':len(seen),'exact_outputs':True}


def run(args,summary):
    require(not torch.cuda.is_initialized(),'CUDA initialized before CPU engineering test')
    torch.set_num_threads(2)
    torch.manual_seed(20261009)
    torch.use_deterministic_algorithms(True)
    cache=Path(args.cache).resolve()
    manifest_path=cache/'manifest.json' if cache.is_dir() else cache
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    require(manifest.get('status')=='ENGINEERING_SUBSET_COMPLETE','test requires engineering subset cache')
    require(manifest.get('engineering_subset') is True,'engineering-subset flag missing')
    require(manifest.get('split')=='val','integration real input must be the engineering VAL cache')
    ids=[int(x['image_id']) for x in manifest['images']]
    require(len(ids)>0,'empty engineering cache')
    if args.max_images is not None: ids=ids[:args.max_images]
    raw=ROIDataset(manifest_path,ids,'raw')
    degraded=ROIDataset(manifest_path,ids,'degraded')
    model=ROIRefiner(query_dim=256,stats_dim=0,initialization_seed=20261009).cpu()
    state_initial=state_sha(model)
    summary.update(cache=str(manifest_path),cache_manifest_sha256=sha(manifest_path),
                   images_tested=len(ids),torch=torch.__version__,numpy=np.__version__,
                   CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   cpu_threads=torch.get_num_threads(),
                   real_optimizer_steps=0,initial_model_state_sha256=state_initial,
                   raw_verification=raw.verification,degraded_verification=degraded.verification,
                   image_checks=[])
    expected_raw,expected_degraded={},{}
    for i in range(len(ids)):
        rb,db=raw[i],degraded[i]
        require(rb['image_id']==db['image_id'],'arms have different images')
        for name in ('boxes0','labels','scores','query_ids','features','output_index','gt_boxes','gt_labels'):
            require(torch.equal(rb[name],db[name]),f'raw/degraded changed {name}')
        for name in ('eligible','det_to_roi','roi_query_ids','roi_boxes','box_roi'):
            require(torch.equal(getattr(rb['rois'],name),getattr(db['rois'],name)),f'arms differ in {name}')
        require(torch.equal(rb['rois'].crops[:,3],db['rois'].crops[:,3]),'valid-pixel masks differ')
        require(torch.equal(rb['bindings'].det_to_gt,db['bindings'].det_to_gt),'arm fixed bindings differ')
        pil_check=exact_native_degraded(degraded,db['image_id'],db['rois'])
        rcheck,rtargets=inspect_real_image(model,rb)
        dcheck,dtargets=inspect_real_image(model,db)
        required_targets={'measured_u','new_count_by_threshold','lost_count_by_threshold',
                          'target_y','identity_valid','local_fixed_gt_delta'}
        require(required_targets <= set(rtargets),'full single-item utility labels missing')
        require(set(rtargets)==set(dtargets),'arm utility fields differ')
        # Same baseline boxes and zero-initialized proposals must have identical
        # full-rematching labels, regardless of the image source for ROI pixels.
        # Learned nonzero raw/degraded proposals are not assumed to be equal.
        for name in rtargets:
            if rtargets[name].is_floating_point():
                torch.testing.assert_close(rtargets[name],dtargets[name],rtol=0,atol=0,equal_nan=True,
                                           msg='raw/degraded utility mismatch: '+name)
            else:
                require(torch.equal(rtargets[name],dtargets[name]),'raw/degraded utility mismatch: '+name)
        stress=real_utility_stress(rb,db)
        summary['image_checks'].append({'image_id':rb['image_id'],'raw':rcheck,'degraded':dcheck,
                                        'same_eligibility_and_query_map':True,
                                        'same_full_utility_labels_for_identical_zero_init_proposals':True,
                                        'utility_fields_checked':sorted(rtargets),'full_utility_stress':stress,**pil_check})
        for batch,container in ((rb,expected_raw),(db,expected_degraded)):
            container[batch['image_id']]={'boxes0':batch['boxes0'].clone(),
                                         'crops':batch['rois'].crops.clone(),
                                         'det_to_gt':batch['bindings'].det_to_gt.clone()}
    require(state_sha(model)==state_initial,'real-data tests changed model state')
    summary['worker_smoke']={'raw':loader_smoke(raw,expected_raw),
                             'degraded':loader_smoke(degraded,expected_degraded)}
    summary['synthetic_adamw']=synthetic_adamw_test()
    summary['pool_equivalence']=fixed_pool_equivalence()
    require(not torch.cuda.is_initialized(),'CPU test initialized CUDA')
    summary['final_real_model_state_sha256']=state_sha(model)
    summary['all_real_model_parameters_unchanged']=state_sha(model)==state_initial
    summary['cuda_initialized']=False
    summary['status']='PASS'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-images',type=int,default=None,
                        help='optional explicit engineering subset of the already-small cache')
    args=parser.parse_args()
    if args.max_images is not None and args.max_images<1: parser.error('--max-images must be positive')
    output=Path(args.output).resolve()
    cache_root=Path(args.cache).resolve()
    if cache_root.is_file(): cache_root=cache_root.parent
    if output==cache_root or cache_root in output.parents:
        parser.error('--output must be outside immutable native cache')
    summary={'status':'RUNNING','scope':'CPU engineering only; real-data backward without any optimizer step',
             'code_sha256':{name:sha(Path(__file__).with_name(name)) for name in
                            ('integration_cpu_test.py','refiner.py','data.py')},
             'real_optimizer_steps':0}
    start=time.perf_counter()
    try:
        run(args,summary)
    except BaseException as exc:
        summary.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        summary['elapsed_seconds']=time.perf_counter()-start
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({'status':summary['status'],'output':str(output),
                          'real_optimizer_steps':0,'elapsed_seconds':summary['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
