"""CPU data-only probe of native, identity and matched source-crop arms."""
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=32)
    a=p.parse_args();assert a.samples>=4 and a.samples%4==0
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1';os.nice(10)
    old=Path(a.prefix)/'DEIM/experiments/fine_residual_screen24_20261009_v2'
    new=Path(a.prefix)/'DEIM/experiments/mosaic_scale_retention12_20261010_v1'
    output=Path(a.output).resolve();assert output.parent==new/'diagnostics' and not output.exists()
    sys.dont_write_bytecode=True;sys.path.insert(0,str(old/'code'))
    from experiment import verify_bundle
    from runtime import activate_training_snapshot
    receipt,plan,paths=verify_bundle(old/'bundles/baseline_seed1')
    activate_training_snapshot(paths['native_root'])
    from evaluator_binding import activate
    activate(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    import numpy as np
    import torch
    from PIL import Image
    from engine.core import YAMLConfig
    from engine.data._misc import convert_to_tv_tensor
    sys.path.insert(0,str(new/'code'))
    from mosaic_crop import CONTRACT,install,MosaicSourceCrop,crop_at
    assert not torch.cuda.is_available()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_default_device('cpu')
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def tensor(v):return v.as_subclass(torch.Tensor)
    def digest(v):return hashlib.sha256(tensor(v).contiguous().numpy().tobytes()).hexdigest()
    def reset(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    def rng():return dict(python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        numpy=hashlib.sha256(np.random.get_state()[1].tobytes()+repr(np.random.get_state()[2:]).encode()).hexdigest(),
        torch=digest(torch.get_rng_state()))
    # Image/box correspondence, field alignment, metadata N=1 and empty targets.
    ramp=np.zeros((640,640,3),dtype=np.uint8)
    ramp[:,:,0]=np.arange(640,dtype=np.uint16)[None,:]%256
    ramp[:,:,1]=np.arange(640,dtype=np.uint16)[:,None]%256
    image=Image.fromarray(ramp)
    target=dict(boxes=convert_to_tv_tensor(torch.tensor([[310.,310.,314.,314.]]),key='boxes',box_format='xyxy',spatial_size=(640,640)),
        labels=torch.tensor([0]),area=torch.tensor([16.]),iscrowd=torch.tensor([0]),image_id=torch.tensor([999]),idx=torch.tensor([123]),orig_size=torch.tensor([640,640]))
    before={k:digest(v) for k,v in target.items()};ci,ct,keep=crop_at(image,target,300,300,320,320)
    assert np.array_equal(np.asarray(ci),ramp[300:620,300:620])
    assert tensor(ct['boxes']).tolist()==[[10.,10.,14.,14.]] and ct['labels'].tolist()==[0]
    assert ct['image_id'].tolist()==[999] and ct['idx'].tolist()==[123] and ct['area'].tolist()==[16.]
    assert before=={k:digest(v) for k,v in target.items()}
    empty=deepcopy(target)
    for k in ('boxes','labels','area','iscrowd'):empty[k]=empty[k][:0]
    ei,et,keep=crop_at(image,empty,0,0,320,320);assert len(et['boxes'])==len(et['labels'])==0
    fake=type('Dataset',(),{'epoch':4})()
    observed=[];hook=MosaicSourceCrop(fake,1,'target_crop',observed.append)
    state=rng();hi,ht=hook(None,(image,target));assert rng()==state and len(ht['boxes'])==1
    assert observed[0]['applied'] and observed[0]['anchor_index']==0
    changed=deepcopy(target);changed['labels']=torch.tensor([54321]);hook.occurrence=0
    hi2,ht2=hook(None,(image,changed));assert np.array_equal(np.asarray(hi),np.asarray(hi2))
    assert torch.equal(tensor(ht['boxes']),tensor(ht2['boxes'])) and ht2['labels'].tolist()==[54321]
    # Actual native dataset/collate, model/optimizer are never constructed.
    cfg=YAMLConfig(str(old/'bundles/baseline_seed1/resolved.yml'))
    loader=cfg.train_dataloader;dataset=loader.dataset;collate=loader.collate_fn
    assert collate.copyblend_prob==0 and not dataset.cache_flag
    positions=[round(i*(len(dataset.ids)-1)/(a.samples-1)) for i in range(a.samples)]
    assert len(set(positions))==len(positions)
    cat={int(k)+1:int(v['category_id']) for k,v in dataset.coco.anns.items()}
    original_load=dataset.load_item;raw={};tagging=False
    def traced_load(idx):
        img,t=original_load(idx);image_id=int(dataset.ids[idx]);w,h=img.size
        anns=[v for v in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[image_id])) if not v.get('iscrowd',0)]
        boxes=torch.tensor([v['bbox'] for v in anns],dtype=torch.float32).reshape(-1,4)
        boxes[:,2:]+=boxes[:,:2];boxes[:,0::2].clamp_(0,w);boxes[:,1::2].clamp_(0,h)
        keep=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
        valid=[v for v,k in zip(anns,keep.tolist()) if k]
        assert torch.equal(boxes[keep],tensor(t['boxes']))
        assert t['labels'].tolist()==[v['category_id'] for v in valid]
        ids=[int(v['id'])+1 for v in valid]
        shorts=((boxes[keep,2:]-boxes[keep,:2])*torch.tensor([640/w,640/h])).min(-1).values.tolist()
        raw[image_id]=dict(zip(ids,shorts))
        if tagging:t['labels']=torch.tensor(ids,dtype=torch.int64)
        return img,t
    dataset.load_item=traced_load
    mosaic=[op for op in dataset._transforms.transforms if type(op).__name__=='Mosaic'][0]
    passive_count=[0]
    passive=mosaic.resize.register_forward_pre_hook(lambda *args:passive_count.__setitem__(0,passive_count[0]+1))
    mode_results={};prefix_checks=[];started=time.monotonic()
    try:
        # Epoch0 skips Mosaic, so installing target crop must keep native bytes.
        loader.set_epoch(0);reset(0);tagging=False
        prefix_items=[dataset[i] for i in positions[:4]]
        base_images,base_targets=collate(prefix_items)
        hook,handle=install(dataset,1,'target_crop')
        reset(0);prefix_items=[dataset[i] for i in positions[:4]]
        trial_images,trial_targets=collate(prefix_items)
        assert digest(base_images)==digest(trial_images)
        assert all(digest(x['boxes'])==digest(y['boxes']) and torch.equal(x['labels'],y['labels'])
            for x,y in zip(base_targets,trial_targets))
        assert hook.occurrence==0;handle.remove()
        prefix_checks.append('epoch0_images_boxes_classes_exact_and_crop_not_called')
        loader.set_epoch(4)
        for seed in (0,1):
            for arm in ('native','identity','target_crop','random_crop'):
                crop_records=[];hook=handle=None
                if arm!='native':hook,handle=install(dataset,seed,arm,crop_records.append)
                normal=[];mode_samples=[];batch_rows=[]
                try:
                    for encoded in (False,True):
                        tagging=encoded;reset(seed);passive_count[0]=0
                        if hook:hook.occurrence=0
                        crop_records.clear()
                        for b in range(a.samples//4):
                            items=[];sample_rows=[]
                            for j,pos in enumerate(positions[b*4:b*4+4]):
                                calls_before=passive_count[0];img,t=dataset[pos]
                                sample_rows.append(dict(position=pos,image_id=int(dataset.ids[pos]),
                                    mosaic=passive_count[0]>calls_before,image=digest(img),boxes=digest(t['boxes']),
                                    ids=t['labels'].tolist(),normalized_wh=tensor(t['boxes'])[:,2:].tolist(),
                                    mosaic_source_image_ids=t.get('mosaic_source_image_ids',torch.empty(0,dtype=torch.int64)).tolist()))
                                items.append((img,t))
                            images,targets=collate(items)
                            batch=dict(image=digest(images),boxes=[digest(t['boxes']) for t in targets],
                                labels=[t['labels'].tolist() for t in targets],shape=list(images.shape),rng_after=rng())
                            assert bool(torch.isfinite(images).all())
                            assert all(bool(torch.isfinite(t['boxes']).all()) and bool((tensor(t['boxes'])[:,2:]>0).all()) for t in targets)
                            if not encoded:normal.append(dict(samples=sample_rows,batch=batch))
                            else:
                                old_batch=normal[b]['batch']
                                assert all(batch[k]==old_batch[k] for k in ('image','boxes','shape','rng_after'))
                                assert [[cat[x] for x in ids] for ids in batch['labels']]==old_batch['labels']
                                for s,prior in zip(sample_rows,normal[b]['samples']):
                                    assert all(s[k]==prior[k] for k in ('position','image_id','mosaic','image','boxes','normalized_wh','mosaic_source_image_ids'))
                                    assert [cat[x] for x in s['ids']]==prior['ids']
                                    occ={}
                                    for gid,wh in zip(s['ids'],s['normalized_wh']):occ.setdefault(gid,[]).append(min(wh)*images.shape[-1])
                                    primary=[dict(annotation_id=gid-1,raw_shortside640=short,
                                        duplicate_occurrences=len(occ.get(gid,[])),survived=gid in occ,
                                        actual_input_shortside=occ[gid][0] if len(occ.get(gid,[]))==1 else None)
                                        for gid,short in raw[s['image_id']].items()]
                                    mode_samples.append(dict(position=s['position'],image_id=s['image_id'],mosaic=s['mosaic'],
                                        mosaic_source_image_ids=s['mosaic_source_image_ids'],primary=primary,
                                        primary_final_GT=sum(r['survived'] for r in primary),target_exposures_before_mixup=len(s['ids'])))
                                batch_rows.append(dict(batch=b,shape=batch['shape'],target_exposures_after_mixup=sum(len(t['labels']) for t in targets),
                                    rng_after=batch['rng_after'],ordinary_tagged_outputs_exact=True))
                        if not encoded:normal_rng=rng()
                    assert rng()==normal_rng
                    primary=[r for s in mode_samples for r in s['primary']]
                    retained=[r for r in primary if r['duplicate_occurrences']==1]
                    summary=dict(raw_primary_GT=len(primary),retained_unique_primary_GT=len(retained),
                        missing_primary_GT=sum(not r['survived'] for r in primary),
                        duplicate_excluded=sum(r['duplicate_occurrences']>1 for r in primary),
                        retained_shortside_lt4=sum(r['actual_input_shortside']<4 for r in retained),
                        retained_shortside_ge4=sum(r['actual_input_shortside']>=4 for r in retained),
                        median_shortside=float(np.median([r['actual_input_shortside'] for r in retained])) if retained else None,
                        target_exposures_before_mixup=sum(s['target_exposures_before_mixup'] for s in mode_samples),
                        target_exposures_after_mixup=sum(b['target_exposures_after_mixup'] for b in batch_rows),
                        crop_calls=len(crop_records),crop_applied_calls=sum(r['applied'] for r in crop_records),
                        crop_source_GT=sum(r['source_GT'] for r in crop_records),crop_kept_GT=sum(r['kept_GT'] for r in crop_records))
                    mode_results[f'{seed}:{arm}']=dict(summary=summary,samples=mode_samples,batches=batch_rows,
                        final_rng=rng(),crop_records=deepcopy(crop_records),ordinary_sample_hashes=normal)
                    print(json.dumps(dict(progress='MODE_COMPLETE',seed=seed,arm=arm,summary=summary)),flush=True)
                finally:
                    if handle:handle.remove()
        comparisons={}
        for seed in (0,1):
            native=mode_results[f'{seed}:native'];identity=mode_results[f'{seed}:identity']
            assert all(native[k]==identity[k] for k in ('samples','batches','final_rng','ordinary_sample_hashes'))
            target=mode_results[f'{seed}:target_crop'];control=mode_results[f'{seed}:random_crop']
            assert all(mode_results[f'{seed}:{arm}']['final_rng']==native['final_rng'] for arm in ('identity','target_crop','random_crop'))
            assert len(target['crop_records'])==len(control['crop_records'])
            for x,y in zip(target['crop_records'],control['crop_records']):
                assert all(x[k]==y[k] for k in ('image_id','occurrence','applied','risky_fraction','source_GT'))
                if x['applied']:assert x['crop'][2:]==y['crop'][2:] and x['local_rng_key_sha256']==y['local_rng_key_sha256']
            for arm in ('target_crop','random_crop'):
                trial=mode_results[f'{seed}:{arm}']
                old={r['annotation_id']:r for s in native['samples'] for r in s['primary']}
                new_gt={r['annotation_id']:r for s in trial['samples'] for r in s['primary']}
                assert set(old)==set(new_gt)
                common=[gid for gid in old if old[gid]['duplicate_occurrences']==new_gt[gid]['duplicate_occurrences']==1]
                comparisons[f'{seed}:{arm}']=dict(same_retained_unique_GT=len(common),
                    common_native_lt4=sum(old[k]['actual_input_shortside']<4 for k in common),
                    common_trial_lt4=sum(new_gt[k]['actual_input_shortside']<4 for k in common),
                    native_retained_trial_missing=sum(r['survived'] and not new_gt[k]['survived'] for k,r in old.items()),
                    native_missing_trial_retained=sum(not r['survived'] and new_gt[k]['survived'] for k,r in old.items()),
                    common_median_scale_ratio=float(np.median([new_gt[k]['actual_input_shortside']/old[k]['actual_input_shortside'] for k in common])) if common else None)
        evidence=dict(status='COMPLETE_CPU_CROP_GEOMETRY_AND_MATCHED_CONTROL_CHECKS',identity=receipt['identity'],
            contract=CONTRACT,sources={name:sha(new/'code'/name) for name in ('mosaic_crop.py','probe_geometry.py')},
            dataset_positions=positions,random_seeds=[0,1],epoch_zero_based=4,
            synthetic_pixel_box_metadata_and_global_RNG_checks='PASS',prefix_checks=prefix_checks,
            native_identity_exact=True,ordinary_tagged_outputs_exact=True,
            target_random_application_size_keys_matched=True,native_global_rng_exact_all_arms=True,
            comparisons=comparisons,modes=mode_results,optimizer_steps=0,model_forward_calls=0,
            model_weights_loaded=False,cuda_used=False,elapsed_seconds=time.monotonic()-started,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            limitations='Source-crop retains intersecting bboxes and changes target density/context. Box size does not prove visibility or AP. '
                'Gate uses original640 shortside<8; native affine and batch multiscale may still reduce final sizes. '
                'Raw GT IDs only in this data diagnostic; no detector/criterion consumes encoded labels. '
                'Uniform dataset-position sample, independently reset augmentation seeds, not actual epoch4 RNG history; not an innovation or AP claim.')
        with output.open('x',encoding='utf8') as f:json.dump(evidence,f,indent=2,allow_nan=False)
        print(json.dumps(dict(status=evidence['status'],comparisons=comparisons,output_sha256=sha(output))),flush=True)
    finally:
        dataset.load_item=original_load;passive.remove()


if __name__=='__main__':main()
