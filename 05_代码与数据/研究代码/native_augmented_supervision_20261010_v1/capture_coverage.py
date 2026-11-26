"""Frozen native EMA + actual augmented batch supervision; no optimizer/backward.

Tracing annotation IDs is confined to a separate CPU data pass. Ordinary and
traced images, boxes, decoded classes and MixUp weights must be byte-identical.
Only restored native classes are ever consumed by the model and criterion.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
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
from training_evidence import TrainingEvidenceRecorder, sha, scalar

CHECKPOINTS={1:{12:'ba3b6351fa8a4f7c769698c616ed6d20acf6a60ba119cf8edc8611851c4d27d4',132:'30febfa18e6e21dc67b16169bd26875b82340eb3435534719a963904dadf68b5'},2:{12:'6198ee5348630ce97c322c16aa4499a50a2f7af0d08dea1ec3a4cb632d9d5046',132:'5d3b838e296c866e0472e07d4c5b3dd59f7566029d1fca947ab8cd8b3fc1fb3a'}}
ANN_SHA='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def tensor(v):return v.as_subclass(torch.Tensor)
def digest(v):return hashlib.sha256(tensor(v).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def reset(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def parameter_sha(model):
    h=hashlib.sha256()
    for k,p in sorted(model.named_parameters()):h.update(k.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class CoverageRecorder(TrainingEvidenceRecorder):
    """Retain only actual indices, normal heads, DN metadata and loss scalars."""
    def _activation(self,name):return lambda *args:None
    def _gradient(self,*args):return None
    def _model_output(self,module,args,outputs):
        if self.active:
            def collect():
                self._register_stages(outputs,'model')
                for stage,value in self.stage_objects(outputs):
                    if '.dn_' not in stage:
                        for k in ('pred_boxes','pred_logits'):self.put(stage+'.'+k,value[k])
                meta=outputs['dn_meta'];self.dn_groups=int(meta['dn_num_group'])
                self.put('dn_meta.num_groups',torch.tensor(self.dn_groups,dtype=torch.int64))
                for b,idx in enumerate(meta['dn_positive_idx']):self.put(f'dn_meta.batch{b}.positive_queries',idx)
                self.metadata['dn_num_split']=[int(x) for x in meta['dn_num_split']]
            self._observe(collect)
    def stage_objects(self,v,path='model'):
        if isinstance(v,dict):
            if 'pred_boxes' in v and 'pred_logits' in v:yield path,v
            for k,x in v.items():yield from self.stage_objects(x,path+'.'+str(k))
        elif isinstance(v,(list,tuple)):
            for i,x in enumerate(v):yield from self.stage_objects(x,path+'.'+str(i))
    def _loss_call(self,loss,outputs,targets,indices,num_boxes,result):
        stage=self.stage(outputs);prefix='loss_call'+str(sum(e['kind']=='loss_call' for e in self.events));pairs=self._pairs(indices,targets,prefix)
        raw={k:scalar(v) for k,v in result.items()}
        if any(v is None for v in raw.values()):raise ValueError('Nonfinite native raw loss')
        self.events.append(dict(kind='loss_call',stage=stage,loss=loss,prefix=prefix,batches=pairs,num_boxes_normalizer=float(num_boxes),native_unweighted_loss_scalars=raw,is_DN='.dn_' in stage,weight_dict={k:float(v) for k,v in self.criterion.weight_dict.items()}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seed',type=int,choices=[1,2],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();recorder=None
    save(out/'start.json',dict(status='RUNNING_NATIVE_AUGMENTED_SUPERVISION',seed=a.seed,epochs_added=0,optimizer_steps=0,automatic_retry=False))
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        root=Path(a.root);native=root/'DEIM/experiments/fine_residual_screen24_20261009_v2/native';sys.path.insert(0,str(native));sources={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        from engine.core import YAMLConfig
        from engine.backbone.hgnetv2 import HGNetv2
        from engine.deim.hybrid_encoder import HybridEncoder
        from engine.deim.dfine_decoder import DFINETransformer
        from engine.deim.deim import DEIM
        from engine.deim.deim_criterion import DEIMCriterion
        from engine.deim.matcher import HungarianMatcher
        run132=root/f'DEIM/experiments/baseline_official_s_640_20261007_v1/seed{a.seed}/run'
        run12=root/f'DEIM/experiments/fine_residual_screen24_20261009_v2/runs/baseline_seed{a.seed}'
        config=copy.deepcopy(json.loads((run132/'args.json').read_text())['yaml_cfg']);early=json.loads((run12/'args.json').read_text())['yaml_cfg']
        for k in ('HGNetv2','HybridEncoder','DFINETransformer','DEIMCriterion','num_classes','train_dataloader'):assert config[k]==early[k],k
        assert config['num_classes']==10 and config['HGNetv2']['pretrained'] is False
        data_cfg=config['train_dataloader'];ann_path=Path(data_cfg['dataset']['ann_file']);assert sha(ann_path)==ANN_SHA
        assert data_cfg['dataset']['transforms']['policy']['epoch']==[4,64,120]
        assert data_cfg['collate_fn']['mixup_prob']==.5 and data_cfg['collate_fn']['mixup_epochs']==[4,64] and data_cfg['total_batch_size']==4
        fixture=copy.deepcopy(config);fixture.pop('__include__',None);fixture.pop('config',None);fixture['train_dataloader']['num_workers']=0;save(out/'fixture.yml',fixture)
        cfg=YAMLConfig(str(out/'fixture.yml'));loader=cfg.train_dataloader;loader.set_epoch(4);dataset=loader.dataset;collate=loader.collate_fn
        assert collate.copyblend_prob==0 and len(dataset)==6471
        mosaics=[op for op in dataset._transforms.transforms if type(op).__name__=='Mosaic'];assert len(mosaics)==1 and mosaics[0].use_cache is False
        positions=[round(i*(len(dataset)-1)/63) for i in range(64)];assert len(set(positions))==64
        def signature(images,targets):
            return dict(image=digest(images),shape=list(images.shape),boxes=[digest(t['boxes']) for t in targets],labels=[tensor(t['labels']).tolist() for t in targets],mixup=[digest(t['mixup']) if 'mixup' in t else None for t in targets])
        ordinary=[];pre_counts=[];reset(a.seed)
        for b in range(16):
            items=[dataset[i] for i in positions[4*b:4*b+4]];pre_counts.append([len(t['labels']) for x,t in items]);images,targets=collate(items);ordinary.append(signature(images,targets))
        original_load=dataset.load_item;cat={int(k)+1:int(v['category_id']) for k,v in dataset.coco.anns.items()};raw_images={}
        def tagged_load(idx):
            image,target=original_load(idx);image_id=int(dataset.ids[idx]);w,h=image.size
            rows=[r for r in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[image_id])) if not r.get('iscrowd',0)]
            boxes=torch.tensor([r['bbox'] for r in rows],dtype=torch.float32).reshape(-1,4);boxes[:,2:]+=boxes[:,:2];boxes[:,0::2].clamp_(0,w);boxes[:,1::2].clamp_(0,h);keep=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1]);valid=[r for r,k in zip(rows,keep.tolist()) if k]
            assert torch.equal(boxes[keep],tensor(target['boxes'])) and tensor(target['labels']).tolist()==[r['category_id'] for r in valid]
            target['labels']=torch.tensor([int(r['id'])+1 for r in valid],dtype=torch.int64)
            if image_id not in raw_images:
                meta=dataset.coco.imgs[image_id];image_path=Path(data_cfg['dataset']['img_folder'])/meta['file_name'];raw_images[image_id]=dict(image_id=image_id,width=w,height=h,file_name=meta['file_name'],source_image_sha256=sha(image_path))
            return image,target
        batches=[];identities=[];dataset.load_item=tagged_load;reset(a.seed)
        try:
            for b in range(16):
                items=[dataset[i] for i in positions[4*b:4*b+4]];assert [len(t['labels']) for x,t in items]==pre_counts[b]
                images,targets=collate(items);ids=[]
                for t in targets:
                    tags=tensor(t['labels']).tolist();ids.append([k-1 for k in tags]);t['labels']=torch.tensor([cat[k] for k in tags],dtype=torch.int64)
                assert signature(images,targets)==ordinary[b], 'Augmentation observer changed native data'
                batches.append((tensor(images).clone(),[{k:tensor(v).clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in t.items()} for t in targets],ids))
                identities.append(dict(batch=b,positions=positions[4*b:4*b+4],pre_collate_counts=pre_counts[b],post_collate_counts=[len(t['labels']) for t in targets],source_annotation_ids=ids,ordinary_and_tagged_outputs_exact=True,**ordinary[b]))
        finally:dataset.load_item=original_load
        save(out/'augmentation_identity.json',dict(train_annotation_sha256=ANN_SHA,augmentation_rng_seed=a.seed,epoch_zero_based=4,primary_positions=positions,primary_image_ids=[int(dataset.ids[i]) for i in positions],batch_identity=identities,loaded_source_images=list(raw_images.values()),ordinary_and_tagged_passes_each_batches=16,global_instance_rows_verified=True,historical_training_stream=False))
        model=DEIM(HGNetv2(**config['HGNetv2']),HybridEncoder(**config['HybridEncoder'],eval_spatial_size=[640,640]),DFINETransformer(**config['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640]),num_classes=10).cuda().train()
        def criterion_new():
            recipe=copy.deepcopy(config['DEIMCriterion']);match=recipe.pop('matcher');match.pop('type',None);return DEIMCriterion(matcher=HungarianMatcher(**match,use_focal_loss=True),num_classes=10,**recipe).cuda()
        criterion=criterion_new();recorder=CoverageRecorder(model,criterion);forward_calls=0;records=[];parameter_checks=[]
        for endpoint,checkpoint in [(12,run12/'checkpoint0011.pth'),(132,run132/'last.pth')]:
            assert sha(checkpoint)==CHECKPOINTS[a.seed][endpoint];payload=torch.load(checkpoint,map_location='cpu',weights_only=False);assert payload['last_epoch']==endpoint-1;weights=payload['ema']['module'];model.load_state_dict(weights,strict=True);before=parameter_sha(model)
            for b,(images,targets,ids) in enumerate(batches):
                model.load_state_dict(weights,strict=True);model_seed=7000+a.seed*100+b;reset(model_seed);x=images.cuda();ts=[{k:v.cuda() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in t.items()} for t in targets]
                if b==0:
                    with torch.no_grad():plain=model(x,targets=ts);plain_losses=criterion(plain,ts,epoch=4,step=b,global_step=b,epoch_step=16)
                    forward_calls+=1;plain_buffers={k:v.detach().clone() for k,v in model.named_buffers()};plain_rng=[torch.get_rng_state().clone()]+[r.clone() for r in torch.cuda.get_rng_state_all()];model.load_state_dict(weights,strict=True);reset(model_seed)
                recorder.begin(dict(endpoint=endpoint,seed=a.seed,batch=b,input_shape=list(x.shape),augmentation_epoch_zero_based=4,augmentation_rng_seed=a.seed,model_rng_seed=model_seed,dtype='CUDA_FP32',AMP=False,source_identity_verified=True,backward_executed=False,purpose='frozen EMA native augmented targets; not actual optimization history'),ts,source_annotation_ids=ids)
                for j,t in enumerate(ts):
                    if 'mixup' in t:recorder.put(f'target.{j}.mixup',t['mixup'])
                with torch.no_grad():outputs=model(x,targets=ts);losses=criterion(outputs,ts,epoch=4,step=b,global_step=b,epoch_step=16)
                forward_calls+=1
                assert outputs['pred_logits'].shape[1]==300 and all(torch.isfinite(v) for v in losses.values()) and not recorder.errors
                if b==0:
                    assert equal(plain,outputs) and equal(plain_losses,losses)
                    assert all(torch.equal(v,plain_buffers[k]) for k,v in model.named_buffers())
                    assert all(torch.equal(x,y) for x,y in zip(plain_rng,[torch.get_rng_state()]+torch.cuda.get_rng_state_all()));del plain,plain_losses,plain_buffers,plain_rng
                evidence=recorder.end(out/f'endpoint{endpoint}/batch_{b:03d}');assert evidence['status']=='CAPTURED';assert all(e['stage']!='UNRESOLVED' for e in evidence['events'] if e['kind'] in ('loss_call','matcher'))
                summary=[]
                for j,t in enumerate(targets):
                    normal_cls=[e for e in evidence['events'] if e['kind']=='loss_call' and not e['is_DN'] and e['loss']=='mal'];normal_reg=[e for e in evidence['events'] if e['kind']=='loss_call' and not e['is_DN'] and e['loss']=='boxes'];dn=[e for e in evidence['events'] if e['kind']=='loss_call' and e['is_DN'] and e['loss']=='boxes']
                    def covered(events):return set(int(g) for e in events for g in recorder.data[e['prefix']+f'.batch{j}.GT_row_index'])
                    final=covered([e for e in normal_cls if e['stage']=='model']);union=covered(normal_reg);any_cls=covered(normal_cls);dn_gt=covered(dn);all_gt=set(range(len(t['labels'])))
                    summary.append(dict(batch_item=j,GT_rows=len(all_gt),final_cls=len(final),any_normal_cls=len(any_cls),normal_regression_union=len(union),DN=len(dn_gt),no_any_supervision=len(all_gt-(any_cls|union|dn_gt)),no_normal_supervision=len(all_gt-(any_cls|union))))
                records.append(dict(endpoint=endpoint,batch=b,input_size=int(x.shape[-1]),counts=summary));save(out/f'progress_endpoint{endpoint}_batch{b:03d}.json',records[-1]);del x,ts,outputs,losses
            after=parameter_sha(model);assert before==after;parameter_checks.append(dict(endpoint=endpoint,parameters_before=before,parameters_after=after,exact=True));model.load_state_dict(weights,strict=True)
        recorder.close();recorder=None
        assert {p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}==sources
        save(out/'source_identity.json',dict(before_and_after_equal=True,native_files_sha256=sources,config=config,source_annotation_sha256=ANN_SHA))
        save(out/'summary.json',dict(samples=records))
        save(out/'completion.json',dict(status='COMPLETE_NATIVE_AUGMENTED_SUPERVISION_CAPTURE',seed=a.seed,model_forward_calls=forward_calls,expected_model_forward_calls=34,model_backward_calls=0,optimizer_steps=0,training_epochs_added=0,observer_plain_outputs_losses_BN_RNG_exact=True,parameter_checks=parameter_checks,native_sources_unchanged=True,all_augmentation_bytes_and_decoded_classes_exact=True,elapsed_seconds=time.monotonic()-started,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),historical_training=False))
        files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()};save(out/'manifest.json',dict(files=files));print(json.dumps(dict(status='COMPLETE',seed=a.seed,elapsed_seconds=time.monotonic()-started,model_forward_calls=forward_calls)),flush=True)
    except BaseException as e:
        if recorder is not None:recorder.close()
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
