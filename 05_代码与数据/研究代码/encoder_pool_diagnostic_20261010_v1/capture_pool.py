"""Observe native TopK and frozen dense encoder-head candidate supply.

The dense head call is diagnostic only. It runs AFTER the original forward,
does not feed candidates into the decoder, and does not update any weight.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import argparse, copy, datetime, hashlib, json, random, sys, time, traceback
import numpy as np
import torch

CKPT={1:{12:'ba3b6351fa8a4f7c769698c616ed6d20acf6a60ba119cf8edc8611851c4d27d4',132:'30febfa18e6e21dc67b16169bd26875b82340eb3435534719a963904dadf68b5'},2:{12:'6198ee5348630ce97c322c16aa4499a50a2f7af0d08dea1ec3a4cb632d9d5046',132:'5d3b838e296c866e0472e07d4c5b3dd59f7566029d1fca947ab8cd8b3fc1fb3a'}}
TRAIN_SHA='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
VAL_SHA='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def plain(v):return v.as_subclass(torch.Tensor) if isinstance(v,torch.Tensor) else v
def digest(v):return hashlib.sha256(plain(v).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def reset(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def param_sha(model):
    h=hashlib.sha256()
    for k,p in sorted(model.named_parameters()):h.update(k.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class PoolObserver:
    def __init__(self,decoder):
        self.decoder=decoder;self.original=decoder._select_topk;self.enabled=False;self.data={};self.hook=decoder.enc_bbox_head.register_forward_hook(self.box_hook)
        def wrapper(memory,logits,anchors,topk):
            result=self.original(memory,logits,anchors,topk)
            if self.enabled:
                assert not self.data and decoder.query_select_method=='default' and topk==300
                idx=logits.max(-1).values.topk(topk,dim=-1).indices
                for original,full in zip(result,[memory,logits if decoder.training else None,anchors]):
                    if full is not None:assert torch.equal(original,full.gather(1,idx.unsqueeze(-1).expand(-1,-1,full.shape[-1])))
                self.data={'memory':memory.detach().clone(),'logits':logits.detach().clone(),'anchors':anchors.detach().clone(),'selected_indices':idx.detach().clone(),'selected_anchors':result[2].detach().clone()}
            return result
        decoder._select_topk=wrapper
    def box_hook(self,module,args,result):
        if self.enabled and args[0].shape[1]==300:
            assert 'selected_residual' not in self.data;self.data['selected_residual']=result.detach().clone()
    def begin(self):self.data={};self.enabled=True
    def end(self):
        self.enabled=False;assert 'selected_residual' in self.data
        d=self.data;native_boxes=(d['selected_residual']+d['selected_anchors']).sigmoid()
        with torch.no_grad():raw=(self.decoder.enc_bbox_head(d['memory'])+d['anchors']).sigmoid()
        raw_selected=raw.gather(1,d['selected_indices'].unsqueeze(-1).expand(-1,-1,4));error=float((raw_selected-native_boxes).abs().max())
        assert error<=2e-6,error
        result={'pool.raw_boxes':raw,'pool.logits':d['logits'],'pool.valid':torch.isfinite(d['anchors']).all(-1),'pool.selected_indices':d['selected_indices'],'encoder.selected_boxes':native_boxes,'encoder.selected_logits':d['logits'].gather(1,d['selected_indices'].unsqueeze(-1).expand(-1,-1,10))}
        self.data={};return result,error
    def close(self):self.decoder._select_topk=self.original;self.hook.remove()

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=[1,2],required=True);a=p.parse_args();prefix=Path(a.prefix);root=prefix/'DEIM/experiments/encoder_pool_diagnostic_20261010_v1';out=root/'capture_v1';out.mkdir(exist_ok=False);started=time.monotonic();observer=None
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);native=prefix/'DEIM/experiments/fine_residual_screen24_20261009_v2/native';sys.path.insert(0,str(native))
        sources={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')};prior=prefix/'DEIM/experiments/native_augmented_supervision_20261010_v1/capture_v1';prior_source=read(prior/'source_identity.json');assert sources==prior_source['native_files_sha256']
        from engine.core import YAMLConfig
        from engine.backbone.hgnetv2 import HGNetv2
        from engine.deim.hybrid_encoder import HybridEncoder
        from engine.deim.dfine_decoder import DFINETransformer
        from engine.deim.deim import DEIM
        from torchvision.ops import box_convert
        config=copy.deepcopy(prior_source['config']);identity=read(prior/'augmentation_identity.json')
        assert sha(Path(config['train_dataloader']['dataset']['ann_file']))==TRAIN_SHA and sha(Path(config['val_dataloader']['dataset']['ann_file']))==VAL_SHA
        fixture=copy.deepcopy(config);fixture.pop('__include__',None);fixture.pop('config',None);fixture['train_dataloader']['num_workers']=0;fixture['val_dataloader']['num_workers']=0;fixture['val_dataloader']['dataset']['transforms']['ops'].append({'type':'ConvertBoxes','fmt':'cxcywh','normalize':True});save(out/'fixture.yml',fixture)
        cfg=YAMLConfig(str(out/'fixture.yml'));train=cfg.train_dataloader;train.set_epoch(4);assert train.collate_fn.copyblend_prob==0
        augmentation=[];reset(a.seed)
        for b in range(16):
            items=[train.dataset[i] for i in identity['primary_positions'][4*b:4*b+4]];x,ts=train.collate_fn(items);expected=identity['batch_identity'][b]
            assert digest(x)==expected['image'] and list(x.shape)==expected['shape']
            assert [digest(t['boxes']) for t in ts]==expected['boxes'] and [plain(t['labels']).tolist() for t in ts]==expected['labels'] and [digest(t['mixup']) if 'mixup' in t else None for t in ts]==expected['mixup']
            if b in (0,5,10,15):augmentation.append((b,plain(x).clone(),[{k:plain(v).clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in t.items()} for t in ts],expected['source_annotation_ids']))
        val=cfg.val_dataloader;dataset=val.dataset;assert len(dataset)==548
        positions=[j*(len(dataset)-1)//31 for j in range(32)];validation=[];val_identity=[]
        for b in range(8):
            items=[];sourceids=[]
            for pos in positions[4*b:4*b+4]:
                image,target=dataset[pos];iid=int(target['image_id']);meta=dataset.coco.imgs[iid];anns=[r for r in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[iid])) if not r.get('iscrowd',0)]
                raw=torch.tensor([r['bbox'] for r in anns],dtype=torch.float32).reshape(-1,4);raw[:,2:]+=raw[:,:2];raw[:,0::2].clamp_(0,meta['width']);raw[:,1::2].clamp_(0,meta['height']);keep=(raw[:,2]>raw[:,0])&(raw[:,3]>raw[:,1]);anns=[r for r,k in zip(anns,keep.tolist()) if k]
                expected=box_convert(raw[keep]*torch.tensor([640/meta['width'],640/meta['height']]*2),'xyxy','cxcywh')/640
                assert torch.allclose(plain(target['boxes']),expected,atol=1e-6,rtol=1e-5) and plain(target['labels']).tolist()==[r['category_id'] for r in anns]
                ids=[r['id'] for r in anns];sourceids.append(ids);items.append((image,target));imagepath=Path(config['val_dataloader']['dataset']['img_folder'])/meta['file_name'];val_identity.append({'position':pos,'image_id':iid,'file_name':meta['file_name'],'original_size':[meta['height'],meta['width']],'source_image_sha256':sha(imagepath),'source_annotation_ids':ids})
            x,ts=val.collate_fn(items);assert list(x.shape)==[4,3,640,640];validation.append((b,plain(x).clone(),[{k:plain(v).clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in t.items()} for t in ts],sourceids))
        save(out/'input_identity.json',{'augmented_batch_indices':[0,5,10,15],'augmented_reconstruction_all16_batches_exact':True,'validation_positions':positions,'validation':val_identity,'train_annotation_sha256':TRAIN_SHA,'val_annotation_sha256':VAL_SHA,'val_train_eval_input_identity':True,'validation_bbox_conversion_only':True})
        model=DEIM(HGNetv2(**config['HGNetv2']),HybridEncoder(**config['HybridEncoder'],eval_spatial_size=[640,640]),DFINETransformer(**config['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640]),num_classes=10).cuda();assert not model.decoder.mgsa_groups and not model.decoder.o2m_t
        observer=PoolObserver(model.decoder);forward_calls=0;dense_head_calls=0;records=[];parameter_checks=[]
        for endpoint in (12,132):
            cp=prefix/(f'DEIM/experiments/fine_residual_screen24_20261009_v2/runs/baseline_seed{a.seed}/checkpoint0011.pth' if endpoint==12 else f'DEIM/experiments/baseline_official_s_640_20261007_v1/seed{a.seed}/run/last.pth')
            assert sha(cp)==CKPT[a.seed][endpoint];payload=torch.load(cp,map_location='cpu',weights_only=False);assert payload['last_epoch']==endpoint-1;weights=payload['ema']['module'];model.load_state_dict(weights,strict=True);before=param_sha(model)
            for context,batches,seedbase,istrain in [('aug_train',augmentation,7000,True),('val_train',validation,9000,True),('val_eval',validation,10000,False)]:
              for k,(b,x,ts,ids) in enumerate(batches):
                model.load_state_dict(weights,strict=True);model.train(istrain);rngseed=seedbase+a.seed*100+b;reset(rngseed);images=x.cuda();targets=[{n:v.cuda() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for n,v in t.items()} for t in ts]
                if k==0:
                    with torch.no_grad():original=model(images,targets=targets)
                    forward_calls+=1;buffers={n:v.detach().clone() for n,v in model.named_buffers()};rng=[torch.get_rng_state().clone()]+[v.clone() for v in torch.cuda.get_rng_state_all()];model.load_state_dict(weights,strict=True);reset(rngseed)
                observer.begin()
                with torch.no_grad():outputs=model(images,targets=targets)
                forward_calls+=1;pool,error=observer.end();dense_head_calls+=1
                if istrain:assert torch.equal(pool['encoder.selected_boxes'],outputs['enc_aux_outputs'][0]['pred_boxes']) and torch.equal(pool['encoder.selected_logits'],outputs['enc_aux_outputs'][0]['pred_logits'])
                if k==0:
                    assert equal(original,outputs) and all(torch.equal(v,buffers[n]) for n,v in model.named_buffers()) and all(torch.equal(v,w) for v,w in zip(rng,[torch.get_rng_state()]+torch.cuda.get_rng_state_all()));del original,buffers,rng
                if context=='aug_train':
                    with np.load(prior/f'endpoint{endpoint}/batch_{b:03d}/arrays.npz') as old:
                        for name in ['pred_boxes','pred_logits']:assert np.array_equal(outputs[name].detach().cpu().numpy(),old['model.'+name])
                        assert np.array_equal(pool['encoder.selected_boxes'].cpu().numpy(),old['model.enc_aux_outputs.0.pred_boxes'])
                values=dict(pool);values.update({'final.boxes':outputs['pred_boxes'],'final.logits':outputs['pred_logits']})
                for j,t in enumerate(targets):
                    for n in ['boxes','labels','image_id','orig_size']:values[f'target.{j}.'+n]=t[n]
                    values[f'target.{j}.source_annotation_ids']=torch.tensor(ids[j])
                    if 'mixup' in t:values[f'target.{j}.mixup']=t['mixup']
                arrays={n:plain(v).detach().cpu().contiguous().numpy() for n,v in values.items()};assert all(np.isfinite(v).all() for n,v in arrays.items() if n!='pool.valid')
                sample=out/f'endpoint{endpoint}/{context}/batch_{b:03d}';sample.mkdir(parents=True,exist_ok=False);np.savez_compressed(sample/'arrays.npz',**arrays)
                description={'endpoint':endpoint,'seed':a.seed,'context':context,'batch':b,'input_shape':list(images.shape),'image_tensor_sha256':digest(images),'model_rng_seed':rngseed,'mode':'train' if istrain else 'eval','dtype':'CUDA_FP32','AMP':False,'pool_tokens':int(pool['pool.raw_boxes'].shape[1]),'selected_count':300,'raw_dense_head_vs_native_selected_max_abs_error':error,'native_selected_rows_override_required_for_pool_superset':True,'original_outputs_BN_RNG_exact_control':k==0,'augmented_outputs_equal_prior_archive':context=='aug_train','GT_counts':[len(t['labels']) for t in targets],'array_count':len(arrays),'arrays':{n:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for n,v in arrays.items()},'model_forwards_so_far':forward_calls,'dense_head_calls_so_far':dense_head_calls,'optimizer_steps':0,'epochs_added':0}
                save(sample/'evidence.json',description);save(sample/'manifest.json',{'files':{n:sha(sample/n) for n in ['arrays.npz','evidence.json']}});records.append({n:v for n,v in description.items() if n!='arrays'});save(out/f'progress_{endpoint}_{context}_{b:03d}.json',records[-1]);del images,targets,outputs,pool,values,arrays
            after=param_sha(model);assert before==after;parameter_checks.append({'endpoint':endpoint,'before':before,'after':after,'exact':True})
        observer.close();observer=None;assert forward_calls==46 and dense_head_calls==40
        assert sources=={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        save(out/'source_identity.json',{'native_files_sha256':sources,'native_sources_unchanged':True,'config':config,'checkpoint_sha256':CKPT[a.seed]})
        save(out/'summary.json',{'records':records});save(out/'completion.json',{'status':'COMPLETE_ENCODER_POOL_CAPTURE','seed':a.seed,'recorded_batches':40,'model_forward_calls':46,'diagnostic_dense_head_calls':40,'model_backwards':0,'optimizer_steps':0,'epochs_added':0,'native_sources_unchanged':True,'parameter_checks':parameter_checks,'elapsed_seconds':time.monotonic()-started,'completed_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_forward_control_parity':True,'all_augmented_prior_outputs_exact':True})
        save(out/'manifest.json',{'files':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}});print(json.dumps({'status':'COMPLETE','seed':a.seed,'forward_calls':forward_calls,'elapsed_seconds':time.monotonic()-started}),flush=True)
    except BaseException as e:
        if observer:observer.close()
        save(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':main()
