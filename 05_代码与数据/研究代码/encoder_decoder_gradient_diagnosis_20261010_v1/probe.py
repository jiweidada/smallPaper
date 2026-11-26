"""Six disposable forward/backward cases, zero optimizer steps and no training.

The four activation cases are analytical interventions, not proposed initializers.
An ordinary validation image is reused in TRAIN mode solely for gradient tracing.
"""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import argparse,copy,datetime,gc,hashlib,json,math,pickle,random,sys,time,traceback
import numpy as np
import torch
CASES=[('common_fine_FP32','fine',False,False,False),('common_native_FP32','native',False,False,False),('common_fine_AMP','fine',True,False,False),('activate_native_regression_FP32','fine',False,True,False),('activate_added_outputs_FP32','fine',False,False,True),('activate_both_FP32','fine',False,True,True)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reset(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def pdigest(m):
    h=hashlib.sha256()
    for n,p in m.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def finitefloat(v):return float(v) if math.isfinite(float(v)) else None
def stat(t):
    if t is None:return {'connected':False}
    t=t.detach();a=t.double();return {'connected':True,'shape':list(t.shape),'dtype':str(t.dtype),'finite':bool(torch.isfinite(t).all()),'nonzero_elements':int(torch.count_nonzero(t)),'max_abs':finitefloat(t.abs().max()) if t.numel() else 0.,'norm32':finitefloat(t.float().norm()),'norm64':finitefloat(a.norm()),'sha256':digest(t)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--runtime-root',required=True);a=p.parse_args()
    root=Path(a.root);out=root/'results';out.mkdir(exist_ok=False);start=time.monotonic();fw=bw=0
    def save(name,obj):
        with (out/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        protocol=json.loads((root/'protocol.json').read_text());assert protocol['probe_sha256']==sha(__file__) and protocol['cases']==[list(x) for x in CASES]
        sys.path.insert(0,str(Path(a.runtime_root)/'code'))
        from runtime import verify_bundle,activate,load_common,COMMON_INIT_SHA
        receipt,plan,paths=verify_bundle(Path(a.runtime_root)/'prepared_seed1');activate(paths['native_root'])
        assert receipt['identity']['protocol_sha256']==protocol['formal_protocol_sha256']
        from engine.core import YAMLConfig
        from engine.backbone.hgnetv2 import HGNetv2
        from engine.deim.hybrid_encoder import HybridEncoder
        from engine.deim.dfine_decoder import DFINETransformer
        from engine.deim.deim import DEIM
        from engine.deim.deim_criterion import DEIMCriterion
        from engine.deim.matcher import HungarianMatcher
        from small_object_model import SmallObjectDEIM,EXTRA_PREFIXES
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        cfgdata=copy.deepcopy(plan['native_baseline_config']);cfgdata.pop('__include__',None);cfgdata.pop('config',None)
        cfgdata['val_dataloader']['dataset']['ann_file']=str(paths['val_ann']);cfgdata['val_dataloader']['dataset']['img_folder']=str(paths['val_images']);cfgdata['val_dataloader']['num_workers']=0
        cfgdata['val_dataloader']['dataset']['transforms']['ops'].append({'type':'ConvertBoxes','fmt':'cxcywh','normalize':True})
        save('fixture.yml',cfgdata);cfg=YAMLConfig(str(out/'fixture.yml'));loader=cfg.val_dataloader;reset(713)
        image,target=loader.dataset[1];image,target=loader.collate_fn([(image,target)])
        image=image.as_subclass(torch.Tensor).cuda();targets=[{k:(v.as_subclass(torch.Tensor).cuda() if isinstance(v,torch.Tensor) else copy.deepcopy(v)) for k,v in t.items()} for t in target]
        assert list(image.shape)==[1,3,640,640] and len(targets[0]['labels'])==81
        save('fixture_identity.json',{'ordinary_validation_position':1,'images':1,'GT_count':81,'image_sha256':digest(image),'boxes_sha256':digest(targets[0]['boxes']),'labels_sha256':digest(targets[0]['labels']),'validation_annotation_sha256':sha(paths['val_ann']),'not_original_training_batch_replay':True})
        assert sha(paths['common_init'])==COMMON_INIT_SHA
        payload=torch.load(paths['common_init'],map_location='cpu',weights_only=False);state=payload['ema']['module'] if 'ema' in payload else payload['model'];del payload
        tiny=torch.tensor([1e-30,1e-30],dtype=torch.float32);save('norm_measurement_unit_case.json',stat(tiny))
        records=[];common_outputs={};common_losses={};optimizer_constructions=0
        for name,variant,amp,activate_native,activate_added in CASES:
            reset(41);recipe=plan['model_recipe']
            if variant=='native':
                model=DEIM(HGNetv2(**recipe['HGNetv2']),HybridEncoder(**recipe['HybridEncoder'],eval_spatial_size=[640,640]),DFINETransformer(**recipe['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640]),num_classes=10)
                model.load_state_dict(state,strict=True)
            else:model=SmallObjectDEIM(recipe,variant='fine',initialization_seed=713);load_common(model,paths['common_init'])
            model.cuda().train();native_heads=['decoder.enc_bbox_head.layers.2.weight','decoder.pre_bbox_head.layers.2.weight']+[f'decoder.dec_bbox_head.{i}.layers.2.weight' for i in range(3)]
            named=dict(model.named_parameters());assert all(n in named for n in native_heads)
            head_initial={n:stat(named[n]) for n in native_heads};assert all(v['nonzero_elements']==0 for v in head_initial.values())
            interventions={}
            with torch.no_grad():
                if activate_native:
                    for n in native_heads:
                        v=torch.arange(named[n].numel(),device='cuda',dtype=torch.float32).reshape_as(named[n]);named[n].copy_(.001*torch.sin(v+1));interventions[n]=stat(named[n])
                if activate_added:
                    for n,m in model.named_modules():
                        if n.endswith('geometry_proposal') or n.startswith('decoder.decoder.geometry_attention.') and n.count('.')==3:
                            v=torch.arange(m.output.weight.numel(),device='cuda',dtype=torch.float32).reshape_as(m.output.weight);m.output.weight.copy_(.001*torch.sin(v+1));interventions[n+'.output.weight']=stat(m.output.weight)
            before=pdigest(model)
            c=copy.deepcopy(plan['criterion_recipe']);matcher=c.pop('matcher');matcher.pop('type',None)
            criterion=DEIMCriterion(matcher=HungarianMatcher(**matcher,use_focal_loss=True),num_classes=10,**c).cuda();criterion.set_model(model);criterion.set_epoch(0)
            optimizer=None;scaler=None
            if amp:
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0004);optimizer_constructions+=1
                def forbidden_step(*args,**kwargs):raise RuntimeError('Diagnostic optimizer.step forbidden')
                optimizer.step=forbidden_step
                scaler=torch.amp.GradScaler('cuda',enabled=True)
            reset(2201)
            with torch.autocast('cuda',enabled=amp):outputs=model(image,targets=targets);fw+=1
            with torch.autocast('cuda',enabled=False):losses=criterion(outputs,targets,epoch=0,step=0,global_step=0,epoch_step=1618);total=sum(losses.values())
            assert bool(torch.isfinite(total))
            if name in ('common_fine_FP32','common_native_FP32'):
                common_outputs[name]={k:outputs[k].detach().cpu().numpy().copy() for k in ['pred_boxes','pred_logits']};common_losses[name]={k:float(v.detach()) for k,v in losses.items()}
            if amp:scaler.scale(total).backward();scaler.unscale_(optimizer)
            else:total.backward()
            bw+=1
            observed={n:p for n,p in model.named_parameters() if n.startswith(EXTRA_PREFIXES) or n in native_heads or n.startswith('decoder.dec_score_head.') or n.startswith('decoder.decoder.layers.')}
            arrays={};before_clip={}
            for n,p in observed.items():
                before_clip[n]=stat(p.grad)
                if p.grad is not None:arrays['before_clip::'+n]=p.grad.detach().cpu().numpy().copy()
            allstats={n:stat(p.grad) for n,p in model.named_parameters() if p.requires_grad}
            finite_global=all(v.get('finite',True) for v in allstats.values())
            global64=math.sqrt(sum(v['norm64']**2 for v in allstats.values() if v.get('norm64') is not None)) if finite_global else None
            clip_return=torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfgdata['clip_max_norm']))
            after_clip={n:stat(p.grad) for n,p in observed.items()}
            for n,p in observed.items():
                if p.grad is not None:arrays['after_clip::'+n]=p.grad.detach().cpu().numpy().copy()
            groups={}
            for prefix in EXTRA_PREFIXES:
                members=[n for n in observed if n.startswith(prefix)]
                if not members:continue
                groups[prefix]={}
                for label,stats in [('before_clip',before_clip),('after_clip',after_clip)]:
                    vals=[stats[n] for n in members];finite=all(v.get('finite',False) for v in vals)
                    groups[prefix][label]={'finite':finite,'nonzero_elements':sum(v.get('nonzero_elements',0) for v in vals),'norm64':math.sqrt(sum(v['norm64']**2 for v in vals)) if finite else None,'max_abs':max(v['max_abs'] for v in vals) if finite else None}
            assert pdigest(model)==before,'Backward/clip changed parameters'
            np.savez_compressed(out/(name+'_gradients.npz'),**arrays)
            row={'case':name,'variant':variant,'AMP_forward_only':amp,'criterion_FP32':True,'losses':{k:float(v.detach()) for k,v in losses.items()},'total_loss':float(total.detach()),'AMP_default_scale':float(scaler.get_scale()) if amp else None,'native_regression_heads_initial':head_initial,'manual_interventions_discarded':interventions,'global_unscaled_all_parameters_finite':finite_global,'global_unscaled_gradient_norm64':global64,'native_clip_norm_return':finitefloat(clip_return),'clip_max_norm':float(cfgdata['clip_max_norm']),'groups':groups,'before_clip':before_clip,'after_clip':after_clip,'parameters_unchanged_after_backward_clip':True,'optimizer_steps':0,'EMA_updates':0}
            save(name+'.json',row);records.append(row)
            del outputs,losses,total,model,criterion,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
        assert fw==6 and bw==6
        n=common_outputs['common_native_FP32'];f=common_outputs['common_fine_FP32'];exact={k:bool(np.array_equal(n[k],f[k])) for k in n};assert all(exact.values()) and common_losses['common_native_FP32']==common_losses['common_fine_FP32']
        np.savez_compressed(out/'common_native_fine_predictions.npz',**{case+'::'+k:v for case,d in common_outputs.items() for k,v in d.items()})
        from runtime import verify_source
        verify_source(paths['native_root'])
        summary={'status':'COMPLETE_SIX_CASE_ZERO_UPDATE_GRADIENT_DIAGNOSIS','forwards':fw,'backwards':bw,'optimizer_constructions_unscale_only':optimizer_constructions,'optimizer_steps':0,'EMA_updates':0,'epochs_added':0,'common_native_fine_outputs_losses_exact':True,'cases':[{k:v for k,v in row.items() if k in ['case','total_loss','groups','global_unscaled_all_parameters_finite','global_unscaled_gradient_norm64','native_clip_norm_return']} for row in records],'bound_formal_protocol_sha256':receipt['identity']['protocol_sha256'],'probe_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-start,'completed_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'claim_boundary':['One ordinary validation image81GT, not actual first10 training batches.','Manual .001 sinusoid activation is a disposable analytic intervention, not a new trained initializer.','No historical terminal parameter tensors exist; cannot prove exact reason for every zero historical norm.','No candidate AP or improved architecture accepted; terminal configuration not resumed/retried.']}
        save('completion.json',summary);save('manifest.json',{'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}})
        print(json.dumps({k:v for k,v in summary.items() if k not in ['cases']},ensure_ascii=False),flush=True)
    except BaseException as e:
        save('failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'completed_forwards':fw,'completed_backwards':bw,'optimizer_steps':0,'epochs_added':0,'automatic_retry':False});raise
if __name__=='__main__':main()
