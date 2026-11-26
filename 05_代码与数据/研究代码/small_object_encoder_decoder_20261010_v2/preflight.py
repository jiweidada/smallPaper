"""One-shot engineering check. No optimizer, training loop or AP evaluation.

Bounded to17 full forwards and6 native-loss backwards on disposable models.
The active-gradient probe deliberately perturbs zero output heads, is logged,
and is discarded. It is not a learning update or efficacy experiment.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import argparse,copy,datetime,gc,hashlib,json,pickle,random,sys,time,traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
COMMON='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
MATURE='30febfa18e6e21dc67b16169bd26875b82340eb3435534719a963904dadf68b5'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def reset(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def rng():return {'python':hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),'numpy':hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),'torch_cpu':digest(torch.get_rng_state()),'torch_cuda':[digest(v) for v in torch.cuda.get_rng_state_all()]}
def flatten(v,path=''):
    if isinstance(v,torch.Tensor):return {path:v.detach().cpu().clone()}
    if isinstance(v,dict):
        out={}
        for k,x in v.items():out.update(flatten(x,path+'.'+str(k)))
        return out
    if isinstance(v,(list,tuple)):
        out={}
        for k,x in enumerate(v):out.update(flatten(x,path+'.'+str(k)))
        return out
    return {path:v}
def identity(v):
    return {k:({'shape':list(x.shape),'dtype':str(x.dtype),'sha256':digest(x)} if isinstance(x,torch.Tensor) else x) for k,x in v.items()}
def compare(a,b):
    assert set(a)==set(b),'Output keys differ'
    diff={}
    for k,x in a.items():
        y=b[k]
        if isinstance(x,torch.Tensor):
            assert x.shape==y.shape and x.dtype==y.dtype,(k,x.shape,y.shape,x.dtype,y.dtype)
            exact=torch.equal(x,y)
            error=float((x.float()-y.float()).abs().max()) if x.numel() else 0.
            diff[k]={'exact':exact,'max_abs_error':error}
            assert exact,(k,error)
        else:assert x==y,(k,x,y)
    return diff
def parameter_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.named_parameters()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);a=p.parse_args()
    prefix=Path(a.prefix);root=prefix/'DEIM/experiments/small_object_encoder_decoder_20261010_v2'
    out=root/'preflight_v2';out.mkdir(exist_ok=False);started=time.monotonic();forward_count=backward_count=0
    try:
        protocol=read(HERE/'preflight_protocol.json')
        assert all(sha(HERE/k)==v for k,v in protocol['implementation_sha256'].items())
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        native=prefix/'DEIM/experiments/fine_residual_screen24_20261009_v2/native'
        source=read(prefix/'DEIM/experiments/native_augmented_supervision_20261010_v1/capture_v1/source_identity.json')
        native_sha={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        assert native_sha==source['native_files_sha256'] and len(native_sha)==468
        sys.path.insert(0,str(native))
        from engine.core import YAMLConfig
        from engine.backbone.hgnetv2 import HGNetv2
        from engine.deim.hybrid_encoder import HybridEncoder
        from engine.deim.dfine_decoder import DFINETransformer
        from engine.deim.deim import DEIM
        from engine.deim.deim_criterion import DEIMCriterion
        from engine.deim.matcher import HungarianMatcher
        from small_object_model import SmallObjectDEIM,load_native_initialization,EXTRA_PREFIXES
        config=copy.deepcopy(source['config']);recipe={k:copy.deepcopy(config[k]) for k in ('HGNetv2','HybridEncoder','DFINETransformer')}
        common_path=prefix/'DEIM/experiments/visdrone_light_20260928_090716/artifacts/common_init_seed0.pth'
        mature_path=prefix/'DEIM/experiments/baseline_official_s_640_20261007_v1/seed1/run/last.pth'
        assert sha(common_path)==COMMON and sha(mature_path)==MATURE
        payload=torch.load(common_path,map_location='cpu',weights_only=False);common_state=payload['ema']['module'] if 'ema' in payload else payload['model'];del payload
        payload=torch.load(mature_path,map_location='cpu',weights_only=False);assert payload['last_epoch']==131;mature_state=payload['ema']['module'];del payload
        ann=Path(config['val_dataloader']['dataset']['ann_file']);assert sha(ann)=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
        fixture=copy.deepcopy(config);fixture.pop('__include__',None);fixture.pop('config',None)
        fixture['val_dataloader']['num_workers']=0
        fixture['val_dataloader']['dataset']['transforms']['ops'].append({'type':'ConvertBoxes','fmt':'cxcywh','normalize':True})
        save(out/'fixture.yml',fixture);cfg=YAMLConfig(str(out/'fixture.yml'));val=cfg.val_dataloader;reset(713)
        image,target=val.dataset[1];image,target=val.collate_fn([(image,target)])
        image=image.as_subclass(torch.Tensor).cuda();targets=[{k:(v.as_subclass(torch.Tensor).cuda() if isinstance(v,torch.Tensor) else copy.deepcopy(v)) for k,v in t.items()} for t in target]
        assert list(image.shape)==[1,3,640,640] and len(targets[0]['labels'])==81
        assert bool(torch.isfinite(image).all()) and bool(torch.isfinite(targets[0]['boxes']).all())
        save(out/'input_identity.json',{'ordinary_eval_image_position':1,'images':1,'GT_count':81,'image_tensor_sha256':digest(image),'box_sha256':digest(targets[0]['boxes']),'label_sha256':digest(targets[0]['labels']),'val_annotation_sha256':sha(ann),'no_augmented_training_trajectory_test':True})

        def criterion_for(model):
            c=copy.deepcopy(config['DEIMCriterion']);m=c.pop('matcher');m.pop('type',None)
            criterion=DEIMCriterion(matcher=HungarianMatcher(**m,use_focal_loss=True),num_classes=10,**c).cuda()
            criterion.set_model(model);criterion.set_epoch(0);return criterion
        baselines={};mature_baseline=None;constructor_rng=None;initializations=[];records=[];gradients=[];arrays={}
        for variant in ('native','fine','degraded'):
            reset(41)
            if variant=='native':model=DEIM(HGNetv2(**recipe['HGNetv2']),HybridEncoder(**recipe['HybridEncoder'],eval_spatial_size=[640,640]),DFINETransformer(**recipe['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640]),num_classes=10)
            else:model=SmallObjectDEIM(recipe,variant)
            after=rng()
            if variant=='native':constructor_rng=after
            else:assert after==constructor_rng,'Constructor changed global RNG'
            if variant=='native':model.load_state_dict(common_state,strict=True);init={'variant':variant,'parameters':sum(p.numel() for p in model.parameters()),'extra_parameters':0}
            else:init=load_native_initialization(model,common_state)
            initializations.append(init);model.cuda();parameter_before=parameter_digest(model)
            for amp in (False,True):
                for train in (False,True):
                    # Restore native BN and weights; added modules unchanged.
                    if variant=='native':model.load_state_dict(common_state,strict=True)
                    else:load_native_initialization(model,common_state)
                    model.train(train);model.zero_grad(set_to_none=True);criterion=criterion_for(model) if train else None
                    reset(2201);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                    with torch.set_grad_enabled(train and variant!='native'),torch.autocast('cuda',enabled=amp):
                        outputs=model(image,targets=targets if train else None);forward_count+=1
                        loss=criterion(outputs,targets) if train else {}
                        total=sum(loss.values()) if train else None
                    torch.cuda.synchronize();elapsed=time.monotonic()-start
                    assert all(bool(torch.isfinite(v).all()) for v in flatten(outputs).values() if isinstance(v,torch.Tensor))
                    assert all(bool(torch.isfinite(v).all()) for v in loss.values())
                    assert outputs['pred_logits'].shape==(1,300,10) and outputs['pred_boxes'].shape==(1,300,4)
                    flat=flatten(outputs);loss_cpu={k:float(v.detach()) for k,v in loss.items()}
                    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers() if not k.startswith(EXTRA_PREFIXES)}
                    key=f'common_amp{int(amp)}_train{int(train)}';after_rng=rng()
                    if variant=='native':baselines[key]={'outputs':flat,'losses':loss_cpu,'buffers':buffers,'rng':after_rng};differences={}
                    else:
                        differences=compare(baselines[key]['outputs'],flat)
                        assert baselines[key]['losses']==loss_cpu and baselines[key]['rng']==after_rng
                        compare(baselines[key]['buffers'],buffers)
                    record={'variant':variant,'case':key,'exact_native_predictions':variant=='native' or all(x['exact'] for x in differences.values()),'output_identity':identity(flat),'comparison':differences,'losses':loss_cpu,'rng_after':after_rng,'native_BN_buffers_exact':True,'forward_seconds_single_probe':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                    if not train:
                        for k in ('pred_logits','pred_boxes'):arrays[variant+'.'+key+'.'+k]=outputs[k].detach().float().cpu().numpy()
                    if train and variant!='native':
                        total.backward();backward_count+=1
                        g={n:{'connected':p.grad is not None,'finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),'norm':None if p.grad is None else float(p.grad.float().norm())} for n,p in model.named_parameters() if n.startswith(EXTRA_PREFIXES)}
                        assert all(x['finite'] for x in g.values()),'Initial added-parameter gradient disconnected/nonfinite'
                        assert any(x['norm']>0 for n,x in g.items() if '.geometry_proposal.output.' in n or '.geometry_attention.' in n and '.output.' in n),'Zero-output heads did not receive a learning signal'
                        gradients.append({'variant':variant,'case':key,'heads_manually_activated':False,'gradients':g})
                    records.append(record);save(out/f'{variant}_{key}.json',record)
                    del outputs,loss,total,criterion,flat,buffers;gc.collect()
            # A trained native weight fixture adds a second identity condition.
            if variant=='native':model.load_state_dict(mature_state,strict=True)
            else:load_native_initialization(model,mature_state)
            model.eval();reset(2201)
            with torch.no_grad():outputs=model(image);forward_count+=1
            flat=flatten(outputs)
            if variant=='native':mature_baseline=flat;differences={}
            else:differences=compare(mature_baseline,flat)
            save(out/f'{variant}_mature_eval.json',{'output_identity':identity(flat),'comparison':differences,'native_checkpoint_sha256':MATURE})
            del outputs,flat
            if variant!='native':
                # Deliberate manual activation on a disposable mature model.
                activation={}
                with torch.no_grad():
                    for n,m in model.named_modules():
                        if n.endswith('geometry_proposal') or n.startswith('decoder.decoder.geometry_attention.') and n.count('.')==3:
                            v=torch.arange(m.output.weight.numel(),device=m.output.weight.device,dtype=torch.float32).reshape_as(m.output.weight)
                            m.output.weight.copy_(torch.sin(v+1)*.001);activation[n+'.output.weight']={'amplitude':.001,'sha256':digest(m.output.weight)}
                assert len(activation)==4
                model.train();model.zero_grad(set_to_none=True);criterion=criterion_for(model);reset(2201)
                with torch.autocast('cuda',enabled=True):outputs=model(image,targets=targets);forward_count+=1;loss=criterion(outputs,targets);total=sum(loss.values())
                assert bool(torch.isfinite(total));total.backward();backward_count+=1
                g={n:{'connected':p.grad is not None,'finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),'norm':None if p.grad is None else float(p.grad.float().norm())} for n,p in model.named_parameters() if n.startswith(EXTRA_PREFIXES)}
                assert all(x['finite'] for x in g.values())
                group_norms={prefix:sum(x['norm']**2 for n,x in g.items() if n.startswith(prefix))**.5 for prefix in EXTRA_PREFIXES}
                assert all(x>0 for x in group_norms.values()),group_norms
                gradients.append({'variant':variant,'case':'mature_AMP_manual_activation','heads_manually_activated':True,'activation':activation,'losses':{k:float(v.detach()) for k,v in loss.items()},'group_gradient_norms':group_norms,'gradients':g,'optimizer_steps':0,'temporary_model_discarded':True})
                del outputs,loss,total,criterion
            # The common/native initial weights were not learned by any probe.
            if variant=='native':model.load_state_dict(common_state,strict=True)
            else:
                with torch.no_grad():
                    for n,m in model.named_modules():
                        if n.endswith('geometry_proposal') or n.startswith('decoder.decoder.geometry_attention.') and n.count('.')==3:m.output.weight.zero_();m.output.bias.zero_()
                load_native_initialization(model,common_state)
            assert parameter_digest(model)==parameter_before,'Probe changed parameters beyond logged/restored manual activation'
            del model;gc.collect();torch.cuda.empty_cache()
        assert forward_count==17 and backward_count==6
        assert native_sha=={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        save(out/'source_identity.json',{'native_files_sha256':native_sha,'native_sources_unchanged':True,'model_recipe':recipe,'common_init_sha256':COMMON,'mature_fixture_sha256':MATURE,'prototype_implementation_sha256':protocol['implementation_sha256'],'protocol_sha256':sha(HERE/'preflight_protocol.json')})
        save(out/'gradients.json',gradients);np.savez_compressed(out/'initial_predictions.npz',**arrays)
        save(out/'completion.json',{'status':'PASS_SMALL_OBJECT_ENCODER_DECODER_ENGINEERING_PREFLIGHT','constructor_CPU_CUDA_Python_NumPy_RNG_exact':True,'common_native_parameters_unchanged':True,'FP32_AMP_eval_train_initial_outputs_exact':True,'native_BN_RNG_loss_checks_exact':True,'mature_eval_initial_outputs_exact':True,'native_losses_and_DN_preserved':True,'new_groups_receive_finite_gradients_after_logged_manual_activation':True,'initializations':initializations,'model_forward_calls':forward_count,'model_backward_calls':backward_count,'optimizer_constructions':0,'optimizer_steps':0,'epochs_added':0,'AP_test':False,'source468_unchanged':True,'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'elapsed_seconds':time.monotonic()-started,'completed_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limits':['One ordinary EVAL image81GT reused in train mode for engineering only.','Initial prediction equality is not training-gradient or training-trajectory equality.','Manual output-weight activation is an explicit disposable gradient probe, not training.','No optimizer, AMP scaler/EMA update, multibatch data alignment or speed benchmark verified.','No manifold, root-cause, novelty or detection-improvement claim.']})
        save(out/'manifest.json',{'files':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}})
        print(json.dumps({'status':'PASS','forwards':forward_count,'backwards':backward_count,'epochs_added':0,'elapsed_seconds':time.monotonic()-started}),flush=True)
    except BaseException as e:
        save(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'forward_calls_completed':forward_count,'backward_calls_completed':backward_count,'optimizer_steps':0,'epochs_added':0,'automatic_retry':False});raise
if __name__=='__main__':main()
