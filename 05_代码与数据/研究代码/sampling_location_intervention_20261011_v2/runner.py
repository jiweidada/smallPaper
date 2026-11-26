"""Frozen-candidate decoder-suffix intervention; no detector/backbone forward."""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def xyxy(a):return np.concatenate((a[...,:2]-a[...,2:]/2,a[...,:2]+a[...,2:]/2),-1)


def iou(gt,box):
    wh=np.maximum(np.minimum(gt[:,None,2:],box[None,:,2:])-np.maximum(gt[:,None,:2],box[None,:,:2]),0)
    inter=wh.prod(-1)
    return inter/np.maximum(np.maximum(gt[:,2:]-gt[:,:2],0).prod(-1)[:,None]+np.maximum(box[:,2:]-box[:,:2],0).prod(-1)[None,:]-inter,1e-30)


def sampling_support(locations,weights):
    loc=locations.detach().double().cpu().numpy();w=weights.detach().double().cpu().numpy()
    masses=[];start=0
    for height,width,n in ((80,80,3),(40,40,6),(20,20,3)):
        pixel=loc[:,start:start+n]*np.array([width,height],dtype=np.float64)-.5
        lower=np.floor(pixel).astype(np.int64);frac=pixel-lower;mass=np.zeros(pixel.shape[:-1])
        for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
            ok=(lower[...,0]+dx>=0)&(lower[...,0]+dx<width)&(lower[...,1]+dy>=0)&(lower[...,1]+dy<height)
            coefficient=(frac[...,0] if dx else 1-frac[...,0])*(frac[...,1] if dy else 1-frac[...,1])
            mass+=coefficient*ok
        masses.append(mass);start+=n
    mass=np.concatenate(masses,axis=-1)
    return float(mass.mean()),float((mass*w).sum(-1).mean())


def main(args):
    p=json.loads(Path(args.protocol).read_text(encoding='utf8'))
    assert p['schema']=='sampling_location_suffix_intervention_v2'
    ready=json.loads(Path(args.preflight).read_text(encoding='utf8'))
    assert ready['status']=='PASS_READ_ONLY_PREFLIGHT_NO_MODEL'
    assert ready['seed']==args.seed and ready['protocol_sha256']==sha(args.protocol)
    assert sha(__file__)==p['code_sha256']['runner.py']
    assert sha(args.inputs)==p['inputs_sha256']
    assert Path(args.inputs).stat().st_size==p['inputs_bytes']
    spec=p['seeds'][str(args.seed)]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    counts=dict(attempted_suffix_calls=0,decoder_suffix_calls=0,control_calls=0,intervention_calls=0,full_detector_forwards=0,
                backbone_forwards=0,encoder_forwards=0,first_self_attention_forwards=0,
                BASE_forwards=0,evaluator_calls=0,optimizer_updates=0,epochs_added=0)
    control_records=[];cases=[];started=time.monotonic()
    write(out/'start.json',dict(utc=utc(),seed=args.seed,protocol_sha256=sha(args.protocol),pid=__import__('os').getpid()))
    try:
        sys.path.insert(0,spec['closeout_tools'])
        from verify_candidate_reload import verify_tools,bound_runtime,assert_trainers_stopped
        verify_tools()
        code,receipt,plan,paths=bound_runtime(Path(spec['bundle']))
        assert plan['arm']=='target_protect' and plan['seed']==args.seed
        assert_trainers_stopped(paths['screen_root'])
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert gpu==spec['gpu_uuid']
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        from runtime import activate,register_model,resolved_config
        from evaluator_binding import activate as bind_evaluator
        activate(paths['native_root']);bind_evaluator(paths['evaluator_root'],receipt['identity']['evaluator_binding']);register_model()
        import torch
        import torch.nn.functional as F
        from engine.core import YAMLConfig
        from engine.misc import dist_utils
        from engine.misc import flash_attention
        from engine.deim.dfine_utils import distance2bbox,weighting_function
        from engine.deim.utils import inverse_sigmoid
        from engine.misc.deformable_detr_msda import should_use_deformable_detr_msda,deformable_detr_msda_core_default
        dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=args.seed)
        cfg_data=resolved_config(plan,dict(paths,output=out))
        write(out/'resolved.yml',cfg_data)
        cfg=YAMLConfig(str(out/'resolved.yml'))
        model=cfg.model.cuda()
        checkpoint=paths['output']/'last.pth'
        assert sha(checkpoint)==spec['checkpoint_sha256']
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['last_epoch']==11 and state['scale_protection12_identity']==receipt['identity']
        ema=state['ema']['module'];model.load_state_dict(ema,strict=True)
        loaded=model.state_dict();assert len(loaded)==len(ema)==794
        assert all(torch.equal(v.cpu(),ema[k].cpu()) for k,v in loaded.items())
        assert sum(x.numel() for x in model.parameters())==10231997 and model.geometry_arm=='baseline'
        del state,ema,loaded
        model.eval();model.requires_grad_(False);torch.set_grad_enabled(False)
        d=model.decoder;decoder=d.decoder
        assert decoder.layers[0].cross_attn.method=='default'
        assert decoder.layers[0].cross_attn.num_points_list==[3,6,3]
        assert len(decoder.layers)==3 and decoder.eval_idx==2
        assert d.num_queries==300 and d.hidden_dim==256
        assert not hasattr(decoder,'project'),'The captured model was not converted to deploy'
        project=weighting_function(decoder.reg_max,d.up,d.reg_scale)
        backend_settings=dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,float32_matmul_precision=torch.get_float32_matmul_precision(),
            native_flash_callable_available=getattr(flash_attention,'flash_attn_func',None) is not None)
        # Hooks are prohibited-path sentinels: these modules must never be called.
        def forbidden(module,inputs):raise AssertionError('Forbidden full-detector/backbone/encoder/first-self-attention call')
        hooks=[m.register_forward_pre_hook(forbidden) for m in (model,model.backbone,model.encoder,decoder.layers[0].self_attn)]
        with zipfile.ZipFile(args.inputs) as z:
            manifest_raw=z.read('manifest.json');assert hashlib.sha256(manifest_raw).hexdigest()==p['inputs_manifest_sha256']
            manifest=json.loads(manifest_raw)
            for name,entry in manifest['files'].items():
                raw=z.read(name);assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
            owners_doc=json.loads(z.read('owners.json'))
            owners=owners_doc['seeds'][str(args.seed)]['owners']
            assert len(owners)==spec['owner_count']
            assert len({(o['image_id'],o['query']) for o in owners})==len(owners)
            assert 2+3*len(owners)<=spec['max_suffix_calls']
            batches={}
            for batch in (25,105):
                prefix=f'seed{args.seed}/capture/batch_{batch:03d}/'
                record=json.loads(z.read(prefix+'record.json'))
                assert record['metadata']['seed']==args.seed and record['metadata']['checkpoint_sha256']==spec['checkpoint_sha256']
                assert record['metadata']['identity']==receipt['identity']
                assert record['metadata']['arm']=='target_protect' and record['metadata']['epoch_zero_based']==11
                assert record['metadata']['evaluated_weights']=='EMA' and record['metadata']['autocast_enabled'] is False
                def event(name,inv=0):
                    es=[e for e in record['invocation_events'] if e['module']==name and e['invocation']==inv]
                    assert len(es)==1;return es[0]['output']
                def array(desc):
                    raw=z.read(prefix+desc['blob_file']);assert hashlib.sha256(raw).hexdigest()==desc['blob_sha256']
                    a=np.load(io.BytesIO(raw),allow_pickle=False)
                    assert list(a.shape)==desc['shape'] and str(a.dtype)==desc['storage_numpy_dtype'] and np.isfinite(a).all()
                    return a
                def tensor(desc):return torch.from_numpy(array(desc).copy()).cuda()
                features=[tensor(event(f'decoder.input_proj.{i}')) for i in range(3)]
                assert [list(f.shape) for f in features]==[[4,256,80,80],[4,256,40,40],[4,256,20,20]]
                memory=torch.cat([f.flatten(2).permute(0,2,1) for f in features],dim=1)
                value=decoder.value_op(memory,None,None,None,[[80,80],[40,40],[20,20]])
                norm1=tensor(event('decoder.decoder.layers.0.norm1'))
                offsets=tensor(event('decoder.decoder.layers.0.cross_attn.sampling_offsets')).reshape(4,300,8,12,2)
                weight_logits=tensor(event('decoder.decoder.layers.0.cross_attn.attention_weights')).reshape(4,300,8,12)
                weights=F.softmax(weight_logits,dim=-1)
                cross_original=tensor(event('decoder.decoder.layers.0.cross_attn'))
                stage=[];targets=[]
                for bi,iid in enumerate(record['metadata']['image_ids']):
                    with np.load(io.BytesIO(z.read(f'seed{args.seed}/stages/image_{iid:06d}.npz')),allow_pickle=False) as f:stage.append({k:f[k].copy() for k in f.files})
                    descs={e['key']['value']:e['value'] for e in record['native_targets']['items'][bi]['entries']}
                    targets.append(array(descs['boxes']).astype(np.float64)/640)
                reference=torch.from_numpy(np.stack([s['encoder_initial_boxes'] for s in stage])).cuda()
                scale=decoder.layers[0].cross_attn.num_points_scale.to(dtype=offsets.dtype).unsqueeze(-1)
                actual_bits=scale.cpu().numpy().reshape(-1).view(np.uint32).tolist()
                assert actual_bits==spec['actual_scale_uint32']
                assert offsets.dtype==weights.dtype==reference.dtype==norm1.dtype==torch.float32
                offset=offsets*scale*reference[:,:,None,None,2:]*decoder.layers[0].cross_attn.offset_scale
                locations=reference[:,:,None,None,:2]+offset
                batches[batch]=dict(record=record,value=value,norm1=norm1,weights=weights,locations=locations,
                                    reference=reference,cross_original=cross_original,stages=stage,targets=targets)
                batches[batch]['immutable_tensors']=[norm1,weights,reference,locations]+list(value)
                batches[batch]['immutable_versions']=[t._version for t in batches[batch]['immutable_tensors']]
            def suffix(batch,owner=None,arm='N'):
                assert counts['attempted_suffix_calls']<spec['max_suffix_calls']
                counts['attempted_suffix_calls']+=1
                data=batches[batch];layer0=decoder.layers[0];attn=layer0.cross_attn
                assert [t._version for t in data['immutable_tensors']]==data['immutable_versions']
                locations=data['locations'].clone()
                boundary={}
                if owner is not None:
                    delta=np.array(owner['delta'],dtype=np.float32)
                    if arm=='Cplus':delta=np.array([-delta[1],delta[0]],dtype=np.float32)
                    elif arm=='Cminus':delta=np.array([delta[1],-delta[0]],dtype=np.float32)
                    else:assert arm=='T'
                    shift=torch.from_numpy(delta).cuda();b=owner['image_offset'];q=owner['query']
                    base=locations[b,q].clone();locations[b,q]+=shift
                    changed=(locations!=data['locations']).any(-1).any(-1).any(-1)
                    changed[b,q]=False
                    assert not changed.any().item(),'A nonselected query sampling location changed'
                    mass_before,attn_before=sampling_support(base,data['weights'][b,q])
                    mass_after,attn_after=sampling_support(locations[b,q],data['weights'][b,q])
                    boundary=dict(delta_applied=delta.tolist(),zero_delta=bool(np.all(delta==0)),
                        normalized_inside_before=int(((base>=0)&(base<=1)).all(-1).sum().item()),
                        normalized_inside_after=int(((locations[b,q]>=0)&(locations[b,q]<=1)).all(-1).sum().item()),
                        points_total=96,bilinear_valid_mass_before=mass_before,bilinear_valid_mass_after=mass_after,
                        attention_valid_mass_before=attn_before,attention_valid_mass_after=attn_after,
                        support_mass_definition='Analytical bilinear zero-padding support from saved FP32 locations, computed in FP64; not semantic object membership.')
                if should_use_deformable_detr_msda(data['value'],locations,data['weights']):
                    cross=deformable_detr_msda_core_default(data['value'],[[80,80],[40,40],[20,20]],locations,data['weights'],[3,6,3])
                    backend='msda_extension'
                else:
                    cross=attn.ms_deformable_attn_core(data['value'],[[80,80],[40,40],[20,20]],locations,data['weights'],[3,6,3])
                    backend='native_grid_sample_core'
                output=layer0.gateway(data['norm1'],layer0.dropout2(cross))
                output=layer0.norm3((output+layer0.dropout4(layer0.forward_ffn(output))).clamp(min=-65504,max=65504))
                pre=F.sigmoid(d.pre_bbox_head(output)+inverse_sigmoid(data['reference']))
                pre_scores=d.dec_score_head[0](output)
                ref_initial=pre.detach()
                corners=d.dec_bbox_head[0](output+0)+0
                inter=distance2bbox(ref_initial,d.integral(corners,project),d.reg_scale)
                previous_output=output.detach()
                for layer_index in (1,2):
                    ref=inter.detach()
                    pos=d.query_pos_head(ref).clamp(min=-10,max=10)
                    output=decoder.layers[layer_index](output,ref.unsqueeze(2),data['value'],[[80,80],[40,40],[20,20]],None,pos)
                    corners=d.dec_bbox_head[layer_index](output+previous_output)+corners
                    inter=distance2bbox(ref_initial,d.integral(corners,project),d.reg_scale)
                    if layer_index==2:
                        scores=decoder.lqe_layers[2](d.dec_score_head[2](output),corners)
                    previous_output=output.detach()
                counts['decoder_suffix_calls']+=1
                counts['control_calls' if owner is None else 'intervention_calls']+=1
                result={k:v.detach().float().cpu().numpy() for k,v in dict(final_boxes=inter,final_logits=scores,pre_boxes=pre,pre_logits=pre_scores).items()}
                assert all(np.isfinite(a).all() for a in result.values())
                assert [t._version for t in data['immutable_tensors']]==data['immutable_versions']
                return result,cross,dict(backend=backend,**boundary)
            # All original-batch controls must pass before any intervention is permitted.
            for batch in (25,105):
                result,cross,info=suffix(batch)
                data=batches[batch];expected_boxes=np.stack([s['final_boxes'] for s in data['stages']]);expected_logits=np.stack([s['final_logits'] for s in data['stages']])
                diff=(cross-data['cross_original']).double()
                ca=float(diff.abs().max().item());cr=float((torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(data['cross_original'].double()).clamp(min=1e-30)).item())
                ba=float(np.max(np.abs(result['final_boxes']-expected_boxes)))
                ld=result['final_logits'].astype(np.float64)-expected_logits.astype(np.float64)
                la=float(np.max(np.abs(ld)));lr=float(np.linalg.norm(ld)/max(np.linalg.norm(expected_logits.astype(np.float64)),1e-30))
                pre_boxes_error=float(np.max(np.abs(result['pre_boxes']-np.stack([s['pre_after_first_layer_boxes'] for s in data['stages']]))))
                pre_logits_error=float(np.max(np.abs(result['pre_logits']-np.stack([s['pre_after_first_layer_logits'] for s in data['stages']]))))
                differences={}
                for threshold in (.5,.75):
                    differences[str(threshold)]=sum(int(np.sum((iou(gt,xyxy(result['final_boxes'][b].astype(np.float64))).max(-1)>=threshold)!=(iou(gt,xyxy(expected_boxes[b].astype(np.float64))).max(-1)>=threshold))) for b,gt in enumerate(data['targets']))
                row=dict(batch=batch,cross_maxabs=ca,cross_relative_L2=cr,final_boxes_maxabs=ba,
                         final_logits_maxabs=la,final_logits_relative_L2=lr,GT_support_differences=differences,
                         pre_boxes_maxabs_diagnostic=pre_boxes_error,pre_logits_maxabs_diagnostic=pre_logits_error,**info)
                control_records.append(row)
                write(out/f'control_{batch:03d}.json',row)
                assert ca<=p['controls']['cross_maxabs'] and cr<=p['controls']['cross_relative_L2'],row
                assert ba<=p['controls']['boxes_maxabs'] and la<=p['controls']['logits_maxabs'] and lr<=p['controls']['logits_relative_L2'],row
                assert all(v==0 for v in differences.values()),row
                np.savez_compressed(out/f'N_batch_{batch:03d}.npz',**result)
            for case_index,owner in enumerate(owners):
                for arm in ('T','Cplus','Cminus'):
                    result,cross,info=suffix(owner['batch'],owner,arm)
                    name=f'case_{case_index:03d}_{arm}.npz'
                    np.savez_compressed(out/name,**result)
                    b=owner['image_offset'];q=owner['query'];base_cross=batches[owner['batch']]['cross_original']
                    row=dict(case_index=case_index,arm=arm,owner=owner,output=name,
                             selected_cross_L2_change=float(torch.linalg.vector_norm((cross[b,q]-base_cross[b,q]).double()).item()),**info)
                    cases.append(row)
                    write(out/f'case_{case_index:03d}_{arm}.json',row)
                write(out/f'progress_{case_index:03d}.json',dict(utc=utc(),completed_owners=case_index+1,counts=counts))
            assert counts['decoder_suffix_calls']==2+3*len(owners)<=spec['max_suffix_calls']
            assert not any(x.grad is not None for x in model.parameters())
            for h in hooks:h.remove()
        report=dict(status='PASS_SAMPLING_LOCATION_SUFFIX_INTERVENTION',seed=args.seed,
                    controls=control_records,counts=counts,owner_count=len(owners),cases=len(cases),
                    checkpoint_sha256=spec['checkpoint_sha256'],protocol_sha256=sha(args.protocol),inputs_sha256=sha(args.inputs),
                    script_sha256=sha(__file__),gpu_uuid=gpu,torch_version=torch.__version__,
                    elapsed_seconds=time.monotonic()-started,completed_remote_UTC=utc(),
                    backend_settings=backend_settings,immutable_saved_inputs_version_checks_passed=True,
                    nonselected_query_sampling_locations_unchanged=True,
                    causal_root_established=False,AP_evaluation=False,trained_method=False)
        write(out/'report.json',report)
        write(out/'manifest.json',dict(files={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in out.iterdir() if f.is_file()}))
        print(json.dumps(report),flush=True)
        dist_utils.cleanup()
    except BaseException:
        write(out/'failure.json',dict(utc=utc(),traceback=traceback.format_exc(),counts=counts,controls=control_records,automatic_retry=False))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(1,2),required=True)
    ap.add_argument('--protocol',required=True);ap.add_argument('--inputs',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--preflight',required=True)
    main(ap.parse_args())
