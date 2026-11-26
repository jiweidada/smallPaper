"""One native capture per case and six exact-author final-layer suffix interventions."""
from pathlib import Path
from datetime import datetime, timezone
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib
import importlib
import json
import sys
import traceback
import numpy as np
import torch
from PIL import Image
from a15_sampling_analysis import analyze,write,ARMS

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT.parent/'dome_assets_v1'
BASE=ROOT.parent/'a12_native_distributions_v1'
sys.path.insert(0,str(ASSETS))
from dome_probe_loader import load_model,state_digest,hash_file

def main():
    p=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    freeze=json.loads((ROOT/'freeze_manifest.json').read_text(encoding='utf-8'))
    for f in freeze['files']:assert hash_file(ROOT/f['file'])==f['sha256']
    for f in p['baseline_files']:assert hash_file(BASE/f['file'])==f['sha256']
    assert hash_file(ASSETS/'dome_probe_loader.py')==p['loader_sha256']
    for rel,digest in p['source_hashes'].items():assert hash_file(ASSETS/'source'/rel)==digest
    with (ROOT/'started.json').open('x',encoding='utf-8') as f:
        json.dump({'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()},f)
    calls=partials=0
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
        model,_,transform,load=load_model(ROOT/'model_load.json')
        assert load['state_digest']==p['state_digest']
        free,total=torch.cuda.mem_get_info();assert free>=2*1024**3
        model=model.to('cuda');before=state_digest(model);assert before==p['state_digest']
        outer=model.decoder;inner=outer.decoder;idx=inner.eval_idx;last=inner.layers[idx]
        assert idx==2 and inner.layer_scale==1 and not model.training
        utils=importlib.import_module('src.zoo.dome.dome_utils')
        assets=json.loads((BASE/'asset_selection.json').read_text(encoding='utf-8'))
        observed={};pred_dir=ROOT/'predictions';pred_dir.mkdir()
        capture_dir=ROOT/'captures';capture_dir.mkdir();gates=[]

        def journal(kind,case,arm=None):
            with (ROOT/'calls.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'kind':kind,
                    'case':case,'arm':arm,'full_forwards':calls,'suffix_forwards':partials})+'\n')
        def forward(image,case,kind):
            nonlocal calls
            assert calls<p['phase_full_forward_cap'];calls+=1;journal(kind,case)
            with torch.inference_mode():return model(image)
        def last_pre(module,args):
            assert len(args)==6
            observed['last_args']=(args[0].detach().clone(),args[1].detach().clone(),
                tuple(x.detach() for x in args[2]),args[3],args[4],args[5].detach().clone())
        def last_post(module,args,out):observed['last_output']=out.detach().clone()
        def integral_hook(module,args,out):
            observed.setdefault('corner_history',[]).append(args[0].detach().clone())
            observed['project']=args[1].detach().clone()
        def inner_post(module,args,out):observed['inner_output']=out
        def ca_pre(module,args):observed['ca_query']=args[0].detach().clone()
        handles=[]
        for case_index,case in enumerate(assets['cases']):
            path=BASE/'inputs'/'images'/case['file'];assert hash_file(path)==case['image_sha256']
            with Image.open(path) as im:image=transform(im.convert('RGB')).to('cuda')
            torch.manual_seed(p['seed']+case_index);torch.cuda.manual_seed_all(p['seed']+case_index)
            if case_index==0:
                out=forward(image,case_index,'UNINSTRUMENTED_IDENTITY')
                plain={k:out[k].detach().clone() for k in ['pred_boxes','pred_logits']};del out
                handles=[last.register_forward_pre_hook(last_pre),last.register_forward_hook(last_post),
                    outer.integral.register_forward_hook(integral_hook),inner.register_forward_hook(inner_post),
                    last.cross_attn.register_forward_pre_hook(ca_pre)]
            observed.clear()
            torch.manual_seed(p['seed']+case_index);torch.cuda.manual_seed_all(p['seed']+case_index)
            out=forward(image,case_index,'NATIVE_CAPTURE')
            if case_index==0:assert all(torch.equal(out[k],plain[k]) for k in plain)
            with np.load(BASE/'predictions'/('case%02d.npz'%case_index),allow_pickle=False) as cache:
                data={k:cache[k].copy() for k in ['raw_boxes','raw_logits','match_q','match_gt',
                    'gt_boxes800','gt_labels','gt_sizes800','gt_uids']}
            assert np.array_equal(out['pred_boxes'][0].cpu().numpy(),data['raw_boxes'])
            assert np.array_equal(out['pred_logits'][0].cpu().numpy(),data['raw_logits'])
            assert len(observed['corner_history'])==3
            # Remove observation hooks before replaying the final layer.
            for h in handles:h.remove()
            handles=[]
            args=observed['last_args'];target,reference,value,shapes,mask,pos=args
            assert reference.shape[2:]==(1,4) and mask is None
            prev=observed['corner_history'][-2];project=observed['project']
            ref_initial=observed['inner_output'][3][-1].detach().clone()
            native_last=observed['last_output'];native_ca=observed['ca_query']
            mq=data['match_q'];mg=data['match_gt'];tiny=data['gt_sizes800']<8
            selected=tiny[mg];qidx=torch.tensor(mq[selected],device='cuda',dtype=torch.long)
            gidx=mg[selected]
            gt=data['gt_boxes800']
            centres=torch.tensor((gt[gidx,:2]+gt[gidx,2:])/(2*800),device='cuda',dtype=reference.dtype)
            changed_reference=reference.clone();changed_reference[0,qidx,0,:2]=centres
            assert torch.equal(changed_reference[...,2:],reference[...,2:])
            changed_pos=outer.query_pos_head(changed_reference.squeeze(2)).clamp(min=-10,max=10)
            unchanged=torch.ones(target.shape[1],device='cuda',dtype=torch.bool);unchanged[qidx]=False
            permuted=[];rolls=[]
            for v,shape in zip(value,shapes):
                hh,ww=map(int,shape);shift=(hh//2+3,ww//2+5)
                rolls.append([shift[0]%hh,shift[1]%ww])
                permuted.append(torch.roll(v.reshape(*v.shape[:-1],hh,ww),shifts=shift,dims=(-2,-1)).reshape_as(v))
            arm_boxes=[];arm_logits=[];source_gates=[]
            for arm in ARMS:
                assert partials<p['phase_suffix_forward_cap'];partials+=1;journal('LAST_LAYER_SUFFIX',case_index,arm)
                ref=changed_reference if arm in ['SAMPLE_GT_CENTER','BOTH_GT_CENTER','SAMPLE_GT_CENTER_PERMUTED_VALUE'] else reference
                qpos=changed_pos if arm in ['POSITION_GT_CENTER','BOTH_GT_CENTER'] else pos
                vals=tuple(permuted) if 'PERMUTED_VALUE' in arm else value
                ca_ob={}
                def replay_ca_hook(module,aa):ca_ob['query']=aa[0].detach().clone()
                h=last.cross_attn.register_forward_pre_hook(replay_ca_hook)
                with torch.inference_mode():
                    suffix=last(target,ref,vals,shapes,mask,qpos)
                    corners=outer.dec_bbox_head[idx](suffix+target)+prev
                    boxes=utils.distance2bbox(ref_initial,outer.integral(corners,project),outer.reg_scale)
                    logits=inner.lqe_layers[idx](outer.dec_score_head[idx](suffix),corners)
                h.remove()
                assert torch.isfinite(boxes).all() and torch.isfinite(logits).all()
                if arm=='REPLAY_NATIVE':
                    assert torch.equal(suffix,native_last)
                    assert torch.equal(boxes,out['pred_boxes']) and torch.equal(logits,out['pred_logits'])
                if arm in ['REPLAY_NATIVE','SAMPLE_GT_CENTER','NATIVE_PERMUTED_VALUE','SAMPLE_GT_CENTER_PERMUTED_VALUE']:
                    assert torch.equal(ca_ob['query'],native_ca),'Source-only sampling intervention changed query/offset branch.'
                if arm=='SAMPLE_GT_CENTER':
                    assert torch.equal(boxes[:,unchanged],out['pred_boxes'][:,unchanged])
                    assert torch.equal(logits[:,unchanged],out['pred_logits'][:,unchanged])
                arm_boxes.append(boxes[0].cpu().numpy().copy());arm_logits.append(logits[0].cpu().numpy().copy())
                source_gates.append({'arm':arm,'cross_query_equal_native':torch.equal(ca_ob['query'],native_ca),
                    'nonpositive_boxes':int((boxes[0,:,2:]<=0).any(-1).sum().item())})
            # Save full FP32 memory and native final-layer inputs for source-faithful auditing.
            capture={'target':target[0].cpu().numpy(),'reference':reference[0].cpu().numpy(),
                'query_pos':pos[0].cpu().numpy(),'ca_query':native_ca[0].cpu().numpy(),
                'previous_corners':prev[0].cpu().numpy(),'fixed_initial_ref':ref_initial[0].cpu().numpy(),
                'project':project.cpu().numpy(),'spatial_shapes':np.asarray(shapes),
                'gt_query_indices':qidx.cpu().numpy(),'gt_centres':centres.cpu().numpy()}
            capture.update({'value_%d'%i:v[0].cpu().numpy() for i,v in enumerate(value)})
            np.savez_compressed(capture_dir/('case%02d.npz'%case_index),**capture)
            data.update({'arm_boxes':np.stack(arm_boxes),'arm_logits':np.stack(arm_logits),
                'prefix':np.asarray(case['prefix'])})
            np.savez_compressed(pred_dir/('case%02d.npz'%case_index),**data)
            gates.append({'case':case_index,'file':case['file'],'queries':int(target.shape[1]),
                'tiny_GT':int(tiny.sum()),'changed_queries':len(gidx),'value_rolls':rolls,
                'identity_exact':True,'sample_only_cross_query_exact':True,
                'nonintervened_query_outputs_exact':True,'arms':source_gates})
            write(ROOT/'progress.json',{'cases_completed':case_index+1,'full_model_forwards':calls,
                'suffix_forwards':partials,'full_cap':p['phase_full_forward_cap'],'suffix_cap':p['phase_suffix_forward_cap']})
            del out,image,observed,args,target,reference,value,pos,permuted,changed_reference,changed_pos
            observed={}
            if case_index<11:
                handles=[last.register_forward_pre_hook(last_pre),last.register_forward_hook(last_post),
                    outer.integral.register_forward_hook(integral_hook),inner.register_forward_hook(inner_post),
                    last.cross_attn.register_forward_pre_hook(ca_pre)]
        assert calls==13 and partials==72
        after=state_digest(model);assert after==before
        for f in freeze['files']:assert hash_file(ROOT/f['file'])==f['sha256']
        analysis=analyze(ROOT)
        summary={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED_BOUNDED_LAST_LAYER_SAMPLING_INTERVENTION',
            'new_full_model_forwards':calls,'last_layer_suffix_forwards':partials,
            'prior_cumulative_full_model_forwards':213,'cumulative_full_model_forwards':213+calls,
            'phase_full_forward_cap':13,'phase_suffix_forward_cap':72,'state_digest_before':before,
            'state_digest_after':after,'identity_exact':True,'gates':gates,'decision':analysis['decision'],
            'peak_own_GPU_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
            'GPU_free_before_model_bytes':free,'GPU_total_bytes':total,'new_fits':0,'main_model_updates':0,
            'mainline_validated':False,'scope':'GT-centre last-layer intervention,notnewalgorithmorAP.'}
        write(ROOT/'summary.json',summary)
        files=[f for f in ROOT.rglob('*') if f.is_file() and f.name!='completed.json']
        write(ROOT/'completed.json',{'status':'COMPLETED','full_model_forwards':calls,'suffix_forwards':partials,
            'files':[{'path':f.relative_to(ROOT).as_posix(),'sha256':hash_file(f),'bytes':f.stat().st_size} for f in files]})
        print(json.dumps({k:summary[k] for k in ['status','new_full_model_forwards',
            'last_layer_suffix_forwards','cumulative_full_model_forwards','decision','peak_own_GPU_allocated_GiB']}),flush=True)
    except BaseException as exc:
        write(ROOT/'failed.json',{'status':'FAILED_NO_AUTOMATIC_RETRY','utc':datetime.now(timezone.utc).isoformat(),
            'error':str(exc),'traceback':traceback.format_exc(),'full_model_forwards':calls,'suffix_forwards':partials})
        raise

if __name__=='__main__':main()
