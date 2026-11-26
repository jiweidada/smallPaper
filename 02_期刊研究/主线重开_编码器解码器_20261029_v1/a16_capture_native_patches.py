"""Native fixed-checkpoint capture on annotation-selected new prefixes; no suffix."""
from pathlib import Path
from datetime import datetime,timezone
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import ast,hashlib,importlib,json,sys,traceback
from typing import Dict
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from a16_spatial_operator import pack
ROOT=Path(__file__).resolve().parent
ASSETS=ROOT.parent/'dome_assets_v1'
sys.path.insert(0,str(ASSETS))
from dome_probe_loader import load_model,state_digest,hash_file
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    p=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    freeze=json.loads((ROOT/'freeze_manifest.json').read_text(encoding='utf-8'))
    for f in freeze['files']:assert hash_file(ROOT/f['file'])==f['sha256']
    assert hash_file(ASSETS/'dome_probe_loader.py')==p['loader_sha256']
    for rel,digest in p['source_hashes'].items():assert hash_file(ASSETS/'source'/rel)==digest
    assert json.loads((ROOT/'source_validation_before_capture.json').read_text())['status']=='PASS_SOURCE_CORE_CPU_DOUBLE_AGREEMENT'
    with (ROOT/'capture_started.json').open('x',encoding='utf-8') as f:json.dump({'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()},f)
    calls=0
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
        model,_,transform,load=load_model(ROOT/'model_load.json')
        assert load['state_digest']==p['state_digest']
        free,total=torch.cuda.mem_get_info();assert free>=2*1024**3
        model=model.to('cuda');before=state_digest(model);assert before==p['state_digest']
        last=model.decoder.decoder.layers[2]
        assert last.cross_attn.num_points_list==[4,4,4,4] and last.cross_attn.num_heads==8 and last.cross_attn.offset_scale==.5
        boxops=importlib.import_module('src.zoo.dome.box_ops')
        matcher_path=ASSETS/'source'/'src/zoo/dome/matcher.py'
        cls=next(n for n in ast.parse(matcher_path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='HungarianMatcher')
        cls.decorator_list=[]
        ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'np':np,'Dict':Dict,
            'linear_sum_assignment':linear_sum_assignment,'box_cxcywh_to_xyxy':boxops.box_cxcywh_to_xyxy,
            'generalized_box_iou':boxops.generalized_box_iou}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(matcher_path),'exec'),ns)
        matcher=ns['HungarianMatcher'](**p['matcher'])
        asset=json.loads((ROOT/'asset_selection.json').read_text(encoding='utf-8'))
        assert len(asset['cases'])==25 and len({c['prefix'] for c in asset['cases']})==25
        with np.load(ROOT/'final_sampler_state.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        for name in ['sampling_offsets','attention_weights']:
            layer=getattr(last.cross_attn,name)
            assert np.array_equal(layer.weight.detach().cpu().numpy(),state[name+'_weight'])
            assert np.array_equal(layer.bias.detach().cpu().numpy(),state[name+'_bias'])
        assert np.array_equal(last.cross_attn.num_points_scale.detach().cpu().numpy().reshape(-1),state['num_points_scale'].reshape(-1))
        outdir=ROOT/'new_patches';outdir.mkdir();observed={};gates=[];handle=None
        def hook(module,args):
            assert len(args)>=4
            observed['q']=args[0].detach().clone();observed['ref']=args[1].detach().clone()
            observed['value']=tuple(x.detach() for x in args[2]);observed['shapes']=np.asarray(args[3],dtype=np.int64)
        def forward(image,index,kind):
            nonlocal calls
            assert calls<p['phase_full_forward_cap'];calls+=1
            torch.manual_seed(p['seed']+index);torch.cuda.manual_seed_all(p['seed']+index)
            with (ROOT/'full_forward_calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'call':calls,'case':index,'kind':kind,'utc':datetime.now(timezone.utc).isoformat()})+'\n')
            with torch.inference_mode():return model(image)
        for index,case in enumerate(asset['cases']):
            path=ROOT/'inputs'/'images'/case['file'];assert hash_file(path)==case['image_sha256']
            with Image.open(path) as im:
                assert im.size==(case['width'],case['height']);image=transform(im.convert('RGB')).to('cuda')
            assert image.shape==(1,3,800,800) and image.dtype==torch.float32
            if index==0:
                plain=forward(image,index,'PLAIN_IDENTITY')
                plain={k:plain[k].detach().cpu().clone() for k in ['pred_boxes','pred_logits']}
                handle=last.cross_attn.register_forward_pre_hook(hook)
            observed.clear();out=forward(image,index,'NATIVE_PATCH_CAPTURE')
            raw_boxes=out['pred_boxes'].detach().cpu();raw_logits=out['pred_logits'].detach().cpu()
            assert torch.isfinite(raw_boxes).all() and torch.isfinite(raw_logits).all()
            if index==0:assert torch.equal(raw_boxes,plain['pred_boxes']) and torch.equal(raw_logits,plain['pred_logits'])
            shapes=observed['shapes'];assert shapes.tolist()==[[200,200],[100,100],[50,50],[25,25]]
            gt=np.asarray([g['box800'] for g in case['GT']],dtype=np.float32)
            labels=np.asarray([g['class_id'] for g in case['GT']],dtype=np.int64)
            sizes=np.asarray([g['size800'] for g in case['GT']],dtype=np.float64)
            norm=torch.tensor(gt/np.float32(800));targets=torch.cat(((norm[:,:2]+norm[:,2:])*.5,norm[:,2:]-norm[:,:2]),-1)
            valid=torch.nonzero((raw_boxes[0,:,2:]>0).all(-1)).flatten()
            match=matcher({'pred_boxes':raw_boxes[:,valid],'pred_logits':raw_logits[:,valid]},[{'boxes':targets,'labels':torch.tensor(labels)}])['indices'][0]
            mq=valid[match[0]].numpy();mg=match[1].numpy();tiny=sizes[mg]<8;qs=mq[tiny];gs=mg[tiny]
            assert len(qs)>0 and len(np.unique(qs))==len(qs) and len(np.unique(gs))==len(gs)
            q=observed['q'][0,qs].cpu().numpy();r=observed['ref'][0,qs].cpu().numpy()
            values=tuple(v[0].cpu().numpy() for v in observed['value'])
            data=pack(q,r,values,shapes,state)
            data.update(query_indices=qs,gt_indices=gs,raw_boxes=raw_boxes[0].numpy(),raw_logits=raw_logits[0].numpy(),
                gt_boxes800=gt,gt_labels=labels,gt_sizes800=sizes,gt_uids=np.asarray([g['uid'] for g in case['GT']]),
                prefix=np.asarray(case['prefix']),image_filename=np.asarray(case['file']))
            np.savez_compressed(outdir/('case%02d.npz'%index),**data)
            if index==0:
                full={'ca_query':q,'reference':r,'spatial_shapes':shapes}
                full.update({'value_%d'%l:v for l,v in enumerate(values)})
                np.savez_compressed(ROOT/'source_validation_case00.npz',**full)
            gates.append({'case':index,'file':case['file'],'prefix':case['prefix'],'queries':len(raw_boxes[0]),
                'valid_queries':len(valid),'GT':len(gt),'tiny_GT':int((sizes<8).sum()),
                'matched_tiny_GT':len(gs),'unmatched_tiny_GT':int((sizes<8).sum()-len(gs)),
                'instrumentation_nonmutating':True})
            write(ROOT/'capture_progress.json',{'cases_completed':index+1,'full_model_forwards':calls,'cap':26})
            del image,out,q,r,values,data
        handle.remove();after=state_digest(model);assert after==before
        assert calls==26
        for f in freeze['files']:assert hash_file(ROOT/f['file'])==f['sha256']
        summary={'status':'COMPLETED_BOUNDED_NATIVE_PATCH_CAPTURE','utc':datetime.now(timezone.utc).isoformat(),
            'new_full_model_forwards':calls,'phase_full_forward_cap':26,'cumulative_full_model_forwards':226+calls,
            'last_layer_suffix_forwards':0,'new_fits':0,'main_model_updates':0,'mainline_validated':False,
            'state_digest_before':before,'state_digest_after':after,'first_plain_identity_exact':True,
            'peak_own_GPU_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
            'GPU_free_before_model_bytes':free,'GPU_total_bytes':total,'cases':gates,
            'scope':'Native observation only. FP64 parameter reconstruction for patches; not a new detector,AP or training-feasibility measurement.'}
        write(ROOT/'capture_summary.json',summary)
        print(json.dumps({k:summary[k] for k in ['status','new_full_model_forwards','cumulative_full_model_forwards','peak_own_GPU_allocated_GiB']}),flush=True)
    except BaseException as exc:
        write(ROOT/'failed.json',{'status':'FAILED_NO_AUTOMATIC_RETRY','error':str(exc),'traceback':traceback.format_exc(),
            'new_full_model_forwards':calls,'cumulative_full_model_forwards':226+calls})
        raise
if __name__=='__main__':main()
