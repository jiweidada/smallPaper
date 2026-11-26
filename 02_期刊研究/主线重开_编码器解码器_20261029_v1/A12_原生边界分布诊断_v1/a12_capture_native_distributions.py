"""One-shot native observation under a separate13-forward cap; no model mutation."""
from pathlib import Path
from datetime import datetime,timezone
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import ast
import hashlib
import importlib
import json
import sys
import traceback
from typing import Dict
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from a12_distribution_analysis import analyze,write

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT.parent/'dome_assets_v1'
sys.path.insert(0,str(ASSETS))
from dome_probe_loader import load_model,state_digest,hash_file


def main():
    p=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'freeze_manifest.json').read_text(encoding='utf-8'))
    for r in frozen['files']:assert hash_file(ROOT/r['file'])==r['sha256']
    assert hash_file(ASSETS/'dome_probe_loader.py')==p['loader_sha256']
    for name,digest in p['source_hashes'].items():assert hash_file(ASSETS/'source'/name)==digest
    with (ROOT/'started.json').open('x',encoding='utf-8') as f:json.dump({'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()},f)
    calls=0
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2)
        model,_,transform,load=load_model(ROOT/'model_load.json')
        assert load['state_digest']==p['state_digest']
        free,total=torch.cuda.mem_get_info()
        assert free>=2*1024**3,'Less than2GiB free;no forward permitted.'
        model=model.to('cuda');before=state_digest(model)
        assert before==p['state_digest']
        config=load['config']
        assert bool(config.get('use_focal_loss',False))==p['matcher']['use_focal_loss']
        box_ops=importlib.import_module('src.zoo.dome.box_ops')
        utils=importlib.import_module('src.zoo.dome.dome_utils')
        matcher_path=ASSETS/'source'/'src/zoo/dome/matcher.py'
        cls=next(n for n in ast.parse(matcher_path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='HungarianMatcher')
        cls.decorator_list=[]  # registry decoration only; numerical class body unchanged.
        ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'np':np,'Dict':Dict,'linear_sum_assignment':linear_sum_assignment,'box_cxcywh_to_xyxy':box_ops.box_cxcywh_to_xyxy,'generalized_box_iou':box_ops.generalized_box_iou}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(matcher_path),'exec'),ns)
        matcher=ns['HungarianMatcher'](**p['matcher'])
        asset=json.loads((ROOT/'asset_selection.json').read_text(encoding='utf-8'))
        pred_dir=ROOT/'predictions';pred_dir.mkdir()
        observed={};plain=None;gates=[]
        def integral_hook(module,args,out):
            observed.setdefault('corners',[]).append(args[0][0].detach().cpu().numpy().copy())
            observed.setdefault('means',[]).append(out[0].detach().cpu().numpy().copy())
            observed.setdefault('grids',[]).append(args[1].detach().cpu().numpy().copy())
        def decoder_hook(module,args,out):
            observed['decoder_output']=[x.detach().cpu() for x in out]
        def call(image,seed,kind,index):
            nonlocal calls
            assert calls<p['phase_forward_cap']
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            calls+=1
            with (ROOT/'forward_calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'call':calls,'case':index,'kind':kind,'utc':datetime.now(timezone.utc).isoformat()})+'\n')
            with torch.inference_mode():return model(image)
        handles=[]
        for index,case in enumerate(asset['cases']):
            path=ROOT/'inputs'/'images'/case['file']
            assert hash_file(path)==case['image_sha256']
            with Image.open(path) as im:
                assert list(im.size)==[case['width'],case['height']]
                image=transform(im.convert('RGB')).to('cuda')
            assert image.shape==(1,3,800,800) and image.dtype==torch.float32
            seed=p['seed']+index
            if index==0:
                native=call(image,seed,'UNINSTRUMENTED_IDENTITY_CONTROL',index)
                plain={k:native[k].detach().cpu().clone() for k in ('pred_boxes','pred_logits')}
                handles=[model.decoder.integral.register_forward_hook(integral_hook),model.decoder.decoder.register_forward_hook(decoder_hook)]
                del native
            observed.clear()
            result=call(image,seed,'NATIVE_DISTRIBUTION_CAPTURE',index)
            if index==0:
                assert all(torch.equal(result[k].detach().cpu(),plain[k]) for k in plain),'Observation hook identity failed.'
            assert len(observed['corners'])==3
            dec=observed['decoder_output']
            raw_boxes=result['pred_boxes'].detach().cpu()
            raw_logits=result['pred_logits'].detach().cpu()
            assert torch.equal(raw_boxes,dec[0][-1]) and torch.equal(raw_logits,dec[1][-1])
            assert torch.isfinite(raw_boxes).all() and torch.isfinite(raw_logits).all()
            ref=dec[3][-1][0]
            assert torch.equal(ref,dec[4][0])
            decoded=[]
            for mean in observed['means']:
                decoded.append(utils.distance2bbox(ref,torch.tensor(mean),torch.tensor([4.])).numpy())
            max_error=float(np.max(np.abs(decoded[-1]-raw_boxes[0].numpy())))
            assert max_error<=2e-6
            assert all(np.array_equal(g,observed['grids'][0]) for g in observed['grids'])
            gt=np.asarray([r['box800'] for r in case['GT']],dtype=np.float32)
            labels=np.asarray([r['class_id'] for r in case['GT']],dtype=np.int64)
            gt_norm=torch.tensor(gt/np.float32(800))
            target_boxes=torch.cat(((gt_norm[:,:2]+gt_norm[:,2:])*.5,gt_norm[:,2:]-gt_norm[:,:2]),-1)
            valid=torch.nonzero((raw_boxes[0,:,2:]>0).all(-1)).flatten()
            matches=matcher({'pred_boxes':raw_boxes[:,valid],'pred_logits':raw_logits[:,valid]},[{'boxes':target_boxes,'labels':torch.tensor(labels)}])['indices'][0]
            mq=valid[matches[0]].numpy();mg=matches[1].numpy()
            np.savez_compressed(pred_dir/('case%02d.npz'%index),raw_boxes=raw_boxes[0].numpy(),raw_logits=raw_logits[0].numpy(),corner_logits=np.stack(observed['corners']).reshape(3,-1,4,33),source_means=np.stack(observed['means']),bin_grid=observed['grids'][0],ref_initial=ref.numpy(),decoded_boxes=np.stack(decoded),match_q=mq,match_gt=mg,gt_boxes800=gt,gt_labels=labels,gt_sizes800=np.asarray([r['size800'] for r in case['GT']]),gt_uids=np.asarray([r['uid'] for r in case['GT']]))
            gates.append({'case':index,'file':case['file'],'queries':raw_boxes.shape[1],'valid_queries':len(valid),'GT':len(gt),'matched_GT':len(mg),'distance2bbox_max_abs_error':max_error})
            write(ROOT/'capture_progress.json',{'cases_complete':index+1,'full_model_forwards':calls,'cap':p['phase_forward_cap']})
            del result,image
        for handle in handles:handle.remove()
        after=state_digest(model);assert after==before
        for r in frozen['files']:assert hash_file(ROOT/r['file'])==r['sha256']
        analysis=analyze(ROOT)
        summary={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED_NATIVE_DISTRIBUTION_CAPTURE','new_full_model_forwards':calls,'prior_cumulative_full_model_forwards':200,'cumulative_full_model_forwards':200+calls,'phase_forward_cap':p['phase_forward_cap'],'state_digest_before':before,'state_digest_after':after,'identity_max_difference':0.,'capture_gates':gates,'decision':analysis['decision'],'new_fits':0,'main_model_updates':0,'mainline_validated':False,'peak_own_GPU_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'GPU_free_before_model_bytes':free,'GPU_total_bytes':total,'scope':'Native frozen observation only;training resources,novelty and causal detection improvements untested.'}
        write(ROOT/'summary.json',summary)
        files=[p for p in ROOT.rglob('*') if p.is_file() and 'inputs' not in p.relative_to(ROOT).parts and p.name!='completed.json']
        write(ROOT/'completed.json',{'status':'COMPLETED','new_full_model_forwards':calls,'cumulative_full_model_forwards':200+calls,'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hash_file(p),'bytes':p.stat().st_size} for p in files]})
        print(json.dumps({k:summary[k] for k in ('status','new_full_model_forwards','cumulative_full_model_forwards','decision','peak_own_GPU_allocated_GiB')}),flush=True)
    except BaseException as exc:
        write(ROOT/'failed.json',{'utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED_NO_AUTOMATIC_RETRY','error':str(exc),'traceback':traceback.format_exc(),'new_full_model_forwards':calls,'cumulative_full_model_forwards':200+calls})
        raise


if __name__=='__main__':main()
