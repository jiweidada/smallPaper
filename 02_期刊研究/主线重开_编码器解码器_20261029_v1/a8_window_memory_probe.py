"""Bounded, frozen-state direct-memory-path diagnostic. No training."""
from pathlib import Path
from datetime import datetime, timezone
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib
import json
import sys
import time
import traceback
import types
import numpy as np
import torch
from PIL import Image
from a8_analysis import masks_for_case,coverage_stats,diagnostic_counts,selfcheck

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT.parent/'dome_assets_v1'
sys.path.insert(0,str(ASSETS))
from dome_probe_loader import load_model,state_digest,hash_file

def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def tensor_digest(value):
    t=value.detach().cpu().contiguous()
    h=hashlib.sha256(str(t.dtype).encode()+str(tuple(t.shape)).encode()+t.numpy().tobytes())
    return h.hexdigest()

def clone_tree(x):
    if isinstance(x,torch.Tensor):return x.detach().clone()
    if isinstance(x,tuple):return tuple(clone_tree(v) for v in x)
    if isinstance(x,list):return [clone_tree(v) for v in x]
    return x

class Probe:
    def __init__(self,model,postprocessor,transform,protocol):
        self.model=model;self.pp=postprocessor;self.transform=transform;self.protocol=protocol
        self.calls=0;self.rows=[];self.cache={};self.case_index=None;self.arm=None;self.current={}
        self.prepare=model.encoder.mwas_processor._prepare_windows
        self.decoder_input=model.decoder._get_decoder_input
        probe=self
        def prepare_windows(module,features,mask,window_size):
            windows,native=probe.prepare(features,mask,window_size)
            c=probe.cache[probe.case_index]
            if probe.arm=='NATIVE':
                c['native_mask']=native[0].detach().cpu().numpy().copy()
                c['p3_digest']=tensor_digest(features);c['input_mask_digest']=tensor_digest(mask)
            else:
                assert tensor_digest(features)==c['p3_digest']
                assert tensor_digest(mask)==c['input_mask_digest']
                assert np.array_equal(native[0].detach().cpu().numpy(),c['native_mask'])
                native=torch.as_tensor(c['masks'][probe.arm][None],device=native.device,dtype=torch.bool)
            probe.current['mask']=native[0].detach().cpu().numpy().copy()
            return windows,native
        def get_decoder_input(module,memory,spatial_shapes,**kwargs):
            c=probe.cache[probe.case_index]
            cpu=memory.detach().cpu()
            probe.current['memory_digest']=tensor_digest(cpu)
            if probe.arm=='NATIVE':
                result=probe.decoder_input(memory,spatial_shapes,**kwargs)
                c['seeds']=clone_tree(result);c['memory_cpu']=cpu.clone()
                probe.current['memory_relative_change']=0.0
                return result
            delta=cpu-c['memory_cpu']
            probe.current['memory_relative_change']=float(delta.norm()/c['memory_cpu'].norm().clamp_min(1e-12))
            return clone_tree(c['seeds'])
        self.model.encoder.mwas_processor._prepare_windows=types.MethodType(prepare_windows,self.model.encoder.mwas_processor)
        self.model.decoder._get_decoder_input=types.MethodType(get_decoder_input,self.model.decoder)
        def window_input(module,args):
            c=probe.cache[probe.case_index];digest=tensor_digest(args[3])
            if probe.arm=='NATIVE':c['global_position_digest']=digest
            else:assert digest==c['global_position_digest']
        self.model.encoder.mwas_processor.register_forward_pre_hook(window_input)

    def call(self,index,arm,case,image):
        assert self.calls<self.protocol['counting']['maximum_new_full_detector_forwards']
        assert 185+self.calls<200
        self.case_index=index;self.arm=arm;self.current={}
        self.calls+=1
        entry={'utc':datetime.now(timezone.utc).isoformat(),'call':self.calls,'global_count':185+self.calls,
               'case':case['file_name'],'arm':arm,'stage':'ENTERED_FULL_DETECTOR_FORWARD'}
        with (ROOT/'forward_counter.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(entry)+'\n');f.flush();os.fsync(f.fileno())
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():result=self.model(image)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated()
        assert result['pred_logits'].shape[-1]==12
        assert torch.isfinite(result['pred_boxes']).all() and torch.isfinite(result['pred_logits']).all()
        raw_logits=result['pred_logits'];q=raw_logits.shape[1]
        assert 250<=q<=500 and q==self.cache[index]['seeds'][4][0]
        native=self.pp(result,torch.tensor([[800,800]],device=image.device))[0]
        scores,flat=torch.topk(raw_logits.sigmoid().flatten(1),q,dim=-1)
        assert torch.equal(native['scores'],scores[0])
        assert torch.equal(native['labels'],flat[0]%12)
        cpu=lambda x:x.detach().cpu().numpy()
        raw_boxes=cpu(result['pred_boxes'][0]);logits=cpu(raw_logits[0]);ids=cpu(flat[0])
        boxes=cpu(native['boxes']);labels=cpu(native['labels']);qids=ids//12
        cache=self.cache[index]
        if arm=='NATIVE':cache['raw_boxes']=raw_boxes.copy();cache['raw_logits']=logits.copy();cache['flat_ids']=ids.copy()
        if arm=='REPLAY_NATIVE':
            max_box=float(np.max(np.abs(raw_boxes-cache['raw_boxes'])))
            max_score=float(np.max(np.abs(logits-cache['raw_logits'])))
            assert max_box<=1e-6 and max_score<=1e-6
            assert np.array_equal(ids,cache['flat_ids'])
            self.current['identity']={'max_raw_box_difference':max_box,'max_raw_logit_difference':max_score,'flat_ids_exact':True}
        gt=cache['gt'];gl=cache['labels'];size=cache['sizes'];mask=self.current['mask']
        if arm not in ['NATIVE','ALL_WINDOWS']:assert mask.sum()==cache['native_mask'].sum()
        path=ROOT/'outputs'/f'case{index}_{arm}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,raw_boxes=raw_boxes,raw_logits=logits,flat_ids=ids,
             boxes800=boxes,labels=labels,query_ids=qids,scores=cpu(native['scores']),
             gt_boxes800=gt,gt_labels=gl,gt_sizes800=size,window_mask=mask)
        row={'case':case['file_name'],'arm':arm,'query_count':q,'output_file':path.name,
             'output_sha256':hash_file(path),'instrumented_model_seconds':elapsed,'own_peak_cuda_allocated_bytes':peak,
             'coverage':coverage_stats(mask,gt,size<8),'counts':diagnostic_counts(gt,gl,size,boxes,labels,qids),
             **{k:v for k,v in self.current.items() if k!='mask'}}
        self.rows.append(row)
        write_json(ROOT/'partial_rows.json',self.rows)
        return row

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'freeze_manifest.json').read_text(encoding='utf-8'))
    for row in frozen['files']:
        p=ASSETS/'dome_probe_loader.py' if row['file']=='dome_probe_loader.py' else ROOT/row['file']
        assert hash_file(p)==row['sha256'],row['file']
    assert protocol['status']=='FROZEN_BEFORE_FIRST_DOME_FORWARD'
    with (ROOT/'started.json').open('x',encoding='utf-8') as f:
        json.dump({'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
                   'protocol_sha256':hash_file(ROOT/'protocol.json'),'prior_full_forwards':185},f)
    probe=None
    try:
        write_json(ROOT/'synthetic_selfcheck.json',selfcheck())
        torch.set_num_threads(2);torch.manual_seed(20261030)
        np.random.seed(20261030)
        import random
        random.seed(20261030)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        model,pp,transform,load=load_model(ROOT/'model_load.json')
        assert load['state_digest']==protocol['strict_CPU_model_digest']
        assert torch.cuda.is_available()
        torch.cuda.manual_seed_all(20261030)
        free,total=torch.cuda.mem_get_info()
        assert free>=8*1024**3,'Less than8GiB free for the bounded inference preflight; no full forward entered.'
        write_json(ROOT/'runtime.json',{'torch':torch.__version__,'numpy':np.__version__,
          'device':torch.cuda.get_device_name(),'free_bytes_before_model_cuda':free,'total_bytes':total,
          'note':'Shared GPU; timings include diagnostic hooks and do not establish native speed or training-memory feasibility.'})
        model=model.to('cuda');pp=pp.to('cuda')
        before=state_digest(model)
        assert before==load['state_digest']
        cases=json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))['cases']
        probe=Probe(model,pp,transform,protocol)
        images=[]
        for i,case in enumerate(cases):
            path=Path('/root/c10_region_restore_20261029_v1/dataset/train/images')/case['file_name']
            assert hash_file(path)==case['image_metadata']['image_sha256']
            with Image.open(path) as im:
                assert list(im.size)==[case['image_metadata']['width'],case['image_metadata']['height']]
                image=transform(im.convert('RGB')).to('cuda')
            assert image.shape==(1,3,800,800) and image.dtype==torch.float32
            assert float(image.min())>=0 and float(image.max())<=1
            images.append(image)
            annotations=case['analysis_gt']
            probe.cache[i]={'gt':np.array([x['box800_xyxy'] for x in annotations],dtype=np.float64),
              'labels':np.array([x['class_id']+1 for x in annotations],dtype=np.int64),
              'sizes':np.array([x['size800_from640'] for x in annotations],dtype=np.float64),
              'input_sha256':tensor_digest(image)}
        for i,case in enumerate(cases):
            probe.call(i,'NATIVE',case,images[i])
            c=probe.cache[i]
            assert c['native_mask'].shape==(10,10)
            c['masks']=masks_for_case(c['native_mask'],c['gt'],c['sizes']<8,i)
        full=all(c['native_mask'].all() for c in probe.cache.values())
        if full:
            decision='ALL_NATIVE_MASKS_FULL_NO_ALLOCATION_CONTRAST_STOP_NO_MODULE_TRAINING'
        else:
            # Complete all identity controls before interpreting any changed mask.
            for i,case in enumerate(cases):probe.call(i,'REPLAY_NATIVE',case,images[i])
            for arm in ['RANDOM_SAME_K','GT_COVERAGE_SAME_K','ALL_WINDOWS']:
                for i,case in enumerate(cases):probe.call(i,arm,case,images[i])
            decision='CASE_DIAGNOSTIC_COMPLETE_NO_NOVELTY_OR_GENERALIZATION_CLAIM'
        after=state_digest(model);assert after==before
        for row in frozen['files']:
            p=ASSETS/'dome_probe_loader.py' if row['file']=='dome_probe_loader.py' else ROOT/row['file']
            assert hash_file(p)==row['sha256']
        summary={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED',
          'decision':decision,'cases':len(cases),'tiny_gt':sum(c['lt8_at_800'] for c in cases),
          'state_digest_before':before,'state_digest_after':after,'new_full_detector_forwards':probe.calls,
          'global_full_detector_forwards':185+probe.calls,'global_cap':200,
          'new_fits':0,'main_detector_updates':0,'rows':probe.rows,
          'scientific_mainline_validated':False,
          'limitations':['Three exploratory author-train cases, selected by GT size/count only; not an independent AP test.',
            'Fixed-query controlled memory-path effect excludes changes mediated through query selection or initialization.',
            'Each size cohort is matched separately with physical query capacity; cohort counts are not a joint partition or official recall/AP.',
            'GT coverage is a known oracle control, all-windows has a different budget, and neither is a new algorithm.',
            'GPU is shared and inference includes tracing overhead; training feasibility remains unverified.']}
        write_json(ROOT/'summary.json',summary)
        write_json(ROOT/'completed.json',{'status':'COMPLETED','summary_sha256':hash_file(ROOT/'summary.json'),
             'new_full_detector_forwards':probe.calls,'global_full_detector_forwards':185+probe.calls,
             'new_fits':0,'main_detector_updates':0})
        print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','limitations']},ensure_ascii=False),flush=True)
    except BaseException as error:
        calls=probe.calls if probe is not None else 0
        write_json(ROOT/'failed.json',{'utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED_NO_AUTOMATIC_RETRY',
          'error':str(error),'traceback':traceback.format_exc(),'new_full_detector_forwards':calls,
          'global_full_detector_forwards':185+calls,'new_fits':0,'main_detector_updates':0})
        raise

if __name__=='__main__':main()
