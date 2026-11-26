"""Frozen 45-call CPU decoder task. No GT, image encoder or training access."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
START=time.monotonic()
COUNTS={'decoder_entries':0,'decoder_completed':0,'validated_outputs':0,
        'layer_entries':0,'forbidden_full_model_attempts':0,'reused_A26_outputs':0}
CURRENT={}

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def atomic(name,value):
    p=ROOT/name;t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(t,p)
def event(kind,**extra):
    with (ROOT/'decoder_calls.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'utc':utc(),'event':kind,**CURRENT,**COUNTS,**extra})+'\n');f.flush()

def main():
    p=json.loads((ROOT/'task_protocol_FROZEN.json').read_text(encoding='utf-8'))
    assert p['status']=='FROZEN_EXPLORATORY_SAME_BUDGET_TASK'
    for row in p['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    for row in p['dependencies']:assert sha(row['path'])==row['sha256'],row['path']
    for case in p['cases']:assert sha(case['cache_file'])==case['cache_sha256']
    fd=os.open(str(ROOT/'decoder_one_shot.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,json.dumps({'pid':os.getpid(),'utc':utc()}).encode());os.close(fd)
    atomic('decoder_started.json',{'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().split()[21]),
      'utc':utc(),'protocol_sha256':sha(ROOT/'task_protocol_FROZEN.json')})
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(p['decoder_seed'])
    assert not torch.cuda.is_initialized()
    spec=importlib.util.spec_from_file_location('a27_pinned_loader',p['loader'])
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    model,_,_,loaded=loader.load_model(str(ROOT/'model_load.json'))
    assert loaded['state_digest']==p['state_digest'] and loaded['weight_sha256']==p['checkpoint_sha256']
    before=loader.state_digest(model);outer=model.decoder;inner=outer.decoder
    assert not model.training and len(inner.layers)==3 and inner.eval_idx==2 and inner.num_head==8
    assert outer.query_select_method=='default'
    handles=[];layers={}
    def forbidden(module,args):
        COUNTS['forbidden_full_model_attempts']+=1
        raise RuntimeError('Detector, backbone, encoder or outer forward forbidden')
    for module in [model,model.backbone,model.encoder,outer]:handles.append(module.register_forward_pre_hook(forbidden))
    def enter(module,args):
        assert COUNTS['decoder_entries']<p['new_complete_decoder_call_cap']
        assert time.monotonic()-START<p['decoder_wall_seconds']
        COUNTS['decoder_entries']+=1;event('COMPLETE_THREE_LAYER_DECODER_ENTER')
    handles.append(inner.register_forward_pre_hook(enter))
    def layer_enter(module,args):
        COUNTS['layer_entries']+=1
        key=str(CURRENT['case_index'])+':'+CURRENT['arm'];layers[key]=layers.get(key,0)+1
    for layer in inner.layers:handles.append(layer.register_forward_pre_hook(layer_enter))
    inventory=[];fidelity=[];outputs=[]
    a26_result=json.loads(Path(p['A26_result']).read_text(encoding='utf-8'))
    (ROOT/'predictions').mkdir(exist_ok=False)
    try:
        for case in p['cases']:
            CURRENT.clear();CURRENT.update(case_index=case['index'],file=case['file'])
            with np.load(case['cache_file'],allow_pickle=False) as z:
                shapes=z['spatial_shapes'].astype(np.int64).tolist()
                values=[z['value_'+str(i)] for i in range(4)]
            assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in values)
            array=np.ascontiguousarray(np.concatenate(values,axis=-1).transpose(2,0,1).reshape(1,-1,256))
            assert hashlib.sha256(array.tobytes()).hexdigest()==case['memory_sha256']
            memory=torch.from_numpy(array)
            roundtrip=inner.value_op(memory,None,None,None,shapes)
            assert all(torch.equal(x[0],torch.from_numpy(v)) for x,v in zip(roundtrip,values))
            del roundtrip,values
            with torch.inference_mode():
                anchors,valid=outer._generate_anchors(shapes,device=memory.device)
                encoded=outer.enc_output(valid.to(memory.dtype)*memory)
                logits=outer.enc_score_head(encoded)
                content,_,selected_anchors=outer._select_topk(encoded,logits,anchors,500)
                ids=logits.max(-1).values.topk(500,dim=-1).indices
                assert torch.equal(content,encoded.gather(1,ids[:,:,None].expand(-1,-1,256)))
                refs=outer.enc_bbox_head(content)+selected_anchors
                content=content.detach();refs=refs.detach()
            assert torch.isfinite(content).all() and not torch.isnan(refs).any()
            assert ids.unique().numel()==500
            fixed={'memory':thash(memory),'content500':thash(content),'refs500':thash(refs),'query_ids500':thash(ids)}
            if case['index']==0:assert fixed==a26_result['input_hashes']
            inventory.append({'case_index':case['index'],'file':case['file'],'input_hashes':fixed,
                              'query_ids500':ids.tolist(),'refs_infinite_count':int(torch.isinf(refs).sum())})
            pred={}
            for name in p['arms']:
                CURRENT['arm']=name;n=250 if name=='BASE250' else 500
                reused=case['index']==0 and name in p['reuse_A26_arms']
                if reused:
                    with np.load(p['A26_predictions'],allow_pickle=False) as z:
                        boxes=z[name+'_boxes'].copy();scores=z[name+'_logits'].copy()
                    COUNTS['reused_A26_outputs']+=1
                else:
                    mask=None
                    if name in ['ONEWAY500','TWOWAY500']:
                        mask=torch.zeros((500,500),dtype=torch.bool);mask[:250,250:]=True
                        if name=='TWOWAY500':mask[250:,:250]=True
                        assert int(mask.sum())==(125000 if name=='TWOWAY500' else 62500)
                    with torch.inference_mode():
                        out=inner(content[:,:n],refs[:,:n],memory,shapes,
                          outer.dec_bbox_head,outer.dec_score_head,outer.query_pos_head,
                          outer.pre_bbox_head,outer.integral,outer.up,outer.reg_scale,
                          attn_mask=mask,dn_meta=None,img_input=None)
                    COUNTS['decoder_completed']+=1
                    boxes=out[0][-1].detach().cpu().numpy();scores=out[1][-1].detach().cpu().numpy()
                    del out
                path=ROOT/'predictions'/('case%02d_%s.npz'%(case['index'],name))
                np.savez_compressed(path,boxes=boxes,logits=scores)
                assert boxes.shape==(1,n,4) and scores.shape==(1,n,12)
                assert np.isfinite(boxes).all() and np.isfinite(scores).all()
                pred[name]=(boxes,scores)
                if not reused:COUNTS['validated_outputs']+=1
                outputs.append({'case_index':case['index'],'arm':name,'path':str(path.relative_to(ROOT)),
                                'sha256':sha(path),'reused_A26_output':reused})
                event('OUTPUT_VALIDATED',reused_A26_output=reused)
                atomic('decoder_progress.json',{'status':'RUNNING',**CURRENT,**COUNTS})
            diffs={}
            for name in ['NATIVE500','TWOWAY500','ONEWAY500']:
                diffs[name]={kind:float(np.max(np.abs(pred[name][i][:,:250]-pred['BASE250'][i])))
                             for i,kind in enumerate(['boxes','logits'])}
            fidelity.append({'case_index':case['index'],'prefix_max_abs_differences':diffs})
            atomic('prefix_fidelity_observed.json',{'cases':fidelity,'tolerances':p['prefix_absolute_tolerance']})
            for name in ['TWOWAY500','ONEWAY500']:
                for kind in ['boxes','logits']:assert diffs[name][kind]<=p['prefix_absolute_tolerance'][kind],(case['index'],name,kind,diffs)
            assert {'memory':thash(memory),'content500':thash(content),'refs500':thash(refs),'query_ids500':thash(ids)}==fixed
            atomic('query_pool_inventory.json',{'cases':inventory})
            del memory,array,encoded,logits,content,refs,pred
    finally:
        for handle in handles:handle.remove()
    after=loader.state_digest(model)
    atomic('decoder_state_integrity.json',{'before':before,'after':after})
    assert before==after==p['state_digest']
    assert COUNTS['decoder_entries']==COUNTS['decoder_completed']==COUNTS['validated_outputs']==45
    assert COUNTS['layer_entries']==135 and len(layers)==45 and set(layers.values())=={3}
    assert COUNTS['reused_A26_outputs']==3 and len(outputs)==48
    assert COUNTS['forbidden_full_model_attempts']==0 and not torch.cuda.is_initialized()
    result={'utc':utc(),'status':'EXPLORATORY_DECODER_TASK_OUTPUTS_COMPLETE','protocol_sha256':sha(ROOT/'task_protocol_FROZEN.json'),
      **COUNTS,'outputs':outputs,'layers_per_new_arm':layers,'whole_detector_calls':0,'optimizer_updates':0,
      'GT_arrays_read':False,'AP_computed':False,'CUDA_initialized':False,'state_digest_before':before,
      'state_digest_after':after,'seconds':time.monotonic()-START,
      'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'author_source_edits':0,'old_training_restarted':False,'followup_jobs_launched':False}
    atomic('decoder_result.json',result);atomic('decoder_progress.json',{'status':result['status'],**COUNTS})
    print(json.dumps({k:v for k,v in result.items() if k not in ['outputs','layers_per_new_arm']}))

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic('decoder_failure.json',{'status':'STOPPED_NO_RETRY',**COUNTS,**CURRENT,
          'traceback':traceback.format_exc(),'seconds':time.monotonic()-START})
        raise
