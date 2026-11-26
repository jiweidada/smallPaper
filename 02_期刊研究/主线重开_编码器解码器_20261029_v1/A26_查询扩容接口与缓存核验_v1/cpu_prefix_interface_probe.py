"""Three complete native decoder replays of one cached real-image memory.

Engineering identity check only. No images, GT, criterion, training or task AP.
"""
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
COUNTS={'decoder_entries':0,'decoder_completed':0,'validated_outputs':0,'layer_entries':0,'forbidden_full_model_attempts':0}
ARM=None
LAYERS={}

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def thash(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def atomic(name,value):
    path=ROOT/name;tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,path)
def event(value):
    with (ROOT/'calls.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'utc':utc(),**value},ensure_ascii=False)+'\n');f.flush()

def run():
    global ARM
    p=json.loads((ROOT/'cpu_interface_protocol_FROZEN.json').read_text(encoding='utf-8'))
    assert p['status']=='FROZEN_CPU_INTERFACE_ONLY'
    for row in p['files']:assert sha(ROOT/row['path'])==row['sha256']
    for row in p['dependencies']:assert sha(row['path'])==row['sha256']
    fd=os.open(str(ROOT/'one_shot.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,json.dumps({'pid':os.getpid(),'utc':utc()}).encode());os.close(fd)
    ticks=int(Path('/proc/self/stat').read_text().split()[21])
    atomic('started.json',{'pid':os.getpid(),'start_ticks':ticks,'utc':utc(),'protocol_sha256':sha(ROOT/'cpu_interface_protocol_FROZEN.json')})
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(p['seed'])
    assert not torch.cuda.is_initialized()
    spec=importlib.util.spec_from_file_location('a26_pinned_dome_loader',p['loader'])
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    model,_,_,loaded=loader.load_model(str(ROOT/'model_load.json'))
    assert loaded['state_digest']==p['state_digest']
    assert loaded['weight_sha256']==p['checkpoint_sha256']
    outer=model.decoder;inner=outer.decoder
    assert outer.query_select_method=='default' and len(inner.layers)==3 and inner.eval_idx==2
    assert inner.num_head==8 and not model.training
    before=loader.state_digest(model)
    handles=[]
    def forbidden(module,args):
        COUNTS['forbidden_full_model_attempts']+=1
        raise RuntimeError('Full model/backbone/encoder/outer forward is forbidden in this decoder-only phase')
    for module in [model,model.backbone,model.encoder,outer]:handles.append(module.register_forward_pre_hook(forbidden))
    def enter(module,args):
        assert COUNTS['decoder_entries']<3
        COUNTS['decoder_entries']+=1
        event({'event':'FULL_THREE_LAYER_DECODER_ENTER','arm':ARM,**COUNTS})
    handles.append(inner.register_forward_pre_hook(enter))
    def layer_enter(module,args):
        COUNTS['layer_entries']+=1;LAYERS[ARM]=LAYERS.get(ARM,0)+1
    for layer in inner.layers:handles.append(layer.register_forward_pre_hook(layer_enter))

    with np.load(p['cache_file'],allow_pickle=False) as z:
        shapes=z['spatial_shapes'].astype(np.int64).tolist()
        values=[z[f'value_{i}'] for i in range(4)]
    assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in values)
    memory_array=np.ascontiguousarray(np.concatenate(values,axis=-1).transpose(2,0,1).reshape(1,-1,256))
    assert hashlib.sha256(memory_array.tobytes()).hexdigest()==p['reconstructed_memory_sha256']
    memory=torch.from_numpy(memory_array)
    roundtrip=inner.value_op(memory,None,None,None,shapes)
    assert all(torch.equal(x[0],torch.from_numpy(v)) for x,v in zip(roundtrip,values))
    del roundtrip,values
    assert memory.shape==(1,53125,256)
    with torch.inference_mode():
        anchors,valid_mask=outer._generate_anchors(shapes,device=memory.device)
        encoded=outer.enc_output(valid_mask.to(memory.dtype)*memory)
        logits=outer.enc_score_head(encoded)
        content,_,selected_anchors=outer._select_topk(encoded,logits,anchors,500)
        ids=logits.max(-1).values.topk(500,dim=-1).indices
        assert torch.equal(content,encoded.gather(1,ids.unsqueeze(-1).expand(-1,-1,256)))
        refs=outer.enc_bbox_head(content)+selected_anchors
        content=content.detach();refs=refs.detach()
    assert torch.isfinite(content).all() and not torch.isnan(refs).any()
    fixed={'memory':thash(memory),'content500':thash(content),'refs500':thash(refs),'query_ids500':thash(ids)}
    selection={'query_ids':ids.tolist(),'selection':'Unchanged native encoder TopK500 and bbox-head operations; PAQI density filtering bypassed for a common fixed pool.',
               'initial_refs_nonfinite_count':int((~torch.isfinite(refs)).sum()),'input_hashes':fixed,
               'source_anchors_allow_positive_infinity':True,'memory_shape':list(memory.shape)}
    atomic('fixed_query_selection.json',selection)
    predictions={};timing={}
    try:
        for name,n,protected in p['arms']:
            ARM=name
            mask=None
            if protected:
                mask=torch.zeros((500,500),dtype=torch.bool)
                mask[:250,250:]=True
                assert int(mask.sum())==62500
            t=time.monotonic()
            with torch.inference_mode():
                output=inner(content[:,:n],refs[:,:n],memory,shapes,
                  outer.dec_bbox_head,outer.dec_score_head,outer.query_pos_head,outer.pre_bbox_head,
                  outer.integral,outer.up,outer.reg_scale,attn_mask=mask,dn_meta=None,img_input=None)
            COUNTS['decoder_completed']+=1
            boxes=output[0][-1].detach().cpu().numpy();scores=output[1][-1].detach().cpu().numpy()
            np.savez_compressed(ROOT/(name+'_raw_returned.npz'),boxes=boxes,logits=scores)
            assert boxes.shape==(1,n,4) and scores.shape==(1,n,12)
            assert np.isfinite(boxes).all() and np.isfinite(scores).all()
            predictions[name+'_boxes']=boxes;predictions[name+'_logits']=scores
            COUNTS['validated_outputs']+=1;timing[name]=time.monotonic()-t
            event({'event':'FULL_THREE_LAYER_DECODER_COMPLETE','arm':ARM,'queries':n,**COUNTS,'seconds':timing[name]})
            atomic('progress.json',{'status':'RUNNING',**COUNTS,'last_completed_arm':ARM})
            del output
    finally:
        for handle in handles:handle.remove()
    assert COUNTS['decoder_entries']==COUNTS['decoder_completed']==COUNTS['validated_outputs']==3
    assert COUNTS['layer_entries']==9 and all(v==3 for v in LAYERS.values())
    assert COUNTS['forbidden_full_model_attempts']==0 and not torch.cuda.is_initialized()
    inputs_after={'memory':thash(memory),'content500':thash(content),'refs500':thash(refs),'query_ids500':thash(ids)}
    after=loader.state_digest(model)
    atomic('state_and_input_integrity_observed.json',{'input_hashes_before':fixed,'input_hashes_after':inputs_after,
      'state_digest_before':before,'state_digest_after':after})
    assert inputs_after==fixed and before==after
    deltas={}
    for arm in ['NATIVE500','ONEWAY500']:
        deltas[arm]={kind:float(np.max(np.abs(predictions[arm+'_'+kind][:,:250]-predictions['BASE250_'+kind])))
                     for kind in ['boxes','logits']}
    np.savez_compressed(ROOT/'predictions_no_GT.npz',**predictions)
    atomic('fidelity_observed.json',{'prefix_max_absolute_differences':deltas,'absolute_tolerance':p['prefix_absolute_tolerance'],
      'predictions_sha256':sha(ROOT/'predictions_no_GT.npz'),**COUNTS})
    assert deltas['ONEWAY500']['boxes']<=p['prefix_absolute_tolerance']['boxes']
    assert deltas['ONEWAY500']['logits']<=p['prefix_absolute_tolerance']['logits']
    result={'utc':utc(),'status':'CPU_THREE_LAYER_PREFIX_INTERFACE_PASS',**COUNTS,
            'complete_detector_forwards':0,'optimizer_updates':0,'raw_image_forwards':0,
            'cached_real_image_contexts':1,'GT_arrays_read':False,'AP_computed':False,
            'CPU_threads':torch.get_num_threads(),'CUDA_initialized':torch.cuda.is_initialized(),
            'layer_calls_per_arm':LAYERS,'prefix_max_absolute_differences':deltas,
            'absolute_tolerance':p['prefix_absolute_tolerance'],'input_hashes':fixed,
            'state_digest_before':before,'state_digest_after':after,
            'per_arm_CPU_seconds':timing,'total_seconds':time.monotonic()-START,
            'peak_process_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'protocol_sha256':sha(ROOT/'cpu_interface_protocol_FROZEN.json'),
            'predictions_sha256':sha(ROOT/'predictions_no_GT.npz'),
            'interpretation':'Fidelity/interface result only. Unmasked output changes are not evidence of harmful task effects; one-way prefix preservation is a known mask property, not a new algorithm or AP gain.',
            'author_source_byte_edits':0,'old_training_restarted':False,'followup_jobs_launched':False}
    atomic('result.json',result);atomic('progress.json',{'status':result['status'],**COUNTS})
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':
    try:run()
    except BaseException:
        result={'utc':utc(),'status':'CPU_INTERFACE_STOPPED_NO_RETRY',**COUNTS,'traceback':traceback.format_exc(),
                'elapsed_seconds':time.monotonic()-START,'old_training_restarted':False,'followup_jobs_launched':False}
        atomic('failure.json',result);atomic('progress.json',{'status':result['status'],**COUNTS})
        raise
