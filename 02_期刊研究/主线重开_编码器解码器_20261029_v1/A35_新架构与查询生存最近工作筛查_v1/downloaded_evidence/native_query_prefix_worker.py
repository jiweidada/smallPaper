"""One-shot CPU replay of original decoder prefixes; no GT or image forward."""
from pathlib import Path
import datetime, hashlib, importlib.util, io, json, os, time, traceback, zipfile
import numpy as np
import torch

D=Path(__file__).resolve().parent
START=time.monotonic()
COUNTS=dict(decoder_entries=0,decoder_completed=0,complete_three_layer_calls=0,
            partial_prefix_calls=0,decoder_layer_entries=0,query_initializers=0,
            whole_detector_forwards=0,image_encoder_forwards=0,optimizer_updates=0)
CURRENT={}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,x):
    p=D/name;t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(t,p)
def event(kind):
    with (D/'calls.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=utc(),kind=kind,current=CURRENT.copy(),counts=COUNTS.copy()))+'\n');f.flush()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);return x

def main():
    cfg=json.loads((D/'native_prefix_protocol_FROZEN.json').read_text())
    manifest=json.loads((D/'execution_inputs_manifest.json').read_text())
    assert cfg['status']=='FROZEN_NATIVE_PREFIX_SUPPORT_DIAGNOSIS'
    for name,h in manifest['remote_files_sha256'].items():assert sha(D/name)==h,name
    for x in cfg['dependencies']:assert sha(x['path'])==x['sha256'],x['path']
    for c in cfg['cases']:assert sha(c['cache_file'])==c['cache_sha256'],c['cache_file']
    with zipfile.ZipFile(D/'native500_reference.zip') as z:
        assert sorted(z.namelist())==sorted(c['reference_entry'] for c in cfg['cases'])
        for c in cfg['cases']:
            assert hashlib.sha256(z.read(c['reference_entry'])).hexdigest()==c['reference_sha256']
    save('inputs_verified.json',dict(status='ALL_FROZEN_INPUT_HASHES_PASS_BEFORE_DECODER_CALLS',utc=utc(),counts=COUNTS.copy()))
    fd=os.open(str(D/'one_shot.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,json.dumps(dict(pid=os.getpid(),utc=utc())).encode());os.close(fd)
    save('started.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],utc=utc()))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(cfg['seed'])
    assert not torch.cuda.is_initialized()
    loader=module('a35_pinned_dome_loader',cfg['loader'])
    metric=module('a35_pinned_metric',str(D/'pinned_o27_metrics.py'))
    model,_,_,loaded=loader.load_model(str(D/'model_load.json'))
    assert loaded['state_digest']==cfg['state_digest'] and loaded['weight_sha256']==cfg['checkpoint_sha256']
    model.requires_grad_(False)
    before=loader.state_digest(model);outer=model.decoder;inner=outer.decoder
    assert not model.training and not inner.training and len(inner.layers)==3 and inner.eval_idx==2
    assert inner.layer_scale==1 and outer.query_select_method=='default'
    assert all(isinstance(x,torch.nn.Linear) for x in outer.dec_score_head)
    handles=[];trace=[]
    def forbid(m,args):raise RuntimeError('Full detector, backbone, image encoder and outer decoder calls forbidden')
    for m in [model,model.backbone,model.encoder,outer]:handles.append(m.register_forward_pre_hook(forbid))
    def enter(m,args):
        assert COUNTS['decoder_entries']<cfg['caps']['decoder_entries']
        assert time.monotonic()-START<cfg['caps']['wall_seconds']
        COUNTS['decoder_entries']+=1;event('PREFIX_ENTER')
    handles.append(inner.register_forward_pre_hook(enter))
    def layer_enter(m,args):COUNTS['decoder_layer_entries']+=1
    def layer_out(m,args,out):trace.append(out.detach().cpu().clone())
    for m in inner.layers:
        handles.append(m.register_forward_pre_hook(layer_enter))
        handles.append(m.register_forward_hook(layer_out))
    (D/'captures').mkdir(exist_ok=False)
    fidelity=[];inventory=[];raw_predictions=[];first_input=None

    def replay(content,refs,memory,shapes,depth):
        trace.clear();inner.eval_idx=depth-1;CURRENT['depth']=depth
        try:
            with torch.inference_mode():
                out=inner(content,refs,memory,shapes,outer.dec_bbox_head,outer.dec_score_head,
                          outer.query_pos_head,outer.pre_bbox_head,outer.integral,outer.up,
                          outer.reg_scale,attn_mask=None,dn_meta=None,img_input=None)
        finally:inner.eval_idx=2
        COUNTS['decoder_completed']+=1
        COUNTS['complete_three_layer_calls' if depth==3 else 'partial_prefix_calls']+=1
        assert len(trace)==depth
        boxes=out[0][-1].detach().cpu().numpy().copy()
        logits=out[1][-1].detach().cpu().numpy().copy()
        assert boxes.shape==(1,500,4) and logits.shape==(1,500,12)
        assert np.isfinite(boxes).all() and np.isfinite(logits).all()
        probs=torch.from_numpy(logits).sigmoid()
        rank=torch.argsort(probs[0,:,1:11].amax(-1),descending=True,stable=True).numpy()
        result=dict(boxes=boxes,logits=logits,probabilities=probs.numpy(),rank_valid=rank,
                    features=trace[-1].numpy().copy())
        event('PREFIX_COMPLETE')
        save('status.json',dict(status='RUNNING',current=CURRENT.copy(),counts=COUNTS.copy(),elapsed_seconds=time.monotonic()-START))
        return result,[x.clone() for x in trace]

    try:
        for c in cfg['cases']:
            CURRENT.clear();CURRENT.update(case_index=c['index'],file=c['file'])
            with np.load(c['cache_file'],allow_pickle=False) as z:
                shapes=z['spatial_shapes'].astype(np.int64).tolist()
                values=[z['value_'+str(i)] for i in range(4)]
            memory_array=np.ascontiguousarray(np.concatenate(values,axis=-1).transpose(2,0,1).reshape(1,-1,256))
            assert hashlib.sha256(memory_array.tobytes()).hexdigest()==c['memory_sha256']
            memory=torch.from_numpy(memory_array)
            with torch.inference_mode():
                anchors,valid=outer._generate_anchors(shapes,device=memory.device)
                encoded=outer.enc_output(valid.to(memory.dtype)*memory)
                enc_logits=outer.enc_score_head(encoded)
                content,_,selected=outer._select_topk(encoded,enc_logits,anchors,500)
                ids=enc_logits.max(-1).values.topk(500,dim=-1).indices
                assert torch.equal(content,encoded.gather(1,ids[:,:,None].expand(-1,-1,256)))
                refs=(outer.enc_bbox_head(content)+selected).detach();content=content.detach()
            COUNTS['query_initializers']+=1
            fixed=dict(memory=thash(memory),content=thash(content),refs=thash(refs))
            three,full_trace=replay(content,refs,memory,shapes,3)
            with zipfile.ZipFile(D/'native500_reference.zip') as z:
                with np.load(io.BytesIO(z.read(c['reference_entry'])),allow_pickle=False) as ref:
                    assert np.array_equal(three['boxes'],ref['boxes'])
                    assert np.array_equal(three['logits'],ref['logits'])
            outputs={3:three};prefix_checks=[]
            for depth in [1,2]:
                pred,part_trace=replay(content,refs,memory,shapes,depth)
                assert all(torch.equal(part_trace[j],full_trace[j]) for j in range(depth))
                prefix_checks.append(dict(depth=depth,all_prefix_features_bitwise_equal=True))
                outputs[depth]=pred
            arrays=dict(encoder_query_ids=ids.numpy())
            arrays['full_call_layer1_features']=full_trace[0].numpy().copy()
            arrays['full_call_layer2_features']=full_trace[1].numpy().copy()
            for depth,pred in outputs.items():
                arrays.update({f'depth{depth}_{key}':value for key,value in pred.items()})
            path='captures/case%02d.npz'%c['index']
            np.savez_compressed(D/path,**arrays)
            decoded=metric.postprocess_fixed(three['boxes'],three['logits'],dict(width=c['width'],height=c['height'],integral=None),cap=500)
            raw_predictions.append(dict(index=c['index'],file=c['file'],**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in decoded.items()}))
            assert fixed==dict(memory=thash(memory),content=thash(content),refs=thash(refs))
            fidelity.append(dict(index=c['index'],reference_head_arrays_bitwise_equal=True,prefix_checks=prefix_checks,inputs_unchanged=True))
            inventory.append(dict(index=c['index'],file=c['file'],path=path,sha256=sha(D/path),input_hashes=fixed))
            if c['index']==0:first_input=(content,refs,memory,shapes,three)
            del values,memory_array,encoded,enc_logits,outputs,arrays,full_trace
        CURRENT.clear();CURRENT.update(case_index=0,repeat=True)
        content,refs,memory,shapes,expected=first_input
        repeated,_=replay(content,refs,memory,shapes,3)
        assert all(np.array_equal(repeated[k],expected[k]) for k in expected)
        np.savez_compressed(D/'repeat_first_full.npz',**repeated)
    finally:
        inner.eval_idx=2
        for h in handles:h.remove()
    assert before==loader.state_digest(model) and not torch.cuda.is_initialized()
    assert COUNTS==dict(decoder_entries=37,decoder_completed=37,complete_three_layer_calls=13,
                       partial_prefix_calls=24,decoder_layer_entries=75,query_initializers=12,
                       whole_detector_forwards=0,image_encoder_forwards=0,optimizer_updates=0)
    for x in cfg['dependencies']:assert sha(x['path'])==x['sha256'],x['path']
    for c in cfg['cases']:assert sha(c['cache_file'])==c['cache_sha256']
    save('raw_native_predictions_without_region_filter.json',raw_predictions)
    save('fidelity.json',dict(cases=fidelity,native_repeat_bitwise_equal=True,weights_unchanged=True,cuda_initialized=False))
    save('capture_inventory.json',dict(cases=inventory))
    save('result.json',dict(status='NATIVE_PREFIX_COLLECTION_COMPLETE',utc=utc(),counts=COUNTS.copy(),
                           cases=12,state_digest=before,elapsed_seconds=time.monotonic()-START,
                           scope='Native trajectory/support diagnosis; not actual pruned-decoder inference or new algorithm evaluation.'))
    excluded={'native_evidence.zip','file_manifest.json','status.json','complete.json','one_shot.lock'}
    files=[p for p in D.rglob('*') if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts]
    file_hashes={str(p.relative_to(D)).replace('\\','/'):sha(p) for p in files}
    save('file_manifest.json',dict(files_sha256=file_hashes))
    with zipfile.ZipFile(D/'native_evidence.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files+[D/'file_manifest.json']:z.write(p,str(p.relative_to(D)).replace('\\','/'))
    done=dict(status='COMPLETED',counts=COUNTS.copy(),archive_sha256=sha(D/'native_evidence.zip'),
              archive_bytes=(D/'native_evidence.zip').stat().st_size,archived_files=len(files)+1,
              elapsed_seconds=time.monotonic()-START)
    save('complete.json',done);save('status.json',done)
    print(json.dumps(done),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('exception.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=COUNTS.copy(),current=CURRENT.copy(),utc=utc()))
        save('status.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',counts=COUNTS.copy(),error=repr(e)))
        raise
