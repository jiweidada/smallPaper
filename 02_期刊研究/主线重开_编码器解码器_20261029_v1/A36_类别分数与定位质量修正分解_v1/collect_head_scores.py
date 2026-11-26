"""12 CPU Linear calls on frozen features; no detector/decoder construction."""
from pathlib import Path
import hashlib,importlib.util,json,os,time,traceback,zipfile
import numpy as np
import torch
import torch.nn.functional as F
D=Path(__file__).resolve().parent;START=time.monotonic();COUNT=0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    p=D/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def main():
    global COUNT
    cfg=json.loads((D/'score_protocol_FROZEN.json').read_text());m=json.loads((D/'execution_inputs_manifest.json').read_text())
    assert cfg['status']=='FROZEN_LQE_SCORE_DECOMPOSITION_TWO_ARMS'
    for n,h in m['remote_files_sha256'].items():assert sha(D/n)==h,n
    assert sha(cfg['checkpoint'])==cfg['checkpoint_sha256']
    assert sha(cfg['native_prediction_reference'])==cfg['native_prediction_reference_sha256']
    for c in cfg['cases']:assert sha(c['capture'])==c['capture_sha256']
    fd=os.open(str(D/'one_shot.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    save('started.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21]))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_initialized()
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)['model']
    W=state['decoder.dec_score_head.2.weight'];bias=state['decoder.dec_score_head.2.bias']
    assert W.shape==(12,256) and bias.shape==(12,) and W.dtype==torch.float32
    np.savez_compressed(D/'head_parameters.npz',weight=W.numpy(),bias=bias.numpy())
    spec=importlib.util.spec_from_file_location('a36_pinned_metrics',str(D/'pinned_o27_metrics.py'))
    metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    reference=json.loads(Path(cfg['native_prediction_reference']).read_text())
    predictions={k:[] for k in cfg['arms']};proof=[];(D/'captures').mkdir(exist_ok=False)
    for c in cfg['cases']:
        with np.load(c['capture'],allow_pickle=False) as z:
            feature=z['depth3_features'].copy();boxes=z['depth3_boxes'].copy();native=z['depth3_logits'].copy();native_prob=z['depth3_probabilities'].copy()
        assert feature.shape==(1,500,256) and native.shape==(1,500,12)
        assert COUNT<12 and time.monotonic()-START<cfg['caps']['wall_seconds']
        COUNT+=1
        with torch.inference_mode():raw=F.linear(torch.from_numpy(feature),W,bias).numpy()
        assert np.isfinite(raw).all()
        delta=native.astype(np.float64)-raw.astype(np.float64)
        spread=float(np.ptp(delta,axis=-1).max())
        bound=float(8*np.finfo(np.float32).eps*max(1,float(np.abs(native).max()),float(np.abs(raw).max())))
        assert spread<=bound,(c['index'],spread,bound)
        rawprob=torch.from_numpy(raw).sigmoid().numpy()
        assert np.array_equal(torch.from_numpy(native).sigmoid().numpy(),native_prob)
        np.savez_compressed(D/'captures'/('case%02d.npz'%c['index']),boxes=boxes,native_logits=native,raw_logits=raw,
                            native_probabilities=native_prob,raw_probabilities=rawprob,offset_per_class=delta)
        for arm,logits in [('NATIVE',native),('RAW_CLASS',raw)]:
            p=metric.postprocess_fixed(boxes,logits,dict(width=c['width'],height=c['height'],integral=None),cap=500)
            row=dict(index=c['index'],file=c['file'],**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items()})
            if arm=='NATIVE':assert row==reference[c['index']]
            predictions[arm].append(row)
        proof.append(dict(index=c['index'],native_predictions_exact=True,class_offset_spread=spread,rounding_bound=bound,
                          argmax_disagreements=int((native.argmax(-1)!=raw.argmax(-1)).sum())))
        save('status.json',dict(status='RUNNING',head_linear_calls=COUNT))
    assert COUNT==12 and not torch.cuda.is_initialized()
    assert sha(cfg['checkpoint'])==cfg['checkpoint_sha256']
    for c in cfg['cases']:assert sha(c['capture'])==c['capture_sha256']
    save('predictions_without_region_filter.json',predictions)
    save('head_result.json',dict(status='COLLECTION_COMPLETE',head_linear_calls=COUNT,full_model_calls=0,decoder_calls=0,
         optimizer_updates=0,GPU_calls=0,fidelity=proof,elapsed_seconds=time.monotonic()-START,
         scope='Functional head ablation at the pinned joint-trained parameterization;not calibrated class/IoU probabilities.'))
    excluded={'file_manifest.json','one_shot.lock','native_score_evidence.zip','complete.json','status.json'}
    files=[p for p in D.rglob('*') if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts]
    save('file_manifest.json',dict(files_sha256={str(p.relative_to(D)).replace('\\','/'):sha(p) for p in files}))
    with zipfile.ZipFile(D/'native_score_evidence.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files+[D/'file_manifest.json']:z.write(p,str(p.relative_to(D)).replace('\\','/'))
    result=dict(status='COMPLETED',head_linear_calls=COUNT,archive_bytes=(D/'native_score_evidence.zip').stat().st_size,
                archive_sha256=sha(D/'native_score_evidence.zip'),archived_files=len(files)+1,elapsed_seconds=time.monotonic()-START)
    save('complete.json',result);save('status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('exception.json',dict(error=repr(e),traceback=traceback.format_exc(),head_linear_calls=COUNT))
        save('status.json',dict(status='FAILED_NO_RETRY',head_linear_calls=COUNT,error=repr(e)));raise
