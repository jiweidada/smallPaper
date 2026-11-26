"""One post-training saved-candidate verification, zero baseline or training calls."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time,traceback


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False)


def verify_tools():
    here=Path(__file__).resolve().parent
    protocol=json.loads((here/'protocol.json').read_text())
    assert protocol['runtime_version']==5 and protocol['candidate_variant']=='degraded'
    for name,h in protocol['files'].items():
        assert hashlib.sha256((here/name).read_bytes()).hexdigest()==h,name


def assert_trainers_stopped(screen):
    for name,pk,tk in [('controller_start.json','pid','start_ticks'),('candidate_start.json','child_pid','child_start_ticks')]:
        record=json.loads((screen/name).read_text());proc=Path(f"/proc/{record[pk]}/stat")
        if proc.exists():
            fields=proc.read_text().split()
            assert fields[21]!=str(record[tk]) or fields[2]=='Z','Owned trainer or supervisor is still live'


def worker(bundle):
    verify_tools()
    code=bundle.parent/'code';sys.path.insert(0,str(code))
    from runtime import verify_bundle,read,sha,activate,register_model,resolved_config
    receipt,plan,paths=verify_bundle(bundle)
    assert plan['variant']=='degraded' and receipt['identity']['arm']=='degraded'
    assert_trainers_stopped(paths['screen_root'])
    original=paths['output'];out=paths['screen_root']/'candidate_reload_v1'
    done=read(original/'training_completion.json')
    assert done['status']=='NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs']==12
    assert done['entered_epochs']==list(range(12)) and done['identity']==receipt['identity']
    assert read(paths['screen_root']/'controller_completion.json')['status']=='COMPLETE_FIXED12'
    assert sha(original/'last.pth')==done['last_checkpoint_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    expected_gpu=read(code/'protocol.json')['GPU_uuid'][str(plan['seed'])]
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==expected_gpu
    from results import load_endpoint,capture
    primary,primary_hits,_=load_endpoint(original/'endpoint_epoch12')
    assert sha(original/'endpoint_epoch12/manifest.json')==done['endpoint_manifest_sha256']
    from stage_trace import StageTrace,TracedLoader,verify_trace
    verify_trace(original/'stages_epoch12',receipt['identity'])
    activate(paths['native_root'])
    from evaluator_binding import activate as bind_evaluator
    bind_evaluator(paths['evaluator_root'],receipt['identity']['evaluator_binding']);register_model()
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver.det_engine import evaluate
    dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
    config=resolved_config(plan,dict(paths,output=out));write(out/'reload_config.yml',config)
    cfg=YAMLConfig(str(out/'reload_config.yml'));model=cfg.model.cuda()
    payload=torch.load(original/'last.pth',map_location='cpu',weights_only=False)
    assert payload['last_epoch']==11 and payload['small_object_encoder_decoder12_identity']==receipt['identity']
    # Full strict EMA reload: never use load_common or recopy the semantic branch.
    model.load_state_dict(payload['ema']['module'],strict=True)
    loaded=model.state_dict();expected=payload['ema']['module']
    assert loaded.keys()==expected.keys()
    assert all(torch.equal(v.detach().cpu(),expected[n].cpu()) for n,v in loaded.items())
    tensor_count=len(loaded);del payload,loaded,expected
    model.requires_grad_(False);assert not any(p.requires_grad for p in model.parameters())
    criterion=cfg.criterion.cuda();criterion.set_model(model);criterion.set_epoch(11)
    post=cfg.postprocessor.cuda();loader=cfg.val_dataloader;evaluator=cfg.evaluator
    trace=StageTrace(model,receipt['identity'],paths['val_ann']);model.eval()
    started=time.monotonic()
    with torch.no_grad():
        stats,obj=evaluate(model,criterion,post,TracedLoader(loader,trace),evaluator,
            torch.device('cuda:0'),test_only=False,output_dir=out,yolo_metrice=config['yolo_metrice'])
    assert obj is evaluator
    trace.finish(out/'stages_epoch12')
    capture(evaluator.coco_eval['bbox'],stats,receipt['identity'],paths['val_ann'],out/'endpoint_epoch12',epoch=11,
            environment={'torch':str(torch.__version__),'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0)})
    replay,replay_hits,_=load_endpoint(out/'endpoint_epoch12')
    error=max(abs(a-b) for a,b in zip(primary['native_stats'],replay['native_stats']))
    hit_equal=all(np.array_equal(primary_hits[n],replay_hits[n]) for n in primary_hits)
    fields={};all_equal=True
    for iid in range(548):
        name=f'image_{iid:06d}.npz'
        with np.load(original/'stages_epoch12'/name,allow_pickle=False) as a,np.load(out/'stages_epoch12'/name,allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                assert a[key].shape==b[key].shape
                exact=np.array_equal(a[key],b[key]);all_equal=all_equal and exact
                fields[key]=max(fields.get(key,0),float(np.max(np.abs(a[key].astype(np.float64)-b[key].astype(np.float64)))))
    result={'status':'PASS_SAVED_CANDIDATE_RELOAD' if error<=1e-8 and hit_equal and all_equal else 'CANDIDATE_RELOAD_MISMATCH_NO_RETRY',
        'seed':plan['seed'],'variant':'degraded','checkpoint_sha256':done['last_checkpoint_sha256'],'strict_loaded_tensors':tensor_count,
        'original_training_completion_sha256':sha(original/'training_completion.json'),
        'original_stats':primary['native_stats'],'reloaded_stats':replay['native_stats'],
        'stats_max_abs_fraction_difference':error,'stats_tolerance':1e-8,'native_GT_hits_exact':hit_equal,
        'all548_stage_arrays_exact':all_equal,'stage_max_abs_difference':fields,
        'images':548,'candidate_model_forwards':137,'baseline_model_forwards':0,'optimizer_updates':0,'epochs_added':0,
        'evaluation_seconds_with_trace_overhead':time.monotonic()-started,'automatic_retry':False,
        'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'completed_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(out/'verification.json',result)
    assert result['status']=='PASS_SAVED_CANDIDATE_RELOAD','Keep original and reload outputs; no auto retry or reference replacement'
    print(json.dumps({'status':result['status'],'seed':plan['seed'],'all548_stage_arrays_exact':all_equal}),flush=True)


def supervise(bundle):
    verify_tools()
    root=bundle.parent
    receipt=json.loads((bundle/'receipt.json').read_text());screen=Path(receipt['paths']['screen_root'])
    assert json.loads((screen/'controller_completion.json').read_text())['status']=='COMPLETE_FIXED12'
    assert_trainers_stopped(screen)
    assert receipt['identity']['arm']=='degraded'
    out=screen/'candidate_reload_v1';out.mkdir(exist_ok=False)
    started=time.monotonic();timeout=False
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--bundle',str(bundle),'--worker']
    with (out/'worker.log').open('xb') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
        write(out/'start.json',{'pid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split()[21],
            'controller_pid':os.getpid(),'command':command,'timeout_seconds':1800,'automatic_retry':False})
        while True:
            try:exit_code=child.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                if time.monotonic()-started<1800:continue
                timeout=True;os.killpg(child.pid,signal.SIGTERM)
                try:exit_code=child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);exit_code=child.wait()
                break
    write(out/'completion.json',{'exit_code':exit_code,'timed_out':timeout,'elapsed_seconds':time.monotonic()-started,'automatic_retry':False})
    if exit_code!=0 or timeout:raise RuntimeError('Candidate checkpoint verification failed; no auto retry')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--worker',action='store_true')
    args=p.parse_args();bundle=Path(args.bundle).resolve()
    if args.worker:worker(bundle)
    else:supervise(bundle)
if __name__=='__main__':main()
