from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
ROOT=Path('/root/c10_region_restore_20261029_v1');PY=str(ROOT/'venv/bin/python')
START=time.monotonic();DEADLINE=START+48*3600
active=None
def save(name,obj):
    path=ROOT/'receipts'/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def terminate_owned(signum,frame):
    if active is not None and active.poll() is None:os.killpg(active.pid,signal.SIGTERM)
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminate_owned);signal.signal(signal.SIGINT,terminate_owned)
def run(label,command,limit):
    global active
    allowed=min(limit,DEADLINE-time.monotonic());assert allowed>0
    log=ROOT/'logs'/(label+'.log');started=time.monotonic()
    with log.open('w') as stream:
        active=subprocess.Popen(command,cwd=ROOT/'assets',env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save('queue_live.json',{'status':'RUNNING','stage':label,'pid':active.pid,'queue_pid':os.getpid(),'started_unix':time.time(),'limit_seconds':allowed})
        while active.poll() is None:
            if time.monotonic()-started>allowed:
                print('HARD_TIMEOUT '+label,flush=True);os.killpg(active.pid,signal.SIGTERM)
                try:active.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
                code=124;break
            time.sleep(5)
        else:code=active.returncode
    record={'stage':label,'command':command,'pid':active.pid,'exit_code':code,'elapsed_seconds':time.monotonic()-started,'log':str(log),'automatic_retry':False}
    save(label+'_receipt.json',record);active=None
    return code

try:
    assert (ROOT/'receipts/environment.json').exists() and (ROOT/'receipts/data_preparation.json').exists()
    save('code_frozen_at_launch.json',{'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'code').glob('*.py'))},
         'protocol_sha256':hashlib.sha256((ROOT/'experiment_protocol.json').read_bytes()).hexdigest(),'frozen_unix':time.time()})
    assert run('prepare_models',[PY,str(ROOT/'code/prepare_models.py')],300)==0
    batch=None
    for candidate in [8,4,2]:
        passed=True
        for arm in ['D0','DA']:
            label='preflight_'+arm+'_b'+str(candidate)
            code=run(label,[PY,str(ROOT/'code/run_train.py'),'--arm',arm,'--batch',str(candidate),'--preflight'],900)
            if code:
                failure=ROOT/'results'/label/'failure.json'
                if failure.exists() and json.loads(failure.read_text()).get('oom'):
                    passed=False;break
                raise RuntimeError(label+' failed; no automatic experiment retry')
        if passed:
            results=[json.loads((ROOT/'results'/('preflight_'+a+'_b'+str(candidate))/'preflight_result.json').read_text()) for a in ['D0','DA']]
            a,b=results[0]['batch_contracts'],results[1]['batch_contracts']
            assert [x['geometry_sha256'] for x in a]==[x['geometry_sha256'] for x in b],'Paired geometric RNG contract failed'
            assert any(x['pixel_sha256']!=y['pixel_sha256'] for x,y in zip(a,b)),'JPEG arm produced no pixel intervention'
            assert results[0]['actual_optimizer_updates']>0 and results[1]['actual_optimizer_updates']>0
            batch=candidate
            save('preflight_pair.json',{'status':'PASS','batch':batch,'steps_each_arm':8,'geometric_batch_hashes_identical':True,
                 'jpeg_pixels_differ':True,'raw_results':results,'scientific_hypothesis_validated':False});break
    assert batch is not None,'No common fitting batch; stop without formal training'
    save('formal_run_budget.json',{'arms':['D0','DA'],'epochs_each':100,'seed':0,'batch':batch,'sequential_single_gpu':True,'primary':'fixed100EMA','extra_module_search':False})
    for arm in ['D0','DA']:
        assert run('train_'+arm,[PY,str(ROOT/'code/run_train.py'),'--arm',arm,'--batch',str(batch)],24*3600)==0,arm+' failed; no restart'
        assert (ROOT/'results'/arm/'completed.json').exists()
    assert run('initial_diagnostic',[PY,str(ROOT/'code/evaluate_fixed.py')],2*3600)==0
    save('queue_completed.json',{'status':'COMPLETED_BASELINES_AND_INITIAL_DIAGNOSTIC','elapsed_seconds':time.monotonic()-START,
         'result':str(ROOT/'evaluation/initial_diagnostic_result.json'),'new_strategy_training':False,'region_restoration_stage_pending_gate_review':True})
    save('queue_live.json',{'status':'COMPLETED','queue_pid':os.getpid()})
except Exception as error:
    import traceback;traceback.print_exc()
    save('queue_live.json',{'status':'FAILED_OR_INCONCLUSIVE','queue_pid':os.getpid(),'reason':str(error),'automatic_retry':False})
    sys.exit(1)
