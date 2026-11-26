"""Bounded one-shot execution of the explicitly authorized two-arm protocol."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time,signal,importlib.metadata as metadata

ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def save(v):
    tmp=ROOT/'queue.json.tmp';tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,ROOT/'queue.json')
def start_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
protocol=json.loads((ROOT/'execution_protocol.json').read_text())
environment=dict(os.environ,PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
assert protocol['authorized'] and protocol['arms']==['BASE','FULL'] and protocol['epochs']==320
hard_timeout=int(protocol['budget']['hard_timeout_seconds_per_arm'])
assert hard_timeout==604800
with (ROOT/'one_shot.lock').open('x') as f:
    f.write(json.dumps({'started_utc':now(),'supervisor_pid':os.getpid(),'start_ticks':start_ticks(os.getpid())})+'\n')
state={'status':'RUNNING','started_utc':now(),'supervisor_pid':os.getpid(),'supervisor_start_ticks':start_ticks(os.getpid()),
       'protocol_sha256':hashlib.sha256((ROOT/'execution_protocol.json').read_bytes()).hexdigest(),
       'automatic_retry':False,'hard_timeout_seconds_per_arm':hard_timeout,'runs':[]}
save(state)
for arm in ['BASE','FULL']:
    expected={k.lower().replace('_','-'):v for k,v in protocol['runtime']['packages'].items()}
    package_names=['torch','torchvision','numpy','Pillow','PyYAML','opencv-python','scipy','faster-coco-eval','timm']
    actual={name:metadata.version(name) for name in package_names}
    mismatch={name:{'expected':expected.get(name.lower().replace('_','-')),'actual':version}
              for name,version in actual.items() if expected.get(name.lower().replace('_','-'))!=version}
    state['last_environment_check']={'arm':arm,'packages':actual,'checked_utc':now()}
    if mismatch:
        state.update(status='STOPPED_ENVIRONMENT_CHANGED_NO_AUTOMATIC_RETRY',environment_mismatches=mismatch);save(state);sys.exit(1)
    command=[sys.executable,'-B','-u',str(ROOT/'fair_runner.py'),'train','--arm',arm]
    with (ROOT/f'{arm}.train.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
        run={'arm':arm,'pid':proc.pid,'start_ticks':start_ticks(proc.pid),'command':command,'started_utc':now(),'status':'RUNNING'}
        state['runs'].append(run);save(state);start=time.monotonic()
        while proc.poll() is None:
            state['updated_utc']=now();run['elapsed_seconds']=time.monotonic()-start
            receipt=ROOT/'runs'/arm/'receipt.json'
            if receipt.exists():
                try:run['latest_receipt']=json.loads(receipt.read_text())
                except json.JSONDecodeError:pass
            save(state)
            if run['elapsed_seconds']>hard_timeout:
                run['status']='HARD_TIMEOUT';save(state)
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(15)
        run['exit_code']=proc.wait();run['ended_utc']=now()
        receipt=ROOT/'runs'/arm/'receipt.json'
        final=json.loads(receipt.read_text()) if receipt.exists() else {}
        completed=run['exit_code']==0 and final.get('status')=='COMPLETED' and final.get('completed_epochs')==320
        run['status']='COMPLETED' if completed else 'FAILED_OR_TIMEOUT';run['final_receipt']=final
        state['status']='RUNNING' if completed else 'STOPPED_NO_AUTOMATIC_RETRY';save(state)
        if not completed:sys.exit(1)
state.update({'status':'COMPLETED','ended_utc':now()});save(state)
