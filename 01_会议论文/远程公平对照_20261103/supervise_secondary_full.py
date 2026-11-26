"""One FULL run on the second authorized GPU under the unchanged frozen recipe.

Separate host-placement receipts supplement, never rewrite, the original
protocol and first-host preparation/environment provenance. No retry occurs.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.metadata as metadata,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA='8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def ticks(pid):return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
def save(state):
    temp=ROOT/'secondary_queue.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');os.replace(temp,ROOT/'secondary_queue.json')
assert sha(ROOT/'execution_protocol.json')==PROTOCOL_SHA
protocol=read(ROOT/'execution_protocol.json')
assert protocol['authorized'] and protocol['arms']==['BASE','FULL'] and protocol['epochs']==320
assert protocol['budget']['hard_timeout_seconds_per_arm']==604800
assert not (ROOT/'runs/FULL').exists() and not (ROOT/'runs/BASE').exists()
preparation=read(ROOT/'secondary_preparation.json')
assert preparation['status']=='READY_FOR_AUTHORIZED_FULL_ONLY'
reservation=read(ROOT/'primary_full_reservation.json')
assert reservation['status']=='FULL_RESERVED_FOR_SECONDARY_HOST_DO_NOT_REMOVE'
assert reservation['secondary_preparation_sha256']==sha(ROOT/'secondary_preparation.json')
assert reservation['addendum_sha256']==sha(ROOT/'parallel_execution_addendum.json')
assert reservation['reserved_log_sha256']==sha(ROOT/'primary_FULL_reserved_log.json')
assert preparation['runtime_sha256']==sha(ROOT/'secondary_runtime.json')
expected={k.lower().replace('_','-'):v for k,v in protocol['runtime']['packages'].items()}
actual={d.metadata['Name'].lower().replace('_','-'):d.version for d in metadata.distributions()}
assert actual==expected,'Package environment changed after preparation'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(', ')
assert gpu[0]=='GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a' and int(gpu[1])<256,gpu
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
for f in protocol['frozen_files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
state={'status':'STARTING','started_utc':now(),'supervisor_pid':os.getpid(),'supervisor_start_ticks':ticks(os.getpid()),
 'protocol_sha256':PROTOCOL_SHA,'automatic_retry':False,'hard_timeout_seconds_per_arm':604800,'runs':[],
 'host_port':30989,'actual_runtime_sha256':sha(ROOT/'secondary_runtime.json'),
 'original_environment_provenance_sha256':sha(ROOT/'environment.current.json'),
 'reservation_sha256':sha(ROOT/'primary_full_reservation.json'),
 'placement_addendum_sha256':sha(ROOT/'parallel_execution_addendum.json'),
 'supervisor_script_sha256':sha(Path(__file__))}
with (ROOT/'secondary_one_shot.lock').open('x') as f:f.write(json.dumps(state)+'\n')
save(state)
environment=dict(os.environ,PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
try:
    command=[sys.executable,'-B','-u',str(ROOT/'fair_runner.py'),'train','--arm','FULL']
    with (ROOT/'FULL.train.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
        run={'arm':'FULL','pid':proc.pid,'start_ticks':ticks(proc.pid),'command':command,'started_utc':now(),'status':'RUNNING'}
        state['runs'].append(run);state['status']='RUNNING';save(state);start=time.monotonic()
        while proc.poll() is None:
            state['updated_utc']=now();run['elapsed_seconds']=time.monotonic()-start
            p=ROOT/'runs/FULL/receipt.json'
            if p.exists():
                try:run['latest_receipt']=read(p)
                except json.JSONDecodeError:pass
            save(state)
            if run['elapsed_seconds']>604800:
                run['status']='HARD_TIMEOUT';save(state);os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(15)
        run['exit_code']=proc.wait();run['ended_utc']=now()
        final=read(ROOT/'runs/FULL/receipt.json') if (ROOT/'runs/FULL/receipt.json').exists() else {}
        complete=(run['exit_code']==0 and final.get('status')=='COMPLETED' and final.get('completed_epochs')==320
                  and final.get('actual_loop_epoch')==319 and final.get('optimizer_step_attempts')==517440)
        run['status']='COMPLETED' if complete else 'FAILED_OR_TIMEOUT';run['final_receipt']=final
        state.update(status='COMPLETED' if complete else 'STOPPED_NO_AUTOMATIC_RETRY',ended_utc=now());save(state)
        if not complete:sys.exit(1)
except Exception as error:
    state.update(status='SUPERVISION_ERROR_NO_AUTOMATIC_RETRY',error=repr(error));save(state);raise
