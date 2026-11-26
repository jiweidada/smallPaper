"""Deliberate preparation retry after an archived registration failure; zero updates."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, signal, subprocess, sys, time
ROOT = Path(__file__).resolve().parent
def now(): return datetime.now(timezone.utc).isoformat()
def ticks(pid): return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
def save(value):
    tmp = ROOT/'preparation_queue_v2.json.tmp'
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,ROOT/'preparation_queue_v2.json')
assert (ROOT/'preparation_attempt1/failure_audit.json').exists()
assert json.loads((ROOT/'cache_verification.json').read_text())['status']=='ALL_INPUT_BYTES_VERIFIED'
assert not (ROOT/'one_shot.lock').exists() and not (ROOT/'runs').exists()
with (ROOT/'preparation_once_v2.lock').open('x') as stream: stream.write(str(os.getpid()))
env = dict(os.environ,PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
           OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
state = {'status':'RUNNING','supervisor_pid':os.getpid(),'start_ticks':ticks(os.getpid()),
         'started_utc':now(),'model_optimizer_steps':0,'steps':[],
         'preparation_attempt':2,'prior_failure':'preparation_attempt1/failure_audit.json','automatic_retry':False}
save(state)
for arm in ['BASE','FULL']:
    command = [sys.executable,'-B','-u',str(ROOT/'fair_runner.py'),'preflight','--arm',arm]
    with (ROOT/f'{arm}_preflight_v2.log').open('x') as log:
        proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        step = {'name':arm+'_preflight','pid':proc.pid,'start_ticks':ticks(proc.pid),'status':'RUNNING',
                'command':command,'started_utc':now()}
        state['steps'].append(step); save(state); started=time.monotonic()
        while proc.poll() is None:
            step['elapsed_seconds']=time.monotonic()-started;state['updated_utc']=now();save(state)
            if step['elapsed_seconds']>1800:
                step['status']='HARD_TIMEOUT';save(state);os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(5)
        step['exit_code']=proc.wait();step['ended_utc']=now()
        report_path=ROOT/'preflight'/f'{arm}.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        passed=step['exit_code']==0 and report.get('status')=='PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE'
        step['status']='COMPLETED' if passed else 'FAILED_OR_TIMEOUT';save(state)
        if not passed:
            state['status']='STOPPED_BEFORE_FORMAL_TRAINING';save(state);sys.exit(1)
state.update(status='PREFLIGHTS_COMPLETED_FORMAL_TRAINING_NOT_STARTED',ended_utc=now());save(state)
