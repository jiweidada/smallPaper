from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time,signal
ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def save(v):
 p=ROOT/'preparation_queue.json.tmp';p.write_text(json.dumps(v,indent=2)+'\n');os.replace(p,ROOT/'preparation_queue.json')
def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
with (ROOT/'preparation_once.lock').open('x') as f:f.write(str(os.getpid()))
env=dict(os.environ,PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
         OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
state={'status':'RUNNING','supervisor_pid':os.getpid(),'start_ticks':ticks(os.getpid()),'started_utc':now(),'model_optimizer_steps':0,'steps':[]}
save(state)
jobs=[('cache',[sys.executable,'-B','-u',str(ROOT/'prepare_cache.py')]),
      ('BASE_preflight',[sys.executable,'-B','-u',str(ROOT/'fair_runner.py'),'preflight','--arm','BASE']),
      ('FULL_preflight',[sys.executable,'-B','-u',str(ROOT/'fair_runner.py'),'preflight','--arm','FULL'])]
for name,cmd in jobs:
 with (ROOT/f'{name}.log').open('x') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  step={'name':name,'pid':p.pid,'start_ticks':ticks(p.pid),'status':'RUNNING','command':cmd,'started_utc':now()}
  state['steps'].append(step);save(state);started=time.monotonic()
  while p.poll() is None:
   step['elapsed_seconds']=time.monotonic()-started;state['updated_utc']=now();save(state)
   if step['elapsed_seconds']>1800:
    step['status']='HARD_TIMEOUT';save(state);os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    break
   time.sleep(5)
  step['exit_code']=p.wait();step['ended_utc']=now();step['status']='COMPLETED' if step['exit_code']==0 else 'FAILED_OR_TIMEOUT';save(state)
  if step['exit_code']!=0:
   state['status']='STOPPED_BEFORE_FORMAL_TRAINING';save(state);sys.exit(1)
state.update({'status':'PREFLIGHTS_COMPLETED_FORMAL_TRAINING_NOT_STARTED','ended_utc':now()});save(state)
