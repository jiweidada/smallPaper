"""Wait for the exact existing Warm trainer/controller then collect once on CPU."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,runpy,sys,time,traceback
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
 with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,indent=2,allow_nan=False)
def state(r,pid='pid',ticks='start_ticks'):
 try:
  f=Path('/proc/'+str(r[pid])+'/stat').read_text().split()
  return dict(r,same_live=f[21]==str(r[ticks]) and f[2]!='Z',state=f[2])
 except (FileNotFoundError,ProcessLookupError):return dict(r,same_live=False)
def main(prefix,seed):
 protocol=read(HERE/'waiter_protocol.json')
 assert seed in protocol['seeds']==[1] and prefix=='/25085400120'
 assert {n:sha(HERE/n) for n in protocol['sources']}==protocol['sources']
 assert sha(HERE/'protocol.json')==protocol['collector_protocol_sha256']
 base=Path(prefix)/'DEIM/experiments';warm=base/'warm_stem12_20261011_v1';screen=warm/f'seed{seed}_warm_stem'
 assert {p.name:sha(p) for p in (warm/'code').iterdir() if p.is_file()}==protocol['warm_files_sha256']
 trainer=read(screen/'candidate_start.json');controller=read(screen/'controller_start.json')
 assert trainer['child_pid']==protocol['trainer']['pid'] and trainer['child_start_ticks']==protocol['trainer']['start_ticks']
 assert controller['pid']==trainer['controller_pid']
 root=HERE.parent
 receipt=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
  source_sha256=sha(__file__),protocol_sha256=sha(HERE/'waiter_protocol.json'),seed=seed,
  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_training_handle=state(trainer,'child_pid','child_start_ticks'),
  actual_controller_handle=state(controller),automatic_retry=False,training_commands=0,new_model_forwards=0,new_evaluator_calls=0)
 write(root/'seed1_waiter_start.json',receipt)
 started=time.monotonic()
 try:
  while state(trainer,'child_pid','child_start_ticks')['same_live'] or state(controller)['same_live']:
   temp=root/'seed1_waiter_live.tmp'
   temp.write_text(json.dumps(dict(phase='waiting_exact_existing_Warm_trainer_and_controller',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    actual_trainer=state(trainer,'child_pid','child_start_ticks'),actual_controller=state(controller))))
   temp.replace(root/'seed1_waiter_live.json')
   if time.monotonic()-started>=protocol['max_wait_seconds']:
    raise TimeoutError('Observation timed out; no signal, restart or automatic collection retry')
   time.sleep(15)
  assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
  exit=read(screen/'candidate_completion.json');assert exit['exit_code']==0 and not exit['timed_out'] and not exit['automatic_retry']
  receipt_bundle=read(screen/'bundle/receipt.json');run=Path(receipt_bundle['paths']['output'])
  done=read(run/'training_completion.json')
  assert done['status']=='NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
  assert done['identity']==receipt_bundle['identity']
  sys.argv=[str(HERE/'collect.py'),'--prefix',prefix,'--seed',str(seed),'--arm','warm_stem']
  runpy.run_path(str(HERE/'collect.py'),run_name='__main__')
  c=read(root/'seed1_collect_completion.json');assert c['exit_code']==0
  assert c['receipt']['pid']==receipt['pid'] and c['receipt']['start_ticks']==receipt['start_ticks']
  assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
  write(root/'seed1_waiter_completion.json',dict(status='EXISTING_WARM12_PRIMARY_COLLECTED_ONCE_AFTER_ACTUAL_HANDLES_STOPPED',
    receipt=receipt,collection_completion_sha256=sha(root/'seed1_collect_completion.json'),
    metadata=c['metadata'],exit_code=0,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,
    existing_process_signals=0,automatic_retry=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 except BaseException:
  write(root/'seed1_waiter_failure.json',dict(status='READ_ONLY_WAIT_OR_COLLECTION_FAILED_NO_RETRY',
    receipt=receipt,traceback=traceback.format_exc(),new_model_forwards=0,new_training_epochs=0,existing_process_signals=0))
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,),required=True)
 a=p.parse_args();main(a.prefix,a.seed)
