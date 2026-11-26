"""Finite already-authorized screen; no automatic experiment retries or training."""
import concurrent.futures,datetime,json,subprocess,sys,time,traceback
from control_isolated_v3 import one,BASE,HERE,ctl
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def event(value):
 v={'client_utc':now(),**value}
 with (BASE/'pipeline.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')
 print(json.dumps(v,ensure_ascii=False),flush=True)
def run(name,args,timeout=300):
 event({'stage':name,'state':'starting'})
 with (BASE/('pipeline_'+name+'.log')).open('xb') as f:
  p=subprocess.Popen([sys.executable,'-s','-W','ignore','-X','utf8','-B']+args,stdout=f,stderr=subprocess.STDOUT)
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=20);raise
 assert rc==0,(name,(BASE/('pipeline_'+name+'.log')).read_text(encoding='utf8',errors='replace')[-6000:])
 event({'stage':name,'state':'complete'})
def control(name):
 # Connector credentials require the normal current-user site, so this action
 # runs without -s; pure local summary programs continue to use -s above.
 with (BASE/('pipeline_'+name+'.log')).open('xb') as f:
  p=subprocess.Popen([sys.executable,'-W','ignore','-X','utf8','-B',str(HERE/'control_isolated_v3.py'),name],stdout=f,stderr=subprocess.STDOUT)
  rc=p.wait(timeout=300)
 assert rc==0,(name,(BASE/('pipeline_'+name+'.log')).read_text(encoding='utf8',errors='replace')[-6000:])
 event({'stage':name,'state':'complete'})
def wait_stage(passwords,phase,done_key,task_key,deadline):
 while True:
  assert time.monotonic()<deadline,'Overall40min background-screen time limit reached; no additional stages'
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={a:pool.submit(one,a,passwords,'status') for a in ['A','B']};states={a:f.result() for a,f in fs.items()}
  compact={a:{'task':s.get(task_key),'progress':s.get('progress'),'failure':s.get('failure'),'done':bool(s.get(done_key))} for a,s in states.items()}
  event({'stage':phase,'accounts':compact})
  for a,s in states.items():
   assert not s.get('failure'),s.get('failure')
   assert (s.get(task_key) or {}).get('state')!='failed',s
  if all(s.get(done_key) and (s.get(task_key) or {}).get('state')=='complete' for s in states.values()):return
  time.sleep(30)
def main():
 with (BASE/'pipeline_once.lock').open('x') as f:f.write(now())
 pw=json.loads(ctl.scheduled_monitor_v4.credentials_bytes());deadline=time.monotonic()+2400
 try:
  wait_stage(pw,'sealed_forward','complete','forward',deadline)
  control('anchor');control('start-development')
  wait_stage(pw,'development_only','dev_complete','dev_task',deadline)
  control('collect-development')
  run('development_summary',[str(HERE/'partition_summary_v3.py'),'--base',str(BASE),'--group','development'])
  control('freeze');control('start-heldout')
  wait_stage(pw,'heldout_after_freeze','heldout_complete','heldout_task',deadline)
  control('collect-heldout')
  run('heldout_summary',[str(HERE/'partition_summary_v3.py'),'--base',str(BASE),'--group','heldout'])
  event({'state':'complete','additional_training':False,'further_research_actions':'none; deliver integer numerator/denominator table and stopping reasons'})
 finally:pw.clear()
if __name__=='__main__':
 try:main()
 except BaseException as e:
  value={'state':'failed','client_utc':now(),'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'automatic_retry':False,'UAVDT_not_interrupted':True}
  (BASE/'pipeline_failure.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(value,ensure_ascii=False),flush=True);raise
