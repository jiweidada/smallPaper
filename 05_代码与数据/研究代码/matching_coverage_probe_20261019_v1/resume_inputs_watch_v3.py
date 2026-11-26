"""Adopt the paused original CPU generator with a new bounded monitoring clock."""
import datetime,json,os,pathlib,signal,sys,time
root=pathlib.Path(sys.argv[1]);hold=json.loads((root/'user_hold_before_discussion_20261020.json').read_text())
identity=hold['child'];pid=identity['pid'];out=root/'task_records';statefile=out/'inputs_resume_v3.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(v):
 p=statefile.with_suffix('.next');p.write_text(json.dumps(v,indent=2));os.replace(p,statefile)
def state():
 p=pathlib.Path('/proc')/str(pid)
 if not p.is_dir():return None
 s=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert s[19]==str(identity['start_ticks']),'Original input PID identity changed'
 return s[0]
with (out/'inputs_resume_v3.lock').open('x') as f:f.write(now())
assert state()=='T' and os.getpgid(pid)==pid
v={'state':'running','started_utc':now(),'pid':pid,'start_ticks':identity['start_ticks'],'same_original_process':True,'timeout_seconds':1800,'automatic_retry':False}
write(v);os.killpg(pid,signal.SIGCONT);start=time.monotonic()
try:
 while state() not in (None,'Z'):
  if time.monotonic()-start>1800:
   os.killpg(pid,signal.SIGTERM);raise TimeoutError('Bounded original CPU input preparation exceeded 1800s after resume')
  v.update(elapsed_seconds=time.monotonic()-start,checked_utc=now());write(v);time.sleep(5)
 done=json.loads((root/'inputs/complete.json').read_text());assert done['state']=='complete' and done['inputs']==64
 v.update(state='complete',finished_utc=now(),optimizer_steps=0,model_forwards=0);write(v);print(json.dumps(v),flush=True)
except BaseException as e:
 v.update(state='failed',error=type(e).__name__+': '+str(e),finished_utc=now());write(v);raise
