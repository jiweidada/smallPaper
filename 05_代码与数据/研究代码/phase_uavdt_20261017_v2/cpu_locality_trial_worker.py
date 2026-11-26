"""Finite reversible OS affinity trial on owned trainers; no math, RNG or training-file changes."""
import datetime,json,os,re,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(CONFIG['root']);ART=Path(__file__).resolve().parent
def write(name,value):
 p=ART/name;tmp=p.with_suffix('.next');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,p)
def ticks(pid):
 try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def alive(entry):return ticks(entry['pid'])==str(entry['start_ticks'])
def thread_scope(owners):
 entries={}
 for owner in owners:
  if not alive(owner):continue
  parent=owner['pid'];children=(Path(f'/proc/{parent}/task/{parent}/children').read_text().split())
  for pid in [parent]+list(map(int,children)):
   path=Path('/proc')/str(pid)/'task'
   if not path.is_dir():continue
   for p in path.iterdir():
    tid=int(p.name);start=ticks(tid)
    if start is not None:
     try:entries[str(tid)]={'pid':tid,'start_ticks':start,'original_affinity':sorted(os.sched_getaffinity(tid)),'owner':parent}
     except ProcessLookupError:pass
 return entries
def restore(original):
 current=thread_scope(original['owners']);errors=[];count=0
 masks={o['pid']:o['original_affinity'] for o in original['owners']}
 for key,entry in current.items():
  old=original['threads'].get(key)
  mask=old['original_affinity'] if old and old['start_ticks']==entry['start_ticks'] else masks[entry['owner']]
  try:
   os.sched_setaffinity(entry['pid'],mask);assert set(os.sched_getaffinity(entry['pid']))==set(mask);count+=1
  except ProcessLookupError:pass
  except Exception as exc:errors.append({'pid':entry['pid'],'error':repr(exc)})
 return {'restored_threads':count,'errors':errors,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def records(job):
 p=ROOT/'logs'/(job+'.log');text=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',p.read_text(errors='replace'));result=[]
 for line in text.splitlines():
  m=re.search(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?Epoch:\s*(\d+)/96\s+\[\s*(\d+)/(\d+)\].*?time:\s*([\d.]+)\s+data:\s*([\d.]+)',line)
  if m:result.append({'time':datetime.datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp(),
   'epoch':int(m[2]),'batch':int(m[3]),'seconds_per_step':float(m[5]),'data_seconds':float(m[6])})
 return result
def observe(label,seconds,owners):
 start=time.time();deadline=time.monotonic()+seconds
 while time.monotonic()<deadline:
  assert all(alive(o) for o in owners),'Owned trainer exited during trial; restore and stop'
  time.sleep(min(3,max(.01,deadline-time.monotonic())))
 end=time.time();results={}
 for o in owners:
  selected=[r for r in records(o['job']) if start+1<=r['time']<=end-1];steps=0;duration=0.;intervals=[]
  for before,after in zip(selected,selected[1:]):
   if after['epoch']==before['epoch'] and after['batch']>before['batch']:
    ds=after['batch']-before['batch'];dt=after['time']-before['time']
    if dt>0:steps+=ds;duration+=dt;intervals.append({'steps':ds,'seconds':dt})
  results[o['job']]={'records':selected,'valid_training_intervals':intervals,'steps':steps,'seconds':duration,
    'images_per_second':4*steps/duration if duration else None}
 result={'label':label,'start_remote_unix':start,'end_remote_unix':end,'jobs':results}
 write(label+'.json',result);return result
def main():
 plan=json.loads((ROOT/'plan.json').read_text());owners=[]
 for arm in plan['jobs']:
  job=f'uavdt_{arm}_full96_seed{plan["seed"]}';o=json.loads((ROOT/'jobs'/(job+'.json')).read_text());assert o['state']=='running' and alive(o)
  args=(Path('/proc')/str(o['pid'])/'cmdline').read_bytes().split(b'\0');assert str(ROOT/'code/train_uavdt.py').encode() in args
  owners.append({'job':job,'pid':o['pid'],'start_ticks':str(o['start_ticks']),'original_affinity':sorted(os.sched_getaffinity(o['pid']))})
 topo=subprocess.check_output(['nvidia-smi','topo','-m'],text=True);m=re.search(r'^GPU0\s+X\s+\S+\s+([0-9,-]+)\s+(\d+)',topo,re.M);assert m,topo
 local=set()
 for part in m[1].split(','):
  if '-' in part:a,b=map(int,part.split('-'));local.update(range(a,b+1))
  else:local.add(int(part))
 local&=set(os.sched_getaffinity(0));assert len(local)>=12
 before=observe('baseline',45,owners)
 original={'owners':owners,'threads':thread_scope(owners),'GPU_local_CPUs':sorted(local),'GPU_NUMA_node':int(m[2]),'expiry_seconds':180}
 write('original_affinities.json',original)
 with (ART/'guardian.log').open('xb') as log:
  guardian=subprocess.Popen([sys.executable,'-u',__file__,'--guardian'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write('guardian.json',{'pid':guardian.pid,'start_ticks':ticks(guardian.pid)})
 for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Trial interrupted; restore affinities')))
 try:
  for entry in original['threads'].values():
   if alive(entry):
    try:os.sched_setaffinity(entry['pid'],local)
    except ProcessLookupError:pass
  write('pinned_state.json',{'state':'temporary_GPU_local_CPU_affinity','started_remote_unix':time.time(),'automatic_restore_within_seconds':180})
  time.sleep(10);pinned=observe('GPU_local_affinity',90,owners)
 finally:
  restored=restore(original);write('restore_receipt.json',restored)
  assert not restored['errors'],restored
 time.sleep(10);after=observe('restored_baseline',45,owners)
 comparison={}
 for o in owners:
  name=o['job'];b=before['jobs'][name]['images_per_second'];p=pinned['jobs'][name]['images_per_second'];a=after['jobs'][name]['images_per_second']
  comparison[name]={'before_images_s':b,'pinned_images_s':p,'restored_images_s':a,
    'gain_vs_combined_baseline_percent':100*(p/((a+b)/2)-1) if a and b and p else None}
 record={'state':'complete_and_original_affinities_restored','account':CONFIG['account'],'comparison':comparison,
 'stages':[before,pinned,after],'restored':restored,'no_training_file_or_recipe_change':True,
 'limit':'Short sequential OS scheduling test, not a randomized benchmark; epoch transitions and input-content variation can confound comparisons.'}
 write('result.json',record);print(json.dumps(record),flush=True)
if '--guardian' in sys.argv:
 original=json.loads((ART/'original_affinities.json').read_text());deadline=time.monotonic()+180
 while time.monotonic()<deadline and not (ART/'restore_receipt.json').is_file():time.sleep(3)
 if not (ART/'restore_receipt.json').is_file():write('guardian_restore_receipt.json',restore(original))
else:
 try:main()
 except BaseException as exc:write('failure.json',{'error':repr(exc)});raise
