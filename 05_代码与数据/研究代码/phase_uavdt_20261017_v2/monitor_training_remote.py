"""Read-only scoped queue monitor; never loads a checkpoint or changes a remote file."""
import datetime,hashlib,json,re,subprocess
from pathlib import Path
root=Path(CONFIG['root'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tail(p,n=262144):
 if not p.is_file():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
def identity(entry):
 if not entry:return None
 proc=Path('/proc')/str(entry['pid']);result={'pid':entry['pid'],'alive':False,'identity_matches':False}
 if not proc.is_dir():return result
 try:
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
  result.update(start_ticks=stat[19],process_state=stat[0],alive=stat[0]!='Z',identity_matches=stat[19]==str(entry['start_ticks']))
  if result['alive'] and result['identity_matches']:
   args=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
   result['owned_command_matches']=any(s.startswith(str(root/'code')) for s in args)
   if result['owned_command_matches']:
    result['command']=args
    ns=re.search(r'^NSpid:\s*(.+)$',(proc/'status').read_text(),re.M)
    result['namespace_pids']=list(map(int,ns.group(1).split())) if ns else None
    env={k:v for s in (proc/'environ').read_bytes().split(b'\0') if b'=' in s for k,v in [s.decode(errors='replace').split('=',1)] if k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONUNBUFFERED','PYTHONDONTWRITEBYTECODE']}
    result['thread_environment']=env
 except FileNotFoundError:result['alive']=False
 return result
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr[:1000]}
plan=read(root/'plan.json');queue=read(root/'queue_state.json');launch=read(root/'launch_receipt.json')
result={'account':CONFIG['account'],'seed':plan['seed'] if plan else None,'root':str(root),'checked_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'plan':plan,'queue':queue,'queue_process':identity(launch),'capacity':read(root/'capacity_receipt.json'),'acceptance':read(root/'final_acceptance.json'),
 'GPU':command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
 'GPU_compute':command(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']),'jobs':[],'collectible':[]}
strip_ansi=re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
for arm in plan['jobs'] if plan else []:
 job=f'uavdt_{arm}_full96_seed{plan["seed"]}';out=root/'outputs'/job;console=root/'logs'/(job+'.log');log=out/'log.txt'
 status=read(root/'jobs'/(job+'.json'));text=strip_ansi.sub('',tail(console));all_rows=[]
 # On reallocation the historical console is retained and appended. Old batches
 # must not be mistaken for progress while the new process loads/evaluates EMA.
 if status and status.get('recovery_id') and 'RESUME_FROM_OWN_CHECKPOINT ' in text:
  text=text.rsplit('RESUME_FROM_OWN_CHECKPOINT ',1)[1]
 if log.is_file():
  for line in log.read_text().splitlines():
   try:row=json.loads(line)
   except ValueError:continue
   if 'test_uavdt_eval_bbox' in row:all_rows.append(row)
 latest=all_rows[-1] if all_rows else None
 epochs=list(re.finditer(r'Epoch:\s*(\d+)\s*/\s*96[^\r\n]*',text));progress=epochs[-1].group(0) if epochs else None
 batch=re.search(r'\[\s*(\d+)\s*/\s*(\d+)\s*\]',progress or '')
 time=re.search(r'\btime:\s*([\d.]+)',progress or '')
 # The native solver prints its resolved output path once, during initialization.
 # A tail-only read loses that line after several epochs and must not imply a path error.
 header=''
 if console.is_file():
  with console.open('rb') as f:header=strip_ansi.sub('',f.read(65536).decode(errors='replace'))
 output_bindings=list(dict.fromkeys(re.findall(r'output_dir:\s*([^\s]+)',header+'\n'+text)))
 item={'job':job,'arm':arm,'seed':plan['seed'],'status':status,'process':identity(status),'output':str(out),'output_exists':out.is_dir(),
       'output_directory_bindings':output_bindings,'output_binding_passed':bool(output_bindings) and all(s==str(out) for s in output_bindings),
       'completed_epochs':len(all_rows),'epoch_zero_based':int(epochs[-1].group(1)) if epochs else None,
       'batch_zero_based':int(batch.group(1)) if batch else None,'batches_per_epoch':int(batch.group(2)) if batch else None,
       'seconds_per_step':float(time.group(1)) if time else None,'latest_progress_line':progress,
       'latest_complete_evaluation':{'epoch_zero_based':latest['epoch'],'AP70':latest['test_uavdt_eval_bbox'][0],'AR70':latest['test_uavdt_eval_bbox'][1]} if latest else None,
       'console_bytes':console.stat().st_size if console.exists() else 0,'console_mtime_remote_unix':console.stat().st_mtime if console.exists() else None,
       'log_bytes':log.stat().st_size if log.exists() else 0}
 if status and status.get('state')=='failed':item['failure_log_last50']='\n'.join(text.splitlines()[-50:])
 if not epochs:item['initialization_log_tail']='\n'.join(text.splitlines()[-8:])
 result['jobs'].append(item)
 if status and status.get('state')=='complete':
  for p in [root/'jobs'/(job+'.json'),log,out/'checkpoint_log_binding.json',root/'results'/job/'val/metrics.json',root/'results'/job/'test/metrics.json']:
   if p.is_file():result['collectible'].append({'relative':str(p.relative_to(root)),'sha256':sha(p),'text':p.read_text(),'bytes':p.stat().st_size})
analysis=root/'analysis'
for p in [analysis/'size_localization.json',analysis/'paired_detection_counts.csv']:
 if p.is_file():result['collectible'].append({'relative':str(p.relative_to(root)),'sha256':sha(p),'text':p.read_text(),'bytes':p.stat().st_size})
if queue and queue.get('state')=='failed':
 controller_name=Path((launch or {}).get('controller','')).name
 current_log={'resume_uavdt_queue_20261019_v2.py':'controller_recovery_20261019_v2.log',
              'resume_uavdt_queue_20261018.py':'controller_recovery_20261018.log'}.get(controller_name,'controller.log')
 result['controller_log_last50']='\n'.join(tail(root/'logs'/current_log).splitlines()[-50:])
print(json.dumps(result),flush=True)
