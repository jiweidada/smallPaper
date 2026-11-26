"""Scoped one-shot stages: sealed forward, development, code freeze, heldout."""
import argparse,concurrent.futures,datetime,hashlib,json,pathlib,subprocess,sys,tarfile
from control_probe import ctl,ROOTS,LOCAL,HERE
from transfer_probe import get,safe_extract
from seal_v3 import sha,write_json
BASE=LOCAL/'isolated_v3';BASE.mkdir(exist_ok=True)
SOURCES=['seal_v3.py','probe_forward_v3.py','analyze_partition_v3.py','partition_summary_v3.py','stage0_thresholds.py','test_seal_v3.py','test_stage0_thresholds.py']
def read(sf,p):
 try:
  with sf.open(p,'rb') as f:return f.read()
 except FileNotFoundError:return None
def put(sf,p,b):
 old=read(sf,p)
 if old is not None:
  assert old==b,'Existing task artifact differs; inspect rather than replace'
  return
 with sf.open(p,'wx') as f:f.write(b)
def execute(c,a,program,timeout=45):
 r=ctl.execute(c,a,'PROBE_ROOT='+repr(ROOTS[a])+'\nSEED='+repr(1 if a=='A' else 2)+'\n'+program,timeout=timeout)
 assert r['returncode']==0,r
 return json.loads(r['stdout'])
def one(a,passwords,action):
 cfg=ctl.ACCOUNTS[a];root=ROOTS[a];seed=1 if a=='A' else 2;c=ctl.connect(cfg['port'],passwords)
 try:
  sf=c.open_sftp()
  if action=='deploy':
   sf.mkdir(root+'/v3');files=[]
   for name in SOURCES:
    raw=(HERE/name).read_bytes();put(sf,root+'/v3/'+name,raw);files.append({'relative':'v3/'+name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
   config=(LOCAL/'05_stage0_stop_thresholds_v1.json').read_bytes();put(sf,root+'/v3/threshold_config.json',config);files.append({'relative':'v3/threshold_config.json','sha256':hashlib.sha256(config).hexdigest(),'bytes':len(config)})
   put(sf,root+'/v3/manifest.json',json.dumps({'files':files,'same_fixed64_inputs':True,'no_optimizer':True,'heldout_sealed_at_generation':True},indent=2).encode())
   return {'account':a,'state':'v3_deployed_not_forwarded','files':len(files)}
  if action=='status':
   result={'account':a}
   for name,rel in [('input_done','inputs/complete.json'),('input_resume','task_records/inputs_resume_v3.json'),('input_progress','inputs/progress.json'),('forward','task_records/forward_isolated_v3_seed'+str(seed)+'.json'),('complete',f'isolated_results_seed{seed}/complete.json'),('failure',f'isolated_results_seed{seed}/failure.json'),('progress',f'isolated_results_seed{seed}/progress.json'),('dev_task',f'task_records/development_analysis_v3_seed{seed}.json'),('heldout_task',f'task_records/heldout_analysis_v3_seed{seed}.json'),('dev_complete',f'isolated_results_seed{seed}/analysis_development_v3/complete.json'),('heldout_complete',f'isolated_results_seed{seed}/analysis_heldout_v3/complete.json')]:
    raw=read(sf,root+'/'+rel)
    if raw:result[name]=json.loads(raw)
   for task in ['forward_isolated_v3','development_analysis_v3','heldout_analysis_v3']:
    raw=read(sf,root+'/task_records/'+task+'_seed'+str(seed)+'.json')
    if raw and json.loads(raw).get('state')=='failed':
     log=read(sf,root+'/task_records/'+task+'_seed'+str(seed)+'.log');result[task+'_failure_tail']=log.decode(errors='replace')[-5000:] if log else None
   return result
  if action=='anchor':
   folder=BASE/'generation_anchor'/a;folder.mkdir(parents=True,exist_ok=False)
   raw=read(sf,root+f'/isolated_results_seed{seed}/complete.json');assert raw
   done=json.loads(raw);assert done['state']=='complete' and not done['outcome_analysis_performed'];(folder/'complete.json').write_bytes(raw)
   for group in ['development','heldout']:
    b=read(sf,root+f'/isolated_results_seed{seed}/{group}/seal.json');assert hashlib.sha256(b).hexdigest()==done['partition_seal_sha256'][group]
    (folder/(group+'_seal.json')).write_bytes(b)
   write_json(folder/'client_anchor.json',{'received_client_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'partition_seal_sha256':done['partition_seal_sha256'],'no_prediction_content_deserialized':True})
   return {'account':a,'state':'generation_SHA_anchored_locally_before_development','seal_SHA':done['partition_seal_sha256']}
  if action.startswith('start-'):
   part=action.removeprefix('start-');forward=part=='forward'
   name=('forward_isolated_v3' if forward else part+'_analysis_v3')+'_seed'+str(seed)
   command=[cfg['python'],'-u','-B',root+'/v3/'+('probe_forward_v3.py' if forward else 'analyze_partition_v3.py'),'--seed',str(seed)]
   if not forward:command+=['--group',part]
   task={'root':root,'name':name,'timeout_seconds':1800 if forward else 900,'cpu_only':not forward,'command':command}
   put(sf,root+'/'+name+'_task.json',json.dumps(task).encode())
   program='''import pathlib,json,subprocess,os,datetime
root=pathlib.Path(PROBE_ROOT)
assert not (root/'task_records'/(NAME+'.lock')).exists()
assert json.loads((root/('preflight_seed'+str(SEED)+'.json')).read_text())['state']=='CPU_preflight_passed'
if PART=='forward':
 assert json.loads((root/'inputs/complete.json').read_text())['inputs']==64
 q=json.loads((pathlib.Path(CONFIG['root'])/'queue_state.json').read_text());assert q['state']=='training' and len(q['active_jobs'])==3,'UAVDT restoration has priority'
else:assert json.loads((root/f'isolated_results_seed{SEED}/complete.json').read_text())['state']=='complete'
with (root/(NAME+'_supervisor.log')).open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u','-B',str(root/'bounded_task.py'),str(root/(NAME+'_task.json'))],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PROBE_OLD_ROOT=CONFIG['root']))
print(json.dumps({'account':CONFIG['account'],'task':NAME,'supervisor_pid':p.pid,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
'''
   return execute(c,a,'NAME='+repr(name)+'\nPART='+repr(part)+'\n'+program)
  if action.startswith('collect-'):
   part=action.removeprefix('collect-');program='''import pathlib,json,tarfile,hashlib
root=pathlib.Path(PROBE_ROOT);folder=root/f'isolated_results_seed{SEED}'/('analysis_'+PART+'_v3')
assert json.loads((folder/'complete.json').read_text())['state']=='complete'
archive=root/(f'collected_{PART}_seed{SEED}_v3.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(folder.iterdir()):
  if p.is_file():t.add(p,arcname=p.name,recursive=False)
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest()}),flush=True)
'''
   meta=execute(c,a,'PART='+repr(part)+'\n'+program,180)
   local=BASE/(a+'_'+part+'.tar.gz');get(sf,meta['path'],local,meta['bytes'],meta['sha256'])
   dest=BASE/a/part;dest.parent.mkdir(exist_ok=True);safe_extract(local,dest)
   return {'account':a,'group':part,'local':str(dest),'archive_sha256':meta['sha256']}
  raise ValueError(action)
 finally:c.close()
def freeze(passwords):
 dev=json.loads((BASE/'development_summary.json').read_text());assert dev['group']=='development'
 generation={str(seed):json.loads((BASE/'generation_anchor'/a/'complete.json').read_text())['partition_seal_sha256'] for a,seed in [('A',1),('B',2)]}
 local={'created_client_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'development_dry_run_passed':True,'development_summary_sha256':sha(BASE/'development_summary.json'),'generation_seal_sha256':generation,'local_code_files':[{'absolute_path':str((HERE/n).resolve()),'sha256':sha(HERE/n)} for n in SOURCES]}
 write_json(BASE/'local_frozen_analysis.json',local);results={}
 for a in ['A','B']:
  c=ctl.connect(ctl.ACCOUNTS[a]['port'],passwords)
  try:
   program='''import pathlib,json,hashlib
root=pathlib.Path(PROBE_ROOT);items={}
for manifest in [root/'package_manifest.json',root/'extension_v2/manifest.json',root/'v3/manifest.json']:
 items[str(manifest.relative_to(root))]={'relative':str(manifest.relative_to(root)),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
 for row in json.loads(manifest.read_text())['files']:
  assert hashlib.sha256((root/row['relative']).read_bytes()).hexdigest()==row['sha256'];items[row['relative']]=row
for extra in ['inputs/manifest.json','make_inputs_registered_v3.py']:
 p=root/extra
 if p.is_file():items[extra]={'relative':extra,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
assert json.loads((root/f'isolated_results_seed{SEED}/analysis_development_v3/complete.json').read_text())['state']=='complete'
value={'both_development_runs_passed':True,'generation_seal_sha256':GENERATION,'local_freeze_sha256':LOCAL_SHA,'files':list(items.values()),'created_client_utc':CLIENT_UTC}
path=root/'analysis_freeze_v3.json'
with path.open('x') as f:json.dump(value,f,indent=2)
digest=hashlib.sha256(path.read_bytes()).hexdigest()
with (root/'analysis_release_v3.json').open('x') as f:json.dump({'freeze_sha256':digest,'local_freeze_sha256':LOCAL_SHA},f,indent=2)
print(json.dumps({'account':CONFIG['account'],'freeze_sha256':digest,'files':len(items),'heldout_still_unread':True}),flush=True)
'''
   results[a]=execute(c,a,'GENERATION='+repr(generation)+'\nLOCAL_SHA='+repr(sha(BASE/'local_frozen_analysis.json'))+'\nCLIENT_UTC='+repr(local['created_client_utc'])+'\n'+program)
  finally:c.close()
 write_json(BASE/'heldout_access_receipt.json',{'local_freeze_sha256':sha(BASE/'local_frozen_analysis.json'),'remote_freezes':results,'development_passed_before_release':True})
 return results
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','status','anchor','start-forward','start-development','start-heldout','collect-development','collect-heldout','freeze']);a=p.parse_args()
 passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes())
 try:
  if a.action=='freeze':results=freeze(passwords)
  else:
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    fs={v:pool.submit(one,v,passwords,a.action) for v in ['A','B']};results={}
    for account,f in fs.items():
     try:results[account]=f.result()
     except Exception as e:results[account]={'error':type(e).__name__+': '+str(e)}
 finally:passwords.clear()
 dest=BASE/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+a.action+'.json');dest.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'saved':str(dest),'results':results},ensure_ascii=False,indent=2))
 if any(isinstance(r,dict) and 'error' in r for r in results.values()):raise SystemExit(1)
