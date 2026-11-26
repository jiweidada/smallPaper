"""User-prioritized continuation of six saved UAVDT jobs; no new arms/recipes."""
import argparse,concurrent.futures,datetime,hashlib,json,pathlib
import control_uavdt as ctl
HERE=pathlib.Path(__file__).resolve().parent
DEST=HERE.parents[1]/'data/07_第二数据集_20261017/C主线恢复_20261020_v3'
REMOTE='recovery_20261020_v3';CONTROLLER='resume_uavdt_queue_20261020_v3.py'
def read(sf,p):
 try:
  with sf.open(p,'rb') as f:return f.read()
 except FileNotFoundError:return None
def put(sf,p,raw):
 old=read(sf,p)
 if old is not None:assert old==raw;return
 with sf.open(p,'wx') as f:f.write(raw)
def one(a,passwords,action):
 cfg=ctl.ACCOUNTS[a];root=cfg['root'];base=root+'/'+REMOTE;client=ctl.connect(cfg['port'],passwords)
 try:
  sf=client.open_sftp()
  if action=='validate':
   sf.mkdir(base)
   config=dict(cfg,validation_result_path=base+'/checkpoint_validation.json',validation_progress_path=base+'/validation_progress.json')
   source=('CONFIG='+repr(config)+'\n').encode()+(HERE/'validate_resume_checkpoints_20261020_v3.py').read_bytes()
   put(sf,base+'/checkpoint_validator.py',source)
   program='''import pathlib,json,subprocess,os,datetime
root=pathlib.Path(CONFIG['root']);base=root/'recovery_20261020_v3'
assert json.loads((root/'pause_user_20261019_v1/release_receipt.json').read_text())['state']=='paused_backup_verified_GPU_released'
with (base/'validation.stdout').open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u','-B',str(base/'checkpoint_validator.py')],cwd=root/'code',env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v={'account':CONFIG['account'],'pid':p.pid,'mode':'deep CPU checkpoint validation only','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(base/'validation_dispatch.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
'''
   return ctl.execute(client,a,program,timeout=30)
  if action=='start':
   raw=read(sf,base+'/checkpoint_validation.json');assert raw
   validation=json.loads(raw);assert validation['all_passed'] and len(validation['jobs'])==3
   pause=json.loads(read(sf,root+'/pause_user_20261019_v1/receipt.json'))
   saved={j['job']:j for j in pause['jobs']}
   proof={'account':a,'seed':0 if a=='A' else 1,'recovery_id':'saved_C_line_20261020_'+a,
          'plan_sha256':hashlib.sha256(read(sf,root+'/plan.json')).hexdigest(),
          'checkpoint_validation_sha256':hashlib.sha256(raw).hexdigest(),'jobs':[]}
   for j in validation['jobs']:
    assert j['checkpoint_sha256']==saved[j['job']]['checkpoint_sha256'] and j['last_epoch']==saved[j['job']]['saved_epoch']
    proof['jobs'].append({k:j[k] for k in ['job','arm','checkpoint','checkpoint_sha256','last_epoch']})
   put(sf,root+'/code/'+CONTROLLER,(HERE/CONTROLLER).read_bytes());put(sf,base+'/resume_preflight.json',json.dumps(proof,indent=2).encode())
   program='''import pathlib,json,subprocess,os,datetime
root=pathlib.Path(CONFIG['root']);base=root/'recovery_20261020_v3'
with (base/'launch_intent.json').open('x') as f:json.dump({'user_authorization':'Restore existing six C-paper UAVDT jobs only','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
with (root/'logs/controller_recovery_20261020_v3.log').open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u','-B',str(root/'code/resume_uavdt_queue_20261020_v3.py')],cwd=root/'code',env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v={'account':CONFIG['account'],'controller_pid':p.pid,'dispatch_only_not_training_verified':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(base/'dispatch_receipt.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
'''
   return ctl.execute(client,a,program,timeout=30)
  result={'account':a}
  for name in ['validation_dispatch.json','validation_progress.json','checkpoint_validation.json','dispatch_receipt.json']:
   raw=read(sf,base+'/'+name)
   if raw:
    result[name]=json.loads(raw)
    (DEST/(a+'_'+name)).write_bytes(raw)
  logs=read(sf,base+'/validation.stdout')
  if logs:result['log_tail']=logs.decode(errors='replace')[-1600:]
  return result
 finally:client.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','status','start']);args=p.parse_args();DEST.mkdir(exist_ok=True)
 passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes())
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={a:pool.submit(one,a,passwords,args.action) for a in ['A','B']};results={}
   for a,f in fs.items():
    try:results[a]=f.result()
    except Exception as e:results[a]={'error':type(e).__name__+': '+str(e)}
 finally:passwords.clear()
 path=DEST/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+args.action+'.json');path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'saved':str(path),'results':results},ensure_ascii=False,indent=2))
