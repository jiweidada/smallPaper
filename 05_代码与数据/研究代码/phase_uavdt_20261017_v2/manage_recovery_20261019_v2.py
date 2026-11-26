"""Scoped second recovery: durable CPU validation, then single-dispatch continuation."""
import argparse,ast,datetime,hashlib,json
from pathlib import Path
from control_uavdt import ACCOUNTS,connect,execute,scheduled_monitor_v4

HERE=Path(__file__).resolve().parent
DEST=HERE.parents[1]/'data/07_第二数据集_20261017/GPU重申请恢复_20261019_v2'
REMOTE='recovery_20261019_v2'
CONTROLLER='resume_uavdt_queue_20261019_v2.py'

def read_remote(sftp,path):
    try:
        with sftp.open(path,'rb') as f:return f.read()
    except FileNotFoundError:return None

def put_new(sftp,path,raw):
    existing=read_remote(sftp,path)
    if existing is not None:
        assert existing==raw,'Existing recovery artifact differs; inspect rather than replace: '+path
        return
    with sftp.open(path,'wbx') as f:f.write(raw)
    assert read_remote(sftp,path)==raw

def validation(client,sftp,cfg):
    base=cfg['root']+'/'+REMOTE
    receipt=read_remote(sftp,base+'/validation_dispatch.json')
    if receipt:return {'already_dispatched':True,'receipt':json.loads(receipt)}
    extra=dict(cfg,validation_result_path=base+'/checkpoint_validation.json',validation_progress_path=base+'/validation_progress.json')
    source=('CONFIG='+repr(extra)+'\n').encode()+(HERE/'validate_resume_checkpoints_remote.py').read_bytes()
    ast.parse(source);put_new(sftp,base+'/checkpoint_validator.py',source)
    program=r'''
import datetime,json,os,subprocess
from pathlib import Path
base=Path(CONFIG['root'])/'recovery_20261019_v2'
with (base/'validation_launch_intent.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
with (base/'validation.stdout').open('xb') as log:
 child=subprocess.Popen([CONFIG['python'],'-u',str(base/'checkpoint_validator.py')],cwd=Path(CONFIG['root'])/'code',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],'account':CONFIG['account']}
with (base/'validation_dispatch.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
'''
    return execute(client,cfg['account'],program,timeout=30)

def status(sftp,cfg):
    base=cfg['root']+'/'+REMOTE;result={}
    for name in ['validation_dispatch.json','validation_progress.json','checkpoint_validation.json','dispatch_receipt.json']:
        raw=read_remote(sftp,base+'/'+name)
        if raw is None:continue
        parsed=json.loads(raw);result[name]=parsed
        path=DEST/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+cfg['account']+'_'+name)
        path.write_bytes(raw)
        if name=='checkpoint_validation.json':
            result['validation_saved']=str(path);result['validation_sha256']=hashlib.sha256(raw).hexdigest()
    raw=read_remote(sftp,base+'/validation.stdout')
    if raw:result['validation_log_tail']=raw.decode(errors='replace')[-1600:]
    return result

def deploy(client,sftp,cfg,preflight):
    base=cfg['root']+'/'+REMOTE
    existing=read_remote(sftp,base+'/dispatch_receipt.json')
    if existing:return {'already_dispatched':True,'receipt':json.loads(existing)}
    source=json.loads(json.loads(preflight.read_text(encoding='utf-8'))['results'][cfg['account']]['stdout'])
    assert not source['processes'] and not source['frozen_mismatches'] and source['cuda_available']
    raw=read_remote(sftp,base+'/checkpoint_validation.json');assert raw
    checked=json.loads(raw);assert checked['all_passed'] and len(checked['jobs'])==3
    deep={j['job']:j for j in checked['jobs']}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+cfg['account']
    registry=json.loads((DEST.parent/'connection_registry.json').read_text(encoding='utf-8'))
    proof={'account':cfg['account'],'seed':source['plan']['seed'],'recovery_id':stamp,
           'plan_sha256':registry[cfg['account']]['plan_sha256'],
           'checkpoint_validation_sha256':hashlib.sha256(raw).hexdigest(),'preflight_source':str(preflight),'jobs':[]}
    for job in source['jobs']:
        check=deep[job['job']]
        assert check['passed'] and check['checkpoint_sha256']==job['checkpoint_sha256'] and check['last_epoch']==job['last_epoch']
        proof['jobs'].append({k:job[k] for k in ['job','arm','checkpoint','checkpoint_sha256','last_epoch']})
    code=(HERE/CONTROLLER).read_bytes();ast.parse(code)
    put_new(sftp,cfg['root']+'/code/'+CONTROLLER,code)
    put_new(sftp,base+'/resume_preflight.json',json.dumps(proof,indent=2).encode())
    program=r'''
import datetime,json,os,subprocess
from pathlib import Path
root=Path(CONFIG['root']);base=root/'recovery_20261019_v2'
with (base/'launch_intent.json').open('x') as f:json.dump({'dispatch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'account':CONFIG['account']},f)
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
with (root/'logs/controller_recovery_20261019_v2.log').open('xb') as log:
 child=subprocess.Popen([CONFIG['python'],'-u',str(root/'code/resume_uavdt_queue_20261019_v2.py')],cwd=root/'code',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'controller_pid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],'account':CONFIG['account'],'dispatched':True,'not_yet_training_verified':True}
with (base/'dispatch_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
'''
    return {'proof':proof,'controller_sha256':hashlib.sha256(code).hexdigest(),'dispatch':execute(client,cfg['account'],program,timeout=30)}

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','status','deploy']);p.add_argument('--account',choices=['A','B','both'],default='both');p.add_argument('--preflight',type=Path)
    args=p.parse_args();assert args.action!='deploy' or args.preflight
    DEST.mkdir(parents=True,exist_ok=True);passwords=json.loads(scheduled_monitor_v4.credentials_bytes());results={}
    try:
        for account in ACCOUNTS if args.account=='both' else [args.account]:
            client=None
            try:
                cfg=ACCOUNTS[account];client=connect(cfg['port'],passwords);sftp=client.open_sftp();base=cfg['root']+'/'+REMOTE
                if args.action!='status':
                    try:sftp.stat(base)
                    except FileNotFoundError:sftp.mkdir(base)
                if args.action=='validate':result=validation(client,sftp,cfg)
                elif args.action=='status':result=status(sftp,cfg)
                else:result=deploy(client,sftp,cfg,args.preflight)
                results[account]=result;sftp.close()
            except Exception as exc:results[account]={'error':type(exc).__name__+': '+str(exc)}
            finally:
                if client:client.close()
    finally:passwords.clear()
    path=DEST/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+args.action+'.json')
    path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'saved':str(path),'results':results},ensure_ascii=False,indent=2))
    if any('error' in v for v in results.values()):raise SystemExit(1)

if __name__=='__main__':main()
