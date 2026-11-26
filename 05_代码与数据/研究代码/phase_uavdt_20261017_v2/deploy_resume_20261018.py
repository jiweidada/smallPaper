"""Add a recovery controller and receipt; never overwrite frozen training files."""
import argparse, ast, datetime, hashlib, json, shlex, sys
from pathlib import Path
from control_uavdt import ACCOUNTS, connect, scheduled_monitor_v4, execute

HERE=Path(__file__).resolve().parent
DEST=HERE.parents[1]/'data/07_第二数据集_20261017/GPU重申请恢复_20261018'

def main():
    p=argparse.ArgumentParser();p.add_argument('--account',choices=['A','B'],required=True);p.add_argument('--preflight',type=Path,required=True)
    args=p.parse_args();cfg=ACCOUNTS[args.account]
    envelope=json.loads(args.preflight.read_text(encoding='utf-8'))['results'][args.account]
    assert envelope['returncode']==0
    source=json.loads(envelope['stdout'])
    assert source['account']==args.account and not source['processes'] and not source['frozen_mismatches'] and source['cuda_available']
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+args.account
    proof={'account':args.account,'seed':source['plan']['seed'],'recovery_id':stamp,
           'plan_sha256':hashlib.sha256(json.dumps(source['plan']).encode()).hexdigest(),
           'preflight_source':str(args.preflight),'jobs':[]}
    for j in source['jobs']:
        assert j['logged_epochs']==list(range(j['last_epoch']+1))
        assert j['optimizer_state_entries']>0 and j['scaler'] and j['ema_updates']>0
        proof['jobs'].append({k:j[k] for k in ['job','arm','checkpoint','checkpoint_sha256','last_epoch']})
    registry=json.loads((DEST.parent/'connection_registry.json').read_text(encoding='utf-8'))
    proof['plan_sha256']=registry[args.account]['plan_sha256']
    controller=(HERE/'resume_uavdt_queue_20261018.py').read_bytes();ast.parse(controller)
    passwords=json.loads(scheduled_monitor_v4.credentials_bytes());client=None
    try:
        client=connect(cfg['port'],passwords);sftp=client.open_sftp()
        directory=cfg['root']+'/recovery_20261018'
        try:sftp.stat(directory)
        except FileNotFoundError:sftp.mkdir(directory)
        files={cfg['root']+'/code/resume_uavdt_queue_20261018.py':controller,
               directory+'/resume_preflight.json':json.dumps(proof,indent=2).encode()}
        for path,raw in files.items():
            try:existing=sftp.stat(path)
            except FileNotFoundError:pass
            else:
                # Paramiko's x-only mode can leave an empty, never-dispatched upload.
                assert existing.st_size==0,'Recovery file exists; inspect rather than repeat deployment: '+path
                try:sftp.stat(directory+'/launch_intent.json')
                except FileNotFoundError:pass
                else:raise RuntimeError('Recovery dispatch already exists')
                sftp.rename(path,path+'.empty_upload_'+stamp)
            with sftp.open(path,'wbx') as f:f.write(raw)
            with sftp.open(path,'rb') as f:assert hashlib.sha256(f.read()).digest()==hashlib.sha256(raw).digest()
        sftp.close()
        program=r'''
import datetime,json,os,subprocess
from pathlib import Path
root=Path(CONFIG['root']);recovery=root/'recovery_20261018'
intent=recovery/'launch_intent.json'
assert not intent.exists(),'Already dispatched; inspect actual receipt, never launch twice'
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
with intent.open('x') as f:json.dump({'dispatch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'account':CONFIG['account']},f)
with (root/'logs/controller_recovery_20261018.log').open('xb') as log:
 child=subprocess.Popen([CONFIG['python'],'-u',str(root/'code/resume_uavdt_queue_20261018.py')],cwd=root/'code',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'controller_pid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],'account':CONFIG['account'],'dispatched':True,'not_yet_training_verified':True}
with (recovery/'dispatch_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
'''
        result=execute(client,args.account,program,timeout=30)
        DEST.mkdir(exist_ok=True)
        record={'account':args.account,'port':cfg['port'],'proof':proof,
                'controller_sha256':hashlib.sha256(controller).hexdigest(),'result':result}
        target=DEST/(stamp+'_resume_dispatch.json');target.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'saved':str(target),'result':result},ensure_ascii=False))
    finally:
        passwords.clear()
        if client:client.close()

if __name__=='__main__':main()
