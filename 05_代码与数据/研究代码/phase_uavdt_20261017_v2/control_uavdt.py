"""Scoped pinned SSH controller; credentials stay in current-user DPAPI and memory."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_second_dataset_20261017_v1'))
from scoped_data_transfer import connect, PYTHONS, scheduled_monitor_v4

DATA=HERE.parents[1]/'data/07_第二数据集_20261017/执行_v2'
ACCOUNTS={'A':{'port':30512,'storage':'/25085400120','old':'/25085400120/DEIM/experiments/phase_mechanism_20261012'},
          'B':{'port':30059,'storage':'/24085403014','old':'/24085403014/experiments/phase_mechanism_20261012'}}
for a,c in ACCOUNTS.items():c.update(account=a,python=PYTHONS[c['port']],root=str(Path(c['old']).parent/'phase_second_dataset_uavdt_20261017_v1').replace('\\','/'))
registry=HERE.parents[1]/'data/07_第二数据集_20261017/connection_registry.json'
if registry.exists():
    for a,endpoint in json.loads(registry.read_text(encoding='utf-8')).items():
        assert endpoint['root']==ACCOUNTS[a]['root'] and endpoint['storage']==ACCOUNTS[a]['storage']
        ACCOUNTS[a]['port']=int(endpoint['port'])

PROBE=r'''
import base64,datetime,hashlib,json,pathlib,shutil,subprocess
P=pathlib.Path;old=P(CONFIG['old']);stage=old/'artifacts/round2_20261014_v1'
pointer=json.loads((stage/'current_execution.json').read_text());qpath=stage/pointer['execution_directory']/'queue_state.json'
code=stage/'execution_v3/code';raw=qpath.read_bytes();q=json.loads(raw)
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr[:500]}
result={'checked_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'account':CONFIG['account'],'old_queue':{'state':q['state'],'sha256':hashlib.sha256(raw).hexdigest(),'training_completed':q.get('training_completed_jobs'),'completed':q.get('completed_jobs')},'new_root_exists':P(CONFIG['root']).exists(),'free_bytes':shutil.disk_usage(CONFIG['storage']).free,'gpu':command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),'gpu_compute':command(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']),'source_files':[]}
for relative in ['engine/data/dataset/coco_dataset.py','engine/data/transforms/container.py','engine/data/transforms/transforms.py','engine/data/dataset/__init__.py']:
 p=code/relative
 if p.is_file():
  b=p.read_bytes();result['source_files'].append({'relative':relative,'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
 else:result['source_files'].append({'relative':relative,'missing':True})
if CONFIG['account']=='B':
 root=P(CONFIG['storage'])/'datasets_verified_20261017/UAVDT_official';receipt=root/'complete_image_replica_v2.json'
 result['data_receipt']=json.loads(receipt.read_text()) if receipt.is_file() else None
print(json.dumps(result),flush=True)
'''

def execute(client,account,program,timeout=180):
    cfg=ACCOUNTS[account]
    command='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(cfg['python'])+' -u -'
    channel=client.get_transport().open_session(timeout=20);channel.exec_command(command)
    channel.sendall(('CONFIG='+repr(cfg)+'\n'+program).encode());channel.shutdown_write()
    out=bytearray();err=bytearray();deadline=time.monotonic()+timeout
    DATA.mkdir(parents=True,exist_ok=True)
    live_path=DATA/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+account+'_live.stdout')
    live=live_path.open('xb')
    while True:
        while channel.recv_ready():
            chunk=channel.recv(65536);out.extend(chunk);live.write(chunk);live.flush()
        while channel.recv_stderr_ready():err.extend(channel.recv_stderr(65536))
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():break
        if time.monotonic()>deadline:
            channel.close();live.close();return {'returncode':None,'timed_out':True,'stdout':out.decode(errors='replace'),'stderr':err.decode(errors='replace')[-3000:],'do_not_repeat_mutation_blindly':True}
        time.sleep(.05)
    live.close()
    return {'returncode':channel.recv_exit_status(),'stdout':out.decode(errors='replace'),'stderr':err.decode(errors='replace'),'live_stdout':str(live_path)}

def run_one(account,passwords,program,timeout):
    client=None
    try:
        client=connect(ACCOUNTS[account]['port'],passwords)
        return execute(client,account,program,timeout)
    except Exception as exc:return {'returncode':None,'error':type(exc).__name__+': '+str(exc)}
    finally:
        if client:client.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['probe','script']);p.add_argument('--account',choices=['A','B','both'],default='both')
    p.add_argument('--script',type=Path);p.add_argument('--timeout',type=int,default=180);p.add_argument('--tag',default='probe')
    args=p.parse_args();assert args.action=='probe' or args.script
    program=PROBE if args.action=='probe' else args.script.read_text(encoding='utf-8')
    accounts=list(ACCOUNTS) if args.account=='both' else [args.account]
    passwords=json.loads(scheduled_monitor_v4.credentials_bytes())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(accounts)) as pool:
            pending={a:pool.submit(run_one,a,passwords,program,args.timeout) for a in accounts}
            results={a:f.result() for a,f in pending.items()}
    finally:passwords.clear()
    now=datetime.datetime.now(datetime.timezone.utc);DATA.mkdir(parents=True,exist_ok=True)
    path=DATA/(now.strftime('%Y%m%dT%H%M%S_%fZ')+'_'+args.tag+'.json')
    record={'checked_utc':now.isoformat(),'action':args.action,'results':results}
    with path.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    if args.action=='probe':
        for account,result in results.items():
            if result.get('returncode')!=0:continue
            parsed=json.loads(result['stdout']);reference=HERE/'frozen_reference'
            if parsed.get('data_receipt'):parsed['data_receipt'].pop('legacy_converted_coverage_comparison',None)
            for entry in parsed.pop('source_files',[]):
                if entry.get('missing'):continue
                raw=__import__('base64').b64decode(entry['base64']);assert hashlib.sha256(raw).hexdigest()==entry['sha256']
                target=reference/entry['relative'];target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():assert target.read_bytes()==raw
                else:target.write_bytes(raw)
            result['summary']=parsed;result.pop('stdout',None)
    else:
        for result in results.values():
            result['stdout_tail']=result.pop('stdout','')[-3000:]
    print(json.dumps({'saved':str(path),'results':results},ensure_ascii=False,indent=2),flush=True)
    if any(result.get('returncode')!=0 for result in results.values()):raise SystemExit(1)

if __name__=='__main__':main()
