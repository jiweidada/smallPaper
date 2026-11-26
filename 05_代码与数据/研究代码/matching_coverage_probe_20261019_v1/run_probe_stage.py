"""Start one explicitly requested bounded probe stage after old backup/release."""
import argparse,datetime,hashlib,json,pathlib,sys
from control_probe import ctl,ROOTS,DATA,LOCAL,HERE
def main(action,account):
    old=ctl.HERE.parents[1]/'data/07_第二数据集_20261017/用户暂停与本地备份_20261019_v1'
    local_proofs={a:json.loads((old/a/'local_backup_verified.json').read_text(encoding='utf8')) for a in ['A','B']}
    assert all(v['all_sha256_match'] for v in local_proofs.values())
    passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes());client=None
    try:
        cfg=ctl.ACCOUNTS[account];root=ROOTS[account];client=ctl.connect(cfg['port'],passwords);sftp=client.open_sftp()
        seed=1 if account=='A' else 2
        task={'root':root,'name':'inputs' if action=='inputs' else f'probe_seed{seed}','timeout_seconds':1800,'cpu_only':action=='inputs'}
        task['command']=[cfg['python'],'-u','-B',root+'/probe/make_inputs.py','--data',DATA[account]] if action=='inputs' else [cfg['python'],'-u','-B',root+'/probe_driver_v2.py','--seed',str(seed)]
        assert action!='inputs' or account=='A'
        helper=root+'/bounded_task.py'
        try:
            with sftp.open(helper,'rb') as f:assert f.read()==(HERE/'bounded_task.py').read_bytes()
        except FileNotFoundError:
            with sftp.open(helper,'wx') as f:f.write((HERE/'bounded_task.py').read_bytes())
        taskpath=root+'/'+task['name']+'_task.json'
        with sftp.open(taskpath,'wx') as f:f.write(json.dumps(task).encode())
        program='''import pathlib,json,subprocess,os,datetime
root=pathlib.Path(PROBE_ROOT);old=pathlib.Path(CONFIG['root'])/'pause_user_20261019_v1/release_receipt.json'
v=json.loads(old.read_text());assert v['state']=='paused_backup_verified_GPU_released'
assert json.loads((root/f'preflight_seed{PROBE_SEED}.json').read_text())['state']=='CPU_preflight_passed'
assert json.loads((root/f'extension_preflight_seed{PROBE_SEED}.json').read_text())['state']=='CPU_extension_preflight_passed'
assert not (root/'task_records'/(TASK['name']+'.lock')).exists()
if not TASK['cpu_only']:assert (root/'inputs/complete.json').is_file()
path=root/(TASK['name']+'_supervisor.log')
with path.open('xb') as f:
 p=subprocess.Popen([CONFIG['python'],'-u','-B',str(root/'bounded_task.py'),str(root/(TASK['name']+'_task.json'))],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 result={'account':CONFIG['account'],'task':TASK['name'],'supervisor_pid':p.pid,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/(TASK['name']+'_launch.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
'''
        result=ctl.execute(client,account,'PROBE_ROOT='+repr(root)+'\nPROBE_SEED='+repr(seed)+'\nTASK='+repr(task)+'\n'+program,timeout=30)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(LOCAL/(stamp+'_'+task['name']+'_dispatch.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(result,ensure_ascii=False,indent=2));assert result['returncode']==0
    finally:
        passwords.clear()
        if client:client.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inputs','forward']);p.add_argument('--account',choices=['A','B'],required=True);a=p.parse_args();main(a.action,a.account)
