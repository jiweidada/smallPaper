from pathlib import Path
import argparse
import concurrent.futures
import datetime as dt
import json
import psutil
import re
from ssh_control import connect,run_script

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
assert re.fullmatch('[A-Za-z0-9_]+',a.label)
target=HERE/('live_pair_'+a.label+'.json');assert not target.exists()
def remote(port):
    c=connect(port)
    try:
        result=run_script(c,(HERE/'observe_remote_run.py').read_text(encoding='utf-8'),timeout=90)
        assert result['exit_code']==0,result
        return {'port':port,**json.loads(result['stdout'])}
    finally:c.close()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(remote,[30715,30048]))
local=[]
for x in json.loads((HERE/'backup_clients_launched_attempt2.json').read_text(encoding='utf-8')):
    root=Path(x['local_control']);record={'arm':x['arm'],'pid':x['pid'],'pid_alive':psutil.pid_exists(x['pid'])}
    if record['pid_alive']:
        proc=psutil.Process(x['pid']);record.update(command_matches=proc.cmdline()==x['command'],created_timestamp=proc.create_time())
    for key,name in [('status','backups/monitor_status.json'),('probe_payload_validation','backups/probe/payload_validation.json')]:
        f=root/name;record[key]=json.loads(f.read_text(encoding='utf-8')) if f.exists() else None
    record['stderr_tail']=(root/'monitor.attempt2.stderr.log').read_text(encoding='utf-8')[-3000:]
    epochs=root/'backups/epochs';record['saved_epoch_directories']=sorted(p.name for p in epochs.iterdir() if p.is_dir()) if epochs.is_dir() else []
    local.append(record)
report={'observed_local_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'remote':rows,'local_backup_clients':local}
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
summary=[]
for r in rows:
    receipt=r['records']['run/receipt.json'] or {};supervisor=r['records']['supervisor.json'] or {}
    summary.append({'arm':r['arm'],'status':receipt.get('status',supervisor.get('status')),'phase':receipt.get('phase'),
                    'completed_rounds':receipt.get('completed_epochs',0),'actual_epoch_0based':receipt.get('actual_loop_epoch'),
                    'batch':receipt.get('batch'),'attempts':receipt.get('optimizer_step_attempts'),'loss':receipt.get('last_loss'),
                    'processes_live':all(x.get('live',False) for x in r['processes'].values()),'gpu':r['gpu'],
                    'actual_first100_rows':r['actual_first100_rows'],'log_tail':r['logs'].get('training.log','')[-600:]})
print(json.dumps({'remote':summary,'local_backup_clients':[{'arm':x['arm'],'pid_alive':x['pid_alive'],'command_matches':x.get('command_matches',False),'status':x['status'],'stderr_tail':x['stderr_tail'],'saved_epoch_directories':x['saved_epoch_directories']} for x in local],'report':str(target)},ensure_ascii=False,indent=2))
