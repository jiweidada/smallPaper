"""Bounded one-shot acceptance for this repair turn; not a recurring automation."""
import concurrent.futures
import datetime
import json
import pathlib
import subprocess
import sys
import time

here=pathlib.Path(__file__).resolve().parent
destination=here.parents[2]/'英文小论文/data/06_第二轮_20261014/恢复执行_20261016_v5'
accounts={'A':(30512,'/25085400120/envs/deim/bin/python'),
          'B':(30059,'/24085403014/deim_migration_runtime/envs/deim/bin/python')}
accepted={}
deadline=time.monotonic()+15*60


def inspect(account):
    port,python=accounts[account]
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=destination/('first_save_probe_'+stamp+'_'+account+'.json')
    command=[sys.executable,'-X','utf8','-B',str(here/'scheduled_monitor_v4.py'),'--remote','--port',str(port),
             '--python',python,'--script',str(here/'verify_resume_v5.py'),'--script-arg',account,
             '--output',str(output),'--timeout','60']
    try:
        p=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=110)
        if p.returncode:
            return account,output,None,p.stderr[-700:]
        data=json.loads(output.read_text(encoding='utf-8'))
        return account,output,data,None
    except Exception as exc:
        return account,output,None,type(exc).__name__+': '+str(exc)


while len(accepted)<2 and time.monotonic()<deadline:
    pending=[a for a in accounts if a not in accepted]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for account,path,data,error in pool.map(inspect,pending):
            if error:
                print(json.dumps({'account':account,'read_error':error},ensure_ascii=False),flush=True)
                continue
            summary={'account':account,'remote_utc':data['remote_utc'],'first_saved_epochs_verified':data['all_first_saved_epochs_verified'],
                     'jobs':[{k:r.get(k) for k in ['job','latest_step','evaluated_epochs','output_path_verified','first_saved_epoch_verified']}
                             for r in data['jobs']]}
            print(json.dumps(summary,ensure_ascii=False),flush=True)
            if data['queue_state']['state'] in ['failed','draining_after_failure'] or data['queue_state'].get('errors'):
                print('QUEUE_FAILURE: preserve evidence and inspect; no retry performed.',flush=True)
                raise SystemExit(2)
            if data['all_first_saved_epochs_verified']:
                accepted[account]=path
    if len(accepted)<2:
        time.sleep(40)
if len(accepted)!=2:
    print(json.dumps({'acceptance_incomplete':True,'accounts_verified':list(accepted)},ensure_ascii=False),flush=True)
    raise SystemExit(3)
result=subprocess.run([sys.executable,'-X','utf8','-B',str(here/'finish_repair_v5.py'),str(accepted['A']),str(accepted['B'])])
raise SystemExit(result.returncode)
