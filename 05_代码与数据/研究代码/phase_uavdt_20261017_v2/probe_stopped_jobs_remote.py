"""Read-only process inventory and stop-site evidence; never launch, signal or edit jobs."""
import datetime,json,os,re,subprocess
from pathlib import Path
root=Path(CONFIG['root']);plan=json.loads((root/'plan.json').read_text())
ansi=re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
def describe(p):
    if not p.exists():return {'exists':False}
    st=p.stat();return {'exists':True,'bytes':st.st_size,'mtime_remote_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat()}
def tail(p,n=50):
    if not p.is_file():return []
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-160000));text=ansi.sub('',f.read().decode(errors='replace'))
    return text.splitlines()[-n:]
result={'account':CONFIG['account'],'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'owned_processes':[],'jobs':[],'controller_log':{},'root':str(root)}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        command=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        if any(s.startswith(str(root/'code')+'/') for s in command):
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            result['owned_processes'].append({'pid':int(proc.name),'state':stat[0],'start_ticks':stat[19],'command':command})
    except (FileNotFoundError,PermissionError):pass
for label,arguments in [('gpu',['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
                        ('gpu_compute',['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'])]:
    p=subprocess.run(arguments,capture_output=True,text=True,timeout=20)
    result[label]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr[:500]}
for name in ['controller.log','controller_recovery_20261018.log']:
    p=root/'logs'/name;result['controller_log'][name]=dict(describe(p),last50=tail(p))
for arm in plan['jobs']:
    job=f'uavdt_{arm}_full96_seed{plan["seed"]}';out=root/'outputs'/job;console=root/'logs'/(job+'.log');log=out/'log.txt'
    rows=[json.loads(s) for s in log.read_text().splitlines() if s.strip()]
    rows=[r for r in rows if 'test_uavdt_eval_bbox' in r]
    recent=tail(console)
    errors=[s for s in recent if re.search(r'Traceback|Error|Killed|out of memory|SIGTERM|SIGKILL',s,re.I)]
    result['jobs'].append({'job':job,'complete_epochs':len(rows),'last_complete_epoch_zero_based':rows[-1]['epoch'] if rows else None,
        'console':describe(console),'log':describe(log),'errors_in_last50':errors,
        'last50':recent,'checkpoint_files':{name:describe(out/name) for name in ['last.pth','best_stg1.pth','best_stg2.pth']}})
print(json.dumps(result),flush=True)
