from pathlib import Path
import datetime
import json
import os
import subprocess

volume=Path('/25085400120') if Path('/25085400120').is_mount() else Path('/24085403014')
arm='BASE_FRESH' if volume.name=='25085400120' else 'FULL_FRESH'
root=Path(globals().get('observed_run_root', volume/'mpsma_fresh_pair_20261112_v1'/arm))
arm=root.name
records={}
for name in ['supervisor.json','run/receipt.json']:
    p=root/name
    records[name]=json.loads(p.read_text()) if p.is_file() else None
supervisor=records['supervisor.json'] or {}
processes={}
for key in ['supervisor_pid','pid']:
    if key not in supervisor:continue
    pid=supervisor[key]
    p=Path('/proc')/str(pid)
    info={'pid':pid,'exists':p.exists()}
    if p.exists():
        fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
        info.update(state=fields[0],start_ticks=int(fields[19]),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip())
        expected=supervisor['supervisor_start_ticks' if key=='supervisor_pid' else 'start_ticks']
        info['identity_matches']=info['start_ticks']==expected
        info['live']=info['identity_matches'] and info['state'] not in ['Z','X']
    processes[key]=info
logs={}
for name in ['supervisor.log','training.log']:
    p=root/name
    if p.is_file():
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4500));logs[name]=f.read().decode(errors='replace')
first=root/'run/actual_first100.jsonl'
first_count=sum(1 for line in first.read_bytes().splitlines() if line) if first.is_file() else 0
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True).strip()
print(json.dumps({'observed_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arm':arm,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  'records':records,'processes':processes,'gpu':gpu,'actual_first100_rows':first_count,'logs':logs}))
