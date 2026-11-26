"""Read-only snapshot; validate live process identity rather than trusting a lock."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
ROOT=Path('/25085400120/mpsma_fair_20261103_v1')
def read(name):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else None
def tail(name,size=2400):
    p=ROOT/name
    if not p.exists():return None
    with p.open('rb') as f:
        f.seek(0,2);length=f.tell();f.seek(max(0,length-size));return f.read().decode(errors='replace')
def proc_identity(pid,expected):
    p=Path(f'/proc/{pid}')
    if not p.exists():return {'pid':pid,'present':False,'live':False}
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        same=str(expected)==stat[19]
        return {'pid':pid,'state':stat[0],'start_ticks':stat[19],'identity_matches':same,
                'live':same and stat[0] not in ['Z','X'],'io':(p/'io').read_text()}
    except FileNotFoundError:return {'pid':pid,'present':False,'live':False}
out={'observed_remote_utc':datetime.now(timezone.utc).isoformat(),'processes':[],'arms':{}}
launch=read('training_launch.json');queue=read('queue.json')
out['launch']=launch;out['queue']=queue
if launch:out['processes'].append(proc_identity(launch['pid'],launch['start_ticks']))
if queue:
    for run in queue.get('runs',[]):out['processes'].append(proc_identity(run['pid'],run['start_ticks']))
for arm in ['BASE','FULL']:
    data={'receipt':read(f'runs/{arm}/receipt.json')}
    rows_path=ROOT/f'runs/{arm}/epochs.jsonl'
    if rows_path.exists():
        rows=[]
        for line in rows_path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
        data['complete_epoch_rows']=len(rows)
        data['latest_complete_epoch']=rows[-1] if rows else None
    first_path=ROOT/f'runs/{arm}/actual_first100.jsonl'
    if first_path.exists():
        rows=[]
        for line in first_path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
        expected=read(f'preflight/{arm}.first100.json')
        data['actual_first100_rows']=len(rows)
        data['available_rows_match_preflight']=rows==expected[:len(rows)]
        data['all_first100_verified']=len(rows)==100 and rows==expected
    data['log_tail']=tail(f'{arm}.train.log')
    out['arms'][arm]=data
protocol=read('execution_protocol.json')
out['protocol_sha256']=hashlib.sha256((ROOT/'execution_protocol.json').read_bytes()).hexdigest()
critical=['fair_runner.py','supervise.py','execution_configs/BASE.yml','execution_configs/FULL.yml','environment.current.json']
manifest={r['path']:r['sha256'] for r in protocol['frozen_files']}
out['critical_frozen_files_match']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==manifest[p] for p in critical}
out['supervisor_log_tail']=tail('training_supervisor.log')
audit_launch=read('early_evidence_audit_launch.json')
if audit_launch:
    out['early_evidence_audit_launch']=audit_launch
    out['processes'].append(proc_identity(audit_launch['pid'],audit_launch['start_ticks']))
    out['evidence_audits']={str(p.parent.relative_to(ROOT)):json.loads(p.read_text())
                            for p in (ROOT/'evidence_audits').glob('*/receipt.json')}
    out['early_evidence_audit_log_tail']=tail('early_evidence_audit.log')
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps(out))
