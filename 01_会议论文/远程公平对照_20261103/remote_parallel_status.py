"""Read-only live snapshot of either host in the split BASE/FULL execution."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
ROOT=Path('/25085400120/mpsma_fair_20261103_v1')
def read(name):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tail(name,size=1800):
    p=ROOT/name
    if not p.exists():return None
    with p.open('rb') as f:
        f.seek(0,2);length=f.tell();f.seek(max(0,length-size));return f.read().decode(errors='replace')
def identity(pid,expected):
    p=Path('/proc/'+str(pid))
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        same=str(expected)==stat[19]
        return {'pid':pid,'state':stat[0],'start_ticks':stat[19],'identity_matches':same,
                'live':same and stat[0] not in ['Z','X']}
    except FileNotFoundError:return {'pid':pid,'present':False,'live':False}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
secondary=gpu.startswith('GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a,')
assert secondary or gpu.startswith('GPU-85eca9af-10f3-4279-d72a-05ff1ecdcd3a,')
queue_name='secondary_queue.json' if secondary else 'queue.json'
queue=read(queue_name)
out={'observed_remote_utc':datetime.now(timezone.utc).isoformat(),'host_port':30989 if secondary else 30166,
     'gpu':gpu,'queue_name':queue_name,'queue':queue,'processes':[],'arms':{},
     'reservation':read('primary_full_reservation.json'),'preparation':read('secondary_preparation.json')}
out['hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip()
if queue:
    out['queue_sha256']=sha(ROOT/queue_name)
    out['processes'].append(identity(queue['supervisor_pid'],queue['supervisor_start_ticks']))
    for run in queue['runs']:out['processes'].append(identity(run['pid'],run['start_ticks']))
arm='FULL' if secondary else 'BASE'
data={'receipt':read('runs/'+arm+'/receipt.json')}
for filename,key in [('epochs.jsonl','complete_epoch_rows'),('actual_first100.jsonl','actual_first100_rows')]:
    p=ROOT/'runs'/arm/filename
    if p.exists():
        rows=[]
        for line in p.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
        data[key]=len(rows)
        if filename=='epochs.jsonl':data['latest_complete_epoch']=rows[-1] if rows else None
        else:
            expected=read('preflight/'+arm+'.first100.json')
            data['available_rows_match_preflight']=rows==expected[:len(rows)]
            data['all_first100_verified']=len(rows)==100 and rows==expected
data['log_tail']=tail(arm+'.train.log');out['arms'][arm]=data
out['unexpected_other_arm_directory']=(ROOT/'runs'/('BASE' if secondary else 'FULL')).exists()
protocol=read('execution_protocol.json')
out['protocol_sha256']=sha(ROOT/'execution_protocol.json')
manifest={f['path']:f['sha256'] for f in protocol['frozen_files']}
critical=['fair_runner.py','supervise.py','execution_configs/BASE.yml','execution_configs/FULL.yml','environment.current.json']
out['critical_frozen_files_match']={f:sha(ROOT/f)==manifest[f] for f in critical}
for name in ['parallel_execution_addendum.json','primary_full_reservation.json','secondary_runtime.json','supervise_secondary_full.py']:
    if (ROOT/name).exists():out[name+'_sha256']=sha(ROOT/name)
if out['reservation']:
    name='primary_FULL_reserved_log.json' if secondary else 'FULL.train.log'
    out['reserved_log_matches']=sha(ROOT/name)==out['reservation']['reserved_log_sha256']
out['supervisor_log_tail']=tail('secondary_supervisor.log' if secondary else 'training_supervisor.log')
print(json.dumps(out))
