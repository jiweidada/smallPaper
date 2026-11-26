"""Validate fresh read-only snapshots, then update the split-host live ledger."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA='8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
arms={};snapshots={};reservations=[]
for arm,port,name in [('BASE',30166,'primary_parallel_live.json'),('FULL',30989,'secondary_parallel_live.json')]:
    p=ROOT/name
    assert time.time()-p.stat().st_mtime<180, 'Stale snapshot: '+name
    raw=json.loads(p.read_text(encoding='utf-8'))
    assert raw['exit_code']==0
    s=json.loads(raw['stdout']);q=s['queue']
    assert s['host_port']==port and s['protocol_sha256']==PROTOCOL_SHA
    uuid='GPU-85eca9af-10f3-4279-d72a-05ff1ecdcd3a' if arm=='BASE' else 'GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a'
    assert s['hardware']=='NVIDIA GeForce RTX 3090, '+uuid+', 550.100, 24576'
    assert q and [v['arm'] for v in q['runs']]==[arm]
    assert q['protocol_sha256']==PROTOCOL_SHA and not q['automatic_retry']
    assert q['hard_timeout_seconds_per_arm']==604800
    assert all(s['critical_frozen_files_match'].values())
    assert s['reserved_log_matches'] and not s['unexpected_other_arm_directory']
    reservations.append(s['primary_full_reservation.json_sha256'])
    run=q['runs'][0];receipt=s['arms'][arm]['receipt']
    assert receipt and receipt['protocol_sha256']==PROTOCOL_SHA
    assert receipt['arm']==arm and receipt['pid']==run['pid']
    if run['status']=='RUNNING':
        for pid,ticks in [(q['supervisor_pid'],q['supervisor_start_ticks']),(run['pid'],run['start_ticks'])]:
            current=next(v for v in s['processes'] if v['pid']==pid)
            assert current['live'] and current['start_ticks']==ticks, 'Missing live identity: '+arm
        assert receipt['status'] in ['INITIALIZING','RUNNING']
    else:
        assert run['status']=='COMPLETED' and run['exit_code']==0 and receipt['status']=='COMPLETED'
        assert receipt['completed_epochs']==320 and receipt['optimizer_step_attempts']==517440
    data=s['arms'][arm]
    if data.get('actual_first100_rows',0)>0:assert data['available_rows_match_preflight']
    if receipt['optimizer_step_attempts']>=100:assert data['all_first100_verified']
    arms[arm]={'port':port,'status':run['status'],'pid':run['pid'],'start_ticks':run['start_ticks'],
               'completed_epochs':receipt.get('completed_epochs',0),'actual_loop_epoch':receipt.get('actual_loop_epoch'),
               'optimizer_step_attempts':receipt['optimizer_step_attempts'],
               'first100_verified':data.get('all_first100_verified',False),'gpu':s['gpu']}
    snapshots[arm]={'path':name,'sha256':sha(p),'observed_remote_utc':s['observed_remote_utc'],
                    'received_local_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()}
assert reservations[0]==reservations[1]==sha(ROOT/'primary_full_reservation.json')
v={'status':'BOTH_TRAINING_COMPLETE_FINAL_EVIDENCE_AUDIT_REQUIRED' if all(a['status']=='COMPLETED' for a in arms.values()) else 'SPLIT_HOST_TRAINING_RUNNING',
   'updated_local_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':PROTOCOL_SHA,
   'placement_addendum_sha256':sha(ROOT/'parallel_execution_addendum.json'),'reservation_sha256':reservations[0],
   'snapshots':snapshots,'arms':arms,'submission_ready':False,
   'authority_note':'Primary per-BASE run plus secondary per-FULL run. Original serial queue top-level RUNNING becomes stale after planned exclusive-log retirement; never interpret it as an additional training run.'}
temp=ROOT/'parallel_execution_ledger.json.tmp';temp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
temp.replace(ROOT/'parallel_execution_ledger.json')
print(json.dumps({'status':v['status'],'arms':arms},ensure_ascii=False))
