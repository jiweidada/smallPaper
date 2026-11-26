"""Reserve the queued FULL slot for the second host, leaving BASE untouched.

Run only after the secondary preparation has passed. The frozen supervisor's
exclusive FULL log open then prevents a duplicate launch. BASE retains its
original parent, timeout watchdog, complete receipt and actual exit code.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
ROOT=Path('/25085400120/mpsma_fair_20261103_v1')
PROTOCOL_SHA='8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def live(pid,ticks):
    a=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
    assert a[19]==ticks and a[0] not in ['Z','X'], 'Process identity mismatch'
assert sha(ROOT/'execution_protocol.json')==PROTOCOL_SHA
queue=read(ROOT/'queue.json')
assert queue['status']=='RUNNING' and len(queue['runs'])==1
base=queue['runs'][0]
assert base['arm']=='BASE' and base['status']=='RUNNING'
assert queue['supervisor_pid']==1193 and queue['supervisor_start_ticks']=='39931198'
assert base['pid']==1194 and base['start_ticks']=='39931668'
live(queue['supervisor_pid'],queue['supervisor_start_ticks']);live(base['pid'],base['start_ticks'])
assert not (ROOT/'runs/FULL').exists()
assert not (ROOT/'FULL.train.log').exists()
preparation=read(ROOT/'secondary_preparation_verified.json')
assert preparation['status']=='READY_FOR_AUTHORIZED_FULL_ONLY'
assert preparation['runtime_matches_frozen_software'] and preparation['dataset_files_verified']==7021
assert preparation['frozen_files_verified']==247 and preparation['protocol_sha256']==PROTOCOL_SHA
test=read(ROOT/'queue_reservation_test.json')
assert test['status']=='SYNTHETIC_RESERVATION_TEST_PASSED' and test['full_process_not_created']
supplement=read(ROOT/'parallel_execution_addendum.json')
assert supplement['status']=='AUTHORIZED_EXECUTION_PLACEMENT_CHANGE_ONLY'
assert supplement['arms']==['BASE','FULL'] and supplement['additional_model_runs']==0
reservation={'status':'FULL_RESERVED_FOR_SECONDARY_HOST_DO_NOT_REMOVE',
 'created_primary_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':PROTOCOL_SHA,
 'primary_port':30166,'secondary_port':30989,'secondary_gpu_uuid':'GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a',
 'original_supervisor':{'pid':1193,'start_ticks':'39931198'},'base':{'pid':1194,'start_ticks':'39931668'},
 'secondary_preparation_sha256':sha(ROOT/'secondary_preparation_verified.json'),
 'addendum_sha256':sha(ROOT/'parallel_execution_addendum.json'),
 'effect':'Native exclusive FULL.train.log open prevents the first host from spawning FULL. BASE continues under its original supervisor.',
 'expected_original_scheduler_end':'After successful BASE exit is recorded, the original scheduler exits with FileExistsError before creating FULL. Its top-level RUNNING status is superseded; inspect BASE run and secondary_queue.json separately.',
 'not_a_training_log':True,'model_runs_started_by_this_action':0}
with (ROOT/'FULL.train.log').open('x') as f:
    f.write(json.dumps(reservation,indent=2)+'\n');f.flush();os.fsync(f.fileno())
reservation_receipt={**reservation,'reserved_log_sha256':sha(ROOT/'FULL.train.log')}
with (ROOT/'primary_full_reservation.json').open('x') as f:f.write(json.dumps(reservation_receipt,indent=2)+'\n')
live(queue['supervisor_pid'],queue['supervisor_start_ticks']);live(base['pid'],base['start_ticks'])
assert read(ROOT/'queue.json')['runs'][0]['pid']==1194 and not (ROOT/'runs/FULL').exists()
print(json.dumps(reservation_receipt))
