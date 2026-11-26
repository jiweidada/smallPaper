"""Maintain a local status ledger from a successful read-only remote snapshot."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('snapshot');args=parser.parse_args()
source=(HERE/args.snapshot).resolve();assert source.parent==HERE
raw=json.loads(source.read_text(encoding='utf-8'));assert raw['exit_code']==0,raw
status=json.loads(raw['stdout'])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
expected=sha(HERE/'execution_protocol.json')
assert status['protocol_sha256']==expected
assert all(status['critical_frozen_files_match'].values())
audit_dir=HERE/'remote_evidence/evidence_audits/BASE_best_epoch0000'
audit=json.loads((audit_dir/'receipt.json').read_text())
assert audit['status']=='SAVED_ENDPOINT_BINDING_VERIFIED'
for name,key in [('checkpoint.pt','checkpoint_sha256'),('predictions.json','prediction_sha256'),('evaluation_arrays.npz','evaluation_arrays_sha256')]:
    assert sha(audit_dir/name)==audit[key],name
arms={arm:{key:value for key,value in state.items() if key!='log_tail'} for arm,state in status['arms'].items()}
live_by_pid={p['pid']:p.get('live',False) for p in status['processes']}
queue_status=status['queue']['status']
recorded_status='REMOTE_TRAINING_'+queue_status
if queue_status=='RUNNING':
    if not live_by_pid.get(status['queue']['supervisor_pid'],False):
        recorded_status='SUPERVISOR_NOT_LIVE_REQUIRES_INSPECTION'
    elif any(not live_by_pid.get(run['pid'],False) for run in status['queue']['runs'] if run['status']=='RUNNING'):
        recorded_status='RUN_TRANSITION_OR_EXIT_REQUIRES_INSPECTION'
report={'status':recorded_status,'snapshot':source.name,'snapshot_sha256':sha(source),
        'recorded_local_utc':datetime.now(timezone.utc).isoformat(),'observed_remote_utc':status['observed_remote_utc'],
        'protocol_sha256':expected,'processes':status['processes'],'arms':arms,
        'first_saved_epoch_audit':audit,'first_saved_epoch_audit_files_verified_after_download':True,
        'manuscript_sha256':sha(HERE.parent/'稿件/MPSMA-DEIM_conference_EN_revised.docx'),
        'manuscript_updated_with_new_results':False,'submission_ready':False,
        'scope_note':'Running-state evidence and an early checkpoint integrity audit. Await both complete fixed-budget runs and final endpoint audit before revising comparative conclusions.'}
tmp=HERE/'training_execution_ledger.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
tmp.replace(HERE/'training_execution_ledger.json')
print(json.dumps({'status':report['status'],'snapshot':source.name,
                  'BASE_completed_epochs':arms['BASE']['receipt'].get('completed_epochs',0),
                  'BASE_updates':arms['BASE']['receipt']['optimizer_step_attempts'],
                  'early_checkpoint_audit_verified_after_download':True}))
