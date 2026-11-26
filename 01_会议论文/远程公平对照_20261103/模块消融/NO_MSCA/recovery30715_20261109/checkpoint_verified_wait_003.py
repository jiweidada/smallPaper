"""Record a bounded live-session wait without touching scientific code or backups."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
ARM, BUNDLE = HERE.parent, HERE.parents[1]
OBSERVER = HERE / 'resume_monitor_20261110_003'
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ARM))
import backup_protocol as protocol

def read(p): return json.loads(p.read_text('utf-8-sig'))
def digest(data): return sha256(data).hexdigest()

p = argparse.ArgumentParser()
p.add_argument('--batch-start', required=True)
p.add_argument('--first-attempts', required=True, type=int)
p.add_argument('--polls', required=True, type=int)
args = p.parse_args()
assert 0 < args.polls <= 60 and 0 <= args.first_attempts <= 517440
assert datetime.fromisoformat(args.batch_start).tzinfo is not None
status = read(OBSERVER / 'observer_status.json')
live_path = OBSERVER / 'latest_live.json'
raw = live_path.read_bytes()
live = json.loads(json.loads(raw)['stdout'])
receipt, supervisor = [live['records'][key] for key in ['run/receipt.json', 'supervisor.json']]
native = read(BACKUPS / 'monitor_status.json')
assert native['status'] not in ['MONITOR_FAILED_NO_FURTHER_ACKS', 'MONITOR_STOPPED_BY_OPERATOR']
assert time.time() - live_path.stat().st_mtime < 100
for pid, created in [(8216, 1794292724.8624065), (40816, 1794312084.181138)]:
    assert abs(psutil.Process(pid).create_time() - created) < .01
assert status['status'] == 'REMOTE_WORKER_CONFIRMED_LIVE' and status['backup_client_alive']
assert live['boot_id'] == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
for key, ticks in [('pid', 26597622), ('supervisor_pid', 26597399)]:
    assert live['processes'][key]['exists'] and live['processes'][key]['start_ticks'] == ticks
assert receipt['first_epoch_in_this_segment'] == 164
assert receipt['runner_sha256'] == supervisor['runner_sha256'] == 'f5f7ef88ace026c5236734dfeeb29cdbc4688617f90fc32c3d6abb70b3fe0e20'
assert receipt['protocol_sha256'] == supervisor['protocol_sha256'] == '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
assert receipt['recovery_addendum_sha256'] == supervisor['recovery_addendum_sha256'] == '5b6fec3b7f4fe2f088cd0176b085fa6074731082307e6215d2ff46ce918a71c9'
assert receipt['optimizer_step_attempts'] >= args.first_attempts
accepted_round = native['epoch'] + 1 if native['status'] == 'EPOCH_BACKUP_VERIFIED' else status['native_accepted_round']
epoch = accepted_round - 1
folder = BACKUPS / f'epochs/epoch{epoch:04d}'
payload = protocol.local_metadata(folder / 'manifest.json')
manifest = protocol.parse_manifest(payload, 'no_msca_20261107_v1', epoch)
assert protocol.local_metadata(folder / 'ack.json') == protocol.ack_bytes(payload, manifest)
assert protocol.local_metadata(folder / 'accepted.json') == protocol.accepted_bytes(payload, manifest)
if native['status'] == 'EPOCH_BACKUP_VERIFIED':
    assert native['manifest_sha256'] == digest(payload)
    assert native['ack_sha256'] == digest(protocol.ack_bytes(payload, manifest))

now = datetime.now(timezone.utc)
stamp = now.strftime('%Y%m%dT%H%M%S%fZ')
snapshot = HERE / f'live_verified_wait_{stamp}.json'
record_path = HERE / f'verified_wait_batch_{stamp}.json'
with snapshot.open('xb') as f: f.write(raw)
evidence_state = read(BUNDLE / 'current_execution_state.json')
record = {
    'goal_turn_classification': 'verified_wait', 'previous_goal_turn_classification': evidence_state['last_goal_turn_classification'],
    'observation_started_utc': args.batch_start, 'observation_finished_utc': now.isoformat(),
    'bounded_poll_count': args.polls, 'specific_live_observer_session': 29581,
    'specific_live_backup_session': 56233,
    'local_process_handles': {'backup': [8216, 1794292724.8624065], 'observer': [40816, 1794312084.181138]},
    'remote_process_handles': {'supervisor': [412, 26597399], 'worker': [413, 26597622]},
    'fresh_remote_snapshot': str(snapshot), 'fresh_remote_snapshot_sha256': digest(raw),
    'native_accepted_round': accepted_round, 'native_acceptance_metadata_verified': True,
    'native_manifest_sha256': digest(payload), 'native_status': native,
    'running_round': receipt['actual_loop_epoch'] + 1, 'phase': receipt.get('phase'),
    'optimizer_step_attempts': receipt['optimizer_step_attempts'],
    'observed_update_attempts_advanced': receipt['optimizer_step_attempts'] - args.first_attempts,
    'next_independent_audit_round': 240, 'independent_lineage_evidence_cutoff': evidence_state['NO_MSCA']['independent_lineage_audit_round'],
    'data_cutoff_round': evidence_state['NO_MSCA']['data_snapshot_independently_verified_round'], 'data_rows': evidence_state['NO_MSCA']['data_handoff']['data_rows'], 'training_actions': 0, 'model_forwards': 0,
    'evaluator_runs': 0, 'full320_completion_pending': True, 'submission_ready': False,
    'goal_last_confirmed_active_by_caller': True,
}
with record_path.open('x', encoding='utf-8') as f:
    json.dump(record, f, ensure_ascii=False, indent=2); f.write('\n')
state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
arm = state['NO_MSCA']
arm['last_accepted_round'] = accepted_round
arm['last_acceptance_utc'] = native.get('observed_utc', arm.get('last_acceptance_utc'))
arm['last_observed_running_round'] = receipt['actual_loop_epoch'] + 1
arm['last_remote_live_observation'] = {
    'file': str(snapshot), 'wrapper_sha256': digest(raw), 'receipt': receipt,
    'gpu': live.get('gpu'), 'observed_local_utc': status['observed_local_utc'],
    'scope': 'Exact process handles and native acceptance metadata verified after bounded session polls',
}
state.update(observed_local_utc=status['observed_local_utc'], last_goal_turn_classification='verified_wait',
             last_verified_wait_record=str(record_path), goal_status='active', goal_objective='resume')
pending = state_path.with_name(state_path.name + '.wait_partial')
pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
pending.replace(state_path)
print(json.dumps({key: record[key] for key in ['native_accepted_round', 'running_round', 'phase',
    'observed_update_attempts_advanced', 'native_acceptance_metadata_verified', 'next_independent_audit_round',
    'training_actions']}, ensure_ascii=True))
