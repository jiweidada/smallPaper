"""Bind an audited saved milestone and a fresh read-only observation window."""
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
ARM, BUNDLE, PAPER = HERE.parent, HERE.parents[1], HERE.parents[3]
OBS = HERE / 'resume_monitor_20261110_004'
PREV = HERE / 'resume_monitor_20261110_003'
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
read = lambda p: json.loads(p.read_text('utf-8-sig'))
digest = lambda p: sha256(p.read_bytes()).hexdigest()
sys.dont_write_bytecode = True
sys.path.insert(0, str(ARM))
import backup_protocol as protocol

def write_new(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

checks = {
    HERE / 'lineage_second_recovery_through_round240.json': 'b210f4fcaf5acb09fffbc8af8cb6f484c73b2a854136352227a087c04e132545',
    HERE / 'segment30715_audit_binding.json': '6de82b5b4bcbf8bbdbd3a61a2fb7ecbd70905b98da0eceefc55c7b451435a41e',
    HERE / 'monitor_authorized_native_run.py': '6530732ad870cb41e73822b8df07374fb71d0cc5045286089c777ac91f35867e',
    HERE / 'read_current_monitor_handles_003.py': 'b39b16f7172373d5e78d0dd2559404917fa642a9f3fdb9b9c3a722a27856bcb5',
    HERE / 'checkpoint_verified_wait_003.py': '3a00581c741f7de9d38272ca0ac0d87b34c38d614a9f6597d28874b5b3d13222',
    PREV / 'milestone_round240_live.json': '809ab33296ee7e61be57f3f4a6967de1c7a1eeedd89ce1550ac030699a5baa1f',
    PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx': '53060de71535e78591fd5856ff537cf65c12a9463bd532e7525de490eb3c2910',
}
for path, expected in checks.items():
    assert digest(path) == expected, path
audit_path = HERE / 'lineage_second_recovery_through_round240.json'
audit = read(audit_path)
assert audit['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE'
assert audit['completed_rounds'] == 240
assert audit['learning_rates'] == {'recorded_attempts': 388080, 'maximum_absolute_difference': 0.0, 'groups': 4}
assert audit['maximum_latest_metric_error'] == 0 and audit['endpoints']['best']['maximum_metric_error'] == 0
assert audit['validation_images'] == 548 and audit['predictions'] == 164400
assert audit['model_forwards'] == audit['evaluator_runs'] == 0
assert audit['recovery']['new_completed_rounds'] == 76
assert audit['recovery']['new_optimizer_update_attempts'] == 122892
goal_path = HERE / 'goal_status_at_round240_continuation_20261110.json'
assert read(goal_path)['goal']['status'] == 'active'
for pid, created in [(8216, 1794292724.8624065), (24304, 1794331842.2949867)]:
    assert abs(psutil.Process(pid).create_time() - created) < .01
assert not (psutil.pid_exists(40816) and abs(psutil.Process(40816).create_time() - 1794312084.181138) < .01)
identity = read(OBS / 'observer_identity.json')
assert identity['observer_pid'] == 24304 and identity['target_round'] == 280
assert identity['observer_source_sha256'] == checks[HERE / 'monitor_authorized_native_run.py']
status = read(OBS / 'observer_status.json')
live_path = OBS / 'latest_live.json'
raw = live_path.read_bytes()
live = json.loads(json.loads(raw)['stdout'])
receipt = live['records']['run/receipt.json']
assert time.time() - live_path.stat().st_mtime < 100
assert status['status'] == 'REMOTE_WORKER_CONFIRMED_LIVE' and status['backup_client_alive']
assert live['boot_id'] == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
for key, ticks in [('pid', 26597622), ('supervisor_pid', 26597399)]:
    assert live['processes'][key]['exists'] and live['processes'][key]['start_ticks'] == ticks
assert receipt['first_epoch_in_this_segment'] == 164 and receipt['completed_epochs'] >= 240
assert receipt['runner_sha256'] == 'f5f7ef88ace026c5236734dfeeb29cdbc4688617f90fc32c3d6abb70b3fe0e20'
assert receipt['protocol_sha256'] == '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
assert receipt['recovery_addendum_sha256'] == '5b6fec3b7f4fe2f088cd0176b085fa6074731082307e6215d2ff46ce918a71c9'
native = read(BACKUPS / 'monitor_status.json')
assert native['status'] == 'EPOCH_BACKUP_VERIFIED' and native['epoch'] >= 239
folder = BACKUPS / f"epochs/epoch{native['epoch']:04d}"
payload = protocol.local_metadata(folder / 'manifest.json')
manifest = protocol.parse_manifest(payload, 'no_msca_20261107_v1', native['epoch'])
assert protocol.local_metadata(folder / 'ack.json') == protocol.ack_bytes(payload, manifest)
assert protocol.local_metadata(folder / 'accepted.json') == protocol.accepted_bytes(payload, manifest)
assert native['manifest_sha256'] == sha256(payload).hexdigest()
assert native['ack_sha256'] == sha256(protocol.ack_bytes(payload, manifest)).hexdigest()

specs = {
    'read_current_monitor_handles_004.py': ('read_current_monitor_handles_003.py', [
        ('resume_monitor_20261110_003', 'resume_monitor_20261110_004'),
        ("('observer', 40816, 1794312084.181138)", "('observer', 24304, 1794331842.2949867)"),
        ('native_round >= 240', 'native_round >= 280'),
        ('milestone_round240_live.json', 'milestone_round280_live.json'),
        ('milestone240_available', 'milestone280_available'),
        ('MILESTONE240_READY_FOR_INDEPENDENT_AUDIT', 'MILESTONE280_READY_FOR_INDEPENDENT_AUDIT'),
    ]),
    'checkpoint_verified_wait_004.py': ('checkpoint_verified_wait_003.py', [
        ('resume_monitor_20261110_003', 'resume_monitor_20261110_004'),
        ('(40816, 1794312084.181138)', '(24304, 1794331842.2949867)'),
        ("'specific_live_observer_session': 29581", "'specific_live_observer_session': 54384"),
        ("'observer': [40816, 1794312084.181138]", "'observer': [24304, 1794331842.2949867]"),
        ("'next_independent_audit_round': 240", "'next_independent_audit_round': 280"),
    ]),
}
outputs = [HERE / n for n in specs] + [HERE / 'live_round240_continuation_20261110.json', HERE / 'round240_observer280_continuation_contract_20261110.json']
assert all(not p.exists() for p in outputs)
prepared = {}
for new_name, (old_name, replacements) in specs.items():
    source = (HERE / old_name).read_text('utf-8')
    for before, after in replacements:
        assert source.count(before) == 1, (old_name, before, source.count(before))
        source = source.replace(before, after)
    ast.parse(source)
    assert '40816' not in source and '29581' not in source
    prepared[HERE / new_name] = source
for path, source in prepared.items():
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(source)
live_snapshot = HERE / 'live_round240_continuation_20261110.json'
with live_snapshot.open('xb') as f:
    f.write(raw)
state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
arm = state['NO_MSCA']
old_observer = state['read_only_observer']
assert old_observer['observer_pid'] == 40816
state.setdefault('read_only_observer_history', []).append({
    'previous_observer': old_observer, 'ended_at_native_round': 240, 'exit_code': 0,
    'training_stopped': False, 'fixed_milestone': str(PREV / 'milestone_round240_live.json'),
    'fixed_milestone_sha256': checks[PREV / 'milestone_round240_live.json'],
})
identity.update(exec_session_id=54384, status_path=str(OBS / 'observer_status.json'),
                native_backup_session_id=56233, native_backup_pid=8216)
state['read_only_observer'] = identity
arm.update(last_accepted_round=native['epoch'] + 1, last_acceptance_utc=native['observed_utc'],
    last_observed_running_round=receipt['actual_loop_epoch'] + 1,
    current_running_round_at_snapshot=receipt['actual_loop_epoch'] + 1,
    independent_state_audit_round=240, independent_lineage_audit_round=240,
    latest_recovered_lineage_audit=str(audit_path), latest_recovered_lineage_audit_sha256=digest(audit_path),
    latest_independent_lineage_audit={'file': str(audit_path), 'sha256': digest(audit_path)},
    next_planned_lineage_audit_round=280, next_independent_audit_round=280,
    consecutive_blocked_goal_turns=0, consecutive_access_block_goal_turns=0,
    last_remote_live_observation={'file': str(live_snapshot), 'wrapper_sha256': digest(live_snapshot),
        'receipt': receipt, 'gpu': live.get('gpu'), 'observed_local_utc': status['observed_local_utc'],
        'scope': 'Exact fresh handles and native acceptance metadata; independent metric cutoff240, data cutoff200 pending refresh'},
    last_local_backup_observation={'monitor_pid': 8216, 'monitor_create_time': 1794292724.8624065,
        'monitor_identity_alive': True, 'monitor_status': native['status'], 'native_status': native})
state['next_action'] = {
    'kind': 'verified_wait_for_native_round280_then_full_CPU_audit_and_data_update',
    'observer_session_id': 54384, 'observer_pid': 24304, 'observer_create_time': 1794331842.2949867,
    'backup_session_id': 56233, 'backup_pid': 8216, 'backup_create_time': 1794292724.8624065,
    'read_monitor_command': str(HERE / 'read_current_monitor_handles_004.py'),
    'checkpoint_command': str(HERE / 'checkpoint_verified_wait_004.py'),
    'audit_bootstrap': str(HERE / 'audit_resume30715_bootstrap.py'),
    'binding_sha256': checks[HERE / 'segment30715_audit_binding.json'],
    'audit_epoch': 279, 'audit_output_exclusive': str(HERE / 'lineage_second_recovery_through_round280.json'),
    'max_tool_wait_ms': 55000, 'no_training_restart_from_observation_timeout': True,
    'completion_requires_full320_and_paper_acceptance': True,
}
contract = {
    'status': 'ROUND240_LINEAGE_VERIFIED_NEW_OBSERVER280_LIVE_DATA_REFRESH_PENDING',
    'observed_local_utc': datetime.now(timezone.utc).isoformat(),
    'audit': {'file': str(audit_path), 'sha256': digest(audit_path), 'completed_rounds': 240,
        'LR_attempts': 388080, 'max_LR_error': 0, 'max_saved_metric_error': 0,
        'model_forwards': 0, 'evaluator_runs': 0},
    'previous_observer': state['read_only_observer_history'][-1], 'new_observer': identity,
    'live_snapshot': {'file': str(live_snapshot), 'sha256': digest(live_snapshot)},
    'helper_files': [{'file': str(p), 'sha256': digest(p)} for p in prepared],
    'current_data_cutoff': arm['data_snapshot_independently_verified_round'],
    'current_data_rows': arm['data_handoff']['data_rows'], 'data240_refresh_pending': True,
    'partial_best_round': audit['best_epoch_index'] + 1,
    'latest240_AP': audit['latest_metrics']['AP'], 'latest240_APs': audit['latest_metrics']['APs'],
    'global_initial_update_witness_unchanged': 'resume_monitor_20261109_001 original fixed first-update proof',
    'goal_receipt': str(goal_path), 'goal_receipt_sha256': digest(goal_path),
    'goal_last_confirmed_active': True, 'submission_ready': False, 'training_actions': 0,
}
contract_path = HERE / 'round240_observer280_continuation_contract_20261110.json'
write_new(contract_path, contract)
state.update(observed_local_utc=status['observed_local_utc'], goal_status='active', goal_objective='resume',
    last_goal_turn_classification='progress', data240_refresh_pending=True,
    goal_tool_status_observed_receipt=str(goal_path), goal_status_observed_local_utc=contract['observed_local_utc'],
    operational_work_status='NO_MSCA_RUNNING_ROUND240_INDEPENDENTLY_VERIFIED_DATA240_REFRESH_PENDING',
    latest_monitoring_continuation_contract=str(contract_path),
    latest_monitoring_continuation_contract_sha256=digest(contract_path))
pending = state_path.with_name(state_path.name + '.round240_partial')
assert not pending.exists()
pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
pending.replace(state_path)
print(json.dumps({'status': contract['status'], 'native_accepted_round': native['epoch'] + 1,
    'running_round': receipt['actual_loop_epoch'] + 1, 'independent_round': 240,
    'data_cutoff': contract['current_data_cutoff'], 'next_audit_round': 280,
    'observer_session_id': 54384, 'training_actions': 0, 'submission_ready': False}))
