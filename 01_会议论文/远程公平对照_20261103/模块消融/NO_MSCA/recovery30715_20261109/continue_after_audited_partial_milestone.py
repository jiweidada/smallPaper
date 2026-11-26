"""Record one audited partial milestone and switch only the read-only observer handle."""
import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
ARM, BUNDLE, PAPER = HERE.parent, HERE.parents[1], HERE.parents[3]
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ARM))
import backup_protocol as protocol

digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text('utf-8-sig'))
def write_new(p, val):
    with p.open('x', encoding='utf-8') as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
        f.write('\n')

p = argparse.ArgumentParser()
p.add_argument('--round', type=int, required=True)
p.add_argument('--next-round', type=int, required=True)
p.add_argument('--stamp', required=True)
p.add_argument('--observer-directory', required=True)
p.add_argument('--observer-session', type=int, required=True)
p.add_argument('--previous-observer-exit-code', type=int, choices=[0], required=True)
p.add_argument('--audit-sha256', required=True)
p.add_argument('--milestone-sha256', required=True)
a = p.parse_args()
r, target, stamp = a.round, a.next_round, a.stamp
assert 164 < r < target <= 320 and re.fullmatch(r'\d{8}', stamp)
assert re.fullmatch(r'resume_monitor_\d{8}_\d{3}', a.observer_directory)
obsdir = HERE / a.observer_directory
binding = '6de82b5b4bcbf8bbdbd3a61a2fb7ecbd70905b98da0eceefc55c7b451435a41e'
assert digest(HERE / 'segment30715_audit_binding.json') == binding
audit_path = HERE / f'lineage_second_recovery_through_round{r}.json'
assert digest(audit_path) == a.audit_sha256
audit = read(audit_path)
assert audit['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and audit['completed_rounds'] == r
assert audit['learning_rates'] == {'recorded_attempts': r * 1617, 'groups': 4, 'maximum_absolute_difference': 0.0}
assert audit['validation_images'] == 548 and audit['predictions'] == 164400
assert audit['model_forwards'] == audit['evaluator_runs'] == audit['maximum_latest_metric_error'] == 0
assert audit['endpoints']['best']['maximum_metric_error'] == 0
assert audit['recovery']['new_completed_rounds'] == r - 164
assert audit['recovery']['new_optimizer_update_attempts'] == (r - 164) * 1617
goal_path = HERE / f'goal_status_at_round{r}_continuation_{stamp}.json'
goal = read(goal_path)
assert goal['goal']['status'] == 'active'

state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
arm, old, next_action = state['NO_MSCA'], state['read_only_observer'], state['next_action']
assert digest(PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx') == state['paper_sha256']
assert old['target_round'] == r and next_action['audit_epoch'] == r - 1
prevdir = Path(old['status_path']).parent
assert prevdir.parent == HERE and prevdir != obsdir
milestone = prevdir / f'milestone_round{r}_live.json'
assert digest(milestone) == a.milestone_sha256
old_pid, old_created = old['observer_pid'], old['observer_create_time']
assert not (psutil.pid_exists(old_pid) and abs(psutil.Process(old_pid).create_time() - old_created) < .01)
identity = read(obsdir / 'observer_identity.json')
new_pid, new_created = identity['observer_pid'], identity['observer_create_time']
assert identity['target_round'] == target and identity['remote_ssh_port'] == 30715
assert identity['observer_source_sha256'] == digest(HERE / 'monitor_authorized_native_run.py') == '6530732ad870cb41e73822b8df07374fb71d0cc5045286089c777ac91f35867e'
assert identity['remote_reader_sha256'] == digest(HERE / 'read_live_recovery.py') == 'ebd4075d17fb884f808a594a49511ba9c37f64eaa5b67915354674362ee332c2'
backup_pid, backup_created, backup_session = [arm[k] for k in ['monitor_pid', 'monitor_create_time', 'monitor_session_id']]
assert identity['backup_client_pid'] == backup_pid and identity['backup_client_create_time'] == backup_created
assert identity['training_control_actions'] == 0
for pid, created in [(backup_pid, backup_created), (new_pid, new_created)]:
    assert abs(psutil.Process(pid).create_time() - created) < .01
status = read(obsdir / 'observer_status.json')
live_path = obsdir / 'latest_live.json'
raw = live_path.read_bytes()
live = json.loads(json.loads(raw)['stdout'])
receipt = live['records']['run/receipt.json']
assert time.time() - live_path.stat().st_mtime < 100
assert status['status'] == 'REMOTE_WORKER_CONFIRMED_LIVE' and status['backup_client_alive']
assert live['boot_id'] == arm['remote_boot_id']
for key, ticks in [('pid', arm['remote_trainer_start_ticks']), ('supervisor_pid', arm['remote_supervisor_start_ticks'])]:
    assert live['processes'][key]['exists'] and live['processes'][key]['start_ticks'] == ticks
assert receipt['first_epoch_in_this_segment'] == 164 and receipt['completed_epochs'] >= r
assert receipt['protocol_sha256'] == arm['protocol_sha256']
assert receipt['recovery_addendum_sha256'] == arm['recovery_addendum_sha256']
assert receipt['runner_sha256'] == 'f5f7ef88ace026c5236734dfeeb29cdbc4688617f90fc32c3d6abb70b3fe0e20'
native = read(BACKUPS / 'monitor_status.json')
assert native['status'] == 'EPOCH_BACKUP_VERIFIED' and native['epoch'] >= r - 1
folder = BACKUPS / f"epochs/epoch{native['epoch']:04d}"
payload = protocol.local_metadata(folder / 'manifest.json')
manifest = protocol.parse_manifest(payload, 'no_msca_20261107_v1', native['epoch'])
assert protocol.local_metadata(folder / 'ack.json') == protocol.ack_bytes(payload, manifest)
assert protocol.local_metadata(folder / 'accepted.json') == protocol.accepted_bytes(payload, manifest)
assert native['manifest_sha256'] == sha256(payload).hexdigest()
assert native['ack_sha256'] == sha256(protocol.ack_bytes(payload, manifest)).hexdigest()

prior_contract = Path(state['latest_monitoring_continuation_contract'])
assert digest(prior_contract) == state['latest_monitoring_continuation_contract_sha256']
previous_helpers = {Path(x['file']): x['sha256'] for x in read(prior_contract)['helper_files']}
common = [(prevdir.name, obsdir.name)]
suffix = obsdir.name.rsplit('_', 1)[1]
specs = {
    f'read_current_monitor_handles_{suffix}.py': ('read_monitor_command', common + [
        (f"('observer', {old_pid}, {old_created})", f"('observer', {new_pid}, {new_created})"),
        (f'native_round >= {r}', f'native_round >= {target}'),
        (f'milestone_round{r}_live.json', f'milestone_round{target}_live.json'),
        (f'milestone{r}_available', f'milestone{target}_available'),
        (f'MILESTONE{r}_READY_FOR_INDEPENDENT_AUDIT', f'MILESTONE{target}_READY_FOR_INDEPENDENT_AUDIT'),
    ]),
    f'checkpoint_verified_wait_{suffix}.py': ('checkpoint_command', common + [
        (f'({old_pid}, {old_created})', f'({new_pid}, {new_created})'),
        (f"'specific_live_observer_session': {old['exec_session_id']}", f"'specific_live_observer_session': {a.observer_session}"),
        (f"'observer': [{old_pid}, {old_created}]", f"'observer': [{new_pid}, {new_created}]"),
        (f"'next_independent_audit_round': {r}", f"'next_independent_audit_round': {target}"),
    ]),
}
live_snapshot = HERE / f'live_round{r}_continuation_{stamp}.json'
contract_path = HERE / f'round{r}_observer{target}_continuation_contract_{stamp}.json'
assert not any(q.exists() for q in [live_snapshot, contract_path, *(HERE / n for n in specs)])
prepared = {}
for name, (state_key, substitutions) in specs.items():
    old_path = Path(next_action[state_key])
    assert digest(old_path) == previous_helpers[old_path]
    source = old_path.read_text('utf-8')
    for before, after in substitutions:
        assert source.count(before) == 1, (old_path, before)
        source = source.replace(before, after)
    ast.parse(source)
    prepared[HERE / name] = source
for path, source in prepared.items():
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(source)
with live_snapshot.open('xb') as f:
    f.write(raw)

old_history = {'previous_observer': old, 'ended_at_native_round': r, 'exit_code': a.previous_observer_exit_code,
    'training_stopped': False, 'fixed_milestone': str(milestone), 'fixed_milestone_sha256': digest(milestone)}
state.setdefault('read_only_observer_history', []).append(old_history)
identity.update(exec_session_id=a.observer_session, status_path=str(obsdir / 'observer_status.json'),
    native_backup_session_id=backup_session, native_backup_pid=backup_pid)
state['read_only_observer'] = identity
running = receipt['actual_loop_epoch'] + 1
arm.update(last_accepted_round=native['epoch'] + 1, last_acceptance_utc=native['observed_utc'],
    last_observed_running_round=running, current_running_round_at_snapshot=running,
    independent_state_audit_round=r, independent_lineage_audit_round=r,
    latest_recovered_lineage_audit=str(audit_path), latest_recovered_lineage_audit_sha256=digest(audit_path),
    latest_independent_lineage_audit={'file': str(audit_path), 'sha256': digest(audit_path)},
    next_planned_lineage_audit_round=target, next_independent_audit_round=target,
    consecutive_blocked_goal_turns=0, consecutive_access_block_goal_turns=0,
    last_remote_live_observation={'file': str(live_snapshot), 'wrapper_sha256': digest(live_snapshot),
        'receipt': receipt, 'gpu': live.get('gpu'), 'observed_local_utc': status['observed_local_utc'],
        'scope': 'Fresh exact live handles and native backup acceptance; data refresh follows independent audit'},
    last_local_backup_observation={'monitor_pid': backup_pid, 'monitor_create_time': backup_created,
        'monitor_identity_alive': True, 'monitor_status': native['status'], 'native_status': native})
next_action.update(kind=f'verified_wait_for_native_round{target}_then_full_CPU_audit_and_data_update',
    observer_session_id=a.observer_session, observer_pid=new_pid, observer_create_time=new_created,
    read_monitor_command=str(HERE / f'read_current_monitor_handles_{suffix}.py'),
    checkpoint_command=str(HERE / f'checkpoint_verified_wait_{suffix}.py'),
    audit_epoch=target - 1, audit_output_exclusive=str(HERE / f'lineage_second_recovery_through_round{target}.json'))
now = datetime.now(timezone.utc).isoformat()
contract = {'status': f'ROUND{r}_LINEAGE_VERIFIED_NEW_OBSERVER{target}_LIVE_DATA_REFRESH_PENDING',
    'observed_local_utc': now, 'previous_goal_turn_classification': state['last_goal_turn_classification'],
    'audit': {'file': str(audit_path), 'sha256': digest(audit_path), 'completed_rounds': r,
        'LR_attempts': r * 1617, 'max_LR_error': 0, 'max_saved_metric_error': 0, 'model_forwards': 0, 'evaluator_runs': 0},
    'previous_observer': old_history, 'new_observer': identity,
    'live_snapshot': {'file': str(live_snapshot), 'sha256': digest(live_snapshot)},
    'helper_files': [{'file': str(path), 'sha256': digest(path)} for path in prepared],
    'current_data_cutoff': arm['data_snapshot_independently_verified_round'], 'current_data_rows': arm['data_handoff']['data_rows'],
    'partial_best_round': audit['best_epoch_index'] + 1, 'latest_AP': audit['latest_metrics']['AP'],
    'goal_receipt': str(goal_path), 'goal_receipt_sha256': digest(goal_path),
    'goal_last_confirmed_active': True, 'original_training_launch_witness_unchanged': True,
    'submission_ready': False, 'training_actions': 0}
write_new(contract_path, contract)
state.update(observed_local_utc=status['observed_local_utc'], goal_status='active', goal_objective=goal['goal']['objective'],
    last_goal_turn_classification='progress', goal_tool_status_observed_receipt=str(goal_path),
    goal_status_observed_local_utc=now, operational_work_status=f'NO_MSCA_RUNNING_ROUND{r}_INDEPENDENTLY_VERIFIED_DATA_REFRESH_PENDING',
    latest_monitoring_continuation_contract=str(contract_path), latest_monitoring_continuation_contract_sha256=digest(contract_path))
state[f'data{r}_refresh_pending'] = True
pending = state_path.with_name(state_path.name + f'.round{r}_partial')
assert not pending.exists()
pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
pending.replace(state_path)
print(json.dumps({'status': contract['status'], 'native_accepted_round': native['epoch'] + 1,
    'running_round': running, 'independent_round': r, 'data_cutoff': contract['current_data_cutoff'],
    'next_audit_round': target, 'observer_session_id': a.observer_session, 'training_actions': 0,
    'submission_ready': False}))
