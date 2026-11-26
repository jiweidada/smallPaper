"""Record existing round160 evidence and the live read-only observer; no ML actions."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1]
PAPER = HERE.parents[3]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


audit_path = HERE / 'lineage_recovered_through_round160.json'
audit = read(audit_path)
assert audit['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE'
assert audit['completed_rounds'] == 160
assert audit['learning_rates']['recorded_attempts'] == 258720
assert audit['learning_rates']['maximum_absolute_difference'] == 0
assert audit['maximum_latest_metric_error'] == 0
assert audit['validation_images'] == 548 and audit['predictions'] == 164400
assert audit['model_forwards'] == audit['evaluator_runs'] == 0
assert audit['recovery']['retained_plus_new_rounds'] == 160
assert audit['recovery']['new_optimizer_update_attempts'] == 211827

observation = HERE / 'goal_monitor_20261109_002'
identity = read(observation / 'observer_identity.json')
live = read(observation / 'observer_status.json')
assert identity['observer_pid'] == 6732
assert abs(identity['observer_create_time'] - 1794283838.1600833) < .01
assert live['status'] == 'REMOTE_WORKER_CONFIRMED_LIVE'
assert live['backup_client_alive'] and live['remote_worker_handle_confirmed_live']
assert live['native_accepted_round'] >= 160
assert identity['training_control_actions'] == 0

snapshot = HERE / 'goal_round160_checkpoint_live_20261109.json'
with snapshot.open('xb') as handle:
    handle.write((observation / 'latest_live.json').read_bytes())
subprocess.run([sys.executable, '-X', 'utf8', '-B',
    str(HERE / 'record_recovery_observation.py'), snapshot.name], check=True)

state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
before = state_path.read_bytes()
with (HERE / 'execution_state_before_round160_checkpoint.json').open('xb') as handle:
    handle.write(before)
arm = state['NO_MSCA']
assert arm['monitor_pid'] == 18924 and arm['monitor_session_id'] == 8984
assert state['goal_status'] == 'active' and state['goal_objective'] == '监测'
arm.update(independent_state_audit_round=160, independent_lineage_audit_round=160,
    latest_recovered_lineage_audit=str(audit_path), latest_recovered_lineage_audit_sha256=digest(audit_path),
    next_planned_lineage_audit_round=200)
observer = {**identity, 'exec_session_id': 84874,
    'status_path': str(observation / 'observer_status.json')}
previous_observer = state['read_only_observer']
state['read_only_observer'] = observer

revision = PAPER / '修订记录_20261103/第三十六轮断点恢复披露_20261109'
master = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
promotion = read(revision / 'promotion_verified.json')
assert digest(master) == promotion['current_manuscript_sha256']
assert state['paper_revision'] == 36
state['paper_sha256'] = digest(master)
for key, filename in [
    ('citation_binding_evidence', 'citation_carryforward_rev36.json'),
    ('current_goal_acceptance_audit', 'current_goal_acceptance_audit_rev36.json')]:
    path = revision / filename
    state[key] = str(path)
    state[key + '_sha256'] = digest(path)

stamp = datetime.now(timezone.utc).isoformat()
summary = {
    'observed_local_utc': stamp, 'user_facing_local_date': '2026-11-09',
    'goal_objective': '监测', 'goal_status': 'active',
    'independently_verified_round': 160, 'target_round': 320,
    'audit': {'file': str(audit_path), 'sha256': digest(audit_path)},
    'native_accepted_round_at_record': arm['last_accepted_round'],
    'remote_running_round_at_snapshot': arm['current_running_round_at_snapshot'],
    'live_snapshot': {'file': str(snapshot), 'sha256': digest(snapshot)},
    'previous_observer': {**previous_observer,
        'outcome': 'Exited successfully after observing accepted round160; training continued'},
    'active_read_only_observer': observer,
    'saved_round160_AP_progress_only': audit['latest_metrics']['AP'],
    'saved_round160_APs_progress_only': audit['latest_metrics']['APs'],
    'data_export_verified_round_at_record': arm['data_snapshot_independently_verified_round'],
    'data_update_to160_pending': True,
    'paper_revision': 36, 'paper_sha256': digest(master),
    'submission_ready': False, 'training_control_actions': 0,
    'next_independent_audit_round': 200,
    'completion_still_requires': [
        '320 native accepted backups and genuine successful worker/supervisor exit',
        'independent complete saved-state/lineage/endpoint audit',
        'data and status reflecting the verified complete result']}
summary_path = HERE / 'status_and_audit_round160_20261109.json'
write(summary_path, summary, exclusive=True)
arm['latest_status_evidence'] = {'file': str(summary_path), 'sha256': digest(summary_path)}
state['observed_local_utc'] = stamp
write(state_path, state)

contract_path = HERE / 'active_monitor_goal_20261109.json'
contract = read(contract_path)
contract.setdefault('observer_history', []).append(summary['previous_observer'])
contract['observer'] = observer
contract['next_independent_audit_round'] = 200
contract['latest_verified_checkpoint'] = arm['latest_status_evidence']
contract['latest_progress_classification'] = 'Independent160-round audit and verified live monitoring toward200'
write(contract_path, contract)
print(json.dumps({'status': 'ACTIVE_MONITOR_GOAL_ROUND160_CHECKPOINT_RECORDED',
    'independent_round': 160, 'native_accepted_round': arm['last_accepted_round'],
    'next_target_round': 200, 'observer_session': 84874,
    'backup_client_session_unchanged': 8984, 'training_control_actions': 0}))
