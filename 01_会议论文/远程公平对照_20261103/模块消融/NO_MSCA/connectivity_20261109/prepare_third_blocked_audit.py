"""Prove the third repeated access block before changing the goal status."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent.parent
PAPER = BUNDLE.parent.parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: sha256(p.read_bytes()).hexdigest()
probe_path = HERE / 'default_provided_30848_04.json'
previous_path = HERE / 'access_audit_turn2_and_resume_cursor.json'
resume_path = HERE / 'backup_client_resume_preflight_round29.json'
state = read(BUNDLE / 'current_execution_state.json')
arm = state['NO_MSCA']
assert read(previous_path)['same_access_block_consecutive_goal_turns'] == 2
assert read(probe_path)['endpoints'][0]['status'] == 'CONNECTION_CLOSED_BEFORE_SSH_BANNER'
assert arm['consecutive_access_block_goal_turns'] == 2
try:
    process = psutil.Process(arm['monitor_pid'])
    old_client_alive = abs(process.create_time() - arm['monitor_create_time']) < 0.001
except psutil.Error:
    old_client_alive = False
assert not old_client_alive
root = Path(arm['backup_root'])
assert read(root / 'monitor_status.json')['status'] == 'MONITOR_FAILED_NO_FURTHER_ACKS'
events = [json.loads(line) for line in (root / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
accepted = next(e for e in reversed(events) if e['status'] == 'EPOCH_BACKUP_VERIFIED')
assert accepted['epoch'] == 28 and arm['independent_lineage_audit_round'] == 29
assert read(resume_path)['checked_object_hash_count'] == 256
assert read(resume_path)['next_expected_backup_round'] == 30
master = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
qa_path = PAPER / '修订记录_20261103/第三十五轮容量与对照边界_20261108/capacity_scope_visual_qa.json'
acceptance_path = PAPER / '修订记录_20261103/第三十五轮容量与对照边界_20261108/current_goal_acceptance_audit_rev35.json'
assert digest(master) == state['paper_sha256'] == read(qa_path)['candidate_sha256']
assert state['paper_revision'] == 35 and read(qa_path)['status'] == 'PASSED'
criteria = read(acceptance_path)['criteria']
assert criteria[0]['status'] == 'PARTIAL_PENDING_NO_MSCA'
assert criteria[3]['status'] == 'CURRENT_TEXT_REVISED_FINAL_MODULE_ANALYSIS_PENDING'
result = {
    'status': 'BLOCKED_STATUS_CRITERIA_PROVED_PENDING_GOAL_TOOL_RESULT',
    'observed_local_utc': datetime.now(timezone.utc).isoformat(),
    'same_access_block_consecutive_goal_turns': 3,
    'block_condition': 'The authorized MSCA SSH endpoint 30848 establishes TCP and closes before an SSH banner; original remote process and project state cannot be checked.',
    'local_backup_client_identity_alive': False, 'verified_live_training_handle_available': False,
    'remote_trainer_state': 'UNKNOWN; neither stopped nor running can be asserted from this connection failure.',
    'safe_local_backup_and_lineage_rounds': 29,
    'independent_work_already_completed': [
        'BASE/FULL and NO_MPS completed endpoints and lineages accepted.',
        'Current revision 35 has source-bound methods, explicit capacity/attribution limits and complete eight-page visual QA.',
        'Current citation mapping and reading limits audited.',
        'NO_MSCA local round-29 state, predictions, evaluation arrays, RNG and 46,893 learning rates verified.',
        'Actual frozen backup-client cursor verified against 256 existing objects without connecting or writing acknowledgements.',
        'The other provided server is reachable and idle, but its checked project path lacks the NO_MSCA run.'
    ],
    'still_required_for_full_objective': [
        'Access to the original MSCA instance to establish the real state and safely recover the existing authorized execution/backup stream.',
        'The complete MSCA 320-round endpoint, lineage, genuine completion and backup verification.',
        'Corresponding conditional analysis, final abstract/conclusion and full-paper review/render.'
    ],
    'why_no_meaningful_independent_work_remains_now': 'Remaining manuscript conclusions depend on the unavailable MSCA outcome. Local recovery materials are already checked. Rewriting prose cannot supply that evidence, and an unverified remote process cannot be treated as terminal or duplicated on the free GPU.',
    'required_external_change': 'Restore the original SSH service/entry or supply the current instance connection information. The existing user information request remains pending.',
    'venue_selection': 'Deferred by the user; it is not the cause of this block.',
    'completion_proved': False, 'submission_ready': False,
    'manuscript_revision': 35, 'manuscript_sha256': digest(master),
    'evidence_sha256': {str(p.relative_to(PAPER)): digest(p) for p in [probe_path, previous_path, resume_path, qa_path, acceptance_path]},
    'new_training_or_evaluation': False,
    'resume_rule': 'If the user later resumes a blocked goal, reset the repeated-block audit and revalidate current state; do not immediately re-block from this old count.'}
with (HERE / 'blocking_audit_turn3.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'consecutive_turns': 3, 'safe_rounds': 29,
                  'remote_trainer_state': 'UNKNOWN', 'completion_proved': False}))
