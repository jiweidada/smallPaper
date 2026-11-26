"""Record the second access audit and the actual local backup-cursor check."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent.parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: sha256(p.read_bytes()).hexdigest()
probe_path = HERE / 'default_provided_30848_03.json'
resume_path = HERE / 'backup_client_resume_preflight_round29.json'
probe, resume = read(probe_path), read(resume_path)
assert probe['endpoints'][0]['status'] == 'CONNECTION_CLOSED_BEFORE_SSH_BANNER'
assert resume['status'] == 'FROZEN_BACKUP_CLIENT_LOCAL_RESUME_CURSOR_VERIFIED'
assert resume['accepted_training_rounds'] == 29 and resume['next_expected_backup_round'] == 30
assert resume['checked_object_hash_count'] == 256
assert not resume['backup_client_started'] and resume['network_connections'] == 0
state_path = BUNDLE / 'current_execution_state.json'
old_bytes = state_path.read_bytes()
state = json.loads(old_bytes.decode('utf-8-sig'))
arm = state['NO_MSCA']
assert arm['status'] == 'REMOTE_UNREACHABLE_BACKUP_CLIENT_FAILED'
assert arm['independent_lineage_audit_round'] == 29
now = datetime.now(timezone.utc).isoformat()
report = {
    'status': 'SECOND_ACCESS_AUDIT_FAILED_LOCAL_BACKUP_CLIENT_CURSOR_VERIFIED',
    'observed_local_utc': now, 'same_access_block_consecutive_goal_turns': 2,
    'latest_probe_sha256': digest(probe_path), 'backup_client_resume_preflight_sha256': digest(resume_path),
    'safe_local_training_rounds': 29, 'actual_frozen_client_next_expected_backup_round': 30,
    'verified_existing_backup_object_count': 256,
    'remaining_block': '30848 still closes before an SSH banner. Current original-instance state and working SSH entry remain unavailable; user information request is already pending.',
    'remote_trainer_state': 'UNKNOWN',
    'meaningful_work_this_turn': 'Invoked the actual frozen client local verification method. It reconstructed its accepted cursor and verified 256 existing content-addressed objects without network access or changing status/event files.',
    'recovery_limits': 'This is a backup transport cursor, not proof that the remote trainer completed round 30 or a training restart instruction. The exclusive client lock and remote project/process must be checked after connectivity returns.',
    'transport_cli_note': 'The frozen monitor main accepts ports 30219 and 30848. If the user supplies a changed entry, a transport-only adapter can honor that explicitly supplied endpoint while preserving the original frozen client and all training settings.',
    'goal_status': 'active; blocked threshold is three consecutive turns and is not yet met',
    'next_action': 'Recheck the supplied endpoint or act on the pending user response; if the same access block persists in the third consecutive goal turn and no meaningful independent work remains, mark the goal blocked.',
    'new_training_or_evaluation': False, 'manuscript_changed': False, 'submission_ready': False}
output = HERE / 'access_audit_turn2_and_resume_cursor.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
arm.update(consecutive_access_block_goal_turns=2, latest_access_audit=str(output),
           latest_access_audit_sha256=digest(output),
           backup_client_resume_preflight=str(resume_path), backup_client_resume_preflight_sha256=digest(resume_path),
           next_expected_backup_epoch_index=29)
state['observed_local_utc'] = now
temporary = state_path.with_suffix('.access2.tmp')
assert state_path.read_bytes() == old_bytes
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, state_path)
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-09 UTC：第二次入口复核与本地备份接续游标\n\n'
                 '30848 仍在 SSH banner 前断开，远端训练状态未知。已直接调用冻结备份客户端的本地 verify_progress，'
                 '256 个已有对象、各轮 manifest/ack/accepted 与硬链接核验通过，本地下一待备份位置为零基29/第30轮。'
                 '未联网、未写确认、未启动备份客户端或训练；这不是远端已完成第30轮的证明。'
                 '已有用户信息请求保持待答，不重复提出新授权问题。当前同一访问阻碍已连续两轮确认，目标保持 active，尚未达到三轮 blocked 门槛。\n')
print(json.dumps({'status': report['status'], 'access_block_goal_turns': 2,
                  'safe_rounds': 29, 'verified_backup_objects': 256, 'training_action': None}))
