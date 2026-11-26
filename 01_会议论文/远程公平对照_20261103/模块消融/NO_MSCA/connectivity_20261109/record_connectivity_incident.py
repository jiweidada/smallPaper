"""Preserve the failed backup evidence and the independently checked safe prefix."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
ARM = HERE.parent
BUNDLE = ARM.parent
PAPER = BUNDLE.parent.parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: sha256(p.read_bytes()).hexdigest()
state_path = BUNDLE / 'current_execution_state.json'
state_bytes = state_path.read_bytes()
state = json.loads(state_bytes.decode('utf-8-sig'))
arm = state['NO_MSCA']
backups = Path(arm['backup_root'])
events_path = backups / 'monitor_events.jsonl'
events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]
accepted_index = max(i for i, e in enumerate(events) if e['status'] == 'EPOCH_BACKUP_VERIFIED')
accepted = events[accepted_index]
failure_status = read(backups / 'monitor_status.json')
assert accepted['epoch'] == 28 and failure_status['status'] == 'MONITOR_FAILED_NO_FURTHER_ACKS'
try:
    process = psutil.Process(arm['monitor_pid'])
    old_client_alive = abs(process.create_time() - arm['monitor_create_time']) < 0.001
except psutil.Error:
    old_client_alive = False
assert not old_client_alive
lineage_path = ARM / 'lineage_through_round029.json'
backup_path = ARM / 'round029_verified_from_D.json'
lineage, backup = read(lineage_path), read(backup_path)
assert lineage['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE'
assert lineage['completed_rounds'] == backup['round'] == 29
assert lineage['protocol_sha256'] == backup['protocol_sha256'] == arm['protocol_sha256']
assert lineage['learning_rates']['recorded_attempts'] == 46893
assert lineage['learning_rates']['maximum_absolute_difference'] == backup['maximum_metric_error'] == 0
assert backup['rng_states_present'] and backup['validation_images'] == 548 and backup['predictions'] == 164400
assert backup['model_forwards'] == backup['evaluator_runs'] == 0
default_path = HERE / 'default_provided_endpoints_01.json'
authorized_path = HERE / 'authorized_channel_30848_01.json'
default, authorized = read(default_path), read(authorized_path)
assert default['endpoints'][0]['port'] == authorized['endpoints'][0]['port'] == 30848
assert default['endpoints'][0]['status'] == authorized['endpoints'][0]['status'] == 'CONNECTION_CLOSED_BEFORE_SSH_BANNER'
assert default['endpoints'][1]['port'] == 30219 and default['endpoints'][1]['status'] == 'SSH_BANNER_OBSERVED'
peer_path = HERE / 'peer_30219_project_presence.json'
peer_wrapper = read(peer_path)
assert peer_wrapper['exit_code'] == 0
peer = json.loads(peer_wrapper['stdout'])
assert peer['boot_id'] != arm['remote_boot_id']
assert peer['arm_paths']['no_mps_20261107_v1']['receipt']['status'] == 'COMPLETED'
assert peer['arm_paths']['no_msca_20261107_v1']['exists'] is False
assert not peer['gpu_processes']
now = datetime.now(timezone.utc).isoformat()
for source, name in [(backups / 'monitor_status.json', 'monitor_status_at_failure.json'),
                     (events_path, 'monitor_events_at_failure.jsonl')]:
    with (HERE / name).open('xb') as stream:
        stream.write(source.read_bytes())
evidence_paths = [default_path, authorized_path, peer_path, lineage_path, backup_path,
                  HERE / 'monitor_status_at_failure.json', HERE / 'monitor_events_at_failure.jsonl']
report = {
    'status': 'BACKUP_CLIENT_TERMINATED_SSH_ENTRY_UNREACHABLE_REMOTE_TRAINER_STATE_UNKNOWN',
    'observed_local_utc': now,
    'last_accepted_local_backup_round': 29, 'last_acceptance_utc': accepted['observed_utc'],
    'latest_independently_verified_state_and_lineage_round': 29,
    'verified_learning_rate_records': 46893, 'maximum_learning_rate_error': 0.0,
    'maximum_metric_error': 0.0, 'rng_states_present': True,
    'verified_round29_rolling_checkpoint_sha256': backup['rolling_checkpoint_sha256'],
    'old_backup_client': {'pid': arm['monitor_pid'], 'create_time': arm['monitor_create_time'],
                          'identity_still_alive': False, 'exec_session_exit_code': 1},
    'failure_status': failure_status,
    'failure_events_after_last_acceptance': events[accepted_index + 1:],
    'diagnosis': 'Local SFTP monitor exhausted its 600-second reconnect window with SSH protocol-banner errors. Both default and authorized network probes establish TCP to port 30848 and then receive EOF before any SSH banner. This does not prove the remote instance or original trainer stopped.',
    'remote_trainer_state': 'UNKNOWN; original last-known trainer identity retained, no terminal inference from connectivity loss.',
    'other_authorized_server': {'port': 30219, 'project_presence': peer['arm_paths'], 'gpu': peer['gpu'],
                               'gpu_processes': peer['gpu_processes'],
                               'scope': 'Its known project path has no NO_MSCA directory; another host namespace cannot establish the original trainer state.'},
    'native_backup_contract': {'client_reconnect_window_seconds': 600, 'trainer_epoch_ack_deadline_seconds': 1800,
                               'scope': 'Frozen source constants; the actual remote phase and deadline remain unobserved.'},
    'next_required_information': 'Current SSH entry and running state of the original MSCA instance; requested asynchronously from the user.',
    'recovery_order': [
        'Verify current endpoint, host identity, boot/GPU identity, frozen project protocol, and original trainer/supervisor identity.',
        'If the original process is alive and awaiting backup, restore only its backup client against the existing checked store and verify the first pending epoch.',
        'If an actual terminal state is proved, inspect all available remote/local checkpoints and remaining authorized update budget before considering continuation; connectivity loss alone is not a restart trigger.'
    ],
    'blocking_condition_first_goal_turn': True,
    'goal_status_at_recording': 'active; first-turn access failure, not marked blocked or complete',
    'manuscript_revision': state['paper_revision'], 'manuscript_sha256': state['paper_sha256'],
    'evidence_sha256': {str(p.relative_to(PAPER)): digest(p) for p in evidence_paths},
    'training_started_stopped_or_retried': False, 'new_model_forwards': 0, 'new_evaluator_runs': 0,
    'submission_ready': False}
output = HERE / 'connectivity_incident_and_safe_round29.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
arm.update(status='REMOTE_UNREACHABLE_BACKUP_CLIENT_FAILED', remote_process_state='UNKNOWN',
           last_accepted_round=29, last_acceptance_utc=accepted['observed_utc'],
           independent_state_audit_round=29, independent_lineage_audit_round=29,
           backup_monitor_terminated=True, latest_connectivity_incident=str(output),
           latest_connectivity_incident_sha256=digest(output),
           pending_information='Original MSCA instance status and current SSH entry')
arm['last_local_backup_observation'] = {
    'observed_local_utc': now, 'monitor_pid': arm['monitor_pid'],
    'monitor_create_time': arm['monitor_create_time'], 'monitor_identity_alive': False,
    'transport_verified_round': 29, 'independent_lineage_verified_round': 29,
    'monitor_status': failure_status['status'], 'scope': 'Local backup client terminal; remote trainer unknown.'}
state['observed_local_utc'] = now
header = (
    '【当前访问异常：2026-11-09 UTC；本地日期2026-11-08】\n'
    'NO_MSCA 本地备份客户端已因 SSH 握手失败、600秒重连窗口耗尽而退出。30848 在普通和授权网络通道均于 SSH banner 前关闭连接，当前无法确认远端原训练是否还在运行；不能继续按“正在正常训练”报告。\n'
    '本地已完整留存至第29轮，原始/EMA/优化器及随机状态、46,893条学习率、548图预测与评价数组已独立核验，误差为0。没有重训、重新推理或运行评价器。'
    '30219 可访问且GPU空闲，但其所查项目路径没有 NO_MSCA 目录；不能据另一主机的进程或文件判断原进程已结束。\n'
    '已向用户请求 MSCA 实例状态与当前 SSH 入口。入口恢复后，先核对原进程、协议、GPU及待备份轮次；若原进程仍在等确认，先恢复备份客户端。'
    '原始进程状态未核实前不把访问失败当作已停训。稿件仍第三十五轮，尚不能投稿；目标 active，本次是首次访问阻碍。\n'
    '证据：远程公平对照_20261103/模块消融/NO_MSCA/connectivity_20261109/connectivity_incident_and_safe_round29.json。以下为此前记录。\n\n')
note = HERE / '访问异常与第29轮安全备份.txt'
with note.open('x', encoding='utf-8') as stream:
    stream.write(header)
for path, name in [(PAPER / '当前修订入口.txt', 'paper_entry_before_incident.txt'),
                   (PAPER / '修订记录_20261103/目标验收与阻塞记录.txt', 'goal_record_before_incident.txt')]:
    with (HERE / name).open('xb') as stream:
        stream.write(path.read_bytes())
    path.write_text(header + path.read_text(encoding='utf-8-sig'), encoding='utf-8')
checklist = PAPER / '远程公平对照_20261103/结果并入与验收清单.txt'
with (HERE / 'acceptance_checklist_before_incident.txt').open('xb') as stream:
    stream.write(checklist.read_bytes())
checklist.write_text(header + checklist.read_text(encoding='utf-8-sig'), encoding='utf-8')
state['acceptance_checklist_sha256'] = digest(checklist)
temporary = state_path.with_suffix('.incident.tmp')
assert state_path.read_bytes() == state_bytes
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, state_path)
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-09 UTC：MSCA SSH入口失联与第29轮本地核验\n\n' + header + '\n')
print(json.dumps({'status': report['status'], 'NO_MSCA_safe_rounds': 29,
                  'independently_verified_learning_rates': 46893, 'remote_trainer_state': 'UNKNOWN',
                  'training_action': None, 'submission_ready': False}))
