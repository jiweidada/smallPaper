"""Preserve current access loss and an audited local cursor; never control training."""
from datetime import datetime, timezone, timedelta
from hashlib import sha256
import json
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
ARM = HERE.parents[1]
BUNDLE = ARM.parent
PAPER = BUNDLE.parents[1]
RECOVERY = ARM / 'recovery30219_r29_20261108'
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, data, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


assert not psutil.pid_exists(18924) and not psutil.pid_exists(6732)
failure = read(BACKUPS / 'monitor_status.json')
assert failure['status'] == 'MONITOR_FAILED_NO_FURTHER_ACKS'
events = [json.loads(v) for v in (BACKUPS / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
accepted = [v for v in events if v['status'] == 'EPOCH_BACKUP_VERIFIED'][-1]
assert accepted['epoch'] == 163
audit_path = RECOVERY / 'lineage_recovered_through_round164.json'
audit = read(audit_path)
assert audit['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and audit['completed_rounds'] == 164
assert audit['learning_rates']['recorded_attempts'] == 265188
assert audit['learning_rates']['maximum_absolute_difference'] == audit['maximum_latest_metric_error'] == 0
assert audit['validation_images'] == 548 and audit['predictions'] == 164400
assert audit['model_forwards'] == audit['evaluator_runs'] == 0
epoch = BACKUPS / 'epochs/epoch0163'
manifest = read(epoch / 'manifest.json')
assert manifest['epoch'] == 163 and manifest['next_epoch'] == 164
assert manifest['optimizer_step_attempts'] == 265188
assert manifest['rolling_sha256'] == digest(epoch / 'next_epoch.pt')
assert accepted['manifest_sha256'] == digest(epoch / 'manifest.json')
for name in ['access_turn1_default.json', 'access_turn1_authorized_network.json']:
    probe = read(HERE / name)
    assert not probe['authentication_attempted'] and probe['training_action'] is None
    assert len(probe['endpoints']) == 1
    assert probe['endpoints'][0]['port'] == 30219
    assert probe['endpoints'][0]['status'] == 'CONNECTION_CLOSED_BEFORE_SSH_BANNER'

state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
assert state['goal_status'] == 'active' and state['goal_objective'] == '监测'
observer = RECOVERY / 'goal_monitor_20261109_002'
for source, name in [
    (state_path, 'execution_state_before_access_loss.json'),
    (BACKUPS / 'monitor_status.json', 'native_monitor_status_at_loss.json'),
    (BACKUPS / 'monitor_events.jsonl', 'native_monitor_events_at_loss.jsonl'),
    (observer / 'observer_status.json', 'observer_status_at_loss.json'),
    (observer / 'observer_terminal.json', 'observer_terminal_at_loss.json'),
    (observer / 'events.jsonl', 'observer_events_at_loss.jsonl'),
    (observer / 'latest_live.json', 'last_successful_remote_snapshot.json')]:
    with (HERE / name).open('xb') as handle:
        handle.write(source.read_bytes())

wrapper = read(HERE / 'last_successful_remote_snapshot.json')
last = json.loads(wrapper['stdout'])
receipt = last['records']['run/receipt.json']
assert last['boot_id'] == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
assert receipt['pid'] == 1779501 and receipt['status'] == 'RUNNING'
assert receipt['completed_epochs'] == 164 and receipt['actual_loop_epoch'] == 164
assert receipt['optimizer_step_attempts'] == 266589
stamp = datetime.now(timezone.utc)
local_date = stamp.astimezone(timezone(timedelta(hours=-8))).date().isoformat()
cursor = {
    'status': 'AUDITED_LOCAL_FALLBACK_CURSOR_NOT_A_RESTART_AUTHORIZATION_OR_COMMAND',
    'last_native_accepted_round': 164, 'checkpoint': str(epoch / 'next_epoch.pt'),
    'checkpoint_sha256': manifest['rolling_sha256'],
    'manifest': str(epoch / 'manifest.json'), 'manifest_sha256': digest(epoch / 'manifest.json'),
    'next_epoch_index_if_continuation_is_appropriate': 164,
    'remaining_retained_rounds': 156, 'remaining_retained_update_attempts': 252252,
    'retained_rounds': 164, 'retained_update_attempts': 265188,
    'saved_model_ema_optimizer_warmup_rng_verified_by_existing_auditor': True,
    'lineage_audit': str(audit_path), 'lineage_audit_sha256': digest(audit_path),
    'protocol_sha256': audit['protocol_sha256'],
    'current_remote_process_state': 'UNKNOWN',
    'required_before_any_continuation': [
        'Re-establish current SSH entry and verify host, boot/GPU, protocol and original process identities',
        'If original worker is live, inspect phase and restore only backup observation/transport when valid',
        'If terminal state is proved, inspect all remote/local artifacts and remaining authorized budget before continuation'],
    'unretained_work_after164': 'Unknown; the last historical observation was training round165',
    'training_actions': 0, 'model_forwards': 0, 'evaluator_runs': 0}
cursor_path = HERE / 'audited_local_cursor_round164.json'
write(cursor_path, cursor, exclusive=True)

report = {
    'status': 'REMOTE_UNREACHABLE_LOCAL_BACKUP_CLIENT_FAILED_REMOTE_TRAINING_UNKNOWN',
    'observed_local_utc': stamp.isoformat(), 'user_facing_local_date': local_date,
    'access_episode': 'goal_monitor002_loss_20261109',
    'consecutive_access_block_goal_turns_this_episode': 1,
    'previous_goal_turn_classification': 'progress:160 audit,1120-row data delivery and immutable snapshot',
    'current_goal_turn_classification': 'verified wait followed by detected access loss and new independent164 audit',
    'last_native_accepted_round': 164, 'last_acceptance_utc': accepted['observed_utc'],
    'independent_state_and_lineage_round': 164, 'learning_rate_records': 265188,
    'historical_last_observed_running_round': 165,
    'historical_last_observed_update_attempts': 266589,
    'historical_last_successful_reader_server_utc': last['observed_server_utc'],
    'historical_snapshot_file_mtime_local_utc': datetime.fromtimestamp(
        (observer / 'latest_live.json').stat().st_mtime, timezone.utc).isoformat(),
    'remote_training_state_now': 'UNKNOWN; access failure does not prove terminal training',
    'backup_client': {'pid': 18924, 'create_time': 1794207481.6664748,
        'alive': False, 'exec_session': 8984, 'exec_exit_code': 1,
        'reason': '600-second SFTP reconnect window exhausted', 'stopped_by_agent': False},
    'observer': {'pid': 6732, 'create_time': 1794283838.1600833,
        'alive': False, 'exec_session': 84874, 'exec_exit_code': 1,
        'reason': 'Backup client handle missing after SSH observation failures', 'stopped_by_agent': False},
    'network_probes': [str(HERE / 'access_turn1_default.json'),
        str(HERE / 'access_turn1_authorized_network.json')],
    'lineage_audit': {'file': str(audit_path), 'sha256': digest(audit_path)},
    'local_fallback_cursor': {'file': str(cursor_path), 'sha256': digest(cursor_path)},
    'data_metric_cutoff_at_record': 160,
    'next_needed_information': 'Current30219 instance state and SSH entry, requested asynchronously',
    'goal_status': 'active; first access-block turn in this new episode',
    'training_started_stopped_retried_or_restarted': False,
    'new_model_forwards': 0, 'new_evaluator_runs': 0, 'submission_ready': False}
report_path = HERE / 'access_loss_and_safe_round164.json'
write(report_path, report, exclusive=True)

arm = state['NO_MSCA']
arm.update(status='REMOTE_UNREACHABLE_BACKUP_CLIENT_FAILED',
    remote_process_state='UNKNOWN', last_accepted_round=164,
    last_acceptance_utc=accepted['observed_utc'], independent_state_audit_round=164,
    independent_lineage_audit_round=164, latest_recovered_lineage_audit=str(audit_path),
    latest_recovered_lineage_audit_sha256=digest(audit_path),
    backup_monitor_terminated=True, backup_monitor_terminated_by_agent=False,
    current_running_round_at_snapshot=None, last_observed_running_round=165,
    pending_information='Current30219 instance status and SSH entry',
    latest_connectivity_incident=str(report_path), latest_connectivity_incident_sha256=digest(report_path),
    active_goal_access_episode='goal_monitor002_loss_20261109',
    consecutive_access_block_goal_turns=1)
arm['last_remote_live_observation'] = {
    'file': str(HERE / 'last_successful_remote_snapshot.json'),
    'wrapper_sha256': digest(HERE / 'last_successful_remote_snapshot.json'),
    'server_observed_utc': last['observed_server_utc'], 'boot_id': last['boot_id'],
    'scope': 'Historical last successful observation; current remote state is UNKNOWN',
    'receipt': receipt, 'current_live_asserted': False}
arm['last_local_backup_observation'] = {
    'observed_local_utc': stamp.isoformat(), 'monitor_pid': 18924,
    'monitor_create_time': 1794207481.6664748, 'monitor_identity_alive': False,
    'monitor_status': failure['status'], 'native_accepted_round': 164,
    'independent_lineage_verified_round': 164, 'native_status': failure}
arm['latest_status_evidence'] = {'file': str(report_path), 'sha256': digest(report_path)}
state['read_only_observer'].update(status='EXITED_REQUIRES_ATTENTION',
    process_alive=False, exec_exit_code=1, terminal_evidence=str(HERE / 'observer_terminal_at_loss.json'))
state['observed_local_utc'] = stamp.isoformat()
write(state_path, state)

goal_path = RECOVERY / 'active_monitor_goal_20261109.json'
goal = read(goal_path)
goal['observer'] = state['read_only_observer']
goal['latest_verified_checkpoint'] = arm['latest_status_evidence']
goal['current_access_episode'] = {'identifier': arm['active_goal_access_episode'],
    'consecutive_goal_turns': 1, 'state': 'Awaiting current SSH entry; trainer state unconfirmed'}
goal['latest_progress_classification'] = report['current_goal_turn_classification']
write(goal_path, goal)
note = (
    f'【当前访问异常与第164轮安全备份：{local_date}】\n'
    '30219在普通和授权网络下均于SSH横幅前断开；原生本地备份客户端18924耗尽600秒重连窗口后退出，只读观察器6732随后退出。远端训练当前状态未知，不能按“正常训练中”或“已经停止”报告。\n'
    'NO_MSCA完整原生备份保留至第164轮，独立状态/轨迹核验通过，265188条学习率误差0，548图保存预测、评价数组和12指标绑定通过；最后一次现场看到第165轮训练中。当前数据交付仍截至160轮，1120行；已完成三组320轮结果保持。\n'
    '已请求用户核对实例与当前SSH入口。入口恢复先核对原进程和待备份状态；访问失败不触发训练重启。已保存第164轮离线安全游标。目标active，这是本次新访问异常的第1个目标轮次；原先失联和blocked计数属于历史。稿件第三十六轮，尚不能投稿。\n'
    '证据：模块消融/NO_MSCA/connectivity_20261109/goal_monitor002_loss_20261109/access_loss_and_safe_round164.json。以下为此前历史记录。\n\n')
for filename in ['当前修订入口.txt', '目标自主执行入口_20261109.txt']:
    path = PAPER / filename
    path.write_text(note + path.read_text(encoding='utf-8'), encoding='utf-8')
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as handle:
    handle.write(f'\n\n## {local_date}：第164轮备份后SSH入口失联\n\n' + note)
print(json.dumps({'status': report['status'], 'safe_rounds': 164,
    'learning_rate_records_verified': 265188, 'remote_state': 'UNKNOWN',
    'goal_status': 'active', 'new_episode_access_turn': 1,
    'training_actions': 0}, ensure_ascii=False))
