"""Pin the independently read-back164 delivery while retaining UNKNOWN remote state."""
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
DATA = PAPER / '实验数据汇总_20261108'
SNAPSHOT = DATA / 'verified_snapshots/NO_MSCA_round164_verified_20261109'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


assert digest(DATA / '交付清单.json') == 'a3420c420b0a185403765120cfeb129a1ea095b7cd49bf4741a88d78dedca7cf'
assert digest(SNAPSHOT / 'snapshot_manifest.json') == '6f4caa3f044cc91d1a3823b58da54ad08b4789394178ed07b2a3826aa93a0011'
assert digest(DATA / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx') == 'd753a4c644e1d2ef451a6b3b93acc4829d35f7cbbae5785f31ee983bb00c7c88'
assert read(DATA / '交付清单.json')['NO_MSCA_accepted_rounds'] == 164
watch = read(HERE / 'access_watch_identity.json')
status = read(HERE / 'access_watch_status.json')
assert status['status'] == 'READ_ONLY_OBSERVATION_WINDOW_EXPIRED_REMOTE_TRAINING_UNKNOWN'
try:
    process = psutil.Process(watch['pid'])
    watch_alive = abs(process.create_time() - watch['create_time']) < .01
except psutil.NoSuchProcess:
    watch_alive = False
assert not watch_alive
state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
arm = state['NO_MSCA']
assert state['goal_status'] == 'active' and arm['remote_process_state'] == 'UNKNOWN'
assert arm['last_accepted_round'] == arm['independent_lineage_audit_round'] == 164
assert arm['active_goal_access_episode'] == 'goal_monitor002_loss_20261109'
assert arm['consecutive_access_block_goal_turns'] == 1
stamp = datetime.now(timezone.utc)
local_date = stamp.astimezone(timezone(timedelta(hours=-8))).date().isoformat()
preview = DATA / '_build/previews/round164_progress_after.png'
protected = read(DATA / '_build/round164_workbook_checks.json')
assert protected['protected_values_formulas_and_sources_identical']
assert protected['protected_sheet_renders_identical'] and protected['final_metric_cells_blank'] == 24
report = {
    'status': 'ROUND164_DELIVERY_PINNED_REMOTE_STATE_STILL_UNKNOWN',
    'observed_local_utc': stamp.isoformat(), 'user_facing_local_date': local_date,
    'goal_status': 'active', 'new_access_episode_goal_turn': 1,
    'independently_verified_round': 164, 'data_rows': 1124,
    'parent_readback_current_file_count': 19, 'parent_readback_snapshot_file_count': 20,
    'parent_verified_previous1120_rows_exact_byte_prefix': True,
    'parent_verified_completed_group_rows': {'BASE': 320, 'FULL': 320, 'NO_MPS': 320},
    'snapshot_delivery_manifest': {'file': str(SNAPSHOT / '交付清单.json'),
        'sha256': digest(SNAPSHOT / '交付清单.json')},
    'snapshot_manifest': {'file': str(SNAPSHOT / 'snapshot_manifest.json'),
        'sha256': digest(SNAPSHOT / 'snapshot_manifest.json')},
    'workbook': {'file': str(DATA / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx'),
        'sha256': digest(DATA / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx')},
    'parent_visual_inspection': {'file': str(preview), 'sha256': digest(preview),
        'result': 'All164-round values and24 blank final cells readable; UNKNOWN remote state and exited clients explicit; no clipping'},
    'protected_sheet_checks': protected['full_protected_render_comparison'],
    'lineage_audit': {'file': arm['latest_recovered_lineage_audit'],
        'sha256': arm['latest_recovered_lineage_audit_sha256']},
    'latest_access_watch': {**watch, 'exec_session_id': 41211,
        'exec_exit_code': 1, 'process_alive': False, 'terminal_status': status},
    'remote_training_state': 'UNKNOWN', 'historical_last_observed_training_round': 165,
    'native_backup_client_exited': True, 'original_live_observer_exited': True,
    'management_page_check': {'observed_window_title': '人工智能平台 - Google Chrome',
        'content_access_result': 'Computer Use app approval timed out',
        'instance_state_or_current_ssh_entry_read': False, 'ui_input_actions': 0},
    'required_external_information': 'Current instance status and working SSH entry; question remains pending',
    'training_control_actions': 0, 'new_model_forwards': 0, 'new_evaluator_runs': 0,
    'submission_ready': False}
report_path = HERE / 'round164_data_handoff_and_access_status.json'
write(report_path, report, exclusive=True)
arm['data_handoff'] = {'path': str(SNAPSHOT / '交付清单.json'),
    'sha256': digest(SNAPSHOT / '交付清单.json'), 'data_rows': 1124,
    'NO_MSCA_verified_round': 164, 'scope': 'Immutable validated snapshot; remote state UNKNOWN'}
arm['current_data_delivery_manifest'] = {'path': str(DATA / '交付清单.json'),
    'sha256': digest(DATA / '交付清单.json'),
    'scope': 'Continuous current delivery; historic hashes resolve via immutable snapshots'}
arm['data_snapshot_independently_verified_round'] = 164
arm['latest_data_snapshot_manifest'] = report['snapshot_manifest']
arm['latest_status_evidence'] = {'file': str(report_path), 'sha256': digest(report_path)}
state['access_recovery_observer'] = report['latest_access_watch']
state['observed_local_utc'] = stamp.isoformat()
state['operational_work_status'] = 'ACCESS_UNAVAILABLE_REMOTE_TRAINING_STATE_UNCONFIRMED'
write(state_path, state)
goal_path = RECOVERY / 'active_monitor_goal_20261109.json'
goal = read(goal_path)
goal['latest_verified_checkpoint'] = arm['latest_status_evidence']
goal['latest_data_handoff'] = arm['data_handoff']
goal['latest_access_recovery_observer'] = report['latest_access_watch']
goal['latest_progress_classification'] = '164-round audit and1124-row export completed; read-only retries expired with remote state unknown'
write(goal_path, goal)
note = (
    f'【第164轮数据已固定，远端状态仍待确认：{local_date}】\n'
    'NO_MSCA独立状态/轨迹与原生完整备份截至164轮；Excel/CSV/JSON已更新至164轮，共1124条逐轮记录，前1120行原字节前缀保持。当前19项交付及清单另存verified_snapshots/NO_MSCA_round164_verified_20261109，20项文件主控回查通过，160轮快照保留；24个320轮终点格仍空。\n'
    '30219 SSH握手前断开；原备份客户端和只读观察器均已退出。额外只读重连观察窗口也已结束，远端训练当前仍UNKNOWN。165轮仅失联前最后观察，不能当作当前正在运行。已打开管理页的内容访问审批超时，未读取实例状态或当前入口，未绕过该访问限制。\n'
    '已向用户请求当前可用SSH入口与实例状态，未得到新信息前不重启训练。目标active，本次新访问异常仍计第1个目标轮次；下一步先恢复对原进程的可验证访问。BASE/FULL/NO_MPS各320轮及稿件第三十六轮保持，尚不能投稿。以下为此前历史记录。\n\n')
for filename in ['当前修订入口.txt', '目标自主执行入口_20261109.txt']:
    path = PAPER / filename
    path.write_text(note + path.read_text(encoding='utf-8'), encoding='utf-8')
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as handle:
    handle.write(f'\n\n## {local_date}：164轮数据固定与只读访问复查\n\n' + note)
print(json.dumps({'status': report['status'], 'data_rows': 1124,
    'safe_round': 164, 'remote_state': 'UNKNOWN', 'goal_status': 'active',
    'access_episode_goal_turn': 1, 'training_actions': 0}, ensure_ascii=False))
