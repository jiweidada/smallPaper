"""Record the user-activated monitor goal and the live read-only observer."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import psutil

here = Path(__file__).resolve().parent
bundle, paper = here.parents[1], here.parents[3]
folder = here / 'goal_monitor_20261109_001'
identity = json.loads((folder / 'observer_identity.json').read_text(encoding='utf-8'))
observer = psutil.Process(identity['observer_pid'])
assert identity['observer_pid'] == 17576 and abs(observer.create_time() - 1794281820.8738992) < .01
backup = psutil.Process(18924)
assert abs(backup.create_time() - 1794207481.6664748) < .01
status = json.loads((folder / 'observer_status.json').read_text(encoding='utf-8'))
assert status['status'] == 'REMOTE_WORKER_CONFIRMED_LIVE' and status['remote_worker_handle_confirmed_live']
snapshot = here / 'goal_initial_live_20261109.json'
with snapshot.open('xb') as output: output.write((folder / 'latest_live.json').read_bytes())
contract = {'goal_objective': '监测', 'goal_status_observed_from_goal_tool': 'active',
    'observed_local_utc': datetime.now(timezone.utc).isoformat(),
    'user_facing_local_date': '2026-11-09',
    'monitor_scope': 'Existing authorized NO_MSCA run to320 with native local backup and saved-state/lineage audits',
    'completion_evidence_required': ['320 native accepted backups',
        'genuine successful worker/supervisor exit and completed receipts',
        'independent full saved-state/trajectory/endpoint audit', 'data and status summary reflecting the verified result'],
    'boundaries': ['no new arm/seed/tuning/evaluation', 'no training restart solely for missing observations',
                  'SSH retries are read-only observation recovery', 'never stop the training worker from this observer'],
    'failure_policy': 'Inspect and report actual terminal/missing handles; never equate observation timeout with training termination',
    'previous_goal_turn_classification': 'progress: round140 independent evidence, data export and source-index preservation',
    'observer': {**identity, 'exec_session_id': 14130, 'status_path': str(folder / 'observer_status.json')},
    'initial_live_snapshot': {'file': str(snapshot), 'sha256': sha256(snapshot.read_bytes()).hexdigest()},
    'next_independent_audit_round': 160, 'new_training_or_remote_writes': False}
with (here / 'active_monitor_goal_20261109.json').open('x', encoding='utf-8') as output:
    json.dump(contract, output, ensure_ascii=False, indent=2)
    output.write('\n')
state_path = bundle / 'current_execution_state.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
state['goal_status'] = 'active'
state['goal_objective'] = '监测'
state['goal_tool_status_note'] = 'User activated the monitoring goal; get_goal returned active. Earlier blocked state is historical.'
state['read_only_observer'] = contract['observer']
state['observed_local_utc'] = contract['observed_local_utc']
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('【监测目标已由用户激活：2026-11-09】\n'
    'goal工具已确认 objective=监测、status=active；此前blocked状态是历史记录，无需继续请求恢复。'
    '已接通只读远端进程/GPU/回执监测，确认原续训进程、监督器与本地备份客户端身份仍一致。'
    '监测到第320轮、成功退出及完整本地备份后，进行完整保存状态和轨迹核验，再据证据结束监测目标。'
    '观察超时不视为训练终止，不自动重试或重启训练。下一独立核验节点160轮。\n'
    '监测会话和首次现场证据：NO_MSCA/recovery30219_r29_20261108/active_monitor_goal_20261109.json。'
    '以下为创建时及此前历史记录。\n\n')
for path in [paper / '当前修订入口.txt', paper / '目标自主执行入口_20261109.txt']:
    path.write_text(note + path.read_text(encoding='utf-8'), encoding='utf-8')
with (bundle / '方案与执行记录.md').open('a', encoding='utf-8') as output:
    output.write('\n\n## 2026-11-09 用户激活监测目标\n\n' + note + '\n')
print(json.dumps({'goal_status': 'active', 'observer_pid': observer.pid,
    'observer_session_id': 14130, 'backup_client_pid': backup.pid, 'target_round': 160}))
