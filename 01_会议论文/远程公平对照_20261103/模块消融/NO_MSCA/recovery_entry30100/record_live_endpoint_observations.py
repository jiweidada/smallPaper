"""Record read-only endpoint observations, preserving prior incident evidence."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

here = Path(__file__).resolve().parent
bundle = here.parents[1]
paper = bundle.parents[1]
now = datetime.now(timezone.utc).isoformat()
record = {'observed_local_utc': now, 'scope': 'Current accessible instances only',
          'training_started': False, 'remote_files_written': False, 'endpoints': {},
          'safe_local_msca_round': 29, 'original_msca_storage_recovered': False}
for port in [30219, 30100]:
    path = here / f'live_{port}_readonly.json'
    wrapper = json.loads(path.read_text(encoding='utf-8'))
    assert wrapper['exit_code'] == 0
    value = json.loads(wrapper['stdout'])
    assert value['gpu']['exit_code'] == 0
    assert value['gpu_compute_processes']['exit_code'] == 0
    assert not value['gpu_compute_processes']['stdout'].strip()
    assert not value['training_or_project_processes']
    record['endpoints'][str(port)] = {'evidence': str(path),
        'sha256': sha256(path.read_bytes()).hexdigest(), 'snapshot': value,
        'observed_gpu_compute_process_count': 0, 'observed_training_process_count': 0}
out = here / 'live_provided_endpoints_summary.json'
with out.open('x', encoding='utf-8') as handle:
    json.dump(record, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
state_path = bundle / 'current_execution_state.json'
before = state_path.read_bytes()
with (here / 'execution_state_before_entry30100_update.json').open('xb') as handle:
    handle.write(before)
state = json.loads(before)
state['observed_local_utc'] = now
arm = state['NO_MSCA']
arm['status'] = 'ORIGINAL_RUN_NOT_FOUND_ON_ACCESSIBLE_INSTANCES'
arm['latest_accessible_endpoints_observation'] = {'file': str(out),
    'sha256': sha256(out.read_bytes()).hexdigest(), 'ports': [30219, 30100],
    'gpu_compute_processes': 0, 'training_processes': 0}
arm['remote_process_state'] = 'NOT_FOUND_ON_PROVIDED_INSTANCES_ORIGINAL_STORAGE_UNRESOLVED'
arm['pending_information'] = 'Original MSCA project-volume availability or exact original instance location'
arm['previous_episode_consecutive_access_block_goal_turns'] = arm.get('consecutive_access_block_goal_turns', 3)
arm['consecutive_access_block_goal_turns'] = 0
arm['resumed_storage_recovery_audit_turns'] = 1
arm['backup_monitor_terminated'] = True
state['goal_tool_status_note'] = 'Last goal-tool result was blocked; current user-directed read-only recovery has progressed; no programmatic resume API is available.'
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = '''【新入口与当前进程实查：2026-11-09 UTC；本地日期2026-11-08】
用户提供30100，并再次指定30219检查。两台SSH均已接通，两个不同UUID的3090各为2 MiB/24576 MiB、0%，无GPU计算进程，也未发现训练/项目进程。该结论仅适用于当前可见实例，不能据此证明缺失项目盘上的原进程何时结束。
30100与原MSCA物理GPU及boot_id一致，但当前挂载/24085403014，原/25085400120、环境和MSCA目录均不可见；另有旧任务GPU保留记录，未擅自采用。30219挂载/25085400120，保留已完成NO_MPS的项目与冻结环境，但对应NO_MSCA目录不存在。
本地MSCA第29轮完整备份、独立状态/轨迹核验与接续游标仍保留。本轮仅查询进程、挂载及候选环境/目录，没有上传或改写服务器文件，没有启动训练、评价或备份客户端。当前目标工具上次状态仍为blocked；新入口接入已完成，不沿用旧访问故障的三轮计数。
证据：远程公平对照_20261103/模块消融/NO_MSCA/recovery_entry30100/live_provided_endpoints_summary.json。稿件仍为第三十五轮，尚不能投稿；以下为历史记录。
'''
entrance = paper / '当前修订入口.txt'
entrance.write_text(note + '\n' + entrance.read_text(encoding='utf-8'), encoding='utf-8')
with (bundle / '方案与执行记录.md').open('a', encoding='utf-8') as handle:
    handle.write('\n\n## 2026-11-09 UTC：新30100与30219当前进程只读核查\n\n' + note + '\n')
print(json.dumps({'summary': str(out), 'ports_checked': [30219, 30100],
                  'training_started': False, 'remote_files_written': False}))
