"""Record the goal tool's actual blocked result; no experiment operations."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent.parent
PAPER = BUNDLE.parent.parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: sha256(p.read_bytes()).hexdigest()
tool_path = HERE / 'goal_update_result.json'
tool = read(tool_path)
assert tool['goal']['status'] == 'blocked'
audit_path = HERE / 'blocking_audit_turn3.json'
audit = read(audit_path)
assert audit['same_access_block_consecutive_goal_turns'] == 3 and not audit['completion_proved']
state_path = BUNDLE / 'current_execution_state.json'
old_bytes = state_path.read_bytes()
state = json.loads(old_bytes.decode('utf-8-sig'))
now = datetime.now(timezone.utc).isoformat()
record = {
    'status': 'blocked', 'observed_local_utc': now,
    'goal_tool_result_sha256': digest(tool_path), 'blocking_audit_sha256': digest(audit_path),
    'reason': 'MSCA SSH entry remains unavailable after three consecutive goal-turn audits; no live remote execution state can be established.',
    'required_external_change': 'Restore the original MSCA instance SSH service or provide its current SSH entry.',
    'local_safe_backed_up_and_lineage_verified_rounds': 29,
    'manuscript_revision': 35, 'manuscript_sha256': state['paper_sha256'],
    'remote_trainer_state': 'UNKNOWN', 'manuscript_complete_for_submission': False,
    'native_training_started_stopped_or_retried': False,
    'resume_policy': 'A user resume starts a fresh blocked audit; first revalidate the endpoint, original process and newest available checkpoints. Recover the existing backup stream before considering any training action.'}
record_path = HERE / 'goal_blocked_status_record.json'
with record_path.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
header = (
    '【目标已标记 blocked：2026-11-09 UTC，本地日期2026-11-08】\n'
    'MSCA 原入口30848连续三个目标轮次仍于SSH握手前断开，本地备份客户端已退出，远端训练实际状态未知。'
    '已完成本地第29轮状态/轨迹核验及256个备份对象的客户端接续检查；当前不能在缺少可用入口的情况下完成剩余320轮终点与相应论文分析。\n'
    'goal 工具已返回 blocked。等待原 MSCA 实例恢复 SSH 或用户提供当前入口；此标记不表示取消或主动终止远端训练。'
    '入口恢复后先核对原进程及待备份产物，再处理备份通道，不能因连接失败推断原训练结束。\n'
    '稿件仍为第三十五轮，尚不能投稿。全文目标和已有数据、断点、原稿与逐轮备份保留。'
    '再次恢复目标时重新开始阻塞审计，不直接沿用本次三轮计数。证据见 NO_MSCA/connectivity_20261109/goal_blocked_status_record.json。以下为历史记录。\n\n')
for path, name in [(PAPER / '当前修订入口.txt', 'paper_entry_before_goal_blocked.txt'),
                   (PAPER / '修订记录_20261103/目标验收与阻塞记录.txt', 'goal_record_before_goal_blocked.txt')]:
    with (HERE / name).open('xb') as stream:
        stream.write(path.read_bytes())
    path.write_text(header + path.read_text(encoding='utf-8-sig'), encoding='utf-8')
checklist = PAPER / '远程公平对照_20261103/结果并入与验收清单.txt'
with (HERE / 'acceptance_checklist_before_goal_blocked.txt').open('xb') as stream:
    stream.write(checklist.read_bytes())
checklist.write_text(header + checklist.read_text(encoding='utf-8-sig'), encoding='utf-8')
state.update(goal_status='blocked', goal_status_observed_local_utc=now,
             goal_blocking_audit=str(audit_path), goal_blocking_audit_sha256=digest(audit_path),
             goal_blocked_status_record=str(record_path), goal_blocked_status_record_sha256=digest(record_path),
             acceptance_checklist_sha256=digest(checklist), observed_local_utc=now)
state['NO_MSCA']['consecutive_access_block_goal_turns'] = 3
temporary = state_path.with_suffix('.blocked.tmp')
assert state_path.read_bytes() == old_bytes
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, state_path)
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-09 UTC：三轮访问阻碍审计后目标标记 blocked\n\n' + header + '\n')
print(json.dumps({'goal_status': 'blocked', 'safe_rounds': 29, 'manuscript_revision': 35,
                  'remote_trainer_state': 'UNKNOWN', 'submission_ready': False}))
