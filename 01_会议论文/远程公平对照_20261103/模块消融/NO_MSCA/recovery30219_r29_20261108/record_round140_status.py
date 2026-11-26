"""Bind the latest retained state audit to the live native run observation."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

here = Path(__file__).resolve().parent
bundle, paper = here.parents[1], here.parents[3]
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return sha256(path.read_bytes()).hexdigest()
audit_path = here / 'lineage_recovered_through_round140.json'
live_path = here / 'live_status_20261109_1710.json'
audit = read(audit_path)
assert audit['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and audit['completed_rounds'] == 140
assert audit['learning_rates']['recorded_attempts'] == 226380
assert audit['learning_rates']['maximum_absolute_difference'] == audit['maximum_latest_metric_error'] == 0
assert audit['validation_images'] == 548 and audit['predictions'] == 164400
assert audit['model_forwards'] == audit['evaluator_runs'] == 0
assert audit['recovery']['new_completed_rounds'] == 111
assert audit['recovery']['new_optimizer_update_attempts'] == 179487
live = json.loads(read(live_path)['stdout'])
receipt = live['records']['run/receipt.json']
assert receipt['pid'] == 1779501 and receipt['status'] == 'RUNNING'
assert receipt['actual_loop_epoch'] == 141 and receipt['optimizer_step_attempts'] >= 227998
assert live['processes']['pid']['start_ticks'] == 18098010
backups = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
rows = [json.loads(line) for line in (backups / 'epochs/epoch0139/epochs.jsonl').read_text().splitlines()]
assert len(rows) == 140
recent = [row['seconds'] for row in rows[-10:]]
state_path = bundle / 'current_execution_state.json'
state = read(state_path)
arm = state['NO_MSCA']
arm.update(independent_state_audit_round=140, independent_lineage_audit_round=140,
    latest_recovered_lineage_audit=str(audit_path), latest_recovered_lineage_audit_sha256=sha(audit_path),
    next_planned_lineage_audit_round=160)
observed = datetime.now(timezone.utc).isoformat()
state['observed_local_utc'] = observed
report = {'status': 'LIVE_RECOVERY_RUNNING_WITH_140_ROUND_INDEPENDENT_AUDIT',
    'observed_local_utc': observed, 'user_facing_local_date': '2026-11-09',
    'ssh_port': 30219, 'same_native_worker_and_supervisor_confirmed': True,
    'current_training_round_at_snapshot': 142, 'native_accepted_round_at_observation': arm['last_accepted_round'],
    'independently_verified_round': 140, 'learning_rate_records_verified': 226380,
    'maximum_learning_rate_error': 0, 'maximum_saved_metric_error': 0,
    'validation_images': 548, 'predictions': 164400,
    'audit': {'file': str(audit_path), 'sha256': sha(audit_path)},
    'live': {'file': str(live_path), 'sha256': sha(live_path)},
    'recent10_round_seconds': recent, 'mean_recent10_round_seconds': sum(recent) / len(recent),
    'rough_remaining_training_hours_at_recent_rate': (320 - 141) * sum(recent) / len(recent) / 3600,
    'estimate_scope': 'Recent saved training/validation durations; not a promised completion time or efficiency benchmark',
    'paper_revision': state['paper_revision'], 'NO_MSCA_target': 320,
    'NO_MSCA_complete_endpoint_available': False, 'submission_ready': False,
    'new_local_model_forwards': 0, 'new_local_evaluator_runs': 0}
path = here / 'status_and_audit_round140_20261109.json'
with path.open('x', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
arm['latest_status_evidence'] = {'file': str(path), 'sha256': sha(path)}
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('【NO_MSCA第140轮独立核验与最新实查：2026-11-09】\n'
    '30219原续训进程仍在运行，现场已进入第142轮，原生本地逐轮备份已接受141轮。'
    '独立状态与轨迹核验补齐到140轮，226380条学习率记录误差0，548图/164400条预测与保存评价数组及12项指标一致。'
    '该核验没有重新推理或执行评价器；原始运行、配置和训练过程未更改。\n'
    'BASE、FULL、NO_MPS均已完成320轮。稿件保持第36轮，等待完整MSCA终点后更新消融结果、摘要和结论；尚不能投稿。'
    '数据汇总本次更新目标为独立核验140轮，现场141轮备份和142轮运行分别记载，不能混入140轮指标。\n'
    '证据：模块消融/NO_MSCA/recovery30219_r29_20261108/status_and_audit_round140_20261109.json。以下为此前记录。\n\n')
entrance = paper / '当前修订入口.txt'
entrance.write_text(note + entrance.read_text(encoding='utf-8'), encoding='utf-8')
with (bundle / '方案与执行记录.md').open('a', encoding='utf-8') as handle:
    handle.write('\n\n## 2026-11-09 第140轮独立核验与续训状态\n\n' + note + '\n')
print(json.dumps({k: report[k] for k in ['status', 'current_training_round_at_snapshot',
    'native_accepted_round_at_observation', 'independently_verified_round',
    'rough_remaining_training_hours_at_recent_rate']}, ensure_ascii=False))
