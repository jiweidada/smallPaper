"""Record the independently verified 20-round prefix and current live identity."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
ARM = HERE / 'NO_MSCA'
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
state_path = HERE / 'current_execution_state.json'
previous_state_sha = digest(state_path)
state = read(state_path)
paper_path = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
paper_sha = '2653ab4164cc454b3fad4d21c4c2e02e80febe00e80e4fc3e13759a7173114ac'
assert digest(paper_path) == state['paper_sha256'] == paper_sha
lineage_path = ARM / 'lineage_through_round020.json'
backup_path = ARM / 'round020_verified_from_D.json'
live_path = ARM / 'live_round020_20261109.json'
lineage, backup, wrapper = read(lineage_path), read(backup_path), read(live_path)
assert lineage['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE'
assert lineage['completed_rounds'] == backup['round'] == 20
assert lineage['protocol_sha256'] == backup['protocol_sha256'] == state['NO_MSCA']['protocol_sha256']
assert lineage['learning_rates'] == {'recorded_attempts': 32340, 'maximum_absolute_difference': 0.0, 'groups': 4}
assert backup['learning_rate_records'] == 32340 and backup['rng_states_present']
assert backup['metrics'] == lineage['latest_metrics']
assert lineage['maximum_latest_metric_error'] == backup['maximum_metric_error'] == 0
assert lineage['validation_images'] == backup['validation_images'] == 548
assert lineage['predictions'] == backup['predictions'] == 164400
assert lineage['model_forwards'] == lineage['evaluator_runs'] == 0
assert wrapper['exit_code'] == 0 and not wrapper['stderr']
snapshot = json.loads(wrapper['stdout'])
live = snapshot['arms']['no_msca_20261107_v1']
supervisor = live['supervisor.json']
arm_state = state['NO_MSCA']
assert snapshot['boot_id'] == arm_state['remote_boot_id']
assert supervisor['protocol_sha256'] == arm_state['protocol_sha256']
assert supervisor['status'] == live['receipt']['status'] == 'RUNNING'
assert supervisor['pid'] == arm_state['remote_trainer_pid'] == 1774399
assert live['1774399'] == {'exists': True, 'start_ticks': arm_state['remote_trainer_start_ticks']}
assert live['1774398'] == {'exists': True, 'start_ticks': supervisor['supervisor_start_ticks']}
assert live['receipt']['completed_epochs'] >= 20
assert arm_state['candidate_gpu_uuid'] in snapshot['gpu']
now = datetime.now(timezone.utc).isoformat()
evidence_paths = [lineage_path, backup_path, live_path, ARM / 'audit_round020_bootstrap.py',
                  HERE / 'msca_parameter_decomposition_20261108.json',
                  HERE / 'parameter_capacity_evidence_supplement_20261108.json']
report = {
    'status': 'NO_MSCA_FIRST20_LINEAGE_VERIFIED_RUNNING_NOT_A_TERMINAL_RESULT',
    'observed_local_utc': now, 'manuscript_content_revision': 34, 'manuscript_sha256': paper_sha,
    'evidence_sha256': {str(p.relative_to(PAPER)): digest(p) for p in evidence_paths},
    'completed_rounds_independently_audited': 20, 'learning_rate_records': 32340,
    'maximum_learning_rate_error': 0.0, 'maximum_metric_error': 0.0,
    'rng_states_present': True, 'validation_images': 548, 'predictions': 164400,
    'current_remote_identity': {'boot_id': snapshot['boot_id'], 'trainer_pid': 1774399,
                                'trainer_start_ticks': live['1774399']['start_ticks'],
                                'supervisor_pid': 1774398, 'supervisor_status': supervisor['status'],
                                'receipt': live['receipt'], 'gpu': snapshot['gpu']},
    'next_planned_prefix_audit_round': 40,
    'scope': 'Independent inspection of saved artifacts and the current live process identity. It is not a new validation, final module outcome, causal attribution or submission certificate.',
    'new_model_forwards': 0, 'new_evaluator_runs': 0, 'new_training_groups': 0,
    'manuscript_changed': False, 'submission_ready': False}
output = HERE / 'msca_round020_evidence_supplement_20261109.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
events = [json.loads(line) for line in (Path(arm_state['backup_root']) / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
accepted = next(e for e in reversed(events) if e['status'] == 'EPOCH_BACKUP_VERIFIED')
arm_state.update(independent_state_audit_round=20, independent_lineage_audit_round=20,
                 last_accepted_round=accepted['epoch'] + 1, last_acceptance_utc=accepted['observed_utc'],
                 next_planned_lineage_audit_round=40,
                 prefix_evidence_supplement=str(output), prefix_evidence_supplement_sha256=digest(output))
arm_state['previous_remote_live_observation'] = arm_state['last_remote_live_observation']
arm_state['last_remote_live_observation'] = {
    'file': str(live_path), 'wrapper_sha256': digest(live_path), 'server_observed_utc': snapshot['observed_utc'],
    'boot_id': snapshot['boot_id'], 'supervisor_status': supervisor['status'],
    'trainer_pid': supervisor['pid'], 'trainer_start_ticks': supervisor['start_ticks'],
    'receipt': live['receipt'], 'gpu': snapshot['gpu'], 'gpu_processes': snapshot['gpu_processes']}
state['observed_local_utc'] = now
temporary = state_path.with_suffix('.round020.tmp')
assert digest(state_path) == previous_state_sha
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, state_path)
header = (
    '【实验与方法证据补核：2026-11-08，稿件仍第三十四轮】\n'
    'NO_MSCA 前 20 个完整轮次的保存状态和轨迹已独立核验：32,340 条学习率误差为 0，548 图/164,400 条预测与保存评价数组、十二指标一致，随机状态保留。'
    'SSH 实查原训练与监督器身份未变，已进入第 21 轮；不是完整 320 轮结果，不据此更新论文性能结论。\n'
    '静态参数核查显示 MSCA 的 3,841,280 个可训练参数均来自末级 Dilated 门；其中上下文投影 3,146,752 个，占约 81.92%。两处 Lite 无可训练参数。'
    '不构建或运行模型，不将参数计数当作精度、时延或显存测量。待完整 MSCA 结果核验后统一修订相应分析。\n'
    '证据为远程公平对照_20261103/模块消融/msca_round020_evidence_supplement_20261109.json 与 msca_parameter_decomposition_20261108.json。'
    '稿件 SHA-256 仍为 ' + paper_sha + '；目标 active，尚不能投稿。以下为此前记录。\n\n')
for path in [PAPER / '当前修订入口.txt', PAPER / '修订记录_20261103/目标验收与阻塞记录.txt']:
    backup_name = 'paper_entry_before_round020_20261109.txt' if path.parent == PAPER else 'goal_acceptance_before_round020_20261109.txt'
    with (ARM / backup_name).open('xb') as stream:
        stream.write(path.read_bytes())
    path.write_text(header + path.read_text(encoding='utf-8-sig'), encoding='utf-8')
with (HERE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-08 本地日期 / 2026-11-09 UTC：MSCA 前 20 轮独立核验\n\n' + header + '\n')
assert digest(paper_path) == paper_sha
print(json.dumps({'status': report['status'], 'NO_MSCA_audited_rounds': 20,
                  'NO_MSCA_latest_backup_round': arm_state['last_accepted_round'],
                  'remote_phase': live['receipt']['phase'], 'remote_actual_round': live['receipt']['actual_loop_epoch'] + 1,
                  'manuscript_changed': False, 'submission_ready': False}))
