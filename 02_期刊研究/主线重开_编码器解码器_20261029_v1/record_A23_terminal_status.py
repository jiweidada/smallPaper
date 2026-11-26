import datetime
import hashlib
import json
import pathlib
import shutil

BASE = pathlib.Path(__file__).resolve().parent
ENG = BASE.parent
OPLOG = ENG / '前期调研_码率受限航拍小目标_20261028_v1' / '远端验证_20261029_v1' / 'ssh_operation_receipts.jsonl'
OP_ID = 'current_mainline_original_job_readonly_20261030'
NAMES = ['01_候选判别表.json', '13_架构检索增补记录.json', '14_架构核心文献矩阵.json', '15_当前状态与结果校验.json', '16_最近工作源码核查.json']
REPORT = BASE / '56_A23当前候选主线与诊断训练终止状态.txt'
RECEIPT = BASE / '57_A23诊断训练终止核验回执.json'
DEST = BASE / 'A23_候选主线与终止核验_v1'
assert not REPORT.exists() and not RECEIPT.exists() and not DEST.exists(), 'One-shot recorder; inspect existing outputs instead of rerunning.'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

ops = []
with OPLOG.open(encoding='utf-8') as source:
    for line in source:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if value.get('id') == OP_ID:
            ops.append(value)
assert len(ops) == 1 and ops[0]['exit_code'] == 0
snapshot = json.loads(ops[0]['stdout'])
p = snapshot['progress']
r = snapshot['training_result']
sr = snapshot['supervisor_result']
assert not snapshot['child']['live'] and not snapshot['supervisor']['live']
assert sr['returncode'] == 1 and not sr['followup_jobs_launched']
assert r['status'] == 'STOPPED_WITH_EVIDENCE_NO_AUTOMATIC_RESUME'
assert r['forward_entries'] == r['forward_completed'] == 2105
assert r['actual_updates'] == r['optimizer_step_calls'] == 526
assert r['optimizer_attempts'] == 527 and r['AMP_skips'] == 0
assert r['validation_calls'] == r['epochs_completed'] == 0
assert 'Nonfinite native raw loss before sanitation' in r['traceback']
for key in ['forward_entries', 'forward_completed', 'actual_updates', 'epochs_completed', 'validation_calls']:
    assert p[key] == r[key] == sr['training_result'][key]
assert snapshot['child']['expected']['pid'] == 13762
assert snapshot['child']['expected']['start_ticks'] == 99433288

DEST.mkdir()
backup = DEST / 'canonical_before_A23'
backup.mkdir()
for name in NAMES:
    shutil.copy2(BASE / name, backup / name)
write_json(DEST / 'original_job_terminal_snapshot.json', snapshot)
write_json(DEST / 'plain_bridge_operation_receipt.json', ops[0])
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
next_step = '先静态定位原始loss非有限的具体分量和数值路径，保留A22终止证据与冻结源码，不自动续跑或重启；N仍是未验证候选，继续核对最近方法差异后才提出新的隔离验证协议。'
phase = {
    'phase': 'A23', 'classification': 'status_and_terminal_verification',
    'recorded_utc': now, 'remote_snapshot_utc': snapshot['utc'],
    'status': 'A22_TERMINATED_RAW_LOSS_NONFINITE_MAINLINE_UNVALIDATED',
    'goal_status': 'active', 'mainline_validated': False,
    'report': str(REPORT), 'receipt': str(RECEIPT),
    'supervisor_PID': 13761, 'supervisor_start_ticks': 99433280,
    'training_PID': 13762, 'training_start_ticks': 99433288,
    'job_identity_verified_live': False, 'terminal_receipts_verified': True,
    'same_original_job_inspected': True, 'new_jobs_launched_this_turn': 0,
    'new_model_forwards_by_this_status_check': 0,
    'new_optimizer_updates_by_this_status_check': 0,
    'A22_detector_forward_entries_terminal': 2105,
    'A22_detector_forward_completed_terminal': 2105,
    'A22_formal_optimizer_updates_terminal': 526,
    'A22_logged_optimizer_attempts_terminal': 527,
    'A22_actual_optimizer_step_calls_terminal': 526,
    'A22_AMP_skips_terminal': 0, 'epochs_completed': 0,
    'actual_eval_calls': 0, 'full_model_forwards': 2365,
    'actual_cumulative_detector_optimizer_updates': 528,
    'formal_detector_training_updates': 526,
    'historical_capacity_updates': 2,
    'cumulative_authorized_protocol_cap': 39092,
    'full_forward_cap_closed': True,
    'closed_scope': 'A22 one-shot allocation terminated; unused budget is not an automatic restart or followup allocation.',
    'counts_terminal': True, 'counts_are_observed_lower_bounds': False,
    'trained_mechanism_effect_estimated': False,
    'algorithm_increment_trained': False, 'complete_mainline': False,
    'failure_interpretation': 'Raw-loss numerical failure, not evidence for or against the shared-state interference hypothesis; exact offending loss component and upstream cause are not yet identified.',
    'next': next_step,
}
evidence_files = [DEST / 'original_job_terminal_snapshot.json', DEST / 'plain_bridge_operation_receipt.json', pathlib.Path(__file__)]
receipt = dict(phase)
receipt['files_sha256'] = {str(path): sha(path) for path in evidence_files}
receipt['remote_training_result'] = r
receipt['remote_supervisor_result'] = sr
write_json(RECEIPT, receipt)

report = '''当前只有待验证的候选主线，尚无可宣称已成立的论文主线。

候选问题：密集无人机航拍场景中，邻近微小目标通过视觉状态空间编码器的共享状态参与全局交互，是否会对受保护目标的定位与实例分离产生有害干扰？这里的“有害干扰”是待验证假设，不是已有实验结论。

数据与宿主：官方VisDrone2019-DET，train6471张、val548张；VSSD-Tiny编码器接Dome-DETR原生颈部与检测解码器。固定12轮训练原本只用于建立可诊断的宿主，不是作者精度复现，也不是候选新算法训练。

算法增量尚未确定。只有先证明共享状态路径存在可重复、可定位的有害交互，并排清最近工作的具体重合，才讨论不依赖GT的写入/读取选择或实例区分机制。更换骨干、组合局部与全局特征，均不能自行构成创新声明。S/M供体、GT掩膜和11组对照是机制诊断，不是可部署算法。

最新实际状态：同一训练PID13762/start_ticks99433288及监督PID13761/start_ticks99433280均已退出。训练结果与监督结果一致：第1轮第2105个微批完成模型前向后，原生criterion原始loss在nan_to_num之前被有限性检查拦截；报错Nonfinite native raw loss before sanitation。原始分量与上游原因尚待静态定位，不将其解释为共享状态假设的肯定或否定。

终止计数：A22共2105次完整检测器前向、526次实际参数更新，AMP跳步0；记录的优化器尝试527次，但最后一次在真正调用step前因loss保护停止。完整训练轮数0，验证集调用0，没有训练后AP或机制效应结果。监督器returncode1、未触发48小时上限、未启动后续任务。本次查询只读，新增模型调用和更新均为0。

此前架构调用260次，加本次任务终止2105次，累计2365次；更新共528次，其中A20临时容量更新2次、A22正式诊断526次。A22原38832次训练调用上限未耗尽即终止，剩余额度不自动转作续跑、新任务或评估。旧冻结源码与科学设计保留，不覆盖，不自动重启。

本轮回答边界：总体编码器、特征交互、解码器选题仍开放；N是当前优先查证的问题，不是被实验确认的主线。接下来先查明诊断训练的数值问题，核对最近方法的具体差异，再决定是否值得新建隔离验证；不把训练恢复或模块训练当成默认下一步。

证据：57_A23诊断训练终止核验回执.json；A23_候选主线与终止核验_v1/original_job_terminal_snapshot.json；此前预先约定的问题与11组对照见52_A21共享状态路径隔离与诊断研究设计.txt。A22运行中报告54保留为历史快照，本报告替代其当前运行状态。
'''
REPORT.write_text(report, encoding='utf-8')

board = read_json(BASE / NAMES[0])
board.update(status='NO_CONFIRMED_MAINLINE_A22_DIAGNOSTIC_TRAINING_STOPPED', mainline_validated=False, latest_update_utc=now, current_priority_question=next_step, latest_report=str(REPORT))
board['A23_evidence'] = phase
for candidate in board['candidates']:
    if candidate['id'] == 'N_NONCAUSAL_STATE_INSTANCE_INTERFERENCE':
        candidate['status'] = 'MECHANISM_UNTESTED_DIAGNOSTIC_BASELINE_STOPPED_RAW_LOSS_NONFINITE'
        candidate['A23_terminal_evidence'] = phase
write_json(BASE / NAMES[0], board)

search = read_json(BASE / NAMES[1])
search['A23_status_verification'] = {'recorded_utc': now, 'new_searches_for_this_status_answer': 0, 'scope': 'Read-only original-job terminal verification; no new paper-read or novelty-clearance claim in this status answer.', 'report': str(REPORT), 'receipt': str(RECEIPT)}
write_json(BASE / NAMES[1], search)
matrix = read_json(BASE / NAMES[2])
assert len(matrix['rows']) == 77
matrix.update(updated_utc=now, latest_report=str(REPORT), latest_empirical_boundary='A22 terminated before completing epoch1:2105 detector forwards,526 actual updates,0val calls. No trained mechanism effect or deployable algorithm increment. Mainline unvalidated.')
write_json(BASE / NAMES[2], matrix)

state = read_json(BASE / NAMES[3])
state.update(recorded_utc=now, goal_status='ACTIVE', mainline_validated=False, diagnostics_complete=False, model_forwards=2365, cap=39092, training_updates=528, formal_detector_training_updates=526, latest_report=str(REPORT), latest_authoritative_phase=phase, current_goal_turn='A23_current_mainline_status_and_original_job_terminal_verification', current_priority_question=next_step, full_forward_cap_closed=True, counts_terminal=True)
state['training_update_scope'] = '2 historical A20 capacity-only updates plus526 A22 formal diagnostic updates; A22 terminal receipts verified. Historical diagnostic fits and suffix calls remain separate.'
state.setdefault('authoritative_phase_history', []).append(phase)
write_json(BASE / NAMES[3], state)
audit = read_json(BASE / NAMES[4])
audit.update(updated_utc=now)
audit['A23_terminal_verification'] = phase
write_json(BASE / NAMES[4], audit)

for path, digest in receipt['files_sha256'].items():
    assert sha(pathlib.Path(path)) == digest
for name in NAMES:
    read_json(BASE / name)
print(json.dumps({'status': 'RECORDED_AND_VERIFIED', 'report': str(REPORT), 'receipt': str(RECEIPT), 'mainline_validated': False, 'training_terminal': True, 'total_detector_calls': 2365, 'formal_updates': 526, 'paper_rows_unchanged': 77}, ensure_ascii=False))
