"""Publish the completed baseline only; keep the three-arm goal active."""
import datetime
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent;B = D.parent;V = D / 'result_verification_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value):p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    previous_path = B / '101_A43启动容量与冻结核验回执.json';previous = read(previous_path)
    for name, expected in previous['files_sha256'].items():assert sha(Path(name)) == expected, name
    proofs = {k: read(V / ('B_epoch12_' + k + '.json')) for k in ['independent_metrics', 'independent_GT_ignore', 'checkpoint_cpu_verification']}
    assert all(p['status'] == 'PASS' for p in proofs.values())
    assert read(V / 'B_controls_verified.json')['status'] == 'PASS'
    dl = read(D / 'completed_arm_B_download_verification.json');assert dl['status'] == 'PASS'
    assert read(V / 'B_CPU_receipt_download_check.json')['status'] == 'PASS'
    raw = read(D / 'completed_results/B/arm_B/result.json')
    assert raw['status'] == 'COMPLETED' and raw['epochs'] == 12 and raw['optimizer_updates'] == 9708
    assert read(D / 'completed_results/B/arm_B/exit_receipt.json')['exit_code'] == 0
    assert raw['full_model_forwards'] == 10159 and raw['teacher_forwards'] == 0
    live = read(D / 'goal_live_latest.json');obs = live['live_observation']
    assert live['receipt_id'] == 'A43_goal_r04_17'
    assert obs['processes']['supervisor']['identity_matches'] and obs['processes']['M']['identity_matches']
    assert obs['processes']['B'].get('missing') and obs['arms']['M']['status'] == 'TRAINING'
    snapshot = D / 'state_before_B_terminal';snapshot.mkdir(exist_ok=False);mapping = {}
    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
        src = B / name;dest = snapshot / name;dest.write_bytes(src.read_bytes())
        mapping[str(src)] = {'snapshot': str(dest), 'sha256': sha(dest)}
    historical = {'status': 'PASS', 'source_receipt': str(previous_path), 'source_receipt_sha256': sha(previous_path),
                  'all_previous_receipt_files_verified': len(previous['files_sha256']), 'remap': mapping}
    write(D / 'startup_state_historical_remap.json', historical)
    m = proofs['independent_metrics']['metrics'];active = obs['arms']['M']
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = B / '102_A43原版终点已核验_区域蒸馏运行中.txt'
    receipt_path = B / '103_A43原版终点独立核验回执.json'
    assert not report.exists() and not receipt_path.exists()
    event = {'phase': 'A43', 'event': 'B_TERMINAL_VERIFIED_M_RUNNING', 'recorded_utc': now,
             'goal_status': 'active', 'status': 'BASELINE_VERIFIED_REGIONAL_DISTILLATION_RUNNING',
             'mainline_validated': False, 'comparative_AP_available': False, 'active_arm': 'M',
             'verified_endpoint_arms': ['B'], 'pending_endpoint_arms': ['M', 'Q'],
             'completed_epochs_by_arm': {'B': 12, 'M': active['completed_epochs'], 'Q': 0},
             'B_endpoint_metrics': m, 'B_checkpoint_sha256': proofs['checkpoint_cpu_verification']['checkpoint_sha256'],
             'B_download_verified_members': dl['member_files_verified'],
             'B_independent_precision_cells': 40400, 'B_independent_recall_cells': 400,
             'B_precision_max_abs_diff': proofs['independent_metrics']['all_precision_cells_max_abs_difference'],
             'B_recall_max_abs_diff': proofs['independent_metrics']['all_recall_cells_max_abs_difference'],
             'B_raw_annotation_files_verified': 548, 'B_retained_predictions_checked': m['predictions_after_ignore'],
             'formal_M_Q_teacher_pairing_verified': False,
             'new_student_forwards_by_verification': 0, 'new_teacher_forwards_by_verification': 0,
             'new_optimizer_updates_by_verification': 0, 'training_protocol_changed': False,
             'latest_live_receipt': live['receipt_id'], 'live_observation': obs,
             'report': str(report), 'receipt': str(receipt_path),
             'next': 'Continue existing M/Q to12, collect/check each endpoint, independently compare Q-B and Q-M at frozen gates. No other-direction training or rescue sweep.'}
    lines = ['## Material Passport', '类型：ARS experiment-agent / run + baseline endpoint verification。目标active，三组任务尚未完成。', '',
        '已完成：A43 fresh原版B，12轮/9708次更新/10159次学生完整前向/0次教师前向，训练退出0。',
        '正在运行：普通区域蒸馏M，第%d轮%d/809步，%d次更新和%d次教师前向；PID22270/start_ticks106117130与/proc匹配。Q仍排队。' % (active['epoch'], active['microbatch'], active['optimizer_updates'], active['teacher_forwards']), '',
        '固定第12轮EMA，本次COCO-style口径（不是VisDrone官方榜单分数）：',
        '组别\t总AP\tAPs\t小目标AP75',
        'B原版\t%.10f\t%.10f\t%.10f' % (m['AP'], m['APs'], m['APs75']), '',
        '核验范围：',
        '- 终点权重、GT/预测、精度矩阵、日志、实际冻结源码及协议已下载；归档%d字节、%d个成员SHA通过。' % (dl['metadata']['bytes'], dl['member_files_verified']),
        '- raw/EMA严格CPU装载为原版9803112参数，参数有限，正常invalid-anchor的inf缓冲区符合源码定义；优化器参数名匹配、step最小/最大均9708。',
        '- 独立NumPy实现未导入训练评价器/torch/pycocotools；40400个精度单元与400个召回值逐值完全一致，标量仅有约1e-15pp求和舍入差。',
        '- 548份原始标注逐文件SHA核对，经另一种矩形并集算法重建38602个评价GT、157个忽略GT；153766条保留预测均检查不落入应忽略区域。',
        '- 范围限制：检查已保存的评价预测，没有重新运行网络或生成被忽略的原始预测。',
        '- 原版12轮预算、输入流记录及采样标量日志通过核对。M/Q完整正式输入/教师目标配对尚未完成，不用容量阶段检查替代。',
        '- CPU回执下载内容与实际远端exit0返回对象一致；全部后处理新增学生/教师前向和更新均0。', '',
        '当前结论：这里只建立了新seed的基线参考。M/Q固定12轮尚未完成，不能据此判断蒸馏、query目标、语义机制或创新成立。',
        '保持冻结判据：Q-B APs≥0.5pp、Q-M APs≥0.3pp、Q-B总AP≥-0.2pp；三项须同时满足。未达即停止直接query目标主张，不追加层数、教师或权重搜索。',
        '本次单seed12轮为探索性投入判断；公开验证集此前用于多次选题，不作显著性或泛化结论。',
        '完整方法和执行协议见100_A43蒸馏三组已启动与固定判据.txt及A43目录startup_evidence/execution_protocol_FROZEN.json。']
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    board_path = B / '01_候选判别表.json';state_path = B / '15_当前状态与结果校验.json'
    board, state = read(board_path), read(state_path)
    for row in board['candidates']:
        if row['id'] == 'A41-B':row.update(status='PILOT_B_VERIFIED_M_RUNNING_Q_PENDING', efficacy_validated=False, evidence=str(report), next=event['next'])
    board.update(goal_status='active', status='A43_B_VERIFIED_M_RUNNING_Q_PENDING', latest_update_utc=now,
                 latest_report=str(report), pilot_validation_status='B_ENDPOINT_VERIFIED_M_Q_PENDING',
                 mainline_validated=False, remaining_deliverable=event['next'], A43_B_terminal_verification=event)
    formal_updates = 9708 + active['optimizer_updates']
    formal_student_calls = 10159 + active['full_model_forwards'];formal_teacher_calls = active['teacher_forwards']
    state.update(recorded_utc=now, goal_status='active', mainline_validated=False,
                 model_forwards=34010 + 12 + formal_student_calls + formal_teacher_calls,
                 training_updates=29664 + 5 + formal_updates, formal_detector_training_updates=29650 + formal_updates,
                 capacity_only_optimizer_updates=19, detector_optimizer_updates_including_capacity=29664 + 5 + formal_updates,
                 teacher_forwards_A43=4 + formal_teacher_calls,
                 formal_training_epochs=36 + 12 + active['completed_epochs'],
                 training_counts_terminal=False, counts_terminal=False, counters_observed_lower_bounds_while_job_live=True,
                 current_native_job_terminal=False, current_user_task_status='A43_B_VERIFIED_M_RUNNING_Q_PENDING',
                 latest_report=str(report), current_goal_turn=event, latest_authoritative_phase=event,
                 current_user_task_phase=event, A43_B_terminal_verification=event)
    state.setdefault('authoritative_phase_history', []).append(event)
    write(board_path, board);write(state_path, state)
    paths = [report, board_path, state_path, D / 'startup_state_historical_remap.json',
             D / 'completed_arm_B_download_verification.json', V / 'B_controls_verified.json',
             V / 'B_epoch12_independent_metrics.json', V / 'B_epoch12_independent_GT_ignore.json',
             V / 'B_epoch12_checkpoint_cpu_verification.json', V / 'B_CPU_receipt_download_check.json']
    receipt = dict(event);receipt.update(previous_canonical_snapshots=historical,
        files_sha256={str(p): sha(p) for p in paths}, this_goal_turn_classification='progress_baseline_terminal_collected_and_independently_verified')
    write(receipt_path, receipt);write(D / 'goal_progress_04_B_terminal_verified_M_running.json', receipt)
    print(json.dumps({'status': event['status'], 'goal_status': 'active', 'B_metrics': m,
                      'M_updates_observed': active['optimizer_updates'], 'report': str(report), 'receipt': str(receipt_path)}))


if __name__ == '__main__':main()
