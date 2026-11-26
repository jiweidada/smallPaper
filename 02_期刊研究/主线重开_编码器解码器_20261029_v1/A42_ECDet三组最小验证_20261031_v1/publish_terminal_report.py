"""Publish the verified negative pilot and its bounded next-candidate handoff."""
import datetime
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
B = D.parent
V = D / 'result_verification_v1'
N = B / 'A43_B训练期语义监督备选_20261031_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    gate = read(V / 'final_practical_gate_verification.json')
    assert gate['status'] == 'PASS' and gate['pilot_pass'] is False
    assert gate['decision'] == 'STOP_THIS_LOCALIZATION_ONLY_MAINLINE_NO_RESCUE_SWEEP'
    controls = read(V / 'B_S_L_controls_verified.json')
    assert controls['status'] == 'PASS'
    subgroup = read(V / 'per_class_descriptive_scope.json')
    summary_download = read(D / 'terminal_summary_download_verification.json')
    assert summary_download['status'] == 'PASS' and summary_download['all_frozen_job_identities_absent']
    process = read(D / 'terminal_summary/final_process_observation.json')
    assert process['status'] == 'ALL_FROZEN_JOB_IDENTITIES_ABSENT'
    assert process['nvidia_compute_processes']['exit_code'] == 0
    assert process['nvidia_compute_processes']['stdout'].strip() == ''
    next_plan = read(N / 'candidate_plan.json')
    assert next_plan['status'] == 'PROPOSED_NOT_IMPLEMENTED_NOT_TRAINED'
    assert next_plan['new_training_allocation'] == next_plan['new_model_forwards'] == next_plan['new_optimizer_updates'] == 0
    assert len(next_plan['nearest_verified']) >= 3 and next_plan['research_question']
    assert (N / next_plan['report']).is_file()
    assert sha(D / 'completed_results/B/arm_B/epoch12.pth') == next_plan['teacher_checkpoint_sha256']
    previous = read(B / '97_A42原版与共享组终点及配对控制核验回执.json')
    remap = read(D / 'B_S_terminal_state_historical_remap.json')
    for name, entry in remap['remap'].items():
        assert sha(Path(name)) == entry['sha256'] == previous['files_sha256'][name]
        assert sha(Path(entry['snapshot'])) == entry['sha256']
    results = {a: read(D / 'completed_results' / a / ('arm_' + a) / 'result.json') for a in ['B', 'S', 'L']}
    initial = {a: read(D / 'completed_results' / a / ('arm_' + a) / 'initialization.json') for a in results}
    for a in results:
        assert read(V / (a + '_epoch12_checkpoint_cpu_verification.json'))['status'] == 'PASS'
        assert read(V / (a + '_epoch12_independent_GT_ignore.json'))['status'] == 'PASS'
        assert read(D / ('completed_arm_' + a + '_download_verification.json'))['status'] == 'PASS'
    m = gate['metrics']
    fallacies = [
        {'type': 'Simpson paradox', 'finding': '10类描述性核查：L-B小目标AP75有8类为正，L-S有3类为正；总体差值不代表各类一致。未发现所有分组同向而总体反向的简单模式，不宣称排除所有异质性。'},
        {'type': 'Ecological fallacy', 'finding': '指标为类别/IoU/召回阈值的宏平均，不推断每个目标、每个场景都改善。'},
        {'type': 'Berkson selection', 'finding': '三组采用相同548图与预定标注规则，未按候选是否成功选图；忽略区及top300限定了估计范围，不能外推到被排除目标。'},
        {'type': 'Collider bias', 'finding': '主指标使用完整预定评价集，不用共同检出/成功匹配子集筛选主结局；不作由后验选择支持的机制因果结论。'},
        {'type': 'Base rate neglect', 'finding': '保留逐类GT与small-GT数量；AP不是PPV或每个目标检出概率，类别宏平均不能替代目标频率加权表现。'},
        {'type': 'Regression to mean', 'finding': '没有挑各组最佳轮次；统一固定12轮。单seed误差未知，不把S的单次正差自动提升为新主线。'},
        {'type': 'Survivorship bias', 'finding': 'B/S/L均正常完成，三组均报告。准备/SSH收集异常保留；没有删除训练失败组后只报幸存者。'},
        {'type': 'Look elsewhere', 'finding': '历史已探索多候选且复用公开验证集，本结果为探索性资源判别；未计算或宣称显著性、未挑有利类别替换终点。'},
        {'type': 'Forking paths', 'finding': 'A42在开训前冻结组别/初始化/配方/门槛，未扫参数、换终点或最佳轮次。A43是另一个未运行的假设，不用它降低A42门槛。'},
        {'type': 'Correlation versus causation', 'finding': '受控结构比较仅支持本次训练方案的资源判断；AP75受置信排序及原生LQE/训练耦合影响，不能单凭它证明几何细节机制。'},
        {'type': 'Reverse causality', 'finding': '结构和门槛先于最终指标确定；但未实测语义机制，不从A42失败反推A43的语义不足假设成立。'},
    ]
    validation = {
        'status': 'ANALYZED_WITH_ENDPOINT_METRICS_INDEPENDENTLY_VERIFIED',
        'fallacy_types_checked': 11, 'fallacy_scan': fallacies,
        'new_statistical_hypothesis_tests': 0, 'p_values': None, 'confidence_intervals': None,
        'practical_gate_is_not_significance_test': True,
        'training_run_repeated': False, 'training_reproducibility_verdict': 'NOT_REPEATED',
        'checkpoint_integrity_and_metric_recalculation': 'VERIFIED_WITH_STATED_SCOPE',
        'new_model_forwards': 0, 'new_optimizer_updates': 0,
    }
    write(V / 'validation_interpretation_11_types.json', validation)
    report = B / '98_A42三组终点结果与主线去留.txt'
    receipt_path = B / '99_A42终点独立核验与去留回执.json'
    assert not report.exists() and not receipt_path.exists(), 'Do not overwrite an existing terminal publication'
    lines = [
        'Material Passport',
        '类型：ARS experiment-agent / run + endpoint validation；单助手。',
        '状态：三组均完成并核验；定位专用读出未过固定投入门槛，停止该主张。',
        '证据级别：文件/权重和已保存预测的指标复算已核验；单seed训练未重复，不称训练可重复性或论文创新已成立。',
        '',
        '1. 结果与决定',
        'VisDrone + ECDet-S，固定第12轮EMA。本次为COCO-style口径，不是VisDrone官方排行榜分数。',
        '组别\t总AP\tAPs\t小目标AP75\t完整模型前向延时中位数(ms)\t参数\t训练峰值GiB',
    ]
    labels = {'B': 'B 原版', 'S': 'S 共享读出', 'L': 'L 定位专用读出'}
    for a in ['B', 'S', 'L']:
        lines.append('%s\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%.4f' % (
            labels[a], m[a]['AP'], m[a]['APs'], m[a]['APs75'], gate['latency_median_ms'][a],
            initial[a]['model_parameters'], results[a]['peak_allocated_bytes'] / 2**30))
    lines += [
        '',
        '差值均为绝对百分点：',
        'S-B：总AP%+.6f；APs%+.6f；小目标AP75%+.6f。' % tuple(gate['deltas_vs_B_pp']['S'][k] for k in ['AP', 'APs', 'APs75']),
        'L-B：总AP%+.6f；APs%+.6f；小目标AP75%+.6f。' % tuple(gate['deltas_vs_B_pp']['L'][k] for k in ['AP', 'APs', 'APs75']),
        'L-S：总AP%+.6f；APs%+.6f；小目标AP75%+.6f。' % tuple(gate['L_minus_S_pp'][k] for k in ['AP', 'APs', 'APs75']),
        '',
        '冻结判据及结果：',
        'L-B 小目标AP75至少+0.5pp：实际%+.6f，未通过。' % gate['deltas_vs_B_pp']['L']['APs75'],
        'L-S 小目标AP75至少+0.3pp：实际%+.6f，未通过。' % gate['L_minus_S_pp']['APs75'],
        'L/B 原生前向延时增幅≤10%%：实际%+.4f%%，通过。该微小负差不构成加速结论。' % (100 * gate['L_latency_fraction_over_B']),
        'L-B 总AP≥-0.2pp：实际%+.6f，通过。' % gate['deltas_vs_B_pp']['L']['AP'],
        '四项须同时满足，本次两项效果门槛未过。停止“定位专用读出优于共享读出”这个主张，不追加层数、采样尺度、阈值、损失权重或更长训练挽救它。',
        '共享读出有本次正向观测，但不能把控制组的收益直接写成已成立论文创新。短测失败也不证明所有定位细节方法在任何充分训练下都无效。',
        '',
        '2. 实际方法与冻结训练',
        'B保留原生ECDet-S。S/L复用已有stride-4 stem，按进入最后解码层时的detach预测参考框取3×3 RoIAlign，经432→192→192小MLP产生残差；S给最终分类/回归输入，L只给最终回归输入。新增120192参数，S/L同形同初始化，末层零初始化。推理不使用GT，不新增骨干或完整decoder调用。',
        '原生LQE及分类质量/回归训练耦合保留；“L不改分类Linear输入”不等于最终类别分数完全独立于回归。',
        '源码固定提交d8752b75c9beaecfcdd0e5cd207bae3b6946ae23；官方COCO权重严格装载后统一转换为十类公共初始化。',
        '三组seed20261031；单卡顺序B→S→L；每组12轮、batch8、809次更新/轮、9708次有效更新。640固定拉伸、确定性水平翻转、ImageNet归一化，无mosaic/mixup。',
        'AdamW：backbone2.5e-5/其他5e-4，wd1e-4（norm/bn/bias为0）；500次warmup及12轮cosine，梯度裁剪0.1；BF16前向、FP32原生criterion、无GradScaler；原生MAL/BBox/GIoU/FGL/DDF权重1/5/2/.15/1.5。EMA固定规则。',
        '第4/8/12轮验证，最终统一取第12轮EMA；不是作者完整COCO训练配方的复现。未从容量权重续训，未超过任一arm12轮。',
        '启动前一次密集batch容量确认后冻结batch8：容量阶段12次更新、138次完整前向，权重丢弃后各组从公共初始化开始。',
        '实际Python3.9.7/Torch2.6.0+cu124/RTX3090 24GiB。三组arm elapsed合计约%.3f小时（含各自评估等，不等于完整租机计费时长）。' % (sum(r['elapsed_seconds'] for r in results.values()) / 3600),
        '',
        '3. 评价范围与独立核验',
        'VisDrone DET官方训练6471图、验证548图。有效类别1..10映射为0..9；ignored区域取矩形并集，GT/预测IoA≥0.5按冻结规则排除；others11不计分类目标。',
        '原图裁剪后GT面积定义small≤1024像素²；maxDet300，原生全局top300输出，不用NMS。38759个有效GT中157个按忽略规则排除，最终38602个。',
        '本地独立NumPy实现未导入训练评价器、torch或pycocotools；三组maxDet300的40400个精度单元和400个召回值逐值完全一致。标量均值仅有约1e-14pp浮点求和差。',
        '548份原始标注逐文件SHA核对，通过另一种矩形并集算法完整重建评价GT。已保留预测数B153873/S153351/L153540，均检查不落入应忽略区域。',
        '范围限制：检查的是保存的评价预测，没有重新生成网络原始输出或被忽略的预测；没有再次运行检测器。',
        '三组raw/EMA在CPU严格装载，参数有限，优化器state均9708；正常invalid-anchor的inf缓冲区按源码处理，不误报为训练发散。S/L读出末层36864个权重非零，四个参数的step均9708。',
        '共同初始化、冻结文件和S/L初始读出摘要一致；全部12轮采样顺序哈希相同，每轮前8个实际输入批次摘要相同。没有直接比对所有实际批次的全部像素，不扩大核验范围。',
        '全部终点权重、GT/预测、矩阵、日志、协议及实际源码已下载：每个arm归档115个成员SHA通过；终点汇总9个文件SHA通过。',
        '延时为同机batch1原生eager CUDA/BF16完整模型前向，10次预热后30次；不含评价后处理，不是机载或TensorRT实测。',
        '',
        '4. 统计解释与偏差检查',
        '11/11类型已逐项检查，完整项见result_verification_v1/validation_interpretation_11_types.json。',
        '没有p值、置信区间或统计显著性结论；预设阈值仅用于分配后续资源。单seed、12轮、同一公开验证集，且此前已探索多候选，不能当作独立确认性测试。',
        '逐类结果仅描述异质性：L-B小目标AP75为正8/10类；L-S为正3/10类，不挑有利类别替换主终点。详见per_class_descriptive_scope.json。',
    ]
    for i, item in enumerate(fallacies, 1):
        lines.append('%d. %s：%s' % (i, item['type'], item['finding']))
    lines += [
        '',
        '5. 执行结束、异常和计数',
        '三组训练及监督器正常结束，exit0；PID/start_ticks复核后均不在/proc。最终GPU计算进程为空，3090显存2MiB、利用率0%。没有新方向自动队列。',
        'SSH观察曾短断并自动重连；不把连接断开当训练失败，未重启训练。S的CPU核验曾丢失远程退出状态，但落盘PASS及SHA复读确认；本轮终点归档再次取得同字节CPU回执。L CPU核验自身exit0。',
        'A42正式合计29124次优化器更新、30477次完整模型前向；加启动容量后A42为29136次更新、30615次前向。每arm只有12轮。',
        '架构研究累计（原有3395次前向、528次更新基础上）：34010次完整前向、29664次含容量更新，其中正式更新29650、容量更新14。36轮为三个独立arm轮数之和，不是任一模型训练36轮。旧D0/DA计数范围不混入；本轮后处理新增检测前向/更新均0。',
        '',
        '6. 下一候选及停止边界',
        '已有备选A41-B具体化为A43-B：同一冻结小型教师下，比较实例query直接监督早期空间表征与普通区域特征蒸馏。',
        '原文已有实例条件、query掩码和跨层语义蒸馏，不能称首次给浅层补语义。拟案仅把“最终匹配query作为直接局部目标”与区域目标的差别作为待验证增量；TSC-DETR等近邻未充分核实，创新尚未确认。',
        '后续草案：fresh原版B、区域蒸馏M、直接query目标Q；同一教师和匹配实例，同一投影/损失，单卡各最多12轮。一次真实容量检查先于任何新训练，若不合格就停止；Q必须优于M才支持目标来源的价值。',
        '这是新的未实现假设，不是从本次L失败证明语义瓶颈成立，也不是降低本次AP75门槛继续跑。没有开训、没有新增教师推理或优化器更新。',
        '完整问题、最近方法逐项差别、11条检索记录、阅读范围、风险与最小验证草案：' + str(N / next_plan['report']),
        'A42之前的选题文献和方法依据保留于90_A41选题纠偏_两条候选与暂定主线.txt。',
        '',
        '7. 关键可核查文件',
        str(V / 'final_practical_gate_verification.json'),
        str(V / 'B_S_L_controls_verified.json'),
        str(D / 'completed_results'),
        str(D / 'terminal_summary/final_process_observation.json'),
        str(N / 'browser_search_and_reading_receipts.json'),
        '',
        '研究诚信：这是一次完成并报告阴性结论的小验证；没有找到已证实可支撑期刊的算法贡献。当前交付目标是完成三组判别并决定去留，已完成；下一候选仅准备方案，不自动实验。',
    ]
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    event = {
        'phase': 'A42', 'event': 'ALL_THREE_TERMINALS_VERIFIED_MAINLINE_RETIRED',
        'recorded_utc': now, 'goal_status': 'complete',
        'status': 'PILOT_COMPLETE_GATES_FAILED_LOCALIZATION_ONLY_RETIRED',
        'mainline_validated': False, 'pilot_pass': False,
        'verified_endpoint_arms': ['B', 'S', 'L'], 'active_arm': None,
        'completed_epochs_by_arm': {'B': 12, 'S': 12, 'L': 12},
        'endpoint_metrics': m, 'deltas_vs_B_pp': gate['deltas_vs_B_pp'],
        'L_minus_S_pp': gate['L_minus_S_pp'], 'practical_gates': gate['practical_gates'],
        'latency_median_ms': gate['latency_median_ms'],
        'formal_optimizer_updates_A42': 29124, 'formal_full_model_forwards_A42': 30477,
        'capacity_optimizer_updates_A42': 12, 'capacity_full_model_forwards_A42': 138,
        'cumulative_architecture_full_forwards': 34010,
        'cumulative_architecture_optimizer_updates_with_capacity': 29664,
        'counts_terminal': True, 'each_arm_max_epochs': 12,
        'new_detector_forwards_by_postprocessing': 0, 'new_optimizer_updates_by_postprocessing': 0,
        'new_direction_training_allocation': 0, 'automatic_training_followup': False,
        'all_job_identities_absent': True, 'gpu_compute_processes_empty': True,
        'training_protocol_changed': False, 'rescue_sweep_started': False,
        'A22_restarted': False, 'old_DEIM_restarted': False, 'compression_branch_reopened': False,
        'report': str(report), 'receipt': str(receipt_path),
        'next_candidate': str(N / 'candidate_plan.json'),
        'next_candidate_status': next_plan['status'],
        'goal_completion_scope': 'Finish frozen B/S/L pilot, independently verify, retire unsupported claim, and prepare a concrete next candidate. No validated journal contribution claimed.'
    }
    board_path = B / '01_候选判别表.json'
    state_path = B / '15_当前状态与结果校验.json'
    board, state = read(board_path), read(state_path)
    for candidate in board['candidates']:
        if candidate['id'] == 'A41-A':
            candidate.update(status='RETIRED_AFTER_FIXED12_PILOT_GATES_FAILED', pilot_recommended=False,
                             efficacy_validated=False, increment_approved=None,
                             pilot_gate_pass=False, evidence=str(report), next='No rescue sweep; retain all negative evidence.')
        if candidate['id'] == 'A41-B':
            candidate.update(status=next_plan['status'], refined_as='A43-B',
                             question=next_plan['research_question'], pilot_recommended=False,
                             efficacy_validated=False, increment_approved=None,
                             evidence=str(N / next_plan['report']), next='Proposal only; capacity/implementation and experiment freezing required before any future launch.')
    board.update(status='A42_COMPLETE_LOCALIZATION_MAINLINE_RETIRED_NEXT_RESERVE_PROPOSED',
                 latest_update_utc=now, goal_status='complete', mainline_validated=False,
                 mainline_selected_for_pilot=False, current_priority_candidate='A41-B',
                 current_priority_question=next_plan['research_question'], latest_report=str(report),
                 pilot_validation_status='A42_COMPLETED_FAILED_FIXED_PRACTICAL_GATES',
                 remaining_deliverable='None for the completed A42 goal; A43-B is a proposed future task, no training allocated.',
                 A42_terminal_result=event, A43_reserve_proposal=str(N / 'candidate_plan.json'))
    state.update(recorded_utc=now, goal_status='complete', mainline_validated=False,
                 mainline_selected_for_pilot=False, model_forwards=34010,
                 training_updates=29664, formal_detector_training_updates=29650,
                 capacity_only_optimizer_updates=14, detector_optimizer_updates_including_capacity=29664,
                 formal_training_epochs=36, training_counts_terminal=True, counts_terminal=True,
                 full_forward_cap_closed=True,
                 counters_observed_lower_bounds_while_job_live=False, current_native_job_terminal=True,
                 current_user_task_status='COMPLETED_PILOT_NEGATIVE_NEXT_PROPOSAL_PREPARED',
                 current_goal_turn=event, latest_authoritative_phase=event,
                 current_user_task_phase=event, latest_report=str(report),
                 current_priority_question=next_plan['research_question'],
                 A42_terminal_verification=event, A43_reserve_proposal=str(N / 'candidate_plan.json'),
                 training_update_scope='Terminal cumulative architecture counts; A42 formal29124 plus capacity12, historical formal526 plus capacity2.',
                 formal_training_epochs_scope='Sum across three independent A42 arms; each12. No arm continued to36.',
                 goal_status_scope='A42 explicit experimental goal complete; research contribution remains unvalidated and A43 has no execution allocation.')
    state.setdefault('authoritative_phase_history', []).append(event)
    write(board_path, board)
    write(state_path, state)
    evidence_paths = [report, board_path, state_path, D / 'B_S_terminal_state_historical_remap.json',
                      D / 'terminal_summary_download_verification.json',
                      V / 'final_practical_gate_verification.json', V / 'B_S_L_controls_verified.json',
                      V / 'validation_interpretation_11_types.json', V / 'per_class_descriptive_scope.json',
                      N / 'candidate_plan.json', N / next_plan['report'], N / 'browser_search_and_reading_receipts.json']
    for a in ['B', 'S', 'L']:
        evidence_paths += [D / ('completed_arm_' + a + '_download_verification.json')]
        evidence_paths += [V / (a + '_epoch12_' + suffix + '.json') for suffix in
                           ['independent_metrics', 'independent_GT_ignore', 'checkpoint_cpu_verification']]
    receipt = dict(event)
    receipt.update(requirements={
        'frozen_three_arm_execution': 'PASS', 'each_arm_12_epoch_cap': 'PASS',
        'source_and_checkpoint_download_integrity': 'PASS', 'independent_endpoint_metrics': 'PASS',
        'raw_GT_and_saved_prediction_ignore_rule': 'PASS_WITH_STATED_SCOPE',
        'paired_initialization_and_input_flow': 'PASS_WITH_STATED_SCOPE',
        'frozen_practical_gates_recomputed': 'PASS_CHECK_NEGATIVE_RESULT',
        'unsupported_claim_retired': 'PASS', 'concrete_next_candidate_prepared': 'PASS_PROPOSAL_ONLY',
        'new_direction_not_launched': 'PASS', 'all_training_processes_exited': 'PASS',
        'journal_contribution_proved': 'NOT_CLAIMED_NOT_THIS_GOAL'},
        previous_state_snapshots=remap, files_sha256={str(p): sha(p) for p in evidence_paths})
    write(receipt_path, receipt)
    write(D / 'goal_live_latest.json', event)
    write(D / 'goal_progress_08_three_arm_completed_negative_verified.json', receipt)
    with (D / 'goal_verified_wait_journal.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'recorded_utc': now, 'turn': 'r08', 'classification': 'progress_terminal_L_verified_and_final_decision',
                            'status': event['status'], 'goal_status': 'complete',
                            'terminal_monitor_receipt': 'A42_goal_r08_wait28'}, ensure_ascii=False) + '\n')
    print(json.dumps({'status': event['status'], 'report': str(report), 'receipt': str(receipt_path),
                      'candidate_count': len(board['candidates']), 'evidence_files_hashed': len(evidence_paths),
                      'goal_deliverable_ready': True, 'new_direction_training': 0}))


if __name__ == '__main__':
    main()
