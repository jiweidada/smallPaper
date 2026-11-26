"""Publish terminal B12 failure of the learning gate, keeping Q explicitly untested."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil

P = Path(__file__).resolve().parent
B = P.parent
R54 = B / 'A54_场景尺度约束候选_20261101_v1'


def load(p):
    return json.loads(p.read_text('utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda: f.read(4 << 20), b''):
            h.update(v)
    return h.hexdigest()


def save(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def replace(p, obj):
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, p)


def main():
    rpc_path = R54 / 'A57_B12_publish_final_rpc.json'
    rpc = load(rpc_path)
    assert rpc['exit'] == 0
    live = json.loads(rpc['stdout'])
    assert live['supervisor']['status'] == 'COMPLETED_BOUNDED_SCREEN'
    assert live['supervisor_exit']['exit_code'] == 0
    assert live['supervisor_exit']['cancelled'] == ['M', 'Q']
    assert live['supervisor_exit']['queue'] == [] and live['supervisor_exit']['active_arm'] is None
    assert all(r['current_identity'] is None and not r['owned_live'] for r in live['owned_jobs'])
    assert live['GPU_readonly'] == '' and live['frozen_inputs_unchanged']
    assert set(live['arms']) == {'B'}
    audit = load(P / 'B12_terminal_local_verification_v1.json')
    assert audit['status'] == 'PASS_TERMINAL_B12_BYTES_CACHED_AP_GT_EVENTS_WEIGHTS_OPTIMIZER'
    assert audit['additional_model_forwards'] == audit['additional_optimizer_updates'] == 0
    metrics = audit['metrics']
    assert metrics == live['arms']['B']['metrics_by_epoch.json']['12']
    assert live['arms']['B']['journal'] == {'entries': 19867, 'completions': 19867, 'updates': 9708, 'epochs_completed': 12}
    prior = load(P / 'A57_B8_preservation_receipt.json')
    mutable = [B / '01_候选判别表.json', B / '15_当前状态与结果校验.json', P / 'CURRENT_A57.json']
    before = P / 'before_B12_terminal_update'
    before.mkdir(exist_ok=False)
    mapping = {}
    for f in mutable:
        row = prior['files'][str(f)]
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256']
        target = before / f.name
        shutil.copy2(f, target)
        mapping[str(f)] = {'snapshot': str(target), 'sha256': sha(target)}
    save(P / 'prior_B8_resolution.json', {
         'reason': 'Advance canonical status from B8 observation to terminal B12; previous receipt resolves via exact immutable snapshots.',
         'files': mapping})
    previous_counters = load(before / '15_当前状态与结果校验.json')
    old_a57 = previous_counters['A57_execution']
    base_f = previous_counters['model_forwards'] - old_a57['formal_model_forwards'] - 3
    base_u = previous_counters['formal_detector_training_updates'] - old_a57['formal_detector_optimizer_updates']
    base_all_u = previous_counters['training_updates'] - old_a57['formal_detector_optimizer_updates']
    base_epochs = previous_counters['formal_training_epochs'] - old_a57['formal_epochs']
    assert (base_f, base_u, base_all_u, base_epochs) == (226281, 194686, 194762, 240)
    totals = {'model_forwards': base_f + 3 + 19867, 'cap': base_f + 3 + 19867,
              'formal_detector_training_updates': base_u + 9708,
              'capacity_only_optimizer_updates': 76, 'training_updates': base_all_u + 9708,
              'detector_optimizer_updates_including_capacity': base_all_u + 9708,
              'formal_training_epochs': base_epochs + 12}
    assert totals['model_forwards'] == 246151 and totals['training_updates'] == 204470
    save(P / 'B12_final_readonly_snapshot_v1.json', live)
    polls = sorted(R54.glob('A57_wait08_B12_poll_*_rpc.json'))
    assert len(polls) == 102
    states = []
    for f in polls:
        r = load(f)
        assert r['exit'] == 0
        states.append(json.loads(r['stdout']))
    assert '12' in states[-1]['arms']['B']['metrics']
    save(P / 'B12_bounded_wait_receipts_v1.json', {
        'polls': len(polls), 'first_remote_utc': states[0]['utc'], 'last_remote_utc': states[-1]['utc'],
        'first_new_metric': 'B12', 'files': {str(f): sha(f) for f in polls},
        'observer_cell': 'g5:1292 completed normally on B12',
        'additional_model_forwards': 0, 'additional_optimizer_updates': 0})
    latency = load(P / 'B12_terminal_remote_v1/arm_B/fixed12_latency.json')
    all_metrics = load(P / 'B12_terminal_remote_v1/arm_B/metrics_by_epoch.json')
    design = load(P / 'pilot_design_FROZEN_v1.json')
    gates = {'B_AP_ge_8': metrics['AP'] >= 8, 'B_APs_ge_3': metrics['APs'] >= 3}
    assert gates == audit['gates'] and not any(gates.values())
    status = 'A57_BASELINE_GATE_FAILED_Q_UNTESTED_RECIPE_STOPPED'
    report = P / '11_B12终点裁决与主线边界.txt'
    terminal = {
        'phase': 'A57', 'status': status, 'goal_status': 'active', 'mainline_validated': False,
        'dataset': 'VisDrone2019-DET train6471/val548',
        'host': 'Pinned author-source MambaSOD compatibility host, common12epoch recipe, ImageNet Vim backbone only; not300epoch paper reproduction',
        'technical_training_success': True, 'verified_endpoint_arms': ['B'],
        'B_fixed12_RAW': metrics, 'B_observations_by_epoch': all_metrics,
        'baseline_learning_gate': design['baseline_learning_gate_at_fixed12'],
        'baseline_gates_passed': gates, 'baseline_learning_gate_passed': False,
        'M_formal_training_started': False, 'Q_formal_training_started': False,
        'candidate_performance_gates_assessed': False, 'candidate_performance_verdict': 'UNTESTED',
        'cancelled_unstarted_arms': ['M', 'Q'], 'owned_jobs_live': False,
        'new_training_queued': False, 'post_training_queue': None, 'no_automatic_retry': True,
        'formal_B_model_forwards': 19867, 'formal_B_optimizer_updates': 9708, 'formal_B_epochs': 12,
        'counts_terminal': True, 'cumulative_counts': totals,
        'reserved_cap_before_gate': 285885, 'unused_cancelled_model_calls': 39734,
        'raw_checkpoint': audit['raw_checkpoint'], 'optimizer_checkpoint': audit['optimizer_checkpoint'],
        'local_verification': str(P / 'B12_terminal_local_verification_v1.json'),
        'final_snapshot': str(P / 'B12_final_readonly_snapshot_v1.json'),
        'report': str(report), 'low_AP_cause_established': False,
        'additional_model_forwards_by_verification': 0, 'additional_optimizer_updates_by_verification': 0,
        'additional_COCO_evaluate_calls': 0}
    save(P / 'A57_B12_terminal_judgment_v1.json', terminal)
    lines = [
        'A57固定B12终点：基线学习门槛未过，M/Q取消；主线尚未成立',
        '',
        '裁决先行',
        'B12 raw AP=0.424583237636，APs=0.162309121731（0–100尺度），均未达到预先固定的AP≥8、APs≥3。',
        '原监督器已取消未启动M/Q并正常退出。终止当前宿主/12轮配方，不自动补轮、换系数、换种子、换宿主移植同一操作或重启。',
        '这不是Q训练阴性：Q与普通全状态读取对照M均未正式训练，没有候选AP、Q−B或Q−M差值。不能据此说query条件状态读取这一理论假设无效。',
        '',
        '问题、算法和阅读边界',
        '候选问题：在匹配完整上下文状态构造和空间参考的情况下，query条件C读出是否优于普通上下文状态编码后采样。',
        '数据为VisDrone2019-DET；宿主为pinned MambaSOD作者源码兼容版；只有ImageNet Vim骨干预训练，没有检测器预训练。本次12轮不是作者300轮论文复现。',
        '最近方法与历史差异见01_候选问题与最近方法对照.txt、05_MambaSOD同名消歧与全文补查.txt。MambaSOD论文全文未读；相关全文选择性章节与源码阅读范围维持原记录，不能将源码核查冒充完整论文复现。',
        '',
        '已证实的原版观察',
    ]
    for epoch in ['4', '8', '12']:
        m = all_metrics[epoch]
        lines.append(f'B{epoch}：AP {m["AP"]:.12f}，AP50 {m["AP50"]:.12f}，APs {m["APs"]:.12f}，APs75 {m["APs75"]:.12f}。')
    lines += [
        f'数据计算：B12−B8 AP +{metrics["AP"]-all_metrics["8"]["AP"]:.12f}pp，APs +{metrics["APs"]-all_metrics["8"]["APs"]:.12f}pp；是同一B随训练变化，不是算法增量。',
        '548张验证图；38602有效GT；157个有效GT被ignore过滤；B12保留155183个预测。采用固定COCO-style、top300、原图裁剪GT面积的小目标口径，不是VisDrone官方排行榜成绩。',
        '',
        '过程与文件核验',
        'B的12轮训练技术上正常完成：19416次训练前向、411次验证前向、40次固定延迟前向，共19867次模型进入/完成；9708次优化器尝试及返回更新；59162条完整事件。B8前缀完整保留，12个epoch完成事件顺序一致。',
        '终点raw模型2011个状态张量的浮点值有限；33个BN计数均19416。优化器1898个活跃参数状态的步数均9708，浮点状态有限；选定骨干/分类/定位/query/参考框权重已相对初始化改变。',
        '位置编码缓存仍为160×160×256，与B8相同。冷加载需先按B4说明扩容该确定性缓存，再strict读入raw权重；不需要检测前向，也不据此恢复训练。',
        '已下载56项小文件、B12 raw和优化器；远端/本地字节数及SHA256匹配。GT与冻结参考一致，缓存precision/recall重算8项指标与原结果差小于1e−12，没有重跑检测模型或COCO评价器。',
        f'raw权重：{audit["raw_checkpoint"]["bytes"]}字节，SHA256 {audit["raw_checkpoint"]["sha256"]}。',
        f'优化器：{audit["optimizer_checkpoint"]["bytes"]}字节，SHA256 {audit["optimizer_checkpoint"]["sha256"]}。',
        f'正式B延迟中位数 {latency["median_ms"]:.6f}ms，batch1/BF16/model-only；不能与其他宿主、容量检查或整条系统吞吐混比。',
        f'监督器单调时钟总耗时 {live["supervisor_exit"]["total_elapsed_seconds"]/3600:.6f}小时；终点UTC {live["supervisor_exit"]["utc"]}。',
        '原监督器17829/start26463967、B worker17830/start26464019现场均已不存在；GPU计算进程为空，原监督队列空，后续队列为空。冻结源码、公共初始化及外部评价依赖哈希未变。',
        '',
        '仍未确定的原因',
        '当前结果只能确认该宿主/短训配方不足以承担本次算法增量对照。低AP的具体原因未证实，不能由权重有变化、loss有限和缓存指标一致反推所有兼容问题都已排除。',
        '源码继承的分类项相对尺度是已记录的优化假设，没有分项梯度/反事实性能证据；不称确定bug、唯一原因或论文创新。旧内核与新宿主的完整端到端等价、充分收敛和作者原文性能均未验证。',
        '',
        '主线与下一步',
        '目前没有经验证成立的论文主线，寻找主线的总目标保持active；本阶段结束不是总目标完成。A57问题保留“未验证”，当前配方退出。',
        '等待期的两类宽泛提法已有限核查，见10_B12等待期两类候选边界核查.txt：泛化的query竞争路由已有近作，简单方差读出可表示成已知非线性矩池化；两者未作为新优先主线，也没有训练阴性结论。',
        '下一候选应给出实质不同的具体操作、与旧编号及最近工作的区别、一个强简单对照；新方案需使用在预算内已可学习的起点，避免再次让冷启动基线遮蔽算法判断。当前没有为任何新方向排队训练。',
        f'累计真实模型前向 {totals["model_forwards"]}，正式训练更新 {totals["formal_detector_training_updates"]}，容量更新76，含容量总更新 {totals["training_updates"]}，正式累计epoch {totals["formal_training_epochs"]}。取消M/Q释放39734次预留模型调用；旧预留上限285885保留历史，本阶段按真实246151闭合，不缩减寻找主线的总目标。',
        '',
    ]
    with report.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    evidence = load(before / 'CURRENT_A57.json')
    evidence.update(status=status, goal_status='active', mainline_validated=False, candidate_selected_for_pilot=False,
        formal_queue={'active': None, 'conditional_pending': [], 'cancelled_unstarted': ['M', 'Q'],
                      'condition': 'B fixed12 AP>=8 and APs>=3 (not passed)', 'post_training_queue': None},
        owned_jobs_live=False, counts_terminal=True, formal_model_forwards=19867,
        formal_model_forward_completions=19867, formal_detector_optimizer_updates=9708, formal_epochs=12,
        observed_utc=live['utc'], counters_observed_lower_bounds_while_job_live=False,
        live_snapshot=terminal['final_snapshot'], report=str(report),
        new_A57_AP_metrics={f'B_epoch{e}_RAW': m for e, m in all_metrics.items()},
        B12_learning_gate_passed=False, candidate_performance_gates_assessed=False,
        Q_formal_training_started=False, M_formal_training_started=False,
        terminal_judgment=str(P / 'A57_B12_terminal_judgment_v1.json'),
        B12_verification=terminal['local_verification'], cancelled_unstarted_arms=['M', 'Q'],
        current_recipe_stopped=True, new_training_queued=False,
        next='Select a substantively different mechanism with a viable learning baseline; no automatic A57 coefficient/epoch/seed/host rescue. Goal remains active, Q untested.')
    replace(P / 'CURRENT_A57.json', evidence)
    now = datetime.now(timezone.utc).isoformat()
    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
        d = load(B / name)
        d.update(status=status, A57_execution=evidence, A57_terminal_result=terminal, goal_status='active',
                 mainline_validated=False, candidate_selected_for_pilot=False, mainline_selected_for_pilot=False,
                 current_priority_candidate=None, latest_report=str(report), latest_update_utc=now,
                 pilot_validation_status='A57 B12 baseline gate failed; M/Q cancelled untrained, current recipe stopped; total mainline goal remains active')
        if name.startswith('01_'):
            for row in d['candidates']:
                if row['id'] == 'A57_FULL_GRID_SPATIAL_STATE_READOUT':
                    row.update(status=status, report=str(report), B12_observation=metrics,
                               baseline_learning_gate_passed=False, candidate_training_status='UNTESTED',
                               Q_formal_training_started=False, M_formal_training_started=False,
                               next=evidence['next'])
            d.update(current_priority_question='No validated mainline; A57 exact host/12epoch recipe stopped after baseline learning gate failure.',
                     next_exploratory_question=evidence['next'])
        else:
            previous_turn = d.get('current_goal_turn')
            d.update(totals)
            d.update(recorded_utc=now, current_phase='A57_TERMINAL_BASELINE_INADEQUATE_Q_UNTESTED',
                     counts_terminal=True, training_counts_terminal=True, current_native_job_terminal=True,
                     full_forward_cap_closed=True, counters_observed_lower_bounds_while_job_live=False,
                     A57_reserved_cap_before_gate=285885, A57_cancelled_unused_forward_budget=39734,
                     previous_goal_turn_classification=previous_turn,
                     current_goal_turn='progress: B12 terminal verified; original processes absent; M/Q cancelled; weights optimizer cached metrics preserved; W1/W2 source screening completed; Q untested, goal active',
                     current_user_task_status='No validated paper mainline. A57 baseline AP0.4245832/APs0.1623091 failed gate; current recipe exited; continue different mechanisms without rescue queue.')
            d.setdefault('authoritative_phase_history', []).append({
                'phase': 'A57', 'utc': now, 'classification': 'progress', 'goal_status': 'active',
                'terminal_judgment': str(P / 'A57_B12_terminal_judgment_v1.json'),
                'B12_metrics': metrics, 'candidate_trained': False, 'cumulative_counts': totals,
                'mainline_validated': False, 'report': str(report), 'source_bound_protocol_unchanged': True})
        replace(B / name, d)
    save(P / 'goal_progress_A57_B12_terminal_verified_v1.json', {
         'classification': 'progress', 'previous_goal_turn_classification': 'progress: B8 verified and canonical status advanced',
         'goal_status': 'active', 'mainline_validated': False, 'baseline_learning_gate_passed': False,
         'candidate_trained': False, 'cancelled_unstarted': ['M', 'Q'], 'terminal_counts': totals,
         'actual_evidence': [terminal['local_verification'], terminal['final_snapshot'], str(report)],
         'recipe_changed_or_retried': False, 'additional_model_forwards': 0, 'additional_optimizer_updates': 0})
    save(P / 'A57_turn_handoff_B12_terminal_v1.json', {
         'phase': 'A57', 'status': status, 'goal_status': 'active', 'mainline_validated': False,
         'previous_goal_turn_classification': 'progress: B8 verified', 'current_goal_turn_classification': 'progress: fixed B12 terminal, gate, files and counters verified; two source-only candidates screened',
         'latest_report': str(report), 'terminal_judgment': str(P / 'A57_B12_terminal_judgment_v1.json'),
         'local_artifacts': str(P / 'B12_terminal_remote_v1'), 'B12_AP': metrics['AP'], 'B12_APs': metrics['APs'],
         'formal_B_model_calls': 19867, 'formal_B_updates': 9708, 'formal_B_epochs': 12,
         'cumulative_counts': totals, 'M_Q': 'UNSTARTED_CANCELLED_NOT_TRAINED_NEGATIVES',
         'owned_jobs': 'supervisor17829/start26463967 and worker17830/start26464019 absent; GPU compute empty',
         'watcher': 'g5:1292 completed on new B12 after102 polls; do not wait or relaunch it',
         'bridge_session': 19671, 'credentials_recorded': False, 'all_training_queues': [],
         'preservation_resolution': str(P / 'prior_B8_resolution.json'),
         'low_AP_cause': 'Unresolved. Source-native classification scaling hypothesis is not established causality. No end-to-end original-kernel equivalence proof.',
         'no_mid_run_or_post_failure_rescue': True, 'new_directions_training_allocated': 0,
         'next': evidence['next'], 'source_only_supplement': str(P / '10_B12等待期两类候选边界核查.txt')})
    for name, row in prior['files'].items():
        f = Path(mapping[name]['snapshot']) if name in mapping else Path(name)
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256'], name
    save(P / 'prior_B8_receipt_reverified.json', {
         'status': 'PASS', 'files_verified': len(prior['files']), 'resolution': str(P / 'prior_B8_resolution.json')})
    supplement = load(P / 'A57_B12_wait_supplement_preservation_receipt.json')
    for name, row in supplement['files'].items():
        f = Path(name)
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256'], name
    files = [f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts] + mutable[:2]
    receipt = P / 'A57_B12_terminal_preservation_receipt_v1.json'
    save(receipt, {'status': 'TERMINAL_B12_BASELINE_GATE_FAILURE_PRESERVED_Q_UNTESTED',
         'goal_status': 'active', 'mainline_validated': False, 'terminal_counts': totals,
         'files': {str(f): {'bytes': f.stat().st_size, 'sha256': sha(f)} for f in sorted(set(files))},
         'additional_model_forwards': 0, 'additional_optimizer_updates': 0})
    print(json.dumps({'status': status, 'B12_AP': metrics['AP'], 'B12_APs': metrics['APs'],
                      'counts': totals, 'preserved_files': len(files), 'receipt_sha256': sha(receipt),
                      'report': str(report), 'goal_status': 'active'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
