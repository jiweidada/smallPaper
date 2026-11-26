"""Publish the original A58 fixed endpoint once; no model execution."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import shutil
from preservation_boundary_A58_v1 import P, B, sha, save_new, begin, finish

R54 = B / 'A54_场景尺度约束候选_20261101_v1'

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def replace(p, value):
    tmp = p.with_suffix(p.suffix + '.terminal.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, p)

def main(index):
    rpc = load(R54 / f'A58_formal_observe_{index:03d}.json')
    assert rpc['exit'] == 0
    obs = json.loads(rpc['stdout'])
    assert obs['source_binding_verified'] and obs['source_files'] == 9
    assert obs['protocol_sha256'] == sha(P / 'protocol_FROZEN_v1.json')
    sup = obs['supervisor']
    assert sup['status'] == 'COMPLETED_BOUNDED_SCREEN'
    assert sup['completed'] == ['B', 'M', 'Q'] and sup['remaining'] == [] and sup['child'] is None
    assert obs['supervisor_exit.json'] == {'exit': 0}
    assert not obs['supervisor_identity']['exists']
    judgment = obs['fixed12_judgment.json']
    assert judgment == sup['judgment']
    bundles, tensors = {}, {}
    for arm in ['B', 'M', 'Q']:
        bundles[arm] = load(P / f'fixed12_{arm}_bundle_local_verification_v1.json')
        tensors[arm] = load(P / f'fixed12_{arm}_checkpoint_local_verification_v1.json')
        assert bundles[arm]['status'] == 'PASS_FIXED12_BUNDLE_COUNTERS_INPUT_FLOW_AND_CPU_MATCHING'
        assert tensors[arm]['status'] == 'PASS_FIXED12_CPU_TENSOR_DATA_AUDIT'
        a = obs['arms'][arm]
        assert a['result.json']['endpoint_metrics'] == bundles[arm]['endpoint_metrics']
        assert a['journal_counts'] == {'forward_start': 10159, 'forward_complete': 10159, 'optimizer_returned': 9708}
        assert a['result.json']['status'] == 'COMPLETED' and len(a['completed_epochs']) == 12
        assert bundles[arm]['input_flow_epochs_verified'] == 12
        if arm != 'B':
            assert bundles[arm]['matched_B_flow']
    worker = load(R54 / 'A58_Q12_original_terminal_identity_v1.json')
    assert worker['exit'] == 0
    identity = json.loads(worker['stdout'])
    assert identity['Q_pid'] == 28541 and identity['supervisor_pid'] == 26447
    assert identity['Q_absent'] and identity['supervisor_absent'] and identity['remaining'] == []
    bm, mm, qm = [bundles[a]['endpoint_metrics'] for a in ['B', 'M', 'Q']]
    delta = {
        'Q_minus_B_APs': qm['APs'] - bm['APs'],
        'Q_minus_M_APs': qm['APs'] - mm['APs'],
        'Q_minus_B_AP': qm['AP'] - bm['AP'],
        'Q_over_M_median_latency': bundles['Q']['latency']['median_ms'] / bundles['M']['latency']['median_ms'],
    }
    gates = {'APs_vs_B': delta['Q_minus_B_APs'] >= .5,
             'APs_vs_M': delta['Q_minus_M_APs'] >= .3,
             'AP_preservation': delta['Q_minus_B_AP'] >= -.2,
             'latency': delta['Q_over_M_median_latency'] <= 1.2}
    assert delta == judgment['delta'] and gates == judgment['gates']
    passed = all(gates.values())
    assert passed == judgment['candidate_has_pilot_support']
    assert judgment['status'] == ('PASSES_MINIMAL_PILOT_REQUIRES_MECHANISM_AND_NOVELTY_EVIDENCE' if passed else 'RETIRED_THIS_FIXED_RECIPE_NO_RESCUE')
    assert not (P / 'A58_fixed12_terminal_preservation_receipt_v1.json').exists()
    begin('A58_fixed12_terminal_v1', 'A58_M12_Q_startup_preservation_receipt_v1.json')
    for i in range(52, index + 1):
        src = R54 / f'A58_formal_observe_{i:03d}.json'
        if not src.exists():
            continue
        target = P / 'formal_observations_v1' / f'{i:03d}.json'
        observation = json.loads(load(src)['stdout'])
        if target.exists():
            assert load(target) == observation
        else:
            save_new(target, observation)
    for old, new in [('A58_fixed12_Q_ready_v1.json', 'fixed12_Q_ready_v1.json'),
                     ('A58_fixed12_Q_small_v1.zip', 'fixed12_Q_small_v1.zip')]:
        target = P / new
        assert not target.exists()
        shutil.copyfile(R54 / 'A57_downloads' / old, target)
    receipts = P / 'transport_receipts_Q12_v1'
    receipts.mkdir(exist_ok=False)
    for src in sorted(R54.glob('A58_Q12_*.json')):
        shutil.copyfile(src, receipts / src.name)
    save_new(P / 'fixed12_judgment_verified_v1.json', dict(native=judgment, recomputed_delta=delta,
        recomputed_gates=gates, original_identities=identity, model_forwards_added=0))
    save_new(P / 'fixed12_saved_audit_usage_B_M_Q_terminal_v1.json', dict(
        completed_arms=['B', 'M', 'Q'], CPU_saved_matching_calls=3, checkpoint_loads=3,
        initialization_loads=3, remaining_each=0, model_forwards_added=0,
        scope_sha256=sha(P / 'fixed12_audit_scope_FROZEN_v1.json')))
    report = P / '16_三组固定12轮终点结论.txt'
    rows = '\n'.join(f"{a}：AP {bundles[a]['endpoint_metrics']['AP']:.9f}，APs {bundles[a]['endpoint_metrics']['APs']:.9f}，small AP75 {bundles[a]['endpoint_metrics']['APs75']:.9f}，模型中位前向 {bundles[a]['latency']['median_ms']:.6f}ms。" for a in ['B', 'M', 'Q'])
    decision = '候选通过预设短测门槛；仍需机制和最近方法差异证据，不能宣称论文主线已成立。' if passed else '候选未同时通过预设门槛，淘汰当前双侧范围路由配方；不补轮、换seed或修改块/topk/系数/kernel抢救。'
    report.write_text(f'''A58三组固定12轮EMA终点：{decision}

真实结果（百分制；差值为绝对百分点）
{rows}
Q−B：AP {delta['Q_minus_B_AP']:+.9f}pp，APs {delta['Q_minus_B_APs']:+.9f}pp。
Q−M：APs {delta['Q_minus_M_APs']:+.9f}pp；Q/M模型中位延迟 {delta['Q_over_M_median_latency']:.9f}。
固定门槛：Q−B APs≥0.5pp；Q−M APs≥0.3pp；Q−B AP≥−0.2pp；Q/M延迟≤1.20。
逐项判定：{json.dumps(gates, ensure_ascii=False)}。

三组可比性与核验
共同初始化、seed20261101、VisDrone6471/548、640stretch、同优化器和训练流程，各12轮/9708更新。12轮轮序、前8批实际输入/RNG及全轮targets流对齐；不是每批全部图像张量的独立哈希证明。
冻结9项源码未变。每组10159次整模型前向（训练9708/验证411/延迟40）、30026行事件；本阶段正式合计30477次前向、29124更新、36轮，另6容量前向/更新不作为训练结果。
三组最终预测各做一次CPU匹配，归档和状态张量按原冻结范围检查通过；核验没有新增检测器前向或优化器更新。原Q进程28541与监督器26447已退出，剩余队列为空。权重、源码、旧失败及所有裁决保留。
评估为冻结COCO-style10类/maxDets300/原图裁剪面积small，并非VisDrone官方挑战指标。时延是同3090、batch1 eager BF16模型前向，不含解码/后处理，不代表机载速度。

结论边界与下一步
{decision}
这是单seed短程筛选，不能作为所有稀疏区域注意力、encoder/decoder改进或无人机小目标方向无效的证据；没有统计显著性、充分收敛或新颖性证明。
原版保存输出表明：缩放后短边≤4像素的5361个小目标，仅4.03%有同类IoU≥0.75的已保存框。该逐GT上限覆盖允许预测复用，不能叫实际召回，也不能定位到encoder或decoder根因。
这项症状与既往A53几何诊断同类，不能包装为新发现或据此重开D007/D013/D024/D025、A42/A44/A47等旧操作。下一候选必须给出旧编号/裁决、本次实质差异、新依据及简单对照；未训练候选仍标未验证。
总目标active，当前没有可直接定稿的已验证新主线。没有追加训练队列或扩大本次预算。

证据：fixed12_judgment_verified_v1.json；fixed12_B/M/Q_bundle_local_verification_v1.json；各组checkpoint_local_verification；protocol_FROZEN_v1.json；报告14/15；原始结果与预测位于fixed12_B/M/Q_remote_v1。
''', encoding='utf-8')
    now = datetime.now(timezone.utc).isoformat()
    status = 'A58_FIXED12_PILOT_SUPPORTED_MAINLINE_UNVALIDATED' if passed else 'A58_RETIRED_EXACT_RECIPE_FIXED12'
    c = load(P / 'CURRENT_A58.json')
    c.update(status=status, observed_utc=now, report=str(report), mainline_validated=False, novelty_cleared=False,
        candidate_Q_completed=True, candidate_has_pilot_support=passed, training_running=False,
        new_training_queued=False, new_training_queue=[], formal_active_arm=None,
        formal_calls_observed=30477, formal_completed_calls_observed=30477, formal_updates_observed=29124,
        formal_completed_epochs_observed=36, new_detector_forwards=30483, new_optimizer_updates=29130,
        Q_fixed12_metrics=qm, Q_fixed12_bundle_verification=str(P / 'fixed12_Q_bundle_local_verification_v1.json'),
        Q_fixed12_checkpoint_verification=str(P / 'fixed12_Q_checkpoint_local_verification_v1.json'),
        fixed12_judgment=judgment, fixed12_CPU_saved_matching_calls=3, fixed12_checkpoint_loads=3,
        original_Q_terminal_verified=True, original_supervisor_terminal_verified=True,
        formal_supervisor_identity=obs['supervisor_identity'], formal_trainer_identity=identity,
        next='Continue mainline selection with a materially distinct mechanism; this pilot is closed, no rescue queue.',
        limits='One-seed fixed12 pilot only. No validated paper mainline, significance, convergence or novelty claim.')
    replace(P / 'CURRENT_A58.json', c)
    terminal = dict(status=status, recorded_utc=now, report=str(report), goal_status='active',
        mainline_validated=False, candidate_has_pilot_support=passed, metrics={a:bundles[a]['endpoint_metrics'] for a in bundles},
        deltas=delta, gates=gates, formal_model_forwards=30477, formal_optimizer_updates=29124,
        original_Q_and_supervisor_absent=True, remaining_training_queue=[])
    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
        x = load(B / name)
        x.update(status=status, goal_status='active', mainline_validated=False, latest_update_utc=now,
            latest_report=str(report), A58_execution=c, A58_terminal_result=terminal,
            current_priority_candidate=None, candidate_selected_for_pilot=False,
            pilot_validation_status=status)
        for row in x.get('candidates', []):
            if row.get('id') == 'A58_RANGE_QK_P3_CONTEXT':
                row.update(status=status, mainline_validated=False, candidate_has_pilot_support=passed,
                    terminal_report=str(report), terminal_delta=delta, terminal_gates=gates)
        if name == '15_当前状态与结果校验.json':
            x.update(phase='A58', model_forwards=276634, cap=276634, training_updates=233600,
                formal_detector_training_updates=233518, capacity_only_optimizer_updates=82,
                detector_optimizer_updates_including_capacity=233600, formal_training_epochs=288,
                discarded_capacity_optimizer_updates_observed=82, formal_optimizer_updates_observed=233518,
                counts_terminal=True, training_counts_terminal=True, current_native_job_terminal=True,
                counters_observed_lower_bounds_while_job_live=False, full_forward_cap_closed=True,
                current_phase=status, current_user_task_phase='MAINLINE_RESEARCH_AFTER_A58_FIXED12',
                current_goal_turn='progress: original A58 B/M/Q fixed12 complete and audited; mainline goal remains active',
                current_user_task_status='A58 pilot closed; continue materially distinct mainline selection; no queued training',
                cap_scope='Historical closed246151 + A58 capacity6 + formal30477 =276634, closed.',
                full_model_forwards_scope='All completed historical detector calls and A58 capacity/formal calls; excludes module fixtures and CPU saved-data audits.',
                formal_training_epochs_scope='Historical252 throughA57 plus A58 B/M/Q36 =288.',
                training_update_scope='Historical formal204394 + A58 formal29124 + all discarded capacity82 =233600.')
        x.setdefault('authoritative_phase_history', []).append(terminal)
        replace(B / name, x)
    save_new(P / 'goal_progress_A58_fixed12_terminal_v1.json', dict(goal_status='active',
        classification='Completed bounded three-arm validation; not completion of the broader mainline goal',
        candidate_has_pilot_support=passed, mainline_validated=False, report=str(report),
        new_audit_model_forwards=0, global_model_forwards=276634, global_optimizer_updates=233600))
    save_new(P / 'A58_turn_handoff_terminal_v1.json', dict(goal_status='active', current=c,
        bridge_session=19671, credentials_recorded=False, remote_root='/root/uav_a58_20261102_v1/pilot_v1',
        last_observer_index=index, all_original_jobs_terminal=True, queued_training=[],
        global_model_forwards=276634, global_optimizer_updates=233600,
        frozen_saved_audits_complete=True, next=c['next']))
    finish('A58_fixed12_terminal_preservation_receipt_v1.json', status,
        formal_model_forwards=30477, formal_updates=29124, candidate_has_pilot_support=passed,
        saved_CPU_matching_calls=3, checkpoint_loads=3, new_audit_detector_forwards=0)
    print(json.dumps(terminal, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observer-index', type=int, required=True)
    main(parser.parse_args().observer_index)
