"""Publish verified B8 observations once, preserving every prior receipt."""
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
        for x in iter(lambda: f.read(4 << 20), b''):
            h.update(x)
    return h.hexdigest()


def save(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def replace(p, value):
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, p)


def main():
    live_files = sorted(R54.glob('A57_B8_publish_live_*_rpc.json'))
    assert len(live_files) == 1
    rpc = load(live_files[0])
    assert rpc['exit'] == 0
    live = json.loads(rpc['stdout'])
    assert live['supervisor']['status'] == 'RUNNING' and live['supervisor']['active_arm'] == 'B'
    assert all(r['owned_live'] for r in live['owned_jobs']) and live['frozen_inputs_unchanged']
    assert {(r['pid'], r['start_ticks']) for r in live['owned_jobs']} == {(17829, 26463967), (17830, 26464019)}
    assert set(live['arms']) == {'B'} and 'supervisor_exit' not in live
    assert 'failure.json' not in live['arms']['B'] and 'exit.json' not in live['arms']['B']
    audit = load(P / 'B_epoch08_cached_verification_v1.json')
    assert audit['status'] == 'PASS_B8_SAVED_METRICS_GT_EVENTS_CHECKPOINT'
    assert audit['GT_semantically_identical'] and audit['all_floating_tensors_finite']
    metrics = audit['metrics']
    previous_metrics = load(P / 'B_epoch04_remote_v1/arm_B/epoch04_RAW_metrics.json')
    assert live['arms']['B']['metrics_by_epoch.json']['8'] == metrics
    assert set(live['arms']['B']['metrics_by_epoch.json']) == {'4', '8'}
    prior = load(P / 'A57_B4_preservation_receipt.json')
    mutable = [B / '01_候选判别表.json', B / '15_当前状态与结果校验.json', P / 'CURRENT_A57.json']
    before = P / 'before_B_epoch08_update'
    before.mkdir(exist_ok=False)
    mapping = {}
    for path in mutable:
        row = prior['files'][str(path)]
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
        dest = before / path.name
        shutil.copy2(path, dest)
        mapping[str(path)] = {'snapshot': str(dest), 'sha256': sha(dest)}
    save(P / 'prior_B4_resolution.json', {
         'reason': 'Canonical status advances to verified B8 observation; resolve previous B4 receipt through exact immutable snapshots.',
         'files': mapping})
    save(P / 'B_epoch08_latest_live_snapshot.json', live)
    polls = sorted(R54.glob('A57_wait06_poll_*_rpc.json'))
    observed = []
    for f in polls:
        r = load(f)
        assert r['exit'] == 0
        observed.append(json.loads(r['stdout']))
    assert observed and '8' in observed[-1]['arms']['B']['metrics']
    save(P / 'B8_readonly_wait_receipts.json', {
         'polls': len(polls), 'first_remote_utc': observed[0]['utc'], 'last_remote_utc': observed[-1]['utc'],
         'new_observation': 'B8', 'files': {str(f): sha(f) for f in polls},
         'additional_model_forwards': 0, 'additional_optimizer_updates': 0,
         'watcher_resume_note': 'Old watcher cell was absent on resume. Its 86 saved receipts contain B8; no training process was restarted.'})
    j = live['arms']['B']['journal']
    progress = live['arms']['B']['progress.json']
    totals = {'model_forwards': 226281 + 3 + j['entries'], 'cap': 285885,
              'formal_detector_training_updates': 194686 + j['updates'],
              'capacity_only_optimizer_updates': 76, 'training_updates': 194762 + j['updates'],
              'detector_optimizer_updates_including_capacity': 194762 + j['updates'],
              'formal_training_epochs': 240 + j['epochs_completed']}
    report = P / '09_B8原版观察与固定判定.txt'
    lines = [
        'A57第8轮原版观察：指标上升，但仍远低于预定学习门槛；当前没有候选增量证据',
        '',
        '数据/宿主/候选：VisDrone2019-DET；pinned MambaSOD作者源码兼容版；M=普通全上下文状态编码后采样，Q=query条件C读取同口径上下文状态。',
        '只有ImageNet Vim骨干预训练，没有检测器预训练。当前统一12轮配方不是作者300轮完整复现；不得与原文性能混报。',
        '',
        '已证实事实（raw，0–100指标尺度）',
        f'B4：AP {previous_metrics["AP"]:.12f}，APs {previous_metrics["APs"]:.12f}。',
        f'B8：AP {metrics["AP"]:.12f}，AP50 {metrics["AP50"]:.12f}，AP75 {metrics["AP75"]:.12f}。',
        f'B8：APs {metrics["APs"]:.12f}，APs75 {metrics["APs75"]:.12f}，ARs {metrics["ARs"]:.12f}。',
        f'数据计算：B8−B4，AP +{metrics["AP"]-previous_metrics["AP"]:.12f}个百分点，APs +{metrics["APs"]-previous_metrics["APs"]:.12f}个百分点。这是同一原版随训练变化，不是Q相对对照的算法增量。',
        'M/Q均未正式训练；没有M/Q的AP值或Q−M结论。不能因B较差而记录Q已训练失败。',
        '',
        '本轮核验与保全',
        '548张验证图，38602有效GT，157个有效GT被ignore过滤；保留156199个预测。缓存GT与冻结参考一致。',
        '下载18项小文件及B8 raw权重，字节数/SHA256逐项匹配；缓存precision/recall重算8项指标与原始结果相差小于1e−12；没有重新运行COCO评价器或检测模型。',
        'B8评价事件前缀完整包含B4前缀：13218次模型进入/完成=12944次训练+274次验证；6472次优化器尝试/返回更新；8轮完成。',
        '2011个状态张量浮点值均有限，33个BN计数均12944；所选骨干/分类/定位/query/参考框权重相对初始化已变化。',
        '位置编码160×160缓存与B4完全相同；冷加载办法沿用B4说明。本轮不新增位置编码构造或模型实例。',
        f'B8权重：{audit["checkpoint"]["bytes"]}字节；SHA256 {audit["checkpoint"]["sha256"]}。',
        '原始缓存、预测、GT、权重及事件前缀见B_epoch08_remote_v1；本轮核验见B_epoch08_cached_verification_v1.json。',
        '',
        '判断与未确定部分',
        '当前证据支持：原版在这套12轮短训配方下截至第8轮仍未学到足够检测能力。',
        '尚未证实：低AP的具体原因、旧内核与当前兼容代码的完整端到端等价、充分收敛后的原文性能，以及query条件状态读取的实际增量。',
        'B4提出的源码分类项相对尺度只是原因假设；目前不改系数、不重启、不追加轮数或种子补救。',
        '',
        '固定裁决',
        '第4/8轮只观察。B仍按原进程完成12轮；B12必须AP≥8且APs≥3，才继续M/Q。B12若不足，取消未启动M/Q，退出当前宿主/短训配方，Q记为未验证。',
        '若B12通过，Q12须同时满足：Q−B APs≥+0.5pp，Q−M APs≥+0.3pp，Q−B AP≥−0.2pp；这些是项目投入门槛，不是统计显著性或投稿保证。',
        f'现场快照：远端UTC {live["utc"]}；B第{progress["epoch"]}轮，完成{j["epochs_completed"]}轮、{j["updates"]}次更新，{j["entries"]}次模型进入/{j["completions"]}次完成。计数为运行中的下界。',
        '原监督器17829/start26463967和B worker17830/start26464019身份一致；冻结源码/初始化哈希未变。',
        '目前尚未找到经验证成立的论文主线，目标保持active。下一步只等B12固定判定，不以B8改写门槛。',
    ]
    with report.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    evidence = load(before / 'CURRENT_A57.json')
    evidence.update(status='A57_B8_VERY_LOW_AP_ORIGINAL_B_CONTINUES', report=str(report), observed_utc=live['utc'],
        formal_model_forwards=j['entries'], formal_model_forward_completions=j['completions'],
        formal_detector_optimizer_updates=j['updates'], formal_epochs=j['epochs_completed'],
        live_snapshot=str(P / 'B_epoch08_latest_live_snapshot.json'),
        new_A57_AP_metrics={'B_epoch4_RAW': previous_metrics, 'B_epoch8_RAW': metrics},
        B8_primary_gate_assessed=False, Q_formal_training_started=False, M_formal_training_started=False,
        B8_exact_validation_prefix={'model_entries': 13218, 'model_completions': 13218, 'optimizer_updates': 6472, 'epochs': 8},
        B8_cache_and_checkpoint_verification=str(P / 'B_epoch08_cached_verification_v1.json'),
        next='Observe original B12 fixed endpoint. Below learning gate => cancel unstarted M/Q; classify host/short-recipe inadequacy, Q untested. No coefficient/epoch/seed/module rescue.')
    replace(P / 'CURRENT_A57.json', evidence)
    now = datetime.now(timezone.utc).isoformat()
    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
        d = load(B / name)
        d.update(status=evidence['status'], A57_execution=evidence, latest_report=str(report), latest_update_utc=now,
                 goal_status='active', mainline_validated=False, candidate_selected_for_pilot=True,
                 mainline_selected_for_pilot=False,
                 pilot_validation_status='B8 very low AP; original B continues to fixed12 learning gate; M/Q formal training not started')
        if name.startswith('01_'):
            for row in d['candidates']:
                if row['id'] == 'A57_FULL_GRID_SPATIAL_STATE_READOUT':
                    row.update(status=evidence['status'], report=str(report), latest_live_snapshot=evidence['live_snapshot'],
                               B8_observation=metrics, Q_formal_training_started=False)
        else:
            previous_turn = d.get('current_goal_turn')
            d.update(totals)
            d.update(recorded_utc=now, current_phase='A57_B_TRAINING_AFTER_VERY_LOW_B8_AP', counts_terminal=False,
                     training_counts_terminal=False, current_native_job_terminal=False, full_forward_cap_closed=False,
                     counters_observed_lower_bounds_while_job_live=True, previous_goal_turn_classification=previous_turn,
                     current_goal_turn='progress: B8 AP and cached metrics/GT/events/checkpoint verified and preserved; original fixed12 continues, Q untested',
                     current_user_task_status='B8 AP0.2808707/APs0.1148618; still very low baseline learning, wait fixed12 learning gate; no Q efficacy claim')
            d.setdefault('authoritative_phase_history', []).append({
                'phase': 'A57', 'utc': now, 'classification': 'progress', 'goal_status': 'active',
                'mainline_validated': False, 'B8_metrics': metrics, 'observed_lower_bound_counts': totals,
                'report': str(report), 'source_bound_protocol_unchanged': True})
        replace(B / name, d)
    save(P / 'goal_progress_A57_B8_verified.json', {
        'classification': 'progress', 'goal_status': 'active', 'mainline_validated': False,
        'B8_metrics': metrics, 'B8_minus_B4_percentage_points': audit['B8_minus_B4_percentage_points'],
        'observed_cumulative_lower_bounds': totals, 'exact_B8_validation_prefix': evidence['B8_exact_validation_prefix'],
        'extra_model_forwards': 0, 'extra_optimizer_updates': 0, 'extra_COCO_evaluate_calls': 0,
        'report': str(report), 'primary_gate_or_recipe_changed': False})
    save(P / 'A57_turn_handoff_B8_v1.json', {
        'phase': 'A57', 'classification': 'progress: verified B8 observation', 'goal_status': 'active',
        'mainline_validated': False, 'latest_live_snapshot': evidence['live_snapshot'],
        'rpc': str(live_files[0]), 'rpc_sha256': sha(live_files[0]), 'latest_report': str(report),
        'B8_AP': metrics['AP'], 'B8_APs': metrics['APs'], 'M_Q_formal_not_started': True,
        'next': evidence['next'], 'only_new_metric_to_wait_for': 'B12 (ignore existing 4/8)',
        'bridge_session': 19671, 'credentials_recorded': False,
        'PID_identities': {'supervisor': [17829, 26463967], 'worker_B': [17830, 26464019]},
        'frozen_protocol_sha256': evidence['protocol_sha256'],
        'terminal_counts_if_only_B12_completes': {'model_forwards': 246151,
             'formal_detector_training_updates': 204394, 'capacity_only_optimizer_updates': 76,
             'training_updates': 204470, 'formal_training_epochs': 252},
        'liveness_note': 'Use supervisor plus active child; completed inactive B child may normally be dead after M starts.',
        'low_AP_cause': 'Unresolved; exact upstream loss scaling is a hypothesis, not an established bug or innovation.',
        'additional_model_forwards_by_this_stage': 0, 'additional_optimizer_updates_by_this_stage': 0})
    for name, row in prior['files'].items():
        path = Path(mapping[name]['snapshot']) if name in mapping else Path(name)
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], name
    save(P / 'prior_B4_receipt_reverified.json', {
        'status': 'PASS', 'files_verified': len(prior['files']), 'resolution': str(P / 'prior_B4_resolution.json')})
    files = [f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts] + mutable[:2]
    receipt = P / 'A57_B8_preservation_receipt.json'
    save(receipt, {'status': 'VERIFIED_B8_OBSERVATION_PRESERVED', 'goal_status': 'active',
         'mainline_validated': False, 'observed_lower_bound_counts': totals,
         'files': {str(f): {'bytes': f.stat().st_size, 'sha256': sha(f)} for f in sorted(set(files))},
         'additional_model_forwards': 0, 'additional_optimizer_updates': 0})
    print(json.dumps({'status': evidence['status'], 'B8_AP': metrics['AP'], 'B8_APs': metrics['APs'],
                      'counts': totals, 'preserved_files': len(files), 'receipt_sha256': sha(receipt),
                      'report': str(report), 'goal_status': 'active'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
