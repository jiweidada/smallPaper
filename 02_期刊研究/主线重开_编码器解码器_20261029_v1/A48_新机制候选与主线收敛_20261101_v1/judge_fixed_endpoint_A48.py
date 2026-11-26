"""Judge three completed, locally verified arms against the frozen allocation gates.

No training/model inference; no early-epoch selection, significance or novelty claim.
Run only after collecting and verifying all B/M/Q completed bundles.
"""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parent
EXPECTED_PROTOCOL = '2294d4e351e42e8760969a3ea06a3f6432591eeb70d17328b3cccec1cd5f4273'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    report = ROOT / 'fixed_endpoint_judgment_A48.json'
    note = ROOT / 'A48_三组固定终点结论.txt'
    assert not report.exists() and not note.exists(), 'Preserve existing judgment; do not overwrite.'
    assert sha(ROOT / 'protocol_FROZEN.json') == EXPECTED_PROTOCOL
    protocol = read(ROOT / 'protocol_FROZEN.json')
    arms, inputs, initializations = {}, {}, {}
    for arm in ['B', 'M', 'Q']:
        folder = ROOT / ('completed_' + arm + '_v1') / ('arm_' + arm)
        verified_path = ROOT / (arm + '_completed_local_verification.json')
        verified = read(verified_path)
        ready_path = ROOT / (arm + '_bundle_ready.json')
        ready = read(ready_path)
        manifest_path = folder.parent / 'bundle_manifest.json'
        manifest = read(manifest_path)
        assert verified['status'] == 'PASS' and verified['arm'] == arm
        assert verified['bundle_sha256'] == ready['sha256']
        # The archive verifier already checked/decompressed all checkpoint bytes.
        # Recheck every JSON used below against the verified payload manifest.
        used = ['result.json', 'exit.json', 'initialization.json',
                'epoch12_EMA_metrics.json', 'epoch12_EMA_GT.json']
        used += ['epoch%02d_input_flow.json' % e for e in range(1, 13)]
        for name in used:
            assert sha(folder / name) == manifest['files']['arm_' + arm + '/' + name]['sha256']
        result = read(folder / 'result.json')
        assert result['arm'] == arm and result['status'] == 'COMPLETED'
        assert result['epochs'] == 12 and result['optimizer_updates'] == 9708
        assert result['full_model_forwards'] == 10159
        assert read(folder / 'exit.json')['exit_code'] == 0
        assert result['protocol_sha256'] == EXPECTED_PROTOCOL
        assert result['endpoint_metrics'] == read(folder / 'epoch12_EMA_metrics.json') == verified['endpoint_metrics']
        initializations[arm] = read(folder / 'initialization.json')
        arms[arm] = result
        inputs[arm] = [read(folder / ('epoch%02d_input_flow.json' % e)) for e in range(1, 13)]
    assert initializations['B'] == initializations['M'] == initializations['Q']
    assert initializations['B']['all_initial_tensors_identical']
    assert initializations['B']['model_parameters'] == 9803112
    control_checks = {}
    stream_keys = ['order_sha256', 'first_eight_actual_input_sha256',
                   'first_eight_pre_forward_rng_sha256', 'all_batches_targets_sha256']
    for arm in ['M', 'Q']:
        rows = []
        for epoch, (base, candidate) in enumerate(zip(inputs['B'], inputs[arm]), 1):
            checks = {key: base[key] == candidate[key] for key in stream_keys}
            assert all(checks.values()), (arm, epoch, checks)
            assert base['optimizer_updates'] == candidate['optimizer_updates'] == 809 * epoch
            rows.append({'epoch': epoch, **checks})
        control_checks[arm + '_vs_B'] = rows
    gt_shas = {a: sha(ROOT / ('completed_' + a + '_v1') / ('arm_' + a) / 'epoch12_EMA_GT.json')
               for a in ['B', 'M', 'Q']}
    assert len(set(gt_shas.values())) == 1
    delta = {}
    for left, right in [('M', 'B'), ('Q', 'B'), ('Q', 'M')]:
        delta[left + '_minus_' + right] = {
            key: arms[left]['endpoint_metrics'][key] - arms[right]['endpoint_metrics'][key]
            for key in ['AP', 'APs', 'APs75']}
    thresholds = protocol['practical_gates_pp']
    gates = {
        'Q_minus_B_APs': {'actual_pp': delta['Q_minus_B']['APs'], 'minimum_pp': thresholds['Q_minus_B_APs_min']},
        'Q_minus_M_APs': {'actual_pp': delta['Q_minus_M']['APs'], 'minimum_pp': thresholds['Q_minus_M_APs_min']},
        'Q_minus_B_AP': {'actual_pp': delta['Q_minus_B']['AP'], 'minimum_pp': thresholds['Q_minus_B_AP_min']}}
    for row in gates.values():
        row['pass'] = row['actual_pp'] >= row['minimum_pp']
    passed = all(row['pass'] for row in gates.values())
    verdict = 'ENTER_NEXT_STAGE_AS_CANDIDATE' if passed else 'RETIRE_EXACT_RECIPE'
    output = {
        'created_local_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'VALID_FIXED_ENDPOINT_COMPARISON', 'protocol_sha256': EXPECTED_PROTOCOL,
        'primary_endpoint': 'epoch12 EMA; no best-epoch selection',
        'metrics_percent': {a: r['endpoint_metrics'] for a, r in arms.items()},
        'deltas_percentage_points': delta, 'frozen_gates': gates,
        'allocation_verdict': verdict, 'all_frozen_gates_pass': passed,
        'control_checks': control_checks, 'endpoint_GT_sha256': gt_shas,
        'resource_use': {'formal_model_forwards': sum(r['full_model_forwards'] for r in arms.values()),
                         'formal_optimizer_updates': sum(r['optimizer_updates'] for r in arms.values()),
                         'formal_epochs': sum(r['epochs'] for r in arms.values()),
                         'capacity_model_forwards': 12, 'capacity_optimizer_updates': 9},
        'goal_status': 'active', 'mainline_confirmed': False,
        'model_forwards_added_by_this_judgment': 0, 'optimizer_updates_added_by_this_judgment': 0,
        'limits': ['One seed and short 12-epoch pilot; allocation gates are not statistical significance.',
                   'COCO-style score with ignore filtering, not official VisDrone score.',
                   'Equal-group weighted normalization is prior art; gains alone do not establish novelty.',
                   'Archive/checkpoint verification is not independent reproduction of predictions.',
                   'A negative retires this exact recipe, not the whole research goal.']}
    lines = ['A48 三组固定第12轮EMA结果与资源分配裁决', '',
             '数据：VisDrone2019-DET；算法：ECDet-S；同初始化/seed/数据流/训练终点。',
             '评估：COCO-style，maxDets300，原图面积小目标及ignore过滤；不是官方VisDrone分数。', '',
             '组别\tAP(%)\tAPs(%)\tAPs75(%)']
    for a in ['B', 'M', 'Q']:
        m = arms[a]['endpoint_metrics']
        lines.append(f"{a}\t{m['AP']:.8f}\t{m['APs']:.8f}\t{m['APs75']:.8f}")
    lines += ['', '预先固定的三个门槛（百分点）：']
    for name, row in gates.items():
        lines.append(f"{name}: {row['actual_pp']:+.8f}，要求至少{row['minimum_pp']:+.2f}，{'通过' if row['pass'] else '未通过'}")
    lines += ['', '裁决：' + ('全部通过，仅建议作为候选进入下一阶段。' if passed else '至少一项未通过，停止本配方，不做lambda/位置/seed救援搜索。'),
              '主线尚未确认，研究目标保持active。小验证的收益不等于创新成立；失败也不等于目标完成。',
              '已有加权归一化与等组贡献先例，最多主张有区别的任务适配；仍须评价与最近工作的差异。',
              '本次只读取已核验结果，无新增模型前向或优化更新。',
              '完整控制核查、资源计数、原始数值见 fixed_endpoint_judgment_A48.json。']
    with report.open('x', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
    with note.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps({'status': output['status'], 'allocation_verdict': verdict,
                      'gates': gates, 'goal_status': 'active', 'report': str(report)}, ensure_ascii=False))
