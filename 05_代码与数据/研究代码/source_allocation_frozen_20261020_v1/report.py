"""Report all predeclared cross-batch comparisons, without changing the gate."""
import csv, datetime, json, math
from pathlib import Path
import diagnostic as d


def weighted(rows, arm, mode, key, normalizer='denominator'):
    numerator, denominator = 0., 0.
    for r in rows:
        value = r['arms'][arm][mode]['metrics']
        weight = value[normalizer]
        numerator += value[key]*weight
        denominator += weight
    return numerator/denominator if denominator else None


def main():
    source_path = d.OUT/'03_全部跨批原始结果.json'
    result = json.loads(source_path.read_text(encoding='utf8'))
    protocol = json.loads((d.OUT/'01_冻结协议与资产.json').read_text(encoding='utf8'))
    assert result['state'] == 'complete' and len(result['rows']) == 16
    summary, capture_rows, csv_rows = [], [], []
    for seed in [1, 2]:
        rows = [r for r in result['rows'] if r['seed'] == seed]
        assert len(rows) == 8 and sorted((r['generator_batch'], r['evaluation_batch']) for r in rows) == [(i, i^1) for i in range(8)]
        captures = [json.loads((d.OUT/f'seed{seed}/batch{b:02d}_capture.json').read_text(encoding='utf8')) for b in range(8)]
        floor = 2*max(c['checks']['zero_functional_loss_abs_error'] for c in captures)
        population = [p for c in captures for p in c['population']]
        seed_summary = {'seed': seed, 'cross_batch_directions': 8, 'numerical_floor': floor,
            'union_pairs': sum(p['union_pairs'] for p in population),
            'transfer_pairs': sum(p['transfer_pairs'] for p in population),
            'GT_appearances': sum(p['GT'] for p in population), 'images_reused_across_seeds': len(population),
            'max_zero_step_box_abs_error': max(c['checks']['zero_functional_final_box_max_abs_error'] for c in captures),
            'max_pre_pair_native_loss_abs_error': max(c['checks']['per_pair_pre_native_loss_abs_error'] for c in captures),
            'modes': {}}
        for mode in ['fixed', 'native']:
            pooled = {arm: weighted(rows, arm, mode, 'weighted_geometry') for arm in d.ARMS}
            contrasts = {'source_minus_original': pooled['source']-pooled['original'],
                         'source_minus_fixed_proportion': pooled['source']-pooled['fixed_proportion'],
                         'fixed_proportion_minus_original': pooled['fixed_proportion']-pooled['original']}
            before_numerator = sum(r['before']['weighted_geometry']*r['before']['denominator'] for r in rows)
            before_denominator = sum(r['before']['denominator'] for r in rows)
            entry = {'pooled_before_checkpoint_geometry': before_numerator/before_denominator,
                'pooled_weighted_geometry': pooled, 'pooled_contrasts': contrasts,
                'pooled_mean_IoU': {a: weighted(rows, a, mode, 'mean_IoU') for a in d.ARMS},
                'pooled_L1': {a: weighted(rows, a, mode, 'L1_original_normalization') for a in d.ARMS},
                'pooled_GIoU_loss': {a: weighted(rows, a, mode, 'GIoU_loss_original_normalization') for a in d.ARMS},
                'per_batch_contrast_signs': {},
                'mean_native_topk_ordered_slot_change_fraction': {
                    a: sum(r['arms'][a][mode]['native_topk_ordered_slots_changed_vs_base'] for r in rows)/
                       sum(r['arms'][a][mode]['native_topk_total_ordered_slots'] for r in rows) for a in d.ARMS}}
            for contrast in contrasts:
                values = [r['contrasts'][mode][contrast] for r in rows]
                entry['per_batch_contrast_signs'][contrast] = {
                    'negative_better': sum(v < -floor for v in values),
                    'positive_worse': sum(v > floor for v in values),
                    'within_numerical_floor': sum(abs(v) <= floor for v in values),
                    'unweighted_batch_mean': sum(values)/8, 'min': min(values), 'max': max(values)}
            seed_summary['modes'][mode] = entry
        seed_summary['primary_passes_screen'] = all(
            seed_summary['modes']['fixed']['pooled_contrasts'][k] < -floor
            for k in ['source_minus_original', 'source_minus_fixed_proportion'])
        seed_summary['native_sensitivity_not_contradictory'] = all(
            seed_summary['modes']['native']['pooled_contrasts'][k] <= floor
            for k in ['source_minus_original', 'source_minus_fixed_proportion'])
        summary.append(seed_summary)
        capture_rows.extend(captures)
        for r in rows:
            for mode in ['fixed', 'native']:
                for a in d.ARMS:
                    m = r['arms'][a][mode]['metrics']
                    csv_rows.append({'seed': seed, 'generator_batch': r['generator_batch'],
                        'evaluation_batch': r['evaluation_batch'], 'query_selection': mode, 'arm': a,
                        **m, 'native_topk_slots_changed': r['arms'][a][mode]['native_topk_ordered_slots_changed_vs_base'],
                        'source_minus_original': r['contrasts'][mode]['source_minus_original'],
                        'source_minus_fixed_proportion': r['contrasts'][mode]['source_minus_fixed_proportion']})
    retain = all(s['primary_passes_screen'] and s['native_sensitivity_not_contradictory'] for s in summary)
    disposition = ('保留为后续短测候选；本诊断不启动训练，也不证明检测收益。' if retain else
                   '本次不给来源分配新增训练依据：未同时通过两个种子的原方案与固定比例比较及原生query敏感性检查。')
    complete = {'created_utc': d.now(), 'protocol_sha256': d.native.sha(d.OUT/'01_冻结协议与资产.json'),
                'results_sha256': d.native.sha(source_path), 'seed_summaries': summary,
                'retain_as_later_short_test_candidate': retain, 'disposition': disposition,
                'unit': 'native weighted geometric loss, not AP, not percentage points',
                'no_new_training': True}
    d.native.save_json(d.OUT/'04_配对汇总与预定筛查.json', complete)
    with (d.OUT/'05_全部三组端点明细.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    local_time = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=-7)))
    lines = ['Material Passport', 'Origin Skill: academic-research-suite / experiment-agent',
             'Origin Mode: run, local frozen full-parameter diagnostic',
             'Verification Status: native formula/gradient synthetic checks and local numerical/source checks passed; not independently replicated',
             'Version Label: source_allocation_frozen_v1', '',
             '来源分配冻结诊断结果', 'America/Los_Angeles: '+local_time.isoformat(), 'UTC: '+d.now(), '',
             '预定筛查处置：'+disposition, '',
             '两个 VisDrone BASE24 在线原模型，last_epoch=23；各8个旧development批次。两个seed复用相同32张增强输入，每seed6629次GT出现。不是新留出验证，也不是64张独立图。',
             '三组：原方案；来源集合等分；同删除集、逐配对转移量、每图各层接收总额的固定比例后移。',
             '488个原可训练张量、10,231,995个参数均进入导数/历史Adam绑定域，含backbone、encoder、共享decoder、各预测头。grad=None者遵照原Adam跳过；未冻结本来可训练的上游以节省计算。',
             '所有组保留同一次基点前向的原生Hungarian、聚合union、num_boxes_go、分类MAL、DN、FGL、DDF和其他损失。只改普通pre的L1/GIoU层间系数。',
             '每个批次每个方案从相同原权重、学习率、Adam历史矩和步数独立产生一次克隆虚拟端点，不累计、不调用optimizer.step、不写回模型。按原全模型梯度做0.1裁剪。',
             '主指标：另一批次final普通查询的5L1+2GIoU，使用其基点union/原分母；原模型FP32预测，指标用同一公式FP64汇总。负差值较好。不能换算为AP或百分点。',
             '主比较固定基点encoder top-k空间query索引，特征及anchor值仍重新计算；另对每个端点执行不修改top-k的原生前向作敏感性检查。两种结果分开报告。', '',
             '一、各seed总分子/总分母的主比较（固定query来源）',
             'seed | 原方案 | 来源分配 | 固定比例 | 来源−原方案 | 来源−固定比例']
    for s in summary:
        p = s['modes']['fixed']['pooled_weighted_geometry']; c = s['modes']['fixed']['pooled_contrasts']
        lines.append(f"{s['seed']} | {p['original']:.10f} | {p['source']:.10f} | {p['fixed_proportion']:.10f} | {c['source_minus_original']:+.10f} | {c['source_minus_fixed_proportion']:+.10f}")
    lines += ['', '二、逐方向与原生query敏感性']
    for s in summary:
        for mode in ['fixed', 'native']:
            m = s['modes'][mode]; c = m['pooled_contrasts']
            a = m['per_batch_contrast_signs']['source_minus_original']; b = m['per_batch_contrast_signs']['source_minus_fixed_proportion']
            lines.append(f"seed{s['seed']} {mode}: 来源−原方案={c['source_minus_original']:+.10f}, 来源−固定比例={c['source_minus_fixed_proportion']:+.10f}; "
                         f"8方向中前者改善/变差/数值未决={a['negative_better']}/{a['positive_worse']}/{a['within_numerical_floor']}, 后者={b['negative_better']}/{b['positive_worse']}/{b['within_numerical_floor']}.")
        lines.append(f"seed{s['seed']} 数值重复性floor={s['numerical_floor']:.3g}, CPU原生可转移配对={s['transfer_pairs']}/{s['union_pairs']}; 原生端点ordered top-k槽变化比例="+
                     json.dumps(s['modes']['native']['mean_native_topk_ordered_slot_change_fraction'], ensure_ascii=False))
    lines += ['', '三、解释边界',
        '方向和大小只描述这些开发输入上、该检查点的一步局部普通几何目标；保留历史Adam动量，不等同于额外loss的纯SGD效应。共享参数与非线性使子集结果不能相加解释独立子集干预。',
        '固定query主比较避免上游更新造成query排序变化后沿旧匹配评估的身份混淆；它也因此是条件性的连续分支诊断，不是完整训练反事实。原生重选敏感性仍沿冻结匹配评估，不是重新Hungarian后的真实验证指标。',
        '没有完整matcher代价最优接收分配对照，没有检测AP，没有多轮收敛或不同检查点验证。本次即使通过，也不支持已确立创新、优于近邻或可投稿的结论。',
        '配对和每图层预算守恒不等于梯度、损失值或优化难度相同。MAL和own匹配次指标见CSV，不能用某个有利次指标替代预定主端点。',
        '不报告p值或等效结论；两个seed、相同增强图及相关批次不足以将局部方向解释为独立重复或普遍规律。未按结果挑批次、换门槛、增组或启动训练。', '',
        '四、执行及证据',
        f"原生基点前向16次；三组虚拟端点48个；交叉批次固定query/原生query端点各48次；另16次零步函数式前向。总模型前向128次，真实训练步0。",
        f"elapsed_seconds={result['elapsed_seconds']:.3f}; CPU resources="+json.dumps(result['resources'], ensure_ascii=False),
        '资源为本地CPU两个线程、低优先级、两个CPU亲和性；GPU/远端/队列/自动监测均未访问。',
        '01协议于新结果前锁定；02启动核验；03逐方向原始JSON；04汇总与预定筛查；05全部96个三组/选择模式端点CSV；seed目录含逐批匹配、系数预算、全损失、分块梯度/更新范数及完整性回执。']
    (d.OUT/'06_结果解释与下一步建议.txt').write_text('\n'.join(lines)+'\n', encoding='utf8')
    verification = {'utc': d.now(), 'unchanged_sources': [], 'seed_checks': [],
                    'protocol_runner_sha_matches': d.native.sha(d.CODE/'diagnostic.py') == protocol['runner_sha256']}
    assert verification['protocol_runner_sha_matches']
    for x in protocol['sources']:
        got = d.native.sha(x['path']); assert got == x['sha256'], x['path']
        verification['unchanged_sources'].append({'path': x['path'], 'sha256': got})
    for seed in [1, 2]:
        v = json.loads((d.OUT/f'seed{seed}/complete.json').read_text(encoding='utf8'))
        assert v['model_unchanged'] and v['Adam_source_states_unchanged'] and v['checkpoint_writes'] == 0
        verification['seed_checks'].append(v)
    d.native.save_json(d.OUT/'07_原始资产与模型完整性核验.json', verification)
    manifest = [{'path': str(p.relative_to(d.OUT)), 'bytes': p.stat().st_size, 'sha256': d.native.sha(p)}
                for p in sorted(d.OUT.rglob('*')) if p.is_file()]
    manifest += [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': d.native.sha(p)} for p in sorted(d.CODE.glob('*.py'))]
    d.native.save_json(d.OUT/'08_结果与代码SHA256清单.json', {'created_utc': d.now(), 'files': manifest})
    print(json.dumps(complete, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
