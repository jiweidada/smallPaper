"""Create corrected v3 and an append-only evidence entry from aggregate results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EN = ROOT.parents[1]
MAIN = EN/'持续寻找研究主线_20261020_v2'
R = json.loads((ROOT/'aggregate_results.json').read_text(encoding='utf-8'))
A = json.loads((EN.parent/'tmp/360_axisA_geo0_width.json').read_text(encoding='utf-8'))
OB = json.loads((MAIN/'348b_tiny_geo0_oracle.json').read_text(encoding='utf-8'))
OC = json.loads((MAIN/'348c_axis_decomposition.json').read_text(encoding='utf-8'))
S1, S2 = R['seeds']['1'], R['seeds']['2']


def pct(value):
    return f'{100*value:.4f}%'


def table(lines, title, values):
    lines += ['', title, '', '| 指标 | seed1 | seed2 |', '|---|---:|---:|']
    for name, first, second in values:
        lines.append(f'| {name} | {first} | {second} |')


def mass_row(key, subgroup):
    values = []
    for s in (S1, S2):
        m = s[key]
        n = s['counts']['matched_criterion2']
        if subgroup == 'all':
            count, rate = m['events_any_four_edges'], m['rate_any_four_edges']
        else:
            count, rate = m['events_horizontal_only'], m['rate_horizontal_only']
        values.append(f'{count}/{n}（{pct(rate)}）')
    return values


def main():
    for seed, s in R['seeds'].items():
        c = s['counts']
        n = c['matched_criterion2']
        assert n == {'1': 1243, '2': 1307}[seed]
        assert c['any_class_IoU_ge_0_1'] == c['same_class_IoU_ge_0_1']+c['overlap_ge_0_1_wrong_class_only']
        assert c['Geo0'] == c['any_class_IoU_ge_0_1']+c['any_class_IoU_lt_0_1']
        assert c['same_class_IoU_ge_0_1'] == n
        for key in ('endpoint', 'outer_two_each_end', 'negative_tail', 'positive_tail'):
            m = s[key]
            assert abs(m['rate_any_four_edges']-m['events_any_four_edges']/n) < 1e-15
            assert abs(m['rate_horizontal_only']-m['events_horizontal_only']/n) < 1e-15
        assert sum(s['argmax']['bin_counts']) == 4*n
        cp = s['checkpoint']
        assert cp['file_sha256_before'] == cp['file_sha256_after']
        assert cp['state_dict_sha256_before'] == cp['state_dict_sha256_after']
        assert not cp['model_branch_equals_ema']
        assert s['integrity']['boxes_max_abs_diff_vs_cache'] == 0
        assert s['integrity']['logits_max_abs_diff_vs_cache'] == 0
    lines = [
        '# 360 线程B：宽度门槛诊断 v3——口径②重算与归因修订', '',
        '日期：2026-10-26。本文替代 v2 与旧 Codex 收尾记录中的机制解释；原文件保留为历史记录。', '',
        '**结论：宽度方法方向仍停止，沿用356 A路径。撤回“B轴已证明表示够用、因而宽度回归没学到位”的归因。**',
        '口径②的近邻子集未见端点或扩展外侧质量明显堆积；该子集的GT横向边界均在各自参考框的理论支撑内。结果不外推到未匹配的约三分之二Geo0。', '',
        '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run',
        '- Origin Date: 2026-10-26',
        '- Verification Status: UNVERIFIED（新聚合未独立完整复跑；预测复现、箱体重建、权重哈希及聚合算术检查通过）',
        '- Version Label: width_gate_criterion2_v3', '',
        '## 0. 自检与本轮范围', '',
        '350第29行：image_id=362、2条负宽度、原图−0.227px；351第29行：loss_assoc为2.13→2.105 / 2.21→2.09。',
        '354第6—8行：UAVDT为AP70，300 query、val选第85轮；test 16592帧/375884 GT，val 5153帧/87719 GT。该85轮信息不属于本次BASE24。',
        '本对话已亲自递归核对LIB为260份PDF（258根+2子目录）；CLAUDE.md有8条硬规则。线程A三篇章节号沿用已存真PDF核查记录，本轮未重开三篇。',
        '本轮只在本机CPU对val548×2 seed各前向一次，另存聚合量和复现元数据。无训练、GPU、远端或test访问。', '',
        '## 1. 样本口径修复', '',
        '- tiny：冻结缓存gt_short_side_640_all<4；共4952。Geo0：全部300个query、不分类别的maxIoU<0.5；3715/3727。',
        '- 本轮只选口径②：query的argmax类别与GT相同、IoU≥0.1，取其中最大IoU；同分取最小query序号。无分类分数门槛。',
        '- 这是多对一诊断配对，不是训练分配，也不保证就是处理该GT的真实query；同一query可对应多个GT，分母不能当独立query数。',
        '- 旧口径①同类最近中心的中位距离17.63/16.15px，确实不适合直接解释tiny目标自身所需的FDR位移。旧B轴统计降格为远近混合邻框的描述，不能支撑原归因。',
    ]
    table(lines, '口径②的局部几何误差（原轴A聚合量，本轮宽度/中心距离复现）：', [
        ('匹配数', 1243, 1307),
        ('宽度绝对误差中位（640px）', f'{S1["geometry_quantiles"]["width_error_px"]["p50"]:.4f}', f'{S2["geometry_quantiles"]["width_error_px"]["p50"]:.4f}'),
        ('±1px比例（旧轴A口径②）', pct(A['1']['c2_aw_le1']), pct(A['2']['c2_aw_le1'])),
        ('中心距离中位（640px）', f'{S1["geometry_quantiles"]["center_distance_px"]["p50"]:.4f}', f'{S2["geometry_quantiles"]["center_distance_px"]["p50"]:.4f}'),
    ])
    lines += ['', '原E1/E2仍保留：口径①w mod4<0.1的比例为3.50%/2.36%；全部final300最小宽度为1.9917/1.7710px。它们未显示4px硬夹紧或原规则下的离散量化，但E1的一侧取模统计不能排除所有形式的量化/表示误差。',
              '', '## 2. 同一近邻子集上的FDR质量统计', '',
              '原定义E=bin0+bin32。补充定义O=bin0+bin1+bin31+bin32；另分别保存两端各自的质量。某侧质量≥0.5记事件。O是本轮补充诊断，不能冒充原预注册判据或作为新方法通过条件。']
    table(lines, '每GT至少一侧发生事件的比例：', [
        ('E：四边至少一边', *mass_row('endpoint', 'all')),
        ('E：仅检查左右边', *mass_row('endpoint', 'x')),
        ('O：四边至少一边', *mass_row('outer_two_each_end', 'all')),
        ('O：仅检查左右边', *mass_row('outer_two_each_end', 'x')),
    ])
    table(lines, '边级质量分布（分母为4×匹配GT数；量的是概率质量，不是argmax频率）：', [
        ('E中位', pct(S1['endpoint']['mass_quantiles_all_edges']['p50']), pct(S2['endpoint']['mass_quantiles_all_edges']['p50'])),
        ('E p90', pct(S1['endpoint']['mass_quantiles_all_edges']['p90']), pct(S2['endpoint']['mass_quantiles_all_edges']['p90'])),
        ('O中位', pct(S1['outer_two_each_end']['mass_quantiles_all_edges']['p50']), pct(S2['outer_two_each_end']['mass_quantiles_all_edges']['p50'])),
        ('O p90', pct(S1['outer_two_each_end']['mass_quantiles_all_edges']['p90']), pct(S2['outer_two_each_end']['mass_quantiles_all_edges']['p90'])),
        ('argmax属于{0,1,31,32}', pct(sum(S1['argmax']['bin_fractions'][i] for i in (0,1,31,32))), pct(sum(S2['argmax']['bin_fractions'][i] for i in (0,1,31,32)))),
    ])
    lines += ['', '旧口径①的外侧四bin argmax比例18.62%/24.77%算术正确，但这不是质量≥0.5的事件率，也不能据此判断饱和。33个bin的完整计数和比例已保存，避免再混淆bin31与bin32。', '',
              '## 3. 真实像素支撑与参考框统计', '',
              '实际checkpoint与hook均确认reg_scale=4、up=0.5、project端点±4。project没有像素单位；每边像素偏移需乘w_ref/4（纵向乘h_ref/4）。',
              '因此不能直接把端点写成统一±8～16px或“ref±32px”。中央相邻bin也不是统一0.3px。',
              '在640平面，横向参考中心为c、参考宽w时：', '',
              '```text',
              'left  = c - (2+d_L) w/4，d_L∈[-4,4]',
              'right = c + (2+d_R) w/4，d_R∈[-4,4]',
              'left范围=[c-1.5w,c+0.5w]；right范围=[c-0.5w,c+1.5w]',
              'pred_width=w*(1+(d_L+d_R)/4)',
              '```', '',
              '正宽度部分可趋近0，不存在统一4px的代数夹紧；这不意味着特征、参数化、学习或有限数据条件下没有精度限制。']
    table(lines, '本轮实际量（口径②）：', [
        ('参考框宽中位（640px）', f'{S1["geometry_quantiles"]["refw_px"]["p50"]:.4f}', f'{S2["geometry_quantiles"]["refw_px"]["p50"]:.4f}'),
        ('ref_w/GT_w中位', f'{S1["geometry_quantiles"]["ref_gt_ratio"]["p50"]:.4f}', f'{S2["geometry_quantiles"]["ref_gt_ratio"]["p50"]:.4f}'),
        ('逐框pred_w/ref_w中位', f'{S1["geometry_quantiles"]["pred_ref_ratio"]["p50"]:.6f}', f'{S2["geometry_quantiles"]["pred_ref_ratio"]["p50"]:.6f}'),
        ('中央bin16→17每边像素间距中位', f'{S1["geometry_quantiles"]["central_step_px"]["p50"]:.4f}', f'{S2["geometry_quantiles"]["central_step_px"]["p50"]:.4f}'),
        ('外侧bin31→32每边像素间距中位', f'{S1["geometry_quantiles"]["outer_gap_px"]["p50"]:.4f}', f'{S2["geometry_quantiles"]["outer_gap_px"]["p50"]:.4f}'),
        ('GT左右两边同时落在该ref的支撑内', f'{S1["GT_horizontal_edges_within_support_count"]}/1243', f'{S2["GT_horizontal_edges_within_support_count"]}/1307'),
        ('GT四边同时落在支撑内', pct(S1['GT_all_edges_within_support_fraction']), pct(S2['GT_all_edges_within_support_fraction'])),
    ])
    lines += ['', '逐框pred/ref中位直接测得0.9630/0.9469，分别对应约3.70%/5.31%的收缩；这是新近邻子集的配对统计，不是两个边际中位数相除，也不追认旧口径①的4～5%估计。',
              '该子集横向支撑检查全部通过，说明这些GT的左右边界没有被该参考框的连续取值范围硬性排除；仍不能把结果外推给未匹配GT，或从中确定回归误差的因果来源。', '',
              '## 4. 修正归因：覆盖、类别与横向区间', '',
              '**一级现象应写为：高IoU几何覆盖不足，同时存在明显的类别条件差异；不能写成“三分之二附近完全没有框”。**',
              'IoU≥0.1只是本文操作阈值，不等同于中心距离意义上的“附近”。query的argmax类别不符，也不自动识别为该GT的分类错误。']
    coverage_rows = []
    for label, key in [('同类且IoU≥0.1', 'same_class_IoU_ge_0_1'),
                       ('IoU≥0.1但仅有argmax类别不符的query', 'overlap_ge_0_1_wrong_class_only'),
                       ('所有类别均无IoU≥0.1的query', 'any_class_IoU_lt_0_1')]:
        vals = []
        for s in (S1, S2):
            c = s['counts']
            vals.append(f'{c[key]}/{c["Geo0"]}（{pct(c[key]/c["Geo0"])}）')
        coverage_rows.append((label, *vals))
    table(lines, 'Geo0按0.1阈值的完整分解：', coverage_rows)
    lines += ['', '不分类别时已有2579/3715、2464/3727个Geo0与某个query达到IoU≥0.1。故同类配不上（2472/2420个）与空间中完全缺少重叠候选不是一回事。', '',
              '348的oracle数字保留，但按代码恢复实际干预定义：']
    table(lines, '具体oracle（均非部署增益）：', [
        ('中心不动、宽高均截到≤1.5×GT（348b）', '378/3715（10.1750%）', '352/3727（9.4446%）'),
        ('最近任意类候选，整个x区间替换为GT（348c）', '2548/3715（68.5868%）', '2470/3727（66.2731%）'),
        ('最近任意类候选，保留中心、替换GT宽高（348c）', '624', '584'),
    ])
    lines += ['', '前两项修改的内容、候选选择也不同：378/352不是“纯宽度最优修正”，不能由10%直接推出“宽度只占10%的成因”；2548/2470同时改横向中心和宽度，也不是训练可实现收益。它们支持的是这项简单收缩远弱于整个横向区间替换。',
              '624/584的旧字段名center_only与源码相反：代码实际保留最近候选中心、替换GT宽高。真正将所有候选中心换成GT中心的348b O_ctr计3209/3124，但它在全部300个候选上再取最大值，不能与固定最近候选的624/584混用。',
              '16:9拉伸下sx/sy=0.5625是坐标事实，不是本轮对漏检成因的因果鉴定；保持宽高比路线既有mixed结论不重开。', '',
              '## 5. 停止理由与A路径', '',
              '- 旧口径①的P_top不能代表近邻tiny子集，旧“表示够用→只是回归没学到位”链条撤回。',
              '- 修复后，近邻子集的端点事件为0/0，扩展外侧事件为0/3；横向GT边界均在连续支撑内。没有取得支持“硬性值域饱和是该子集宽度误差原因”的证据。端部检验的统计检验力没有单独标定，不把未发生事件写成对全部表示问题的否定。',
              '- 这个近邻子集只占Geo0的33.46%/35.07%；其余样本不能据此归因。宽度偏差、类别条件覆盖、中心偏移及其他因素并未被完整因果分解。',
              '- 因此继续停止宽度方法方向，不进入§3.6、不提出新回归头、不启动训练。按356 A路径保留受控实证与机制分析，以上仅作为带边界的诊断/负结果。', '',
              '## 6. 权重、复现与交付', '',
              '原脚本实际从327索引的last.pth加载checkpoint[model]，last_epoch=23，即第24轮；与EMA分支的state_dict哈希不同。v2的“BASE24 EMA、第85轮”标注撤回。',
              f'两seed合计1096次CPU前向，墙钟{R["execution"]["wall_seconds"]:.3f}秒。新boxes和logits与冻结缓存逐元素最大差均为0.0。',
              f'ref+dist重建最大误差为{S1["integrity"]["max_normalized_box_reconstruction_error"]:.12g} / {S2["integrity"]["max_normalized_box_reconstruction_error"]:.12g}（归一化坐标）。',
              '两seed的checkpoint文件SHA256和加载后完整state_dict哈希，前后均相同。输出复现与权重身份核验各有作用，不能互相替代。',
              '落盘仅聚合量、脚本/协议和哈希元数据，无per-query角点张量或encoder特征；总量低于20MB。原v2与旧数据未覆盖。', '',
              '产物目录：`C:/Users/jiwei/Desktop/小论文投稿/英文小论文/code/width_gate_nearby_recheck_20261026_v1/`。',
              '完整数值：`aggregate_results.json`；协议：`protocol.json`、`frozen_protocol.sha256`；输入/权重核验：`input_manifest.json`及各seed的checkpoint字段；执行记录：`run.log`、`run_metadata.json`。',
              '脚本：同目录`run_recheck.py`（历史输出采用排他创建，不可在原输出目录直接覆盖复跑）。归因修订脚本：`build_closeout.py`。', '',
              '未做：GPU训练、test、远端操作、对未匹配GT的query身份鉴定、端部事件检验力的独立标定、对全部表示限制的排除、新聚合的第二次独立模型前向复跑。']
    # Derive displayed oracle percentages from the actual source counts.
    for idx, line in enumerate(lines):
        if '中心不动、宽高均截到' in line and line.startswith('|'):
            values = []
            for seed in ('1', '2'):
                b = OB['seed'+seed]
                values.append(f'{b["tiny_geo0_Oshrink1.5_ge.5"]}/{b["tiny_geo0"]}（{pct(b["tiny_geo0_Oshrink1.5_ge.5"]/b["tiny_geo0"])}）')
            lines[idx] = '| 中心不动、宽高均截到≤1.5×GT（348b） | '+' | '.join(values)+' |'
        if '最近任意类候选，整个x区间替换' in line and line.startswith('|'):
            values = []
            for seed in ('1', '2'):
                b = OC['seed'+seed]
                n = b['tiny_geo0_nn_Ox_fix_ge.5']
                values.append(f'{n}/{b["tiny_geo0"]}（{pct(n/b["tiny_geo0"])}）')
            lines[idx] = '| 最近任意类候选，整个x区间替换为GT（348c） | '+' | '.join(values)+' |'
    report_path = MAIN/'360_线程B_宽度门槛诊断_20261026_v3.md'
    with report_path.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    entry = {
        'status': 'width_direction_stopped; route_356_A_retained',
        'supersedes_interpretation': [str(MAIN/'360_线程B_宽度门槛诊断_20261026_v2.md'),
                                     str(MAIN/'360_线程B_收尾核对_20261026_Codex.txt')],
        'revised_report': str(report_path), 'aggregate_results': str(ROOT/'aggregate_results.json'),
        'primary_matching': 'criterion2_same_argmax_class_IoU_ge_0.1_max',
        'sample_counts': [1243, 1307], 'endpoint_events': [0, 0],
        'outer_two_each_end_events': [0, 3], 'outer_horizontal_events': [0, 1],
        'withdrawn_claims': ['Adequate representation inferred from old criterion1 P_top',
                             'Reference prior plus weak regression is the identified cause',
                             'Two-thirds of Geo0 have no nearby prediction',
                             '378/352 is a pure-width intervention', '624/584 is a center-only intervention',
                             'BASE24 EMA at epoch85 is the evaluated checkpoint'],
        'new_observation': 'All criterion2 GT horizontal edges are within each selected reference box support; this is conditional, not a global exclusion of representation limitations.',
        'model_source': '24-epoch checkpoint model branch, not EMA',
        'no_new_method_or_training': True,
    }
    with (MAIN/'360_线程B_v3_证据修订账本_20261026.json').open('x', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    fallacy_scan = {
        'material_passport': {'origin_skill': 'experiment-agent', 'origin_mode': 'validate',
                              'verification_status': 'ANALYZED', 'version': 'width_nearby_validation_v1'},
        'scope': 'Arithmetic/integrity and inference boundaries; new aggregates not independently forward-replicated',
        'coverage': '11/11 checked',
        'checks': {
            '1_Simpson': 'Two seeds reported separately; no subgroup-generalization claim.',
            '2_Ecological': 'Paired pred/ref ratios now directly measured; quotient of marginal medians withdrawn.',
            '3_Berkson': 'Geo0 and IoU>=0.1 selection limit inference to that conditional subset.',
            '4_Collider': 'No causal inference from this selected observational population.',
            '5_Base_rate': 'Geo0, matched, wrong-class overlap and no-overlap denominators all reported.',
            '6_Regression_to_mean': 'No fitted intervention or before/after improvement claim.',
            '7_Survivorship': 'Unmatched majority explicitly reported, not silently discarded in the interpretation.',
            '8_Look_elsewhere': 'Both mass definitions and all seeds reported; no significant-result selection.',
            '9_Forking_paths': 'Matching correction and exploratory outer-bin definition disclosed and frozen before the rerun.',
            '10_Causality': 'Reference-prior/optimization causal attribution withdrawn; oracle results not AP gains.',
            '11_Reverse_causality': 'No causal direction identified from a final-model snapshot.'},
        'inference_tests': 'None; descriptive counts and quantiles only, no p-values or population CI.',
    }
    with (ROOT/'validation_note.json').open('x', encoding='utf-8') as f:
        json.dump(fallacy_scan, f, ensure_ascii=False, indent=2)
    print(json.dumps({'report': str(report_path), 'report_lines': len(lines),
                      'ledger': entry['status']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
