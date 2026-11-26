"""Read-only source/arithmetic audit for closing width diagnosis; no model import."""
import hashlib
import json
import re
from pathlib import Path

BASE = Path(r'C:\Users\jiwei\Desktop\小论文投稿')
EN = BASE/'英文小论文'
MAIN = EN/'持续寻找研究主线_20261020_v2'
ROOT = Path(__file__).resolve().parent
paths = {
    'rules': EN/'CLAUDE.md',
    'route': MAIN/'356_355b子带裁剪预检_主控核对与路线结论.txt',
    'protocol': MAIN/'360_创新点整理_提示词_20261026_v5.md',
    'report': MAIN/'360_线程B_宽度门槛诊断_20261026_v2.md',
    'thread_A_prior_verification': MAIN/'360_线程A_补核_三篇章节号_20261026_v1.md',
    'axis_A': BASE/'tmp/360_axisA_geo0_width.json',
    'axis_B': BASE/'tmp/360_axisB_fdr_top_v1.json',
    'axis_A_code': BASE/'tmp/360_axisA_geo0_width.py',
    'axis_B_code': BASE/'tmp/360_axisB_fdr_top_v1.py',
    'selfcheck_350': MAIN/'350_349预检_主控核对.txt',
    'selfcheck_351': MAIN/'351_主线val548结果_主控核对.txt',
    'selfcheck_354': MAIN/'354_UAVDT六run结果核对_论文用.txt',
}
sources = {name: {'path': str(p), 'bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for name, p in paths.items()}
A = json.loads(paths['axis_A'].read_text(encoding='utf-8'))
B = json.loads(paths['axis_B'].read_text(encoding='utf-8'))
seeds = {}
for seed in ('1', '2'):
    a, b = A[seed], B[seed]
    assert a['n_gt'] == b['n_gt'] == 38759
    assert a['tiny'] == b['tiny'] == 4952
    assert a['tiny_geo0'] == b['tiny_geo0']
    assert a['matched_c1'] == b['g0']['matched']
    a_pass = a['c1_aw']['p50'] < 2 and a['c1_aw_le1'] > .7
    ptop = b['g0']['top']/b['g0']['matched']
    b_class = '顶' if ptop >= .3 else '未顶' if ptop < .1 else 'mixed'
    assert not a_pass and b_class == '未顶'
    hist = b['g0']['argmax_bin_hist']
    assert len(hist) == 33 and abs(sum(hist)-1) < .002
    assert b['max_pb_diff_vs_cache'] == 0
    seeds[seed] = {
        'Geo0': a['tiny_geo0'], 'matched_nearest_same_class': a['matched_c1'],
        'matched_IoU_at_least_0_1': a['matched_c2'],
        'nearby_subset_fraction_of_Geo0': a['matched_c2']/a['tiny_geo0'],
        'width_abs_error_median_640_px': a['c1_aw']['p50'],
        'within_1px_fraction': a['c1_aw_le1'], 'relative_width_error_median': a['c1_rel']['p50'],
        'nearest_center_distance_median_640_px': a['c1_cd']['p50'],
        'nearby_width_error_median_640_px': a['c2_aw']['p50'],
        'nearby_center_distance_median_640_px': a['c2_cd']['p50'],
        'P_top_exact': ptop, 'P_top_exact_percent': 100*ptop,
        'P_top_count': b['g0']['top'], 'P_top_denominator': b['g0']['matched'],
        'cell': ['不准', b_class],
        'argmax_bin0_percent': 100*hist[0], 'argmax_bin1_percent': 100*hist[1],
        'argmax_bin31_percent': 100*hist[31], 'argmax_bin32_percent': 100*hist[32],
        'reference_width_over_GT_median': b['g0']['ratio_refw_pct']['p50'],
        'quotient_of_medians_NOT_paired_shrinkage': (1+a['c1_rel']['p50'])/b['g0']['ratio_refw_pct']['p50'],
        'P_mod4_below_0_1': a['mod4_1_lt0.1'], 'minimum_width_640_px': a['pred_w_min'],
        'reported_prediction_difference_from_cache': b['max_pb_diff_vs_cache'],
        'reported_box_reconstruction_error': b['max_box_recon_err'],
    }
rules_text = paths['rules'].read_text(encoding='utf-8')
core = [line for line in rules_text.splitlines() if re.match(r'^\d+\.\s', line)]
prohibited = [line for line in rules_text.split('## 绝对禁止', 1)[1].split('## 参考资料', 1)[0].splitlines() if line.startswith('- ')]
assert len(core) == len(prohibited) == 4
lib = EN/'文献_相关原始论文_20261004'
pdfs = sorted(p for p in lib.rglob('*') if p.is_file() and p.suffix.lower() == '.pdf')
root_pdfs = [p for p in pdfs if p.parent == lib]
assert len(pdfs) == 260 and len(root_pdfs) == 258
text350 = paths['selfcheck_350'].read_text(encoding='utf-8').splitlines()
text351 = paths['selfcheck_351'].read_text(encoding='utf-8').splitlines()
text354 = paths['selfcheck_354'].read_text(encoding='utf-8').splitlines()
checks = {'350_line29': text350[28], '351_line29': text351[28],
          '354_lines6_8': text354[5:8], 'LIB_root_pdf': len(root_pdfs),
          'LIB_recursive_pdf': len(pdfs), 'hard_rules_count': len(core)+len(prohibited),
          'hard_rules': core+prohibited}
result = {
    'decision': 'close_width_direction_at_gate; adopt_356_route_A',
    'verification_level': 'source reading and arithmetic only; no independent forward replay',
    'source_files': sources, 'selfcheck': checks, 'seeds': seeds,
    'corrections': [
        'bin32 argmax is 8.05%/6.80%, not near zero; low end-bin total mass and frozen P_top classification still stand.',
        'A ratio of marginal medians does not identify the median paired prediction/reference width ratio; 4-5% is not an audited shrinkage statistic.',
        'No saturation by the frozen threshold does not prove adequate representation or causally isolate reference priors/optimization as the source.',
        'Identical finite-sample predictions demonstrate output reproduction, not equality or immutability of every weight tensor.',
    ],
    'matching_limit': 'IoU>=0.1 evidence covers only 33.46%/35.07% of Geo0. The same-class nearest-center result includes distant boxes; it is not a known training assignment.',
    'thread_A_scope': 'Read the previous section verification note; did not reopen the three PDFs or change its conclusions.',
    'execution': {'model_loads': 0, 'model_forwards': 0, 'training': False, 'GPU': False, 'test_access': False},
}
with (ROOT/'closeout_audit.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps({'decision': result['decision'], 'selfcheck': checks, 'seeds': seeds}, ensure_ascii=True, indent=2))
