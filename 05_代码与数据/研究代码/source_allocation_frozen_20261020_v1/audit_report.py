"""Independent stdlib-only reporting checks, no model/data tensor execution."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/10_来源分配冻结诊断_20261020_v1'
OLD = ROOT/'data/08_匹配聚合覆盖探针_20261019_v1/isolated_v3'


def pairs(pair):
    return set(zip(*pair))


def main():
    raw = json.loads((OUT/'03_全部跨批原始结果.json').read_text(encoding='utf8'))
    summary = json.loads((OUT/'04_配对汇总与预定筛查.json').read_text(encoding='utf8'))
    report_checks, comparisons, gradient_blocks = [], [], []
    for seed in [1, 2]:
        subset = [r for r in raw['rows'] if r['seed'] == seed]
        s = next(x for x in summary['seed_summaries'] if x['seed'] == seed)
        for mode in ['fixed', 'native']:
            # Independent pool of differences directly, instead of subtracting pooled arm means.
            for name, left, right in [('source_minus_original', 'source', 'original'),
                                      ('source_minus_fixed_proportion', 'source', 'fixed_proportion')]:
                numerator = sum((r['arms'][left][mode]['metrics']['weighted_geometry']-
                                 r['arms'][right][mode]['metrics']['weighted_geometry'])*r['before']['denominator'] for r in subset)
                denominator = sum(r['before']['denominator'] for r in subset)
                expected = numerator/denominator
                got = s['modes'][mode]['pooled_contrasts'][name]
                assert abs(expected-got) < 2e-15
                report_checks.append({'seed': seed, 'mode': mode, 'contrast': name,
                                      'independent_pooled_difference': expected, 'reported_difference': got,
                                      'absolute_error': abs(expected-got)})
        side = 'A' if seed == 1 else 'B'
        for batch in range(8):
            c = json.loads((OUT/f'seed{seed}/batch{batch:02d}_capture.json').read_text(encoding='utf8'))
            old = json.loads((OLD/side/f'development/batch{batch:02d}_analysis.json').read_text(encoding='utf8'))
            branch = {k: 0 for k in ['final', 'aux_0', 'aux_1', 'pre', 'enc_0']}
            union_difference = 0
            for b, old_matches in enumerate(old['branch_matches']):
                for stage in branch:
                    branch[stage] += len(pairs(old_matches[stage]) ^ pairs(c['branch_matches'][b][stage]))
                pre = old['inputs'][b]['branches']['pre']['native_union']
                union_difference += len(set(zip(pre['pair_query'], pre['pair_GT'])) ^ pairs(c['union_indices'][b]))
            comparisons.append({'seed': seed, 'batch': batch,
                'branch_matching_symmetric_difference_vs_old_GPU_predictions': branch,
                'union_pair_symmetric_difference_vs_old_GPU_predictions': union_difference})
            for arm, u in c['updates'].items():
                assert u['active_tensors'] > 0
                for block in ['backbone_shared_upstream', 'hybrid_encoder_shared_upstream',
                              'decoder_attention_FFN_shared', 'pre_bbox_head', 'decoder_bbox_heads']:
                    assert u['blocks'][block]['active_tensors'] > 0
                    assert u['blocks'][block]['gradient_squared_norm'] > 0
                    assert u['blocks'][block]['update_squared_norm'] > 0
                gradient_blocks.append({'seed': seed, 'batch': batch, 'arm': arm,
                                        'active_tensors': u['active_tensors'],
                                        'required_shared_and_head_blocks_reached': True})
    sums = []
    for seed in [1, 2]:
        subset = [r for r in comparisons if r['seed'] == seed]
        sums.append({'seed': seed, 'union_pair_symmetric_difference': sum(r['union_pair_symmetric_difference_vs_old_GPU_predictions'] for r in subset),
                     'branch_pair_symmetric_difference': {stage: sum(r['branch_matching_symmetric_difference_vs_old_GPU_predictions'][stage] for r in subset)
                                                          for stage in ['final', 'aux_0', 'aux_1', 'pre', 'enc_0']}})
    value = {'pooled_contrast_independent_arithmetic_checks': report_checks,
        'all_48_endpoint_gradients_reach_shared_parameters_and_heads': len(gradient_blocks) == 48,
        'gradient_domain_checks': gradient_blocks,
        'comparison_to_prior_GPU_prediction_matching': sums, 'per_batch_matching_differences': comparisons,
        'interpretation': 'All arms use current base native CPU matching, not substituted GPU matching. '
                          'Prior GPU prediction analysis differs; this is a CPU frozen diagnostic, not a bitwise GPU forward replay. '
                          'Counts 875/879 should not be silently substituted for the earlier 897/899 cache audit. '
                          'Exact reason for individual matching differences was not isolated. Native CPU matcher/tie rule was not changed.'}
    with (OUT/'09_独立算术与匹配差异复核.json').open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    print(json.dumps({'arithmetic_checks': len(report_checks), 'gradient_checks': len(gradient_blocks),
                      'prior_GPU_matching_comparison': sums}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
