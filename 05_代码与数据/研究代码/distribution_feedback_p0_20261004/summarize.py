"""Summarize retained real-image probes; descriptive statistics, not AP."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spearman_without_blas(x, y):
    # Avoid this local NumPy build's unavailable matrix-product DLL. Definition:
    # Pearson correlation of average ranks, using only elementwise reductions.
    x, y = rankdata(x), rankdata(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator else float('nan')


def describe(value):
    value = np.asarray(value).astype(float).reshape(-1)
    value = value[np.isfinite(value)]
    if not len(value):
        return {'n': 0}
    return {'n': len(value), 'mean': float(value.mean()), 'median': float(np.median(value)),
            'p10': float(np.quantile(value, .1)), 'p90': float(np.quantile(value, .9)),
            'min': float(value.min()), 'max': float(value.max())}


def summarize(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    complete = json.loads((root / 'completion.json').read_text(encoding='utf8'))
    traces = sorted((root / 'traces').glob('*.npz'))
    assert len(traces) == complete['crops']
    scalar_keys = ['match_iou', 'match_annotation_id', 'original_gt_area', 'truncated',
                   'entropy_bits', 'variance_px2', 'two_bin_variance_px2', 'excess_variance_px2',
                   'initial_edge_size_px', 'mean_edge_error_px', 'q3_span_px', 'e3_span_px',
                   'q3_span_local_cells', 'q3_span_global_cells', 'q3_all_identical',
                   'q3_oracle_edge_error_px', 'e3_oracle_edge_error_px', 'fixed3_oracle_edge_error_px',
                   'q3_oracle_gain_over_e3_px', 'q3_oracle_gain_over_fixed3_px']
    for view in ['local', 'global', 'local_p4', 'global_p3']:
        scalar_keys += [view + '_' + k for k in ['p_mean_relative_l2', 'p_e_relative_l2',
            'e_mean_relative_l2', 'p_feature_rms_relative', 'q3_e3_relative_l2', 'p_valid_mass',
            'e_valid_mass', 'q3_valid_all', 'q3_same_interpolation_cell', 'q3_feature_spread_relative']]
    values = {k: [] for k in scalar_keys}
    image_ids = []
    for path in traces:
        assert sha(path) == complete['trace_hashes'][path.name]
        with np.load(path) as data:
            for key in scalar_keys:
                values[key].append(data[key])
            image_ids.extend([int(data['image_id'])] * len(data['match_iou']))
    values = {k: np.concatenate(v) for k, v in values.items()}
    values['image_id'] = np.array(image_ids)
    area = values['original_gt_area']
    iou = values['match_iou']
    small = (area > 0) & (area < 1024)
    masks = {
        'all_queries': np.ones_like(iou, dtype=bool),
        'hungarian_all': iou >= 0,
        'matched_iou_ge_0.1': iou >= .1,
        'matched_iou_ge_0.5': iou >= .5,
        'small_iou_ge_0.1': small & (iou >= .1),
        'small_iou_ge_0.5': small & (iou >= .5),
        'small_nontruncated_iou_ge_0.1': small & (iou >= .1) & ~values['truncated'],
        'small_truncated_iou_ge_0.1': small & (iou >= .1) & values['truncated'],
        'medium_large_iou_ge_0.1': (area >= 1024) & (iou >= .1),
    }
    groups = {}
    for name, mask in masks.items():
        group = {'query_instances': int(mask.sum()), 'images': int(len(np.unique(values['image_id'][mask]))),
                 'unique_image_annotation_pairs': len(set(zip(values['image_id'][mask & (iou >= 0)].tolist(),
                                                  values['match_annotation_id'][mask & (iou >= 0)].tolist())))}
        for key in scalar_keys:
            if key in ['match_annotation_id', 'original_gt_area', 'truncated', 'initial_edge_size_px']:
                continue
            v = values[key][mask]
            if key.endswith('same_interpolation_cell'):
                prefix = key.removesuffix('_q3_same_interpolation_cell')
                v = v[values[prefix + '_q3_valid_all'][mask]]
            group[key] = describe(v)
            if 'oracle_gain' in key:
                finite = v[np.isfinite(v)]
                if len(finite):
                    group[key].update({'fraction_positive': float((finite > 1e-5).mean()),
                                       'fraction_negative': float((finite < -1e-5).mean()),
                                       'fraction_tie_1e-5_px': float((np.abs(finite) <= 1e-5).mean())})
        for key in ['local_p_e_relative_l2', 'global_p_e_relative_l2']:
            v = values[key][mask]
            valid = np.isfinite(v)
            group[key + '_fraction_below_0.01'] = float((v[valid] < .01).mean()) if valid.any() else None
        normalized_error = values['mean_edge_error_px'][mask] / np.maximum(values['initial_edge_size_px'][mask], 1e-6)
        normalized_spread = np.sqrt(np.maximum(values['excess_variance_px2'][mask], 0)) / np.maximum(values['initial_edge_size_px'][mask], 1e-6)
        for key, x, y in [
            ('entropy_vs_normalized_error', values['entropy_bits'][mask], normalized_error),
            ('excess_sd_normalized_vs_normalized_error', normalized_spread, normalized_error),
        ]:
            valid = np.isfinite(x) & np.isfinite(y)
            corr = spearman_without_blas(x[valid], y[valid]) if valid.sum() > 2 else float('nan')
            group[key] = {'spearman_descriptive': corr if np.isfinite(corr) else None, 'edge_instances': int(valid.sum()),
                          'note': 'Correlated edges and repeated GT across windows; no IID p-value or causal interpretation.'}
        groups[name] = group
    primary = masks['small_nontruncated_iou_ge_0.1']
    per_image = []
    for image_id in np.unique(values['image_id']):
        selected = primary & (values['image_id'] == image_id)
        row = {'image_id': int(image_id), 'query_instances': int(selected.sum())}
        for key in ['q3_oracle_gain_over_e3_px', 'q3_oracle_gain_over_fixed3_px',
                    'local_p_e_relative_l2', 'global_p_e_relative_l2']:
            row[key] = describe(values[key][selected])
        per_image.append(row)
    result = {'status': 'descriptive_real_image_probe_not_candidate_efficacy',
              'manifest_sha256': sha(root / 'manifest.json'), 'completion_sha256': sha(root / 'completion.json'),
              'summarizer_sha256': sha(__file__), 'images': complete['images'], 'crops': complete['crops'],
              'groups': groups, 'per_image_primary_small_nontruncated': per_image,
              'important_limits': ['No trained feedback module, no new AP or causal global-information result.',
                                   '1% relative-L2 cutoff is descriptive, not a preregistered success criterion.',
                                   'Fixed3 and nearest-GT candidate metrics are geometry-only oracle comparisons.',
                                   '32 hash-selected validation images and one partly trained EMA checkpoint.',
                                   'Query/edge/window observations are correlated; not independent sample sizes.']}
    with (root / 'summary.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: {'n': v['query_instances'], 'unique_gt': v['unique_image_annotation_pairs'],
        'span_px_median': v['q3_span_px'].get('median'),
        'local_samecell': v['local_q3_same_interpolation_cell'].get('mean'),
        'global_samecell': v['global_q3_same_interpolation_cell'].get('mean'),
        'local_p_e_l2_median': v['local_p_e_relative_l2'].get('median'),
        'global_p_e_l2_median': v['global_p_e_relative_l2'].get('median'),
        'q3_oracle_gain_e3_mean_px': v['q3_oracle_gain_over_e3_px'].get('mean'),
        'q3_oracle_gain_fixed_mean_px': v['q3_oracle_gain_over_fixed3_px'].get('mean'),
        'excess_sd_correlation': v['excess_sd_normalized_vs_normalized_error']['spearman_descriptive']}
         for k, v in groups.items()}, indent=2))
    make_figure(root, values, primary)


def make_figure(root, values, primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    def ecdf(ax, value, label, color):
        v = np.asarray(value).reshape(-1)
        v = np.sort(v[np.isfinite(v)])
        if len(v):
            ax.plot(np.maximum(v, 1e-6), np.arange(1, len(v) + 1) / len(v), label=label, color=color)
    colors = ['#2369a2', '#d46e32']
    ecdf(axes[0], values['q3_span_local_cells'][primary], 'Local P3', colors[0])
    ecdf(axes[0], values['q3_span_global_cells'][primary], 'Global P4', colors[1])
    axes[0].set(xscale='log', xlabel='10-90% span / feature stride', ylabel='Fraction of edge instances', title='Geometric sampling span')
    axes[0].axvline(1, color='#888888', ls='--', lw=1)
    axes[0].legend()
    ecdf(axes[1], values['local_p_e_relative_l2'][primary], 'Local P3', colors[0])
    ecdf(axes[1], values['global_p_e_relative_l2'][primary], 'Global P4', colors[1])
    axes[1].set(xscale='log', xlabel='Relative L2: full p vs same-mean two-bin', ylabel='Fraction of valid edge instances', title='Frozen feature-read difference')
    axes[1].legend()
    names = ['Mean', 'Two-bin q3', 'Full-p q3', 'Fixed q3']
    keys = ['mean_edge_error_px', 'e3_oracle_edge_error_px', 'q3_oracle_edge_error_px', 'fixed3_oracle_edge_error_px']
    medians = [float(np.nanmedian(values[k][primary])) for k in keys]
    axes[2].bar(names, medians, color=['#8997a5', '#b7c6d4', '#2369a2', '#d46e32'])
    axes[2].set(ylabel='Median nearest-candidate edge error (px)', title='GT-oracle geometry only; not AP')
    fig.suptitle('P0 | 32 images | epoch-54 EMA | small, non-truncated, first-layer IoU >= 0.1', fontsize=12)
    fig.savefig(root / 'diagnostic_summary.png', dpi=180)
    fig.savefig(root / 'diagnostic_summary.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)
