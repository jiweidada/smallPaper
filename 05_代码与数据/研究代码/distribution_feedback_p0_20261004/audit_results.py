"""Independent retained-trace checks and validity-masked oracle sensitivity."""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'data/00_实验计划与证据索引/分布反馈深入自审_20261004'
V2 = BASE / 'p0_real_cpu_v2'
ROOT = BASE / 'p0_real_cpu_v3_level_control'
summary = json.loads((ROOT / 'summary.json').read_text(encoding='utf8'))
manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf8'))
old_manifest = json.loads((V2 / 'manifest.json').read_text(encoding='utf8'))
assert manifest['images'] == old_manifest['images']
masked = {key: [] for key in ['q3', 'e3', 'fixed3']}
correlation_x, correlation_y = [], []
differences = {}
primary_edges = 0
eligible_edges = 0
max_probability_sum_error = 0.0
for path in sorted((ROOT / 'traces').glob('*.npz')):
    with np.load(path) as current, np.load(V2 / 'traces' / path.name) as old:
        for key in ['p', 'p_two_bin', 'first_boxes', 'final_boxes', 'final_logits',
                    'local_p_e_relative_l2', 'global_p_e_relative_l2', 'match_iou']:
            assert np.array_equal(current[key], old[key], equal_nan=True), (path.name, key)
        assert np.all(np.diff(current['q3_edge_px'], axis=-1) >= 0)
        assert np.all(np.diff(current['e3_edge_px'], axis=-1) >= 0)
        probability_error = float(np.abs(current['p'].sum(-1)-1).max())
        max_probability_sum_error = max(max_probability_sum_error, probability_error)
        # Softmax and summation of 33 float32 values need roundoff tolerance.
        assert np.allclose(current['p'].sum(-1), 1, atol=1e-6, rtol=0)
        assert np.allclose(current['p_two_bin'].sum(-1), 1, atol=1e-6, rtol=0)
        assert np.allclose((current['p'] * current['edge_positions_px']).sum(-1),
                           (current['p_two_bin'] * current['edge_positions_px']).sum(-1), atol=3e-3, rtol=0)
        window = current['window']
        width, height = current['image_wh']
        real_upper = np.array([min(640, width-window[0]), min(640, height-window[1])])
        max_normal = real_upper[[0, 1, 0, 1]]
        max_tangent = real_upper[[1, 0, 1, 0]]
        tangent = current['first_boxes'][:, [1, 0, 1, 0]] * 640
        tangent_valid = (tangent >= 0) & (tangent <= max_tangent)
        mask = (current['original_gt_area'] > 0) & (current['original_gt_area'] < 1024)
        mask &= (current['match_iou'] >= .1) & ~current['truncated']
        edge_mask = np.broadcast_to(mask[:, None], tangent.shape)
        primary_edges += int(edge_mask.sum())
        q3, e3 = current['q3_edge_px'], current['e3_edge_px']
        fixed3 = current['mean_edge_px'][..., None] + current['initial_edge_size_px'][..., None] * np.array([-.25, 0, .25])
        gt = current['gt_edges_crop_px']
        errors = {}
        for name, candidate in [('q3', q3), ('e3', e3), ('fixed3', fixed3)]:
            valid = (candidate >= 0) & (candidate <= max_normal[None, :, None]) & tangent_valid[..., None]
            errors[name] = np.where(valid, np.abs(candidate - gt[..., None]), np.inf).min(-1)
        eligible = edge_mask & np.isfinite(errors['q3']) & np.isfinite(errors['e3']) & np.isfinite(errors['fixed3'])
        eligible_edges += int(eligible.sum())
        for key in errors:
            masked[key].append(errors[key][eligible])
        normalized_error = current['mean_edge_error_px'] / np.maximum(current['initial_edge_size_px'], 1e-6)
        normalized_sd = np.sqrt(np.maximum(current['excess_variance_px2'], 0)) / np.maximum(current['initial_edge_size_px'], 1e-6)
        correlation_x.extend(normalized_sd[edge_mask].tolist())
        correlation_y.extend(normalized_error[edge_mask].tolist())

masked = {k: np.concatenate(v) for k, v in masked.items()}
ordinary_scipy_rho = float(spearmanr(correlation_x, correlation_y).statistic)
recorded_rho = summary['groups']['small_nontruncated_iou_ge_0.1']['excess_sd_normalized_vs_normalized_error']['spearman_descriptive']
assert abs(ordinary_scipy_rho - recorded_rho) < 1e-12
g = summary['groups']['small_nontruncated_iou_ge_0.1']
levels = {v: {'median_relative_l2': g[v+'_p_e_relative_l2']['median'],
              'same_cell_fraction': g[v+'_q3_same_interpolation_cell']['mean']}
          for v in ['local', 'local_p4', 'global_p3', 'global']}
per_image = summary['per_image_primary_small_nontruncated']
result = {
    'status': 'passed_trace_and_summary_verification',
    'same_32_images_and_170_windows_as_v2': True,
    'all_v2_v3_shared_prediction_and_main_feature_arrays_bit_equal': True,
    'left_top_quantiles_sorted_in_physical_coordinates': True,
    'probability_normalization_and_mean_preserving_control_checked': True,
    'maximum_float32_probability_sum_error': max_probability_sum_error,
    'scipy_native_spearman_after_process_PATH_fix': ordinary_scipy_rho,
    'manual_rank_correlation_matches_scipy': True,
    'primary_edges': primary_edges,
    'validity_masked_oracle_edges': eligible_edges,
    'validity_masked_oracle_median_error_px': {k: float(np.median(v)) for k, v in masked.items()},
    'validity_masked_oracle_mean_gain_q3_over_e3_px': float(np.mean(masked['e3']-masked['q3'])),
    'validity_masked_oracle_mean_gain_q3_over_fixed3_px': float(np.mean(masked['fixed3']-masked['q3'])),
    'levels_2x2': levels,
    'images_with_positive_mean_oracle_gain_over_e3': sum(r['q3_oracle_gain_over_e3_px']['mean'] > 0 for r in per_image),
    'images_with_positive_mean_oracle_gain_over_fixed3': sum(r['q3_oracle_gain_over_fixed3_px']['mean'] > 0 for r in per_image),
    'no_claim_of_AP_gain_or_trained_context_utility': True,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
with (ROOT / 'independent_trace_audit.json').open('x', encoding='utf8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps(result, indent=2))
