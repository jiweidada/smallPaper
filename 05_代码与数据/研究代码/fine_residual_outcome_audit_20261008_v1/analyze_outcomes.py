"""Expose early-geometry/TP damage in future bound six-run stage outputs.

This supplements the existing v1 stage analysis; no model or evaluator is called.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
THRESHOLDS = np.linspace(.5, .95, 10)
TRANSITIONS = ('absent_in_both', 'gained', 'lost', 'present_in_both')
STAGES = ('initial', 'pre_after_first_layer', 'final')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_pair(control, candidate):
    n = len(control['annotation_id'])
    for key in ('annotation_id', 'image_id', 'category_id', 'original_area'):
        if np.asarray(control[key]).shape != (n,) or not np.array_equal(control[key], candidate[key]):
            raise ValueError('Pair populations differ: ' + key)
    if len(np.unique(control['annotation_id'])) != n:
        raise ValueError('Duplicate annotation identifiers')
    area = np.asarray(control['original_area'])
    if not np.isfinite(area).all() or np.any((area < 0) | (area > 1e10)):
        raise ValueError('Area outside bound native all-area range')
    if not np.isin(control['category_id'], np.arange(10)).all():
        raise ValueError('Unknown category')
    for value in (control, candidate):
        m, tp = np.asarray(value['max_iou']), np.asarray(value['native_TP'])
        if m.shape != (n, 3) or not np.isfinite(m).all() or np.any((m < 0) | (m > 1)):
            raise ValueError('Malformed stage IoU array')
        if tp.shape != (n, 10) or tp.dtype != np.dtype(bool):
            raise ValueError('Malformed native TP array')


def continuous_summary(delta):
    n = len(delta)
    return {
        'count': n,
        'increased_by_stage': (delta > 0).sum(0).tolist(),
        'decreased_by_stage': (delta < 0).sum(0).tolist(),
        'exactly_equal_by_stage': (delta == 0).sum(0).tolist(),
        'abs_delta_le_1e_minus_6_by_stage': (np.abs(delta) <= 1e-6).sum(0).tolist(),
        'mean_delta_by_stage': delta.mean(0).tolist() if n else [None]*3,
        'min_delta_by_stage': delta.min(0).tolist() if n else [None]*3,
        'max_delta_by_stage': delta.max(0).tolist() if n else [None]*3,
    }


def group_metrics(control, candidate, mask, *, include_ids=False):
    mask = np.asarray(mask)
    if mask.shape != (len(control['annotation_id']),) or mask.dtype != np.dtype(bool):
        raise ValueError('Malformed group mask')
    a, b = control['max_iou'][mask], candidate['max_iou'][mask]
    ta, tb = control['native_TP'][mask], candidate['native_TP'][mask]
    ids = control['annotation_id'][mask]
    initial_a, initial_b = a[:, 0, None] >= THRESHOLDS, b[:, 0, None] >= THRESHOLDS
    early_state = 2*initial_a.astype(np.int8) + initial_b.astype(np.int8)
    tp_state = 2*ta.astype(np.int8) + tb.astype(np.int8)
    table = np.stack([np.bincount(4*early_state[:, t]+tp_state[:, t], minlength=16).reshape(4, 4)
                      for t in range(10)])
    if not np.all(table.sum((1, 2)) == len(ids)):
        raise AssertionError('Transition table must account for every GT exactly once')
    by_threshold = {}
    for t, threshold in enumerate(THRESHOLDS):
        cell = {
            'initial_support_transition_by_native_TP_transition': table[t].tolist(),
            'continuous_geometry_delta_by_native_TP_transition': {
                name: continuous_summary((b-a)[tp_state[:, t] == state])
                for state, name in enumerate(TRANSITIONS)},
            'initial_support_gain_with_native_TP_loss': int(table[t, 1, 2]),
            'initial_support_loss_with_native_TP_gain': int(table[t, 2, 1]),
        }
        if include_ids:
            cell['initial_support_gain_with_native_TP_loss_ids'] = ids[(early_state[:, t] == 1) & (tp_state[:, t] == 2)].tolist()
            cell['initial_support_loss_with_native_TP_gain_ids'] = ids[(early_state[:, t] == 2) & (tp_state[:, t] == 1)].tolist()
        by_threshold[format(threshold, '.2f')] = cell
    return {'GT_count': len(ids), 'continuous_geometry_delta': continuous_summary(b-a),
            'by_IoU_threshold': by_threshold}


def audit_pair(control, candidate):
    validate_pair(control, candidate)
    n, area = len(control['annotation_id']), control['original_area']
    # These follow inclusive COCO GT area boundaries and can overlap at 1024/9216.
    # TP flags remain from all-area matching, never from size-specific evaluation.
    areas = {'small': (area >= 0) & (area <= 1024),
             'medium': (area >= 1024) & (area <= 9216),
             'large': (area >= 9216) & (area <= 1e10)}
    cohorts = {
        'control_native_TP_at_0_50': control['native_TP'][:, 0],
        'control_native_miss_at_0_50': ~control['native_TP'][:, 0],
        'control_initial_no_geometry_support_at_0_50': control['max_iou'][:, 0] < .5,
        'control_final_no_geometry_support_at_0_50': control['max_iou'][:, 2] < .5,
    }
    return {
        'transition_order_for_both_axes': list(TRANSITIONS),
        'geometry_stage_order': list(STAGES),
        'all': group_metrics(control, candidate, np.ones(n, dtype=bool), include_ids=True),
        'by_class': {str(c): group_metrics(control, candidate, control['category_id'] == c) for c in range(10)},
        'by_original_GT_area': {name: group_metrics(control, candidate, mask) for name, mask in areas.items()},
        'by_control_defined_cohort': {name: group_metrics(control, candidate, mask) for name, mask in cohorts.items()},
        'GT_area_boundary_counts': {'exactly_1024': int((area == 1024).sum()), 'exactly_9216': int((area == 9216).sum())},
        'group_scope': 'Cohorts may overlap; size groups condition all-area native TP on GT area. They are not COCO area-specific TP, APs/APm/APl or a partition at shared boundaries.',
        'interpretation': 'Descriptive paired changes, not causal mediation. Positive early IoU delta or support gain does not imply a final TP or AP gain.',
    }


def load_bound_stage():
    binding = read(HERE/'stage_binding.json')
    root = HERE.parent/binding['directory']
    for name, expected in binding['files_sha256'].items():
        if sha(root/name) != expected:
            raise ValueError('Bound stage implementation changed: ' + name)
    name = 'bound_fine_residual_stage_analysis_v1'
    spec = importlib.util.spec_from_file_location(name, root/'analyze_stages.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def analyze(experiment_root, endpoints, annotation_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    stage_api = load_bound_stage()
    # Original gate precedes creation: six native endpoints, bound v5, same EMA
    # epoch/evaluator/environment/initialization and intact checkpoint/stage files.
    original = stage_api.analyze(experiment_root, endpoints, annotation_path, output/'stage_analysis')
    arrays = {}
    for key in original['runs']:
        with np.load(output/'stage_analysis'/f'{key}_per_GT.npz', allow_pickle=False) as data:
            arrays[key] = {name: data[name].copy() for name in data.files}
    audits = {}
    for seed in (1, 2):
        candidate = arrays[f'fine_residual_seed{seed}']
        for arm in ('baseline', 'fused_residual'):
            key = f'fine_residual_minus_{arm}_seed{seed}'
            audit = audit_pair(arrays[f'{arm}_seed{seed}'], candidate)
            prior = original['paired_stage_changes'][key]['all']
            for t, threshold in enumerate(THRESHOLDS):
                table = np.asarray(audit['all']['by_IoU_threshold'][format(threshold, '.2f')]['initial_support_transition_by_native_TP_transition'])
                if (int(table[:, 1].sum()) != prior['new_native_TP'][t]
                        or int(table[:, 2].sum()) != prior['lost_native_TP'][t]
                        or int(table[1].sum()) != prior['new_geometric_support_by_stage_and_threshold'][0][t]):
                    raise ValueError('Outcome table differs from bound stage/native counts')
            audits[key] = audit
    report = {'status': 'OUTCOME_AUDIT_REQUIRES_NUMERICAL_REVIEW' if original['native_TP_geometry_disagreement_cells'] else 'OUTCOME_AUDIT_JOINED_TO_BOUND_STAGE_ANALYSIS',
              'stage_analysis_status': original['status'], 'comparisons': audits,
              'main_AP_results': original['native_endpoint_comparison'],
              'model_calls': 0, 'evaluator_calls': 0, 'native_TP_flags_recomputed': False,
              'automatic_adoption_decision': False, 'causal_mechanism_proven': False,
              'actual_six_run_results_required': True}
    write(output/'report.json', report)
    write(output/'manifest.json', {'code_sha256': sha(__file__), 'binding_sha256': sha(HERE/'stage_binding.json'),
                                  'files': {'report.json': sha(output/'report.json'),
                                            'stage_analysis/manifest.json': sha(output/'stage_analysis'/'manifest.json')}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--endpoints', nargs=6, required=True)
    parser.add_argument('--annotation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = analyze(args.experiment_root, args.endpoints, args.annotation, args.output)
    print(json.dumps({'status': report['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
