"""Synthetic outcome counterexamples only; never read old research arrays."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import analyze_outcomes as a

HERE = Path(__file__).resolve().parent
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def rejects(name, call):
    try:
        call()
    except (ValueError, RuntimeError):
        checks.append(name)
        return
    raise AssertionError('Expected rejection: ' + name)


def main():
    # Each early-support transition occurs once with each native TP transition.
    c = {'annotation_id': np.arange(100, 116), 'image_id': np.zeros(16, dtype=int),
         'category_id': np.arange(16) % 9,
         'original_area': np.array([1023, 1024, 1025, 9215, 9216, 9217]+[10]*10, dtype=float),
         'max_iou': np.tile([.1, .8, 1.], (16, 1)), 'native_TP': np.zeros((16, 10), dtype=bool)}
    d = deepcopy(c)
    for early in range(4):
        for tp in range(4):
            i = early*4 + tp
            c['max_iou'][i, 0] = 1. if early//2 else .1
            d['max_iou'][i, 0] = 1. if early % 2 else .1
            c['native_TP'][i] = bool(tp//2)
            d['native_TP'][i] = bool(tp % 2)
    saved_c, saved_d = deepcopy(c), deepcopy(d)
    report = a.audit_pair(c, d)
    for threshold in ('0.50', '0.95'):
        item = report['all']['by_IoU_threshold'][threshold]
        check('all sixteen distinct transitions remain separate at '+threshold,
              np.array_equal(item['initial_support_transition_by_native_TP_transition'], np.ones((4, 4), dtype=int)))
        check('early improvement with lost native TP explicitly identified at '+threshold,
              item['initial_support_gain_with_native_TP_loss'] == 1
              and item['initial_support_gain_with_native_TP_loss_ids'] == [106])
        check('early damage with gained native TP does not become mechanism proof at '+threshold,
              item['initial_support_loss_with_native_TP_gain_ids'] == [109])
    check('every threshold accounts for all GT exactly once', all(
        np.asarray(x['initial_support_transition_by_native_TP_transition']).sum() == 16
        for x in report['all']['by_IoU_threshold'].values()))
    check('native gains and losses equal known independent construction', all(
        np.asarray(x['initial_support_transition_by_native_TP_transition']).sum(0).tolist() == [4]*4
        for x in report['all']['by_IoU_threshold'].values()))
    check('area boundaries retain native overlap instead of silent reassignment',
          [x['GT_count'] for x in report['by_original_GT_area'].values()] == [12, 4, 2]
          and report['GT_area_boundary_counts'] == {'exactly_1024': 1, 'exactly_9216': 1})
    check('old success and old miss form distinct fixed baseline cohorts',
          report['by_control_defined_cohort']['control_native_TP_at_0_50']['GT_count'] == 8
          and report['by_control_defined_cohort']['control_native_miss_at_0_50']['GT_count'] == 8)
    check('empty class and empty failure cohort remain well-defined',
          report['by_class']['9']['continuous_geometry_delta']['mean_delta_by_stage'] == [None]*3
          and report['by_control_defined_cohort']['control_final_no_geometry_support_at_0_50']['GT_count'] == 0)
    check('class counts exactly partition all targets', sum(x['GT_count'] for x in report['by_class'].values()) == 16)
    check('control and candidate input arrays are unchanged', all(
        np.array_equal(before[k], after[k]) for before, after in ((saved_c, c), (saved_d, d)) for k in before))

    # Equal mean improvement can conceal damage to one target and benefit to another.
    u = {k: v[:2].copy() for k, v in c.items()}
    v = deepcopy(u)
    u['max_iou'][:, 0] = [.125, .25]
    v['max_iou'][:, 0] = [.25, .125]
    u['native_TP'][:] = v['native_TP'][:] = True
    detail = a.audit_pair(u, v)['all']['by_IoU_threshold']['0.50']
    continuous = detail['continuous_geometry_delta_by_native_TP_transition']['present_in_both']
    check('zero mean does not conceal opposite continuous geometry changes',
          continuous['mean_delta_by_stage'][0] == 0
          and continuous['increased_by_stage'][0] == continuous['decreased_by_stage'][0] == 1
          and continuous['min_delta_by_stage'][0] == -.125 and continuous['max_delta_by_stage'][0] == .125)
    check('within-threshold continuous improvement is not counted as support gain',
          np.asarray(detail['initial_support_transition_by_native_TP_transition'])[1].sum() == 0)
    u['annotation_id'][0] = v['annotation_id'][0] = 0
    u['native_TP'][0] = v['native_TP'][0] = False
    zero = a.audit_pair(u, v)
    check('native false TP for annotation zero preserved',
          np.asarray(zero['all']['by_IoU_threshold']['0.50']['initial_support_transition_by_native_TP_transition'])[:, 0].sum() == 1
          and not v['native_TP'][0].any())

    malformed = deepcopy(d); malformed['annotation_id'][0] = 999
    rejects('mismatched population rejected', lambda: a.audit_pair(c, malformed))
    malformed = deepcopy(d); malformed['native_TP'] = malformed['native_TP'].astype(int)
    rejects('integer pseudo TP flags rejected', lambda: a.audit_pair(c, malformed))
    malformed = deepcopy(d); malformed['max_iou'][0, 0] = np.nan
    rejects('nonfinite geometry rejected', lambda: a.audit_pair(c, malformed))
    malformed = deepcopy(d); malformed['max_iou'][0, 0] = 1.01
    rejects('out of range geometry rejected', lambda: a.audit_pair(c, malformed))
    x, y = deepcopy(c), deepcopy(d); x['annotation_id'][1] = y['annotation_id'][1] = 100
    rejects('duplicate GT identities rejected', lambda: a.audit_pair(x, y))
    rejects('nonboolean cohort mask rejected', lambda: a.group_metrics(c, d, np.arange(16)))
    check('full report is finite JSON', bool(json.dumps(report, allow_nan=False)))
    with tempfile.TemporaryDirectory(prefix='outcome_gate_') as temporary:
        output = Path(temporary)/'result'
        rejects('real bound v5 gate rejects missing six-run results', lambda: a.analyze(
            HERE.parent/'fine_residual_geometry_20261008_v5', [], Path(temporary)/'unused.json', output))
        check('failed experiment eligibility leaves no output', not output.exists())
    check('tests imported neither torch nor evaluator', 'torch' not in sys.modules
          and not any(k.startswith('pycocoeval') for k in sys.modules))
    result = {'status': 'PASS_SYNTHETIC_OUTCOME_COUNTEREXAMPLES', 'checks': checks, 'check_count': len(checks),
              'model_calls': 0, 'evaluator_calls': 0, 'optimizer_or_parameter_updates': 0,
              'real_dataset_arrays_read': 0, 'real_research_images_read': 0, 'remote_accesses': 0,
              'formal_six_run_end_to_end_execution_verified': False,
              'input_code_sha256': {p.name: a.sha(p) for p in (HERE/'analyze_outcomes.py', Path(__file__), HERE/'stage_binding.json')}}
    a.write(HERE/'engineering_synthetic.json', result)
    print(json.dumps({'status': result['status'], 'checks': len(checks)}))


if __name__ == '__main__':
    main()
