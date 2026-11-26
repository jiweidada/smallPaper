"""CPU-only REAL-pycocoeval contract fixture; never imports/runs a detector.

Example (use the already approved binding; this test never manufactures one):
  ROI12_SOURCE_BINDINGS=/path/source_bindings_proposed.json \
    python evaluate_contract_test.py --output-dir /path/new_fixture_directory

Optional --evaluator-root is the actual bound pycocoeval package directory.
Requires NumPy and the original ABI-compatible pycocoeval binary. No torch/GPU.
Synthetic results are diagnostic fixtures, never experimental performance.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

import numpy as np

import evaluate as api


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_rejection(callback, label, exception=(AssertionError, ValueError)):
    try:
        callback()
    except exception:
        return
    raise AssertionError('Expected rejection: ' + label)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def fixture():
    """GT0, strict-small, area==1024, medium, large, crowd, and score ties."""
    definitions = [
        (0, 11, 0, [10., 10., 20., 20.]),
        (1, 11, 1, [50., 50., 20., 20.]),
        (2, 11, 2, [100., 10., 32., 32.]),
        (3, 11, 3, [170., 10., 64., 64.]),
        (4, 11, 4, [270., 10., 100., 100.]),
        (5, 29, 5, [10., 10., 12., 12.]),
        (6, 29, 6, [40., 10., 20., 20.]),
        (7, 29, 7, [80., 10., 32., 32.]),
        (8, 29, 8, [140., 10., 40., 40.]),
        (9, 29, 9, [220., 10., 110., 110.]),
    ]
    annotations, predictions = [], []
    rowids = {}
    for gid, frame, category, box in definitions:
        annotations.append(dict(id=gid, image_id=frame, category_id=category,
                                bbox=box, area=box[2] * box[3], iscrowd=0))
        row = rowids.get(frame, 0)
        rowids[frame] = row + 1
        predictions.append(dict(image_id=frame, rowid=row, query_id=100 + gid,
                                category_id=category, score=0.9 - row * 0.01,
                                bbox=list(box)))
    annotations.append(dict(id=10, image_id=11, category_id=1,
                            bbox=[10., 250., 50., 50.], area=2500., iscrowd=1))
    predictions.extend([
        dict(image_id=11, rowid=5, query_id=200, category_id=1, score=.7,
             bbox=[10., 250., 50., 50.]),  # genuine native ignored crowd match
        dict(image_id=11, rowid=6, query_id=101, category_id=2, score=.2,
             bbox=[400., 400., 10., 10.]),  # multi-class query retained, false box
        dict(image_id=11, rowid=7, query_id=201, category_id=3, score=.001,
             bbox=[410., 410., 10., 10.]),  # exactly native threshold: excluded
        dict(image_id=11, rowid=8, query_id=202, category_id=1, score=.89,
             bbox=[420., 420., 10., 10.]),  # tied score after the true row1 box
    ])
    annotation = dict(info={'description': 'SYNTHETIC ROI12 evaluation contract ONLY'},
                      licenses=[],
                      images=[dict(id=i, width=512, height=512, file_name=f'fixture_{i}.png')
                              for i in (11, 29)],
                      categories=[dict(id=i, name=f'fixture_class_{i}') for i in range(10)],
                      annotations=annotations)
    return annotation, predictions


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def zero_changes(report):
    require(all(v == 0 for v in report['delta_metrics_pp'].values()), 'Same metrics must be EXACT')
    for group in report['GT_changes'].values():
        for counts in group['thresholds'].values():
            require(counts['new'] == counts['lost'] == counts['net'] == 0,
                    'Identical predictions have nonzero GT changes')
    require(report['screening_decision']['passed'] is None,
            'Synthetic/calibration comparison must never qualify as experiment pass')


def run(output_dir, evaluator_root=None):
    binding = os.environ.get('ROI12_SOURCE_BINDINGS')
    if not binding or not Path(binding).is_file():
        raise RuntimeError('Set ROI12_SOURCE_BINDINGS to the CURRENT approved real evaluator binding')
    require(Path(api.__file__).resolve() == Path(__file__).with_name('evaluate.py').resolve(),
            'Imported an unrelated evaluate module')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    annotation, predictions = fixture()
    annotation_path = output / 'fixture_annotations.json'
    write_json(annotation_path, annotation)
    checks = []

    # Both calls use the actual binary-bound original evaluator. Sorting by fixed
    # frame/rowid must make serialized-list order immaterial, including score ties.
    baseline = api.evaluate_predictions(annotation_path, predictions, output / 'baseline', evaluator_root)
    same = api.evaluate_predictions(annotation_path, list(reversed(copy.deepcopy(predictions))),
                                    output / 'same', evaluator_root)
    require(baseline['native_stats'] == same['native_stats'], 'Repeated native stats differ')
    require(baseline['categories_percent'] == same['categories_percent'], 'Per-class metrics differ')
    require(len(baseline['native_stats']) == 12, 'Missing native twelve statistics')
    require(baseline['protocol']['max_dets'] == [1, 10, 100], 'Wrong maxDets')
    require(baseline['evaluated_items'] == len(predictions) - 1, 'score>.001 rule changed')
    for name in ('precision_recall.npz', 'GT_hits.npz'):
        a, b = load_npz(output / 'baseline' / name), load_npz(output / 'same' / name)
        require(a.keys() == b.keys(), 'Changed NPZ fields')
        for key in a:
            require(np.array_equal(a[key], b[key]), 'Repeat arrays differ: ' + name + '/' + key)
    report = api.compare_evaluations(output / 'baseline', output / 'same',
                                     output / 'same_pair.json', epoch=12, verify_reference=False)
    zero_changes(report)
    stored_rows = json.loads((output / 'baseline' / 'predictions.json').read_text())
    keys = [(r['image_id'], r['rowid']) for r in stored_rows]
    require(keys == sorted(keys), 'Canonical frame+rowid ordering not saved')
    require(sum(r['query_id'] == 101 for r in stored_rows) == 2, 'Multi-class query was deleted')
    checks.append('real_evaluator_exact_repeat_zero_gain_harm_and_frame_rowid_order')

    flags = load_npz(output / 'baseline' / 'GT_hits.npz')
    locate = {int(g): i for i, g in enumerate(flags['annotation_id'])}
    require(not flags['hits'][locate[0]].any(), 'GT id0 sentinel was reinterpreted as a true hit')
    require(flags['hits'][locate[1]].all(), 'Ordinary exact GT match must be detected at all thresholds')
    require(not flags['eligible_by_area'][locate[10]].any(), 'Native crowd-ignore behavior changed')
    require(not flags['hits'][locate[10]].any(), 'Ignored crowd GT counted as ordinary TP')
    for boundary_id in (2, 7):
        j = locate[boundary_id]
        require(flags['area'][j] == 1024, 'Boundary area missing')
        require(flags['eligible_by_area'][j, 1] and flags['eligible_by_area'][j, 2],
                'Native inclusive small/medium boundary was altered')
    require(int(flags['eligible_by_area'][:, 1].sum()) == 6, 'Unexpected native-small denominator')
    require(report['GT_changes']['small_original_area_lt1024']['GT_count'] == 4,
            'Strict diagnostic area<1024 wrongly includes boundary GT')
    require(report['GT_changes']['all']['GT_count'] == 10, 'Ignored GT included in all denominator')
    require(report['GT_changes']['all']['thresholds']['0.75']['baseline_hits'] == 9,
            'Real native TP count must preserve GT0 loss')
    checks.append('real_GT0_sentinel_crowd_and_strict_vs_native_small_area_boundary')

    # Genuine geometry-only harm; reversing the pair is a genuine rescue. This
    # verifies exported match flags rather than merely checking equality to self.
    damaged = copy.deepcopy(predictions)
    victim = next(r for r in damaged if r['image_id'] == 11 and r['rowid'] == 1)
    victim['bbox'] = [380., 20., 20., 20.]
    api.assert_frozen_identity(predictions, damaged, require_query_ids=True)
    api.evaluate_predictions(annotation_path, damaged, output / 'damaged', evaluator_root)
    harm = api.compare_evaluations(output / 'baseline', output / 'damaged',
                                   output / 'harm_pair.json', epoch=12, verify_reference=False)
    gain = api.compare_evaluations(output / 'damaged', output / 'baseline',
                                   output / 'gain_pair.json', epoch=12, verify_reference=False)
    for threshold in ('0.50', '0.75'):
        for group in ('all', 'small_original_area_lt1024'):
            lost = harm['GT_changes'][group]['thresholds'][threshold]
            won = gain['GT_changes'][group]['thresholds'][threshold]
            require(lost['lost_GT_ids'] == [1] and lost['new'] == 0 and lost['net'] == -1,
                    'Expected exactly one true native loss')
            require(won['new_GT_ids'] == [1] and won['lost'] == 0 and won['net'] == 1,
                    'Expected exactly one true native rescue')
    checks.append('real_geometry_only_gain_and_harm_use_same_GT_identity')

    for field, value in [('score', .123), ('category_id', 9), ('query_id', 999), ('rowid', 999)]:
        bad = copy.deepcopy(predictions)
        bad[0][field] = value
        expect_rejection(lambda bad=bad: api.assert_frozen_identity(predictions, bad, require_query_ids=True), field)
    reordered = copy.deepcopy(predictions)
    reordered[0]['rowid'], reordered[1]['rowid'] = reordered[1]['rowid'], reordered[0]['rowid']
    expect_rejection(lambda: api.assert_frozen_identity(predictions, reordered, require_query_ids=True),
                     'semantic output-order permutation')
    expect_rejection(lambda: api.assert_frozen_identity(predictions, predictions[:-1]), 'item deletion')
    expect_rejection(lambda: api.assert_frozen_identity(predictions, damaged, require_same_boxes=True),
                     'off/reject-all path geometry change')
    checks.append('score_class_query_rowid_order_count_and_reject_all_mutations_rejected')

    # A filtered calibration JSON keeps ALL ten categories and original IDs/area.
    # Predictions must be filtered to its images; zero-GT categories remain valid.
    subset = copy.deepcopy(annotation)
    subset['images'] = [i for i in subset['images'] if i['id'] == 29]
    subset['annotations'] = [g for g in subset['annotations'] if g['image_id'] == 29]
    subset_path = output / 'calibration_C_annotations.json'
    write_json(subset_path, subset)
    c_predictions = [r for r in predictions if r['image_id'] == 29]
    c_summary = api.evaluate_predictions(subset_path, c_predictions, output / 'calibration_C', evaluator_root)
    require(c_summary['protocol']['image_ids'] == [29], 'Calibration evaluator leaked full image population')
    require(c_summary['GT_count'] == 5, 'Calibration GT subset was not preserved')
    require(all(c_summary['categories_percent'][i]['AP'] is None for i in range(5)),
            'Absent calibration classes must have null AP, not fabricated zero')
    c_pair = api.compare_evaluations(output / 'calibration_C', output / 'calibration_C',
                                     output / 'calibration_pair.json', verify_reference=False)
    zero_changes(c_pair)
    expect_rejection(lambda: api.evaluate_predictions(subset_path, predictions,
                                                       output / 'invalid_foreign_images', evaluator_root),
                     'non-C prediction image')
    pruned = copy.deepcopy(subset)
    pruned['categories'] = [c for c in pruned['categories'] if c['id'] >= 5]
    pruned_path = output / 'invalid_pruned_categories.json'
    write_json(pruned_path, pruned)
    expect_rejection(lambda: api.evaluate_predictions(pruned_path, c_predictions,
                                                       output / 'invalid_categories', evaluator_root),
                     'calibration metadata missing original ten categories')
    expect_rejection(lambda: api.assert_bound_baseline(c_summary), 'C cannot pass full-val reference check')
    checks.append('real_filtered_C_annotation_with_ten_categories_and_no_reference_bypass')

    base_metrics = {'AP': .25, 'APs': .15}
    require(api.screening_decision(base_metrics, {'AP': .25, 'APs': .1500000001}, 12)['passed'],
            'Strict positive unrounded APs improvement must pass when AP is unchanged')
    for candidate in ({'AP': .25, 'APs': .15}, {'AP': .26, 'APs': .14}, {'AP': .249, 'APs': .16}):
        require(api.screening_decision(base_metrics, candidate, 12)['passed'] is False,
                'Endpoint passed without APs>base AND AP>=base')
    for epoch in (0, 8, 11):
        require(api.screening_decision(base_metrics, {'AP': .3, 'APs': .2}, epoch)['passed'] is None,
                'Interim checkpoint selected as final pass')
    expect_rejection(lambda: api.screening_decision(base_metrics, base_metrics, 13), 'epoch cap')
    require(not api.screening_decision(base_metrics, {'AP': .3, 'APs': .2}, 12)
            ['additional_training_authorized_by_this_result'], 'Passing screen must not authorize extension')
    checks.append('fixed_epoch12_unrounded_screen_rule_and_no_extension')

    receipt = {'status': 'PASS_REAL_NATIVE_EVALUATOR_SYNTHETIC_CPU_FIXTURE',
               'not_experimental_results': True, 'no_model_forward': True, 'checks': checks,
               'evaluate_code_sha256': api._sha(api.__file__),
               'test_code_sha256': api._sha(__file__), 'binding_sha256': api._sha(binding),
               'evaluator_source': baseline['evaluator_source'], 'output_dir': str(output),
               'calibration_API_restrictions': [
                   'Keep categories 0..9 in C JSON, even if some have no GT.',
                   'Keep original GT IDs and area; filter images/annotations/predictions to C.',
                   'Use a fresh output directory per tau/evaluation; existing artifacts are refused.',
                   'Use compare_evaluations(..., verify_reference=False) for C; never a final pass.',
                   'No automatic threshold scan in this API; runner selects tau only on C.',
                   'No binding rewrite or full-val reference bypass is needed for C evaluation.',
               ]}
    write_json(output / 'contract_test_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='New, nonexistent directory; artifacts are retained')
    parser.add_argument('--evaluator-root', type=Path, default=os.environ.get('ROI12_EVALUATOR_ROOT'))
    args = parser.parse_args()
    if args.output_dir is None:
        # mkdtemp creates a parent; run owns a new child to enforce no overwrites.
        args.output_dir = Path(tempfile.mkdtemp(prefix='roi12_evaluator_contract_')) / 'fixture'
    receipt = run(args.output_dir, args.evaluator_root)
    print(json.dumps({'status': receipt['status'], 'checks': len(receipt['checks']),
                      'receipt': str(args.output_dir / 'contract_test_receipt.json')}, indent=2))


if __name__ == '__main__':
    main()
