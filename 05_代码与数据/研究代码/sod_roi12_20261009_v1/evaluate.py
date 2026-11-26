"""ROI12 evaluation of saved predictions only; no model forward or training.

Use the SHA-bound original ``pycocoeval`` including its compiled _mask backend.
Never fall back to pycocotools or renumber GT id 0. Prediction input is either
COCO rows (xywh) or {image_id: {boxes: xyxy, scores, labels, query_ids?, rowids?}}.
Row IDs must originate from the frozen final output, not from re-sorting boxes.
Set ROI12_SOURCE_BINDINGS or place source_bindings_proposed.json beside this file.
All metrics in native_stats/stats are fractions; *_percent and deltas_pp are points.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


STAT_NAMES = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
              'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl')
AREAS = [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]
AREA_NAMES = ['all', 'small', 'medium', 'large']
MAX_EPOCHS = 12


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False,
                  default=lambda x: x.tolist())


def _bindings(path=None):
    explicit = path or os.environ.get('ROI12_SOURCE_BINDINGS')
    if explicit:
        source = Path(explicit).resolve()
    else:
        here = Path(__file__).resolve()
        candidates = [here.parent / 'source_bindings_proposed.json']
        candidates.extend(p / 'outputs' / 'sod_roi12_20261009' /
                          'source_bindings_proposed.json' for p in here.parents)
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError('Set ROI12_SOURCE_BINDINGS to the approved binding JSON')
    data = _read(source)
    if not data.get('seeds'):
        raise ValueError('No seeds in source binding')
    return data, source


def _activate_evaluator(evaluator_root=None):
    bindings, source = _bindings()
    expected = bindings['seeds'][0]['evaluator']['file_sha256']
    if any(s['evaluator']['package'] != 'pycocoeval' or
           s['evaluator']['file_sha256'] != expected for s in bindings['seeds']):
        raise ValueError('Ambiguous evaluator binding across seeds')
    if evaluator_root is not None:
        root = Path(evaluator_root).resolve()
        if root.name != 'pycocoeval' or not root.is_dir():
            raise ValueError('evaluator_root must be the actual pycocoeval package directory')
        loaded = sys.modules.get('pycocoeval')
        if loaded and Path(loaded.__file__).resolve().parent != root:
            raise RuntimeError('A different pycocoeval is already loaded')
        sys.path.insert(0, str(root.parent))
        importlib.invalidate_caches()
    else:
        spec = importlib.util.find_spec('pycocoeval')
        if spec is None or not spec.origin:
            raise ImportError('Original bound pycocoeval is required; no fallback allowed')
        root = Path(spec.origin).resolve().parent
    # Check source and binary bytes before importing them.
    for filename, digest in expected.items():
        if Path(filename).name != filename or _sha(root / filename) != digest:
            raise RuntimeError('Evaluator binding mismatch: ' + filename)
    coco_module = importlib.import_module('pycocoeval.coco')
    eval_module = importlib.import_module('pycocoeval.cocoeval')
    mask_module = importlib.import_module('pycocoeval.mask')
    backend = importlib.import_module('pycocoeval._mask')
    for module in (sys.modules['pycocoeval'], coco_module, eval_module, mask_module, backend):
        path = Path(module.__file__).resolve()
        if path.parent != root or path.name not in expected or _sha(path) != expected[path.name]:
            raise RuntimeError('Unexpected loaded evaluator module: ' + str(path))
    if eval_module.maskUtils is not mask_module or mask_module.iou is not backend.iou:
        raise RuntimeError('COCO bbox IoU is not using the bound native backend')
    fingerprint = {'package': 'pycocoeval', 'root': str(root), 'files': expected,
                   'binding_path': str(source), 'binding_sha256': _sha(source)}
    return coco_module.COCO, eval_module.COCOeval, fingerprint


def _tolist(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, 'tolist') else list(value)


def _integer(value, name):
    result = int(value)
    if isinstance(value, float) and (not math.isfinite(value) or value != result):
        raise ValueError('Noninteger ' + name)
    return result


def _xyxy_to_xywh(boxes):
    # Native convert_to_xywh subtracts in the tensor dtype, NOT Python float64.
    # JSON xyxy arrays are defined as native float32; other precisions should be
    # supplied as typed arrays/tensors or preconverted COCO xywh rows.
    if hasattr(boxes, 'detach'):
        boxes = boxes.detach().cpu()
    if not hasattr(boxes, 'dtype'):
        import numpy as np
        boxes = np.asarray(boxes, dtype=np.float32)
    if len(boxes) == 0:
        return []
    if len(boxes.shape) != 2 or boxes.shape[1] != 4:
        raise ValueError('Boxes must have shape [N,4]')
    corners, sizes = _tolist(boxes), _tolist(boxes[:, 2:] - boxes[:, :2])
    return [[b[0], b[1], size[0], size[1]] for b, size in zip(corners, sizes)]


def canonical_predictions(predictions):
    """Copy/validate predictions; retain fixed frame+rowid order and all scores.

    No NMS, clipping, quality rescore, top-k, or threshold is applied here.
    Explicit rows may use image_id/frame, rowid/row_id/output_index, query_id.
    """
    if isinstance(predictions, (str, Path)):
        predictions = _read(predictions)
    rows = []
    if isinstance(predictions, dict):
        for frame, item in predictions.items():
            if not item:
                continue
            boxes = _xyxy_to_xywh(item['boxes'])
            scores, labels = map(_tolist, (item['scores'], item['labels']))
            rowids = _tolist(item.get('rowids', item.get('output_indices', range(len(boxes)))))
            queries = _tolist(item['query_ids']) if 'query_ids' in item else [None] * len(boxes)
            if len({len(x) for x in (boxes, scores, labels, rowids, queries)}) != 1:
                raise ValueError('Mismatched prediction array lengths')
            for b, s, c, r, q in zip(boxes, scores, labels, rowids, queries):
                row = {'image_id': frame, 'category_id': c, 'score': s, 'rowid': r,
                       'bbox': b}
                if q is not None:
                    row['query_id'] = q
                rows.append(row)
    else:
        counters = {}
        for item in predictions:
            frame = item['image_id'] if 'image_id' in item else item['frame']
            frame = _integer(frame, 'image_id')
            ordinal = counters.get(frame, 0)
            counters[frame] = ordinal + 1
            row = dict(item)
            row['image_id'] = frame
            row['rowid'] = row.get('rowid', row.get('row_id', row.get('output_index', ordinal)))
            rows.append(row)
    result, seen, counts = [], set(), {}
    for row in rows:
        frame = _integer(row['image_id'], 'image_id')
        rowid = _integer(row['rowid'], 'rowid')
        cat = _integer(row['category_id'], 'category_id')
        score = float(row['score'])
        box = [float(x) for x in row['bbox']]
        if rowid < 0 or cat not in range(10) or len(box) != 4:
            raise ValueError('Invalid row id, category, or bbox shape')
        if not all(math.isfinite(x) for x in [score, *box]) or not 0 <= score <= 1:
            raise ValueError('Nonfinite bbox/score or score outside [0,1]')
        if box[2] < 0 or box[3] < 0:
            raise ValueError('Negative width/height; evaluator must not silently repair geometry')
        key = (frame, rowid)
        if key in seen:
            raise ValueError('Duplicate final output identity: ' + str(key))
        seen.add(key)
        counts[frame] = counts.get(frame, 0) + 1
        if counts[frame] > 300:
            raise ValueError('More than native top300 output items in a frame')
        normalized = {'image_id': frame, 'rowid': rowid, 'category_id': cat,
                      'score': score, 'bbox': box}
        if row.get('query_id') is not None:
            normalized['query_id'] = _integer(row['query_id'], 'query_id')
        result.append(normalized)
    return sorted(result, key=lambda row: (row['image_id'], row['rowid']))


def assert_frozen_identity(baseline, candidate, *, require_same_boxes=False, require_query_ids=False):
    """Reject additions/deletions/relabeling/rescoring/reindexing; boxes alone may vary."""
    left, right = canonical_predictions(baseline), canonical_predictions(candidate)
    if len(left) != len(right):
        raise AssertionError('Final output count changed')
    fields = ('image_id', 'rowid', 'category_id', 'score', 'query_id')
    for a, b in zip(left, right):
        if require_query_ids and ('query_id' not in a or 'query_id' not in b):
            raise AssertionError('Query identity verification requires query_id metadata')
        if any(a.get(k) != b.get(k) for k in fields):
            raise AssertionError('Frozen output identity changed at ' + str((a['image_id'], a['rowid'])))
        if require_same_boxes and a['bbox'] != b['bbox']:
            raise AssertionError('Reject-all/off path changed a box')
    return {'items': len(left), 'frames_with_predictions': len({r['image_id'] for r in left}),
            'identity_fields': list(fields), 'same_boxes_required': require_same_boxes,
            'query_id_verified': bool(left) and all('query_id' in r for r in left)}


def _extract_hits(evaluator, annotation):
    """Read the native dtMatches sentinel, NEVER reinterpret gtMatches[id=0]."""
    import numpy as np
    gt = {int(a['id']): a for a in annotation['annotations']}
    if len(gt) != len(annotation['annotations']):
        raise ValueError('Duplicate annotation IDs')
    ids = sorted(gt, key=lambda g: (int(gt[g]['image_id']), g))
    index = {g: i for i, g in enumerate(ids)}
    hits = np.zeros((len(ids), 10, 4), dtype=bool)
    eligible = np.zeros((len(ids), 4), dtype=bool)
    fp = np.zeros((10, 4), dtype=np.int64)
    cells, seen = set(), [set() for _ in AREAS]
    for item in evaluator.evalImgs:
        if item is None:
            continue
        area = AREAS.index(list(item['aRng']))
        frame, cat = int(item['image_id']), int(item['category_id'])
        cell = (frame, cat, area)
        if cell in cells or item['maxDet'] != 100 or len(item['dtIds']) > 100:
            raise ValueError('Duplicate matching cell or incorrect maxDet')
        cells.add(cell)
        g_ignore = {int(g): bool(i) for g, i in zip(item['gtIds'], item['gtIgnore'])}
        if len(g_ignore) != len(item['gtIds']):
            raise ValueError('Duplicate GT in native matching cell')
        for g, ignore in g_ignore.items():
            if g not in gt or g in seen[area] or (int(gt[g]['image_id']), int(gt[g]['category_id'])) != (frame, cat):
                raise ValueError('Foreign or repeated ground truth')
            seen[area].add(g)
            eligible[index[g], area] = not ignore
        matches = np.asarray(item['dtMatches'])
        ignored = np.asarray(item['dtIgnore'], dtype=bool)
        if matches.shape != (10, len(item['dtIds'])) or ignored.shape != matches.shape:
            raise ValueError('Malformed native detection matching arrays')
        if not np.isfinite(matches).all() or not np.equal(matches, np.floor(matches)).all():
            raise ValueError('Noninteger detection match identity')
        for t in range(10):
            fp[t, area] += int(((matches[t] == 0) & ~ignored[t]).sum())
            for value in matches[t][(matches[t] != 0) & ~ignored[t]]:
                g = int(value)
                if g not in g_ignore or g_ignore[g] or hits[index[g], t, area]:
                    raise ValueError('Invalid or repeated nonignored native TP')
                hits[index[g], t, area] = True
    if any(s != set(gt) for s in seen):
        raise ValueError('Incomplete per-GT evaluation coverage')
    return {'annotation_id': np.asarray(ids, dtype=np.int64),
            'image_id': np.asarray([gt[g]['image_id'] for g in ids], dtype=np.int64),
            'category_id': np.asarray([gt[g]['category_id'] for g in ids], dtype=np.int64),
            'area': np.asarray([gt[g]['area'] for g in ids], dtype=np.float64),
            'hits': hits[:, :, 0], 'hits_by_area': hits, 'eligible_by_area': eligible,
            'false_positives_by_area': fp, 'iou_thresholds': np.asarray(evaluator.params.iouThrs)}


def evaluate_predictions(annotation_path, predictions, output_dir, evaluator_root=None):
    """Run/export the bound original bbox evaluator and return summary.json content.

    score>.001 is the bound PostProcessor rule, applied once to the unchanged
    scores. All supplied identities, including filtered rows, remain in the audit.
    A subset annotation file is permitted for train-only calibration; full-val
    screening separately requires the bound annotation SHA via assert_bound_baseline.
    """
    import numpy as np
    annotation_path, output = Path(annotation_path), Path(output_dir)
    artifact_names = ('summary.json', 'predictions.json', 'precision_recall.npz',
                      'GT_hits.npz', 'native_matching.json.gz', 'manifest.json')
    if any((output / name).exists() for name in artifact_names):
        raise FileExistsError('Evaluation artifacts already exist: ' + str(output))
    annotation = _read(annotation_path)
    names = {int(c['id']): c['name'] for c in annotation['categories']}
    image_ids = sorted(int(i['id']) for i in annotation['images'])
    if sorted(names) != list(range(10)) or len(image_ids) != len(set(image_ids)):
        raise ValueError('Expected ten original categories 0..9 and unique images')
    if any(int(g['category_id']) not in names or int(g['image_id']) not in image_ids
           for g in annotation['annotations']):
        raise ValueError('Annotation outside evaluation population')
    rows = canonical_predictions(predictions)
    if any(r['image_id'] not in image_ids for r in rows):
        raise ValueError('Prediction references an image outside annotation file')
    COCO, COCOeval, fingerprint = _activate_evaluator(evaluator_root)
    bindings, _ = _bindings()
    score_threshold = float(bindings['evaluation_binding']['native_score_threshold'])
    if score_threshold != 0.001:
        raise ValueError('Native score contract changed')
    results = [{k: r[k] for k in ('image_id', 'category_id', 'bbox', 'score')}
               for r in rows if r['score'] > score_threshold]
    coco_gt = COCO(str(annotation_path))
    if results:
        coco_dt = coco_gt.loadRes(copy.deepcopy(results))
    else:
        # Native loadRes indexes anns[0]; this is the original empty-result path.
        coco_dt = COCO()
        coco_dt.dataset = {'images': copy.deepcopy(annotation['images']),
                           'categories': copy.deepcopy(annotation['categories']), 'annotations': []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox', using_tiny_metrice=False,
                         using_vtiny_metrice=False, max_dets=100)
    p = evaluator.params
    if (p.maxDets != [1, 10, 100] or list(p.catIds) != list(range(10)) or
            list(p.imgIds) != image_ids or list(p.areaRng) != AREAS or
            list(p.areaRngLbl) != AREA_NAMES or p.useCats != 1 or
            not np.allclose(p.iouThrs, np.linspace(.5, .95, 10), atol=1e-12, rtol=0) or
            not np.allclose(p.recThrs, np.linspace(0, 1, 101), atol=1e-12, rtol=0)):
        raise ValueError('Native evaluator parameters differ from standard bound contract')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision, recall = np.asarray(evaluator.eval['precision']), np.asarray(evaluator.eval['recall'])
    stats = np.asarray(evaluator.stats, dtype=float)
    if precision.shape != (10, 101, 10, 4, 3) or recall.shape != (10, 10, 4, 3) or stats.shape != (12,):
        raise ValueError('Unexpected metric shapes')
    if not all(np.isfinite(a).all() for a in (precision, recall, stats)):
        raise ValueError('Nonfinite COCO metrics')
    def average(values):
        valid = values[values > -1]
        return float(valid.mean() * 100) if valid.size else None
    per_class = [{'class_id': c, 'name': names[c], 'AP': average(precision[:, :, c, 0, 2]),
                  'APs': average(precision[:, :, c, 1, 2]),
                  'AP75': average(precision[5, :, c, 0, 2])} for c in range(10)]
    checks = [average(precision[:, :, :, 0, 2]), average(precision[0, :, :, 0, 2]),
              average(precision[5, :, :, 0, 2]),
              *[average(precision[:, :, :, a, 2]) for a in (1, 2, 3)]]
    for i, value in enumerate(checks):
        expected = -1.0 if value is None else value / 100
        if abs(expected - stats[i]) > 1e-12:
            raise ValueError('Saved precision does not reproduce native stats')
    flags = _extract_hits(evaluator, annotation)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'precision_recall.npz', precision=precision, recall=recall,
                        iou_thresholds=p.iouThrs, recall_thresholds=p.recThrs,
                        category_ids=np.asarray(p.catIds), area_ranges=np.asarray(p.areaRng),
                        max_dets=np.asarray(p.maxDets))
    np.savez_compressed(output / 'GT_hits.npz', **flags)
    with gzip.open(output / 'native_matching.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(evaluator.evalImgs, stream, allow_nan=False, default=lambda x: x.tolist())
    _write(output / 'predictions.json', rows)
    protocol = {'category_ids': list(range(10)), 'image_ids': image_ids, 'iou_type': 'bbox',
                'iou_thresholds': list(map(float, p.iouThrs)), 'max_dets': list(p.maxDets),
                'area_ranges': AREAS, 'area_labels': AREA_NAMES, 'score_threshold': score_threshold,
                'recall_thresholds': list(map(float, p.recThrs)),
                'prediction_order': 'image_id,rowid; original score stable ties',
                'GT_zero_policy': 'native dtMatches==0 means unmatched; never renumbered'}
    summary = {'status': 'EVALUATED_SAVED_PREDICTIONS_WITH_BOUND_NATIVE_EVALUATOR',
               'output_dir': str(output.resolve()), 'annotation_sha256': _sha(annotation_path),
               'annotation_path': str(annotation_path.resolve()), 'evaluator_source': fingerprint,
               'protocol': protocol, 'native_stats': stats.tolist(),
               'stats': dict(zip(STAT_NAMES, stats.tolist())),
               'overall_percent': {k: float(v * 100) for k, v in zip(STAT_NAMES, stats)},
               'categories_percent': per_class, 'input_items': len(rows), 'evaluated_items': len(results),
               'GT_count': len(flags['annotation_id']), 'annotation_zero_present': bool(0 in flags['annotation_id']),
               'TP_by_IoU_all_area': flags['hits'].sum(axis=0).tolist(),
               'FP_by_IoU_all_area': flags['false_positives_by_area'][:, 0].tolist(),
               'independent_baseline_reproduction_verified': False}
    _write(output / 'summary.json', summary)
    _write(output / 'manifest.json', {'files': {name: _sha(output / name) for name in artifact_names
                                              if name != 'manifest.json'},
                                      'evaluate_code_sha256': _sha(__file__)})
    return summary


def _load_endpoint(directory):
    import numpy as np
    root = Path(directory)
    manifest = _read(root / 'manifest.json')
    expected = {'summary.json', 'predictions.json', 'precision_recall.npz', 'GT_hits.npz', 'native_matching.json.gz'}
    if set(manifest['files']) != expected:
        raise ValueError('Incomplete evaluation artifact set')
    for name, digest in manifest['files'].items():
        if _sha(root / name) != digest:
            raise ValueError('Evaluation artifact was changed: ' + name)
    if manifest['evaluate_code_sha256'] != _sha(__file__):
        raise ValueError('Evaluation used a different export implementation')
    with np.load(root / 'GT_hits.npz', allow_pickle=False) as values:
        flags = {name: values[name] for name in values.files}
    return _read(root / 'summary.json'), flags, _read(root / 'predictions.json')


def assert_bound_baseline(summary, bindings_path=None, *, seed=1, tolerance_pp=0.0001):
    """Check full-val checkpoint-reference AP values; no historical 24-epoch constants."""
    if not math.isfinite(tolerance_pp) or tolerance_pp < 0:
        raise ValueError('Invalid reproduction tolerance')
    bindings, source = _bindings(bindings_path)
    record = next(s for s in bindings['seeds'] if s['seed'] == seed)
    if summary['annotation_sha256'] != record['data']['val']['ann_sha256']:
        raise ValueError('Reference check requires the bound FULL validation annotations')
    if summary['evaluator_source']['files'] != record['evaluator']['file_sha256']:
        raise ValueError('Reference check uses a different evaluator')
    reference = record['frozen_detector_proposed']['metrics_percent']
    errors = {k: summary['overall_percent'][k] - value for k, value in reference.items()}
    if any(not math.isfinite(v) or abs(v) > tolerance_pp for v in errors.values()):
        raise ValueError('Frozen baseline numerical reproduction failed: ' + repr(errors))
    return {'seed': seed, 'reference_checkpoint_sha256': record['frozen_detector_proposed']['sha256'],
            'reference_binding_sha256': _sha(source), 'errors_pp': errors,
            'tolerance_pp': tolerance_pp, 'checkpoint_identity_must_also_be_verified_by_runner': True}


def screening_decision(baseline, candidate, epoch):
    """Pre-registered final rule: APs increases strictly AND overall AP does not fall."""
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch <= MAX_EPOCHS:
        raise ValueError('Global epoch must be 0..12; no phase counter reset')
    ap_delta = float(candidate['AP']) - float(baseline['AP'])
    aps_delta = float(candidate['APs']) - float(baseline['APs'])
    if not all(math.isfinite(v) for v in (ap_delta, aps_delta)):
        raise ValueError('Nonfinite screening metrics')
    passed = epoch == MAX_EPOCHS and aps_delta > 0 and ap_delta >= 0
    return {'epoch': epoch, 'passed': passed if epoch == MAX_EPOCHS else None,
            'status': ('PASS_SHORT_SCREEN_STOP_AT_CAP' if passed else 'STOP_NO_EXTENSION')
                      if epoch == MAX_EPOCHS else 'INTERIM_NOT_FINAL_SCREEN',
            'delta_AP': ap_delta, 'delta_APs': aps_delta,
            'rule': 'epoch12 APs>same-protocol baseline AND AP>=baseline; unrounded values',
            'maximum_epochs_per_configuration': MAX_EPOCHS,
            'additional_training_authorized_by_this_result': False}


def compare_evaluations(baseline_dir, candidate_dir, output_path=None, *, epoch=12,
                        bindings_path=None, seed=1, verify_reference=True):
    """Pair native per-GT hits, preserving the native id0 sentinel and fixed outputs.

    small_original_area_lt1024 filters ALL-area native matches by original GT area;
    native_small_matching additionally exposes area-specific matching (not AP itself).
    verify_reference=False is solely for train-internal calibration/diagnostics and
    never yields a final-screen pass.
    """
    import numpy as np
    left, lf, lp = _load_endpoint(baseline_dir)
    right, rf, rp = _load_endpoint(candidate_dir)
    if left['annotation_sha256'] != right['annotation_sha256'] or left['protocol'] != right['protocol']:
        raise ValueError('Paired annotation populations or COCO protocols differ')
    if left['evaluator_source']['files'] != right['evaluator_source']['files']:
        raise ValueError('Paired evaluators differ')
    identity = assert_frozen_identity(lp, rp)
    for name in ('annotation_id', 'image_id', 'category_id', 'area', 'eligible_by_area', 'iou_thresholds'):
        if not np.array_equal(lf[name], rf[name]):
            raise ValueError('Paired GT identities/areas/eligibility differ: ' + name)
    for flags in (lf, rf):
        if flags['hits_by_area'].shape != (len(flags['annotation_id']), 10, 4) or flags['hits_by_area'].dtype != bool:
            raise ValueError('Malformed per-GT hit flags')
    changes = {}
    def recount(mask, area):
        result = {'GT_count': int(mask.sum()), 'matching_area': AREA_NAMES[area], 'thresholds': {}}
        for threshold, t in (('0.50', 0), ('0.75', 5)):
            a, b = lf['hits_by_area'][:, t, area], rf['hits_by_area'][:, t, area]
            gained, lost = mask & ~a & b, mask & a & ~b
            result['thresholds'][threshold] = {
                'baseline_hits': int((mask & a).sum()), 'candidate_hits': int((mask & b).sum()),
                'new': int(gained.sum()), 'lost': int(lost.sum()),
                'net': int(gained.sum() - lost.sum()), 'kept': int((mask & a & b).sum()),
                'new_GT_ids': lf['annotation_id'][gained].tolist(),
                'lost_GT_ids': lf['annotation_id'][lost].tolist()}
        return result
    valid = lf['eligible_by_area'][:, 0]
    changes['all'] = recount(valid, 0)
    changes['small_original_area_lt1024'] = recount(valid & (lf['area'] < 1024), 0)
    changes['native_small_matching_original_area_lt1024'] = recount(
        lf['eligible_by_area'][:, 1] & (lf['area'] < 1024), 1)
    reference = assert_bound_baseline(left, bindings_path, seed=seed) if verify_reference else None
    decision = screening_decision(left['stats'], right['stats'], epoch)
    if not verify_reference:
        decision.update(passed=None, status='DIAGNOSTIC_ONLY_REFERENCE_NOT_VERIFIED')
    report = {'identity_check': identity, 'reference_reproduction': reference,
              'baseline_dir': str(Path(baseline_dir).resolve()),
              'candidate_dir': str(Path(candidate_dir).resolve()),
              'delta_metrics_pp': {k: 100 * (right['stats'][k] - left['stats'][k]) for k in STAT_NAMES},
              'GT_changes': changes, 'screening_decision': decision,
              'GT_zero_policy': left['protocol']['GT_zero_policy'],
              'scope': 'Native per-GT matches, not geometric maximum-IoU support; small<1024 uses original annotation area'}
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        _write(output_path, report)
    return report
