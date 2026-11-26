"""Read/capture already computed evaluation results. Never runs a model or evaluator."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ARMS = ('baseline', 'fused_residual', 'fine_residual')
METRICS = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=json_default)


def capture(evaluator, test_stats, identity, annotation_path, output, *, epoch,
            mode='native_runtime', environment=None):
    """Export a completed COCO evaluation without calling evaluate/accumulate/summarize.

    archived_fixture is deliberately ineligible for a real six-run comparison.
    Native runtime capture is observation, not independent numerical reproduction.
    """
    import numpy as np
    if epoch != 131 or mode not in ('native_runtime', 'archived_fixture'):
        raise ValueError('Only fixed final epoch 132 (zero-based 131) is accepted')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if identity['arm'] not in ARMS or identity['seed'] not in (1, 2):
        raise ValueError('Unknown experiment identity')
    annotation_path = Path(annotation_path)
    if sha(annotation_path) != identity['val_annotation_sha256']:
        raise ValueError('Annotation fingerprint differs from experiment identity')
    annotations = read(annotation_path)
    gt = {int(a['id']): a for a in annotations['annotations']}
    if len(gt) != len(annotations['annotations']):
        raise ValueError('Duplicate annotation identifiers')
    p = evaluator.params
    protocol = dict(iou_type=p.iouType, iou_thresholds=np.asarray(p.iouThrs).tolist(),
        recall_thresholds=np.asarray(p.recThrs).tolist(), max_dets=list(p.maxDets),
        area_ranges=np.asarray(p.areaRng).tolist(), area_labels=list(p.areaRngLbl),
        category_ids=list(map(int, p.catIds)), image_ids=sorted(map(int, p.imgIds)),
        use_categories=int(p.useCats))
    expected_area = [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]
    if (protocol['iou_type'] != 'bbox' or protocol['max_dets'] != [1, 10, 100]
        or protocol['area_ranges'] != expected_area
        or protocol['area_labels'] != ['all', 'small', 'medium', 'large']
        or protocol['category_ids'] != list(range(10)) or protocol['use_categories'] != 1
        or protocol['image_ids'] != sorted(int(x['id']) for x in annotations['images'])
        or not np.allclose(p.iouThrs, np.linspace(.5, .95, 10), atol=1e-12, rtol=0)
        or not np.allclose(p.recThrs, np.linspace(0, 1, 101), atol=1e-12, rtol=0)):
        raise ValueError('Evaluation protocol differs from the bound standard COCO contract')
    precision = np.asarray(evaluator.eval['precision'])
    recall = np.asarray(evaluator.eval['recall'])
    if precision.shape != (10, 101, 10, 4, 3) or recall.shape != (10, 10, 4, 3):
        raise ValueError('Unexpected evaluation tensor shape')
    if not np.isfinite(precision).all() or not np.isfinite(recall).all():
        raise ValueError('Non-finite evaluation arrays')
    if np.any((precision < -1) | (precision > 1)) or np.any((recall < -1) | (recall > 1)):
        raise ValueError('Evaluation values outside COCO ranges')
    def avg(a):
        a = a[a > -1]
        return float(a.mean() * 100) if len(a) else None
    def metrics(k=slice(None)):
        return dict(zip(METRICS, [avg(precision[:, :, k, 0, 2]), avg(precision[0, :, k, 0, 2]),
            avg(precision[5, :, k, 0, 2]), *[avg(precision[:, :, k, a, 2]) for a in (1, 2, 3)]]))
    overall = metrics()
    stats = np.asarray(test_stats['coco_eval_bbox'], dtype=float)
    if stats.shape != (12,) or not np.allclose(stats, evaluator.stats, atol=1e-12, rtol=0):
        raise ValueError('Returned native statistics differ from evaluator state')
    if not np.allclose([overall[k] for k in METRICS], stats[:6]*100, atol=1e-9, rtol=0):
        raise ValueError('Saved precision does not reproduce current native AP statistics')
    names = {int(c['id']): c['name'] for c in annotations['categories']}
    categories = [dict(class_id=k, name=names[k], **metrics(k)) for k in range(10)]
    if abs(sum(c['AP'] for c in categories)/10 - overall['AP']) > 1e-9:
        raise ValueError('Macro AP differs from equal-weight category AP')
    ids = sorted(gt)
    index = {g: i for i, g in enumerate(ids)}
    hits = np.zeros((len(ids), 10), dtype=bool)
    seen, cells, records = set(), set(), []
    fp = np.zeros(10, dtype=np.int64)
    for item in evaluator.evalImgs:
        if item is None or list(item['aRng']) != expected_area[0]:
            continue
        key = (int(item['image_id']), int(item['category_id']))
        if key in cells or item['maxDet'] != 100 or len(item['dtIds']) > 100:
            raise ValueError('Duplicate image/category cell or wrong detection limit')
        cells.add(key)
        if key[0] not in protocol['image_ids'] or key[1] not in range(10):
            raise ValueError('Foreign image or category in matching records')
        if len(item['gtIds']) != len(item['gtIgnore']):
            raise ValueError('Malformed GT ignore flags')
        for g, ignore in zip(item['gtIds'], item['gtIgnore']):
            g = int(g)
            if g in seen or g not in gt or ignore:
                raise ValueError('Unexpected duplicate, foreign, or ignored GT in this bound dataset')
            if (gt[g]['image_id'], gt[g]['category_id']) != key:
                raise ValueError('GT belongs to another image/category')
            seen.add(g)
        dm, di = np.asarray(item['dtMatches']), np.asarray(item['dtIgnore'], dtype=bool)
        if dm.shape != (10, len(item['dtIds'])) or di.shape != dm.shape:
            raise ValueError('Malformed matching arrays')
        if not np.isfinite(dm).all() or not np.equal(dm, np.floor(dm)).all():
            raise ValueError('Non-integer or non-finite matching identities')
        for t in range(10):
            fp[t] += int(((dm[t] == 0) & ~di[t]).sum())
            for g in dm[t][(dm[t] != 0) & ~di[t]]:
                g = int(g)
                if g not in item['gtIds'] or g not in gt or (gt[g]['image_id'], gt[g]['category_id']) != key or hits[index[g], t]:
                    raise ValueError('Invalid or repeated TP identity')
                hits[index[g], t] = True
        # Read-only: conversion/copy is performed by the JSON encoder when writing.
        records.append(item)
    if seen != set(gt):
        raise ValueError('Incomplete validation GT coverage')
    source = {'module': type(evaluator).__module__, 'class': type(evaluator).__name__}
    module = sys.modules.get(source['module'])
    module_file = getattr(module, '__file__', None)
    if mode == 'native_runtime':
        if source['module'] != 'pycocoeval.cocoeval' or not module_file:
            raise ValueError('Expected the original custom pycocoeval evaluator')
        source.update(file=str(Path(module_file).resolve()), sha256=sha(module_file))
        from evaluator_binding import loaded_fingerprint, validate_result_source
        source.update(loaded_fingerprint(identity['evaluator_binding']))
        validate_result_source(source, identity)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'precision_recall.npz', precision=precision, recall=recall)
    np.savez_compressed(output/'GT_hits.npz', annotation_id=np.array(ids),
        category_id=np.array([gt[g]['category_id'] for g in ids]), hits=hits)
    with gzip.open(output/'matching_all_area.json.gz', 'xt', encoding='utf-8') as stream:
        json.dump(records, stream, ensure_ascii=False, allow_nan=False, default=json_default)
    summary = dict(status='CAPTURED_CURRENT_EVALUATION_NOT_INDEPENDENTLY_REPRODUCED',
        mode=mode, epoch_zero_based=epoch, epoch_one_based=epoch+1, evaluated_weights='EMA',
        identity=identity, protocol=protocol, evaluator_source=source, environment=environment,
        overall_percent=overall, categories_percent=categories, native_stats=stats.tolist(),
        GT_count=len(ids), TP_by_IoU=hits.sum(axis=0).tolist(), FP_by_IoU=fp.tolist(),
        annotation_zero_present=0 in gt, annotation_zero_TP_policy='native dtMatches zero sentinel retained',
        independent_reproduction_verified=False, latency_and_memory_verified=False)
    write(output/'summary.json', summary)
    write(output/'manifest.json', {'files':{f.name:sha(f) for f in output.iterdir() if f.is_file()},
        'capture_code_sha256':sha(__file__), 'mode':mode})
    return summary


def load_endpoint(path):
    import numpy as np
    path = Path(path)
    manifest = read(path/'manifest.json')
    expected = {'summary.json', 'precision_recall.npz', 'GT_hits.npz', 'matching_all_area.json.gz'}
    if set(manifest['files']) != expected or any(sha(path/k) != v for k,v in manifest['files'].items()):
        raise ValueError('Endpoint file set or fingerprints changed')
    summary = read(path/'summary.json')
    if summary['mode'] != manifest['mode'] or summary['epoch_zero_based'] != 131:
        raise ValueError('Endpoint mode or final epoch mismatch')
    with np.load(path/'GT_hits.npz', allow_pickle=False) as values:
        flags = {k:values[k] for k in values.files}
    return summary, flags, manifest


def pair(left, right, left_hits, right_hits):
    """right minus left; matches are per GT at the SAME IoU, not shared query indices."""
    import numpy as np
    if left['protocol'] != right['protocol']:
        raise ValueError('Pair uses different evaluation protocols')
    for summary, flags in ((left, left_hits), (right, right_hits)):
        if ([c['class_id'] for c in summary['categories_percent']] != list(range(10))
            or flags['hits'].shape != (len(flags['annotation_id']), 10)
            or flags['hits'].dtype != np.dtype(bool)):
            raise ValueError('Malformed category order or GT flags')
    if [c['name'] for c in left['categories_percent']] != [c['name'] for c in right['categories_percent']]:
        raise ValueError('Category names differ')
    for field in ('annotation_id','category_id'):
        if not np.array_equal(left_hits[field], right_hits[field]):
            raise ValueError('GT populations/order differ')
    a, b = left_hits['hits'], right_hits['hits']
    class_changes = []
    for c in left['categories_percent']:
        mask = left_hits['category_id'] == c['class_id']
        class_changes.append(dict(class_id=c['class_id'], name=c['name'],
            new_TP_by_IoU=((~a[mask])&b[mask]).sum(0).tolist(),
            lost_TP_by_IoU=(a[mask]&~b[mask]).sum(0).tolist()))
    return {'delta_AP_metrics_pp':{k:right['overall_percent'][k]-left['overall_percent'][k] for k in METRICS},
        'class_delta_AP_pp':[{'class_id':c['class_id'],'name':c['name'],
            'delta':right['categories_percent'][i]['AP']-c['AP']} for i,c in enumerate(left['categories_percent'])],
        'GT_new_TP_by_IoU':((~a)&b).sum(0).tolist(),
        'GT_lost_TP_by_IoU':(a&(~b)).sum(0).tolist(),
        'GT_kept_TP_by_IoU':(a&b).sum(0).tolist(),
        'IoU_thresholds':left['protocol']['iou_thresholds'],
        'class_GT_changes':class_changes,
        'delta_FP_by_IoU':(np.asarray(right['FP_by_IoU'])-np.asarray(left['FP_by_IoU'])).tolist(),
        'lost_GT_ids_by_IoU':{format(left['protocol']['iou_thresholds'][t], '.2f'):
            left_hits['annotation_id'][a[:,t]&~b[:,t]].tolist() for t in range(10)}}


def compare(paths):
    from runtime import implementation_hashes, COMMON_INIT_SHA
    here = Path(__file__).resolve().parent
    endpoints = {}
    for path in paths:
        s, flags, manifest = load_endpoint(path)
        key = (s['identity']['arm'], s['identity']['seed'])
        if s['mode'] != 'native_runtime':
            raise ValueError('Archived engineering fixtures cannot serve as new experimental results')
        from evaluator_binding import validate_result_source
        validate_result_source(s['evaluator_source'], s['identity'])
        if manifest['capture_code_sha256'] != sha(__file__):
            raise ValueError('Capture version differs from this implementation')
        completion = read(Path(path)/'training_completion.json')
        if (completion['status'] != 'NATIVE_FIT_RETURNED_AND_LAST_CHECKPOINT_PRESENT'
            or completion['identity'] != s['identity']
            or completion['capture_manifest_sha256'] != sha(Path(path)/'manifest.json')):
            raise ValueError('Missing or inconsistent native completion receipt')
        if not Path(completion['last_checkpoint']).is_file() or sha(completion['last_checkpoint']) != completion['last_checkpoint_sha256']:
            raise ValueError('Completed native checkpoint is missing or changed')
        from stage_trace import verify_trace
        trace_manifest = Path(completion['stage_trace_manifest'])
        if sha(trace_manifest) != completion['stage_trace_manifest_sha256']:
            raise ValueError('Stage trace completion fingerprint differs')
        verify_trace(trace_manifest.parent,s['identity'])
        plan_path = here/'configs'/f'{key[0]}_seed{key[1]}.json'
        if (s['identity']['plan_sha256'] != sha(plan_path)
            or s['identity']['implementation'] != implementation_hashes()
            or s['identity']['common_init_sha256'] != COMMON_INIT_SHA
            or s['identity']['training_binding_sha256'] != sha(here/'training_source_binding.json')):
            raise ValueError('Result differs from the current bound experiment package')
        if key in endpoints:
            raise ValueError('Duplicate arm/seed')
        endpoints[key] = (s, flags, manifest, str(Path(path).resolve()))
    if set(endpoints) != {(a,s) for a in ARMS for s in (1,2)}:
        raise ValueError('All three arms and both seeds are required')
    reference = endpoints[('baseline',1)][0]
    for s, flags, manifest, path in endpoints.values():
        if s['protocol'] != reference['protocol'] or s['evaluated_weights'] != 'EMA':
            raise ValueError('Evaluation protocol or evaluated weights differ')
        for field in ('common_init_sha256','implementation','training_binding_sha256',
                      'train_annotation_sha256','val_annotation_sha256','native_profiling'):
            if s['identity'][field] != reference['identity'][field]:
                raise ValueError('Experiment lineage mismatch: '+field)
        from evaluator_binding import same_result_evaluator
        same_result_evaluator(s, reference)
        if s['environment'] != reference['environment']:
            raise ValueError('Runtime environments differ; separate review is required')
        if manifest['capture_code_sha256'] != endpoints[('baseline',1)][2]['capture_code_sha256']:
            raise ValueError('Capture versions differ')
    comparisons = {}
    for seed in (1,2):
        for control in ('baseline','fused_residual'):
            a,af,_,_ = endpoints[(control,seed)]
            b,bf,_,_ = endpoints[('fine_residual',seed)]
            comparisons[f'fine_residual_minus_{control}_seed{seed}'] = pair(a,b,af,bf)
    return {'status':'PAIRED_OBSERVED_RESULTS_REPRODUCTION_AND_COST_REVIEW_REQUIRED',
        'comparisons':comparisons, 'source_directories':[v[3] for v in endpoints.values()],
        'claim_boundary':'Observed deltas only; no automatic efficacy, novelty, significance, or adoption decision',
        'independent_reproduction_verified':False, 'cost_verified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoints', nargs=6, required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    write(args.output, compare(args.endpoints))


if __name__ == '__main__':
    main()
