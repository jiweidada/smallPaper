"""Offline N/D/R engineering screen on immutable BASE12 original_stem caches.

No model calls. A strict native-baseline replay gate precedes candidate metrics.
"""
import contextlib, datetime, gc, hashlib, io, json, os
from pathlib import Path
import sys, time, traceback

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
OUT = ROOT/'postprocessing_probe'
COLLECTION = WORK/'outputs/native_selection_collection12_20261011_v2'
ANN = WORK/'outputs/small_object_encoder_decoder12_20261010_v5/terminal_collection/seed1/extracted/reference/val_annotations.json'
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(ROOT/'runtime_pycoco'), str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1/runtime')]
import numpy as np
import torch
import psutil
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.cocoeval as coco_source


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def choose(probabilities, k):
    """Native topk; prune original prefix; refill by one highest class/query.

    If an argmax is tied, retain the first native-prefix representative when
    available; new queries use torch.max's first-class tie. Scores never change.
    """
    scores, flat = torch.topk(probabilities.flatten(), k)
    keep = scores > .001
    native = flat[keep].tolist()
    C = probabilities.shape[1]
    seen, deleted = set(), []
    for f in native:
        q = f//C
        if q not in seen:
            deleted.append(f); seen.add(q)
    maximum, label = probabilities.max(-1)
    label = label.clone()
    for f in deleted:
        assert probabilities[f//C, f%C] == maximum[f//C]
        label[f//C] = f%C
    prefix_rank = {f//C: i for i, f in enumerate(deleted)}
    eligible = [q for q in range(len(maximum)) if maximum[q] > .001]
    eligible.sort(key=lambda q: (-float(maximum[q]), prefix_rank.get(q, k+q)))
    refill = [q*C+int(label[q]) for q in eligible[:k]]
    assert set(deleted).issubset(native) and set(deleted).issubset(refill)
    assert len({f//C for f in refill}) == len(refill)
    assert not ({f//C for f in refill}-{f//C for f in native}) & {f//C for f in native}
    return {'N': native, 'D': deleted, 'R': refill}


def xywh_original(boxes, hw):
    cx, cy, w, h = boxes.unbind(-1)
    xyxy = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)
    hh, ww = int(hw[0]), int(hw[1])
    xyxy = xyxy*torch.tensor([ww, hh, ww, hh], dtype=boxes.dtype)
    xywh = xyxy.clone()
    xywh[:, 2:] -= xyxy[:, :2]
    return xywh


def evaluate(coco, predictions, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(predictions)
        evaluation = COCOeval(coco, dt, 'bbox')
        evaluation.params.imgIds = image_ids
        evaluation.params.maxDets = [1, 10, 100]
        evaluation.evaluate(); evaluation.accumulate(); evaluation.summarize()
    precision = evaluation.eval['precision']
    per_class = {}
    for ci, cid in enumerate(evaluation.params.catIds):
        a = precision[:, :, ci, 0, -1]; s = precision[:, :, ci, 1, -1]
        per_class[str(cid)] = {'AP': float(a[a > -1].mean()) if np.any(a > -1) else None,
                              'APs': float(s[s > -1].mean()) if np.any(s > -1) else None}
    hits = {}
    counts = {}
    for threshold in [.5, .75]:
        ti = int(np.flatnonzero(np.isclose(evaluation.params.iouThrs, threshold))[0])
        gt_hits, tp, fp = set(), 0, 0
        for item in evaluation.evalImgs:
            if item is None or item['aRng'] != [0, 1e10]:
                continue
            valid = ~item['dtIgnore'][ti].astype(bool)
            match = item['dtMatches'][ti]
            gt_hits.update(int(v) for v in match[valid] if v > 0)
            tp += int(((match > 0) & valid).sum())
            fp += int(((match == 0) & valid).sum())
        hits[str(threshold)] = gt_hits
        counts[str(threshold)] = {'TP': tp, 'FP': fp, 'unique_GT_hits': len(gt_hits)}
    return {'stats': evaluation.stats.tolist(), 'per_class': per_class,
            'matching_counts': counts}, hits


def main():
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    process = psutil.Process(); process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-1::-2][:2])
    assert torch.version.cuda is None
    toy = torch.tensor([[.9, .8], [.7, .1], [.6, .2]])
    toy_selection = choose(toy, 3)
    assert toy_selection == {'N': [0, 1, 2], 'D': [0, 2], 'R': [0, 2, 4]}
    assert sha(ANN) == '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    OUT.mkdir(exist_ok=False)
    protocol = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'models': 'original_stem12 EMA seed1/2, epoch11; not current BASE24 baseline or C96',
        'cohort': 'same548 val images, original annotation SHA checked,38759GT',
        'groups': {'N': 'native sigmoid flatten top300 then score>.001',
                   'D': 'one highest-class representative per query within N, no refill',
                   'R': 'all final queries highest class, original scores, refill up to300, same>.001'},
        'ties': 'Native torch.topk preserved for N. D keeps first native-prefix representative. R resolves argmax ties to D when present, else first class; shared tied scores prioritize original-prefix query order, then queryID.',
        'coordinates': 'FP32 normalizedCXCYWH -> originalXYXY -> originalXYWH; no clipping or box modification',
        'evaluator': 'pinned upstream pycocotools2.0.11 Windows backend, maxDets1/10/100, all original categories and annotationIDs including0 sentinel',
        'native_replay_gate': 'all12 native baseline stats max abs error<=1e-8 before any D/R evaluation; failure stops candidate AP, no tolerance relaxation',
        'decision': 'R must beat N and D in overallAP in BOTH endpoints and yield new matchedGT to retain the occupancy explanation. Ordinary argmax/equivalent exact-coordinate dedup remains an engineering baseline, not a novel method or goal completion.',
        'outputs': 'full12 metrics, per-classAP/APs, output/query counts, each-group fresh matching, GT gains/losses at.5/.75, size/class denominators',
        'new_model_forwards': 0, 'training_steps': 0, 'heldout_reads': False, 'remote_operations': False,
        'script_sha256': sha(__file__), 'annotation_path': str(ANN), 'annotation_sha256': sha(ANN),
        'eval_python_sha256': sha(coco_source.__file__), 'synthetic_selection_test_pass': True,
        'scope_limit': 'Exploratory old model validation engineering probe; not official VisDrone ignore-region re-evaluation or independent hypothesis confirmation.'}
    dump(OUT/'01_protocol.json', protocol)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ANN))
    image_ids = sorted(coco.getImgIds())
    anns = {int(a['id']): a for a in coco.dataset['annotations']}
    result = {'protocol': protocol, 'seeds': []}
    for seed in [1, 2]:
        folder = COLLECTION/f'seed{seed}_original_stem/extracted'
        packet = json.loads((folder/'packet_manifest.json').read_text(encoding='utf-8-sig'))
        expected = json.loads((folder/'diagnostic/endpoint_epoch12/summary.json').read_text(encoding='utf-8-sig'))
        predictions = {g: [] for g in ['N', 'D', 'R']}
        image_rows, source_checks = [], []
        for iid in image_ids:
            relative = f'diagnostic/stages_epoch12/image_{iid:06d}.npz'
            path = folder/relative
            digest = sha(path); assert digest == packet['files'][relative]['sha256']
            source_checks.append({'path': str(path), 'sha256': digest})
            with np.load(path, allow_pickle=False) as z:
                boxes = torch.from_numpy(z['final_boxes'].copy())
                logits = torch.from_numpy(z['final_logits'].copy())
                hw = z['original_hw'].copy()
            assert tuple(boxes.shape) == (300, 4) and tuple(logits.shape) == (300, 10)
            probability = logits.sigmoid()
            selected = choose(probability, 300)
            unscaled = xywh_original(boxes, hw)
            for g, flat in selected.items():
                for f in flat:
                    q, c = divmod(f, 10)
                    predictions[g].append({'image_id': iid, 'category_id': c,
                                           'bbox': unscaled[q].tolist(), 'score': float(probability[q, c])})
            unique_coordinates = len({tuple(v) for v in boxes.tolist()})
            image_rows.append({'image_id': iid, 'output_counts': {g: len(v) for g, v in selected.items()},
                'unique_queries_N': len({f//10 for f in selected['N']}),
                'duplicate_query_slots_N': len(selected['N'])-len({f//10 for f in selected['N']}),
                'new_queries_R': len({f//10 for f in selected['R']}-{f//10 for f in selected['N']}),
                'all300_boxes_exactly_distinct': unique_coordinates == 300,
                'exact_coordinate_dedup_refill_equals_R_if_distinct': unique_coordinates == 300})
        dump(OUT/f'seed{seed}_source_identity.json', {'files': source_checks, 'packet_sha256': sha(folder/'packet_manifest.json')})
        print(json.dumps({'seed': seed, 'stage': 'original_COCO_replay_start', 'predictions': len(predictions['N'])}), flush=True)
        metrics, hits = {}, {}
        metrics['N'], hits['N'] = evaluate(coco, predictions['N'], image_ids)
        error = max(abs(a-b) for a, b in zip(metrics['N']['stats'], expected['native_stats']))
        dump(OUT/f'seed{seed}_native_replay.json', {'max_abs_error': error, 'tolerance': 1e-8,
             'pass': error <= 1e-8, 'expected': expected['native_stats'], 'current': metrics['N']['stats']})
        assert error <= 1e-8, f'Native baseline mismatch seed{seed}: {error}; candidate evaluation not run'
        for group in ['D', 'R']:
            print(json.dumps({'seed': seed, 'stage': 'candidate_COCO_start', 'group': group}), flush=True)
            metrics[group], hits[group] = evaluate(coco, predictions[group], image_ids)
        changes = {}
        for t in ['0.5', '0.75']:
            changes[t] = {}
            for a, b in [('R', 'N'), ('D', 'N'), ('R', 'D')]:
                gain, loss = hits[a][t]-hits[b][t], hits[b][t]-hits[a][t]
                changes[t][a+'-'+b] = {'gained_GT': len(gain), 'lost_GT': len(loss),
                    'gained_small_GT': sum(anns[g]['area'] < 1024 for g in gain),
                    'lost_small_GT': sum(anns[g]['area'] < 1024 for g in loss),
                    'gained_by_class': {str(c): sum(anns[g]['category_id'] == c for g in gain) for c in range(10)},
                    'lost_by_class': {str(c): sum(anns[g]['category_id'] == c for g in loss) for c in range(10)}}
        one = {'seed': seed, 'metrics': metrics, 'GT_changes_not_AP_decomposition': changes,
               'native_replay_max_abs_error': error, 'images': image_rows,
               'total_duplicate_slots_N': sum(r['duplicate_query_slots_N'] for r in image_rows),
               'total_new_queries_R': sum(r['new_queries_R'] for r in image_rows),
               'images_where_exact_coordinate_dedup_is_the_same_argmax_control': sum(r['all300_boxes_exactly_distinct'] for r in image_rows),
               'R_minus_N_AP_pp': 100*(metrics['R']['stats'][0]-metrics['N']['stats'][0]),
               'R_minus_D_AP_pp': 100*(metrics['R']['stats'][0]-metrics['D']['stats'][0])}
        dump(OUT/f'seed{seed}_results.json', one)
        result['seeds'].append(one)
        print(json.dumps({'seed': seed, 'stage': 'complete', 'APs_percent': {g: 100*m['stats'][0] for g, m in metrics.items()},
                          'R_minus_N_AP_pp': one['R_minus_N_AP_pp'], 'R_minus_D_AP_pp': one['R_minus_D_AP_pp']}), flush=True)
        del predictions; gc.collect()
    result['occupancy_probe_passes'] = all(x['R_minus_N_AP_pp'] > 0 and x['R_minus_D_AP_pp'] > 0 and
        x['GT_changes_not_AP_decomposition']['0.5']['R-D']['gained_GT'] > 0 for x in result['seeds'])
    result['novel_method_established'] = False
    dump(OUT/'02_results.json', result)


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        if OUT.exists():
            dump(OUT/'failure.json', {'error': type(e).__name__+': '+str(e),
                 'traceback': traceback.format_exc(), 'no_automatic_identical_retry': True,
                 'new_model_forwards': 0, 'training_steps': 0})
        raise
