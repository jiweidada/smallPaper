"""CPU-only image-cluster bootstrap of existing project-COCO APl predictions.

No model, torch, network, remote access, or new inference.  Repeated images are
independent copies; exact score ties retain image/copy/detection order.  Uses
official COCOeval for matching, and validates the cached accumulator against it.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import csv
import datetime as dt
import gc
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
import numpy as np
from numba import njit

CODE = Path(__file__).resolve().parent
WORK = CODE.parents[1]
PAPER = WORK / '相位机制论文_20261018'
V2 = PAPER / '修订_20261018_v2'
CSV = PAPER / '修订_20261018_v5/supplementary/full12_original_precision.csv'
INDEX = V2 / 'data/prediction_index.json'
OUT = WORK / 'data/00_实验计划与证据索引/C_APl抽样与种子核查_20261020_v1'
sys.path.insert(0, str(CODE / 'runtime'))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(p, rows):
    if not rows:
        return
    with Path(p).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


@njit(cache=True)
def bootstrap_category(img, tp, ignored, group_ends, gt, counts):
    """Exactly expand each score-tie/image group in image-copy order.

    Ignore-only detections can be omitted without changing precision/recall.
    Returns NaN when a resample contains no nonignored GT of this category.
    """
    reps = counts.shape[0]
    ts, nd = tp.shape
    result = np.full((reps, ts), np.nan)
    for b in range(reps):
        ng = 0
        for j in range(gt.size):
            ng += gt[j] * counts[b, j]
        if ng == 0:
            continue
        capacity = 0
        for j in range(nd):
            capacity += counts[b, img[j]]
        rec = np.empty(capacity, np.float64)
        pre = np.empty(capacity, np.float64)
        for t in range(ts):
            nt, nf, used, start = 0, 0, 0, 0
            for end in group_ends:
                # One equal-score / same-source-image group per segment.
                multiplicity = counts[b, img[start]]
                for unused_copy in range(multiplicity):
                    for j in range(start, end):
                        if not ignored[t, j]:
                            if tp[t, j]:
                                nt += 1
                            else:
                                nf += 1
                            rec[used] = nt / ng
                            pre[used] = nt / (nt + nf + np.spacing(1.0))
                            used += 1
                start = end
            if used == 0:
                result[b, t] = 0.0
                continue
            for j in range(used - 2, -1, -1):
                if pre[j + 1] > pre[j]:
                    pre[j] = pre[j + 1]
            value, pos = 0.0, 0
            for r in range(101):
                threshold = r * 0.01
                while pos < used and rec[pos] < threshold:
                    pos += 1
                if pos < used:
                    value += pre[pos]
            result[b, t] = value / 101.0
    return result


def category_cache(entries, image_count):
    scores, imgs, matches, ignores = [], [], [], []
    gt = np.zeros(image_count, dtype=np.int64)
    for i, e in enumerate(entries):
        if e is None:
            continue
        gt[i] = np.count_nonzero(~np.asarray(e['gtIgnore'], dtype=bool))
        scores.extend(e['dtScores'])
        imgs.extend([i] * len(e['dtScores']))
        matches.append(np.asarray(e['dtMatches']) > 0)
        ignores.append(np.asarray(e['dtIgnore'], dtype=bool))
    if not scores:
        return (np.zeros(0, np.int64), np.zeros((10, 0), bool),
                np.zeros((10, 0), bool), np.zeros(0, np.int64), gt)
    score = np.asarray(scores)
    img = np.asarray(imgs, dtype=np.int64)
    tp = np.concatenate(matches, axis=1)
    ignored = np.concatenate(ignores, axis=1)
    order = np.argsort(-score, kind='mergesort')
    order = order[~np.all(ignored[:, order], axis=0)]
    score, img = score[order], img[order]
    tp, ignored = tp[:, order], ignored[:, order]
    # Equal score order is original image, then copy, then detection order.
    ends = np.flatnonzero((score[1:] != score[:-1]) | (img[1:] != img[:-1])) + 1
    ends = np.r_[ends, len(score)].astype(np.int64) if len(score) else np.zeros(0, np.int64)
    return img, np.ascontiguousarray(tp), np.ascontiguousarray(ignored), ends, gt


def cached_ap(caches, counts):
    bycat = [bootstrap_category(*c, counts) for c in caches]
    arr = np.stack(bycat, axis=1)  # replicate x category x IoU
    valid = np.isfinite(arr)
    denom = valid.sum(axis=(1, 2))
    result = np.nansum(arr, axis=(1, 2)) / np.where(denom, denom, 1)
    result[denom == 0] = np.nan
    return result, np.all(valid, axis=2).sum(axis=1)


def official_repeated_ap(entries_by_cat, counts):
    """COCO's own accumulator on explicitly duplicated per-image matches."""
    indices = np.repeat(np.arange(len(counts)), counts)
    e = COCOeval(None, None, 'bbox')
    e.params.imgIds = list(range(len(indices)))
    e.params.catIds = list(range(len(entries_by_cat)))
    e.params.areaRng = [[96**2, 1e10]]
    e.params.areaRngLbl = ['large']
    e.params.maxDets = [100]
    e._paramsEval = copy.deepcopy(e.params)
    e.evalImgs = [items[int(i)] for items in entries_by_cat for i in indices]
    with contextlib.redirect_stdout(io.StringIO()):
        e.accumulate()
    precision = e.eval['precision']
    return float(precision[precision >= 0].mean())


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'models').mkdir()
    rows = list(csv.DictReader(CSV.open(encoding='utf-8-sig')))
    records = json.loads(INDEX.read_text(encoding='utf-8'))
    assert len(records) == len(rows) == 40
    by = {(r['split'], int(r['seed']), r['arm']): r for r in rows}
    assert len(by) == 40
    descriptive, paired = [], []
    for split in ['val', 'testdev']:
        for arm in dict.fromkeys(r['arm'] for r in rows):
            rs = [r for r in rows if r['split'] == split and r['arm'] == arm]
            values = np.array([float(r['APl_percent']) for r in rs])
            small = np.array([float(r['APs_percent']) for r in rs])
            descriptive.append({'split': split, 'arm': arm, 'n_seeds': len(rs),
                'seeds': '/'.join(r['seed'] for r in rs),
                'APl_by_seed_percent': '/'.join(str(x) for x in values),
                'APl_mean_percent': values.mean(), 'APl_sample_sd_pp': values.std(ddof=1),
                'APl_range_pp': np.ptp(values), 'APs_range_pp': np.ptp(small)})
        for arm in dict.fromkeys(r['arm'] for r in rows):
            seeds = sorted(k[1] for k in by if k[0] == split and k[2] == arm)
            for a, b in itertools.combinations(seeds, 2):
                paired.append({'split': split, 'kind': 'within_arm_seed_difference',
                    'candidate': f'{arm}/seed{b}', 'reference': f'{arm}/seed{a}',
                    'APl_delta_pp': float(by[split,b,arm]['APl_percent']) - float(by[split,a,arm]['APl_percent'])})
        for seed in [0, 1, 2]:
            arms = [k[2] for k in by if k[0] == split and k[1] == seed]
            for a, b in itertools.combinations(arms, 2):
                paired.append({'split': split, 'kind': 'same_seed_between_arm_difference',
                    'candidate': f'{b}/seed{seed}', 'reference': f'{a}/seed{seed}',
                    'APl_delta_pp': float(by[split,seed,b]['APl_percent']) - float(by[split,seed,a]['APl_percent'])})
    write_csv(OUT / '全部臂_种子描述统计.csv', descriptive)
    write_csv(OUT / '全部臂间与臂内差值.csv', paired)
    inputs = {}
    for split, ds in [('val', 'val'), ('testdev', 'test-dev')]:
        p = V2 / f'data/annotations/{ds}/converted.json'
        raw = p.read_bytes()
        annotation = json.loads(raw)
        n = len(annotation['images'])
        rng = np.random.default_rng(20261020 + int(split == 'testdev'))
        counts = rng.multinomial(n, np.full(n, 1/n), size=2000).astype(np.int32)
        np.savez_compressed(OUT / f'{split}_shared_image_multiplicities.npz', counts=counts,
                            image_ids=np.array(sorted(i['id'] for i in annotation['images'])))
        inputs[split] = {'annotation': str(p), 'sha256': hashlib.sha256(raw).hexdigest(),
            'image_count': n, 'large_gt_count': sum(not g.get('iscrowd',0) and 96**2 <= g['area'] <= 1e10 for g in annotation['annotations']),
            'multiplicities_sha256': sha(OUT / f'{split}_shared_image_multiplicities.npz')}
    protocol = {'created_utc': now(), 'status': 'exploratory_protocol_frozen_before_bootstrap',
        'repetitions': 2000, 'rng': 'numpy default_rng multinomial; val20261020/testdev20261021',
        'sampling': 'All original images, replacement, original image count; same draws for all models and seeds within a split. Repeated image copies retain all predictions and GT.',
        'metric': 'Original-coordinate project COCO APl, IoU .50:.05:.95, 101 recall points, per-class macro mean, maxDets100, unchanged annotation/ignore semantics, top300 score0 stored predictions.',
        'interval': 'Percentile 2.5/97.5, numpy linear quantile, differences computed within each paired image resample, not by subtracting marginal CI endpoints.',
        'main_comparisons': {'C-baseline': [0,1], 'C-C-mean': [0,1,2]},
        'secondary': 'All arm pairs at each common seed; every within-arm seed pair. Report all, no significance-selection.',
        'mean_pairs': 'Mean of matched-seed differences, same image weights across seeds. Conditional on observed trained models; does not include population training-seed uncertainty.',
        'zero_GT': 'Category absent in a resample follows COCO exclusion; report present-category count. All categories absent => NA, never zero.',
        'limits': ['Exploratory analysis of already inspected outcomes, not preregistered confirmation.',
            'Two/three training seeds cannot establish equivalence, population variance or systematic scale cost.',
            'Image bootstrap assumes independent image clusters; correlated frames/scenes can make intervals too narrow. No scene independence claim.',
            'Validation was used for checkpoint selection; its intervals condition on selected weights and do not include selection uncertainty.',
            'No additive decomposition of sampling and seed ranges; CI overlap or inclusion of zero does not prove noise/no effect.',
            'Unadjusted 95% intervals are descriptive; many comparisons are not family-wise significance tests.'],
        'inputs': inputs, 'csv_sha256': sha(CSV), 'index_sha256': sha(INDEX),
        'code_sha256': sha(__file__), 'new_training': False, 'new_model_inference': False,
        'remote_access': False, 'coco_source': 'https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py'}
    dump(OUT / '冻结分析口径.json', protocol)
    print(json.dumps({'prepared': str(OUT), 'models': len(records), 'protocol_sha256': sha(OUT/'冻结分析口径.json')}, ensure_ascii=False), flush=True)


def worker(rec):
    start = time.monotonic()
    protocol = json.loads((OUT/'冻结分析口径.json').read_text())
    assert sha(__file__) == protocol['code_sha256']
    tag = f"{rec['split']}_seed{rec['seed']}_{rec['arm']}"
    assert not (OUT/'models'/f'{tag}.json').exists()
    inp = protocol['inputs'][rec['split']]
    assert inp['sha256'] == rec['export_manifest']['annotations_sha256']
    assert sha(inp['annotation']) == inp['sha256']
    annotation = json.loads(Path(inp['annotation']).read_text(encoding='utf-8'))
    with zipfile.ZipFile(rec['archive']) as z:
        raw = z.read(rec['member'])
    assert hashlib.sha256(raw).hexdigest() == rec['predictions_sha256']
    predictions = json.loads(raw)
    del raw
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = annotation; gt.createIndex()
        det = gt.loadRes(predictions)
        ev = COCOeval(gt, det, 'bbox')
        ev.params.areaRng = [[96**2, 1e10]]
        ev.params.areaRngLbl = ['large']
        ev.params.maxDets = [100]
        ev.evaluate(); ev.accumulate()
    precision = ev.eval['precision']
    point = float(precision[precision >= 0].mean())
    expected = next(r for r in csv.DictReader(CSV.open(encoding='utf-8-sig'))
                    if r['split'] == rec['split'] and int(r['seed']) == rec['seed'] and r['arm'] == rec['arm'])
    assert abs(point - float(expected['APl_raw'])) < 1e-10, (tag, point, expected['APl_raw'])
    n = len(ev.params.imgIds)
    entries = [ev.evalImgs[k*n:(k+1)*n] for k in range(len(ev.params.catIds))]
    caches = [category_cache(e,n) for e in entries]
    with np.load(OUT/f"{rec['split']}_shared_image_multiplicities.npz") as z:
        counts = z['counts']
        assert z['image_ids'].tolist() == ev.params.imgIds
    checks = np.vstack([np.ones((1,n),dtype=np.int32), counts[:2]])
    checked,_ = cached_ap(caches, checks)
    references = [point] + [official_repeated_ap(entries,c) for c in checks[1:]]
    discrepancy = float(np.max(np.abs(checked - references)))
    assert discrepancy < 1e-10, (tag, checked, references)
    # Release boxes/IoU matrices before the repeated accumulation.
    del gt, det, ev, predictions, annotation, entries
    gc.collect()
    vals, present = cached_ap(caches, counts)
    np.savez_compressed(OUT/'models'/f'{tag}_draws.npz', APl_raw=vals, present_categories=present)
    finite = vals[np.isfinite(vals)]
    ci = np.quantile(finite,[.025,.975]) * 100 if finite.size else [None,None]
    result = {'split': rec['split'], 'seed': rec['seed'], 'arm': rec['arm'],
        'point_percent': point*100, 'ci95_low_percent': float(ci[0]), 'ci95_high_percent': float(ci[1]),
        'replicates': len(vals), 'NA_replicates': int((~np.isfinite(vals)).sum()),
        'present_category_min': int(present.min()), 'present_category_max': int(present.max()),
        'replicates_missing_any_category': int((present < len(caches)).sum()),
        'CSV_point_max_abs_error_raw': abs(point - float(expected['APl_raw'])),
        'official_duplicate_accumulation_max_abs_error_raw': discrepancy,
        'prediction_sha256': rec['predictions_sha256'], 'annotation_sha256': inp['sha256'],
        'checkpoint_sha256': rec['export_manifest']['checkpoint_sha256'],
        'draws_sha256': sha(OUT/'models'/f'{tag}_draws.npz'), 'elapsed_seconds': time.monotonic()-start}
    dump(OUT/'models'/f'{tag}.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def run():
    protocol = json.loads((OUT/'冻结分析口径.json').read_text())
    assert sha(__file__) == protocol['code_sha256']
    records = json.loads(INDEX.read_text())
    # Fixed two CPU workers only. Never touches either remote GPU server.
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs = {pool.submit(worker,rec): rec for rec in records}
        for f in as_completed(fs):
            f.result()
    summarize()


def interval_row(split, kind, cand, ref, point, draws, seeds):
    finite = draws[np.isfinite(draws)]
    q = np.quantile(finite,[.025,.975])*100
    return {'split':split,'kind':kind,'candidate':cand,'reference':ref,'seeds':seeds,
        'point_delta_pp':point,'ci95_low_pp':q[0],'ci95_high_pp':q[1],
        'replicates':len(draws),'NA_replicates':len(draws)-len(finite),
        'conditional_on_observed_checkpoints':True}


def summarize():
    models = [json.loads(p.read_text()) for p in sorted((OUT/'models').glob('*.json'))]
    assert len(models) == 40, len(models)
    by = {(r['split'],r['seed'],r['arm']):r for r in models}
    draws = {}
    for key,r in by.items():
        tag=f"{key[0]}_seed{key[1]}_{key[2]}"
        p=OUT/'models'/f'{tag}_draws.npz'
        assert sha(p)==r['draws_sha256']
        with np.load(p) as z: draws[key]=z['APl_raw']
    rows=[]
    for split in ['val','testdev']:
        for seed in [0,1,2]:
            arms=[k[2] for k in by if k[0]==split and k[1]==seed]
            for a,b in itertools.combinations(arms,2):
                ka,kb=(split,seed,a),(split,seed,b)
                rows.append(interval_row(split,'all_same_seed_arm_pairs',b,a,
                    by[kb]['point_percent']-by[ka]['point_percent'],draws[kb]-draws[ka],str(seed)))
        for arm in dict.fromkeys(k[2] for k in by if k[0]==split):
            seeds=sorted(k[1] for k in by if k[0]==split and k[2]==arm)
            for a,b in itertools.combinations(seeds,2):
                ka,kb=(split,a,arm),(split,b,arm)
                rows.append(interval_row(split,'within_arm_seed_pairs',f'{arm}/seed{b}',f'{arm}/seed{a}',
                    by[kb]['point_percent']-by[ka]['point_percent'],draws[kb]-draws[ka],f'{a}/{b}'))
    main=[]
    for split in ['val','testdev']:
        for ref,seeds in [('baseline',[0,1]),('C-mean',[0,1,2])]:
            ps,ds=[],[]
            for seed in seeds:
                kc,kr=(split,seed,'C'),(split,seed,ref)
                p=by[kc]['point_percent']-by[kr]['point_percent']
                d=draws[kc]-draws[kr]
                main.append(interval_row(split,'main_same_seed_pair','C',ref,p,d,str(seed)))
                ps.append(p);ds.append(d)
            main.append(interval_row(split,'mean_of_common_seed_pairs','C',ref,float(np.mean(ps)),
                                     np.mean(ds,axis=0),'/'.join(map(str,seeds))))
    write_csv(OUT/'各模型_APl图像抽样区间.csv',models)
    write_csv(OUT/'全部配对差值_图像抽样区间.csv',rows)
    write_csv(OUT/'主比较_同种子与共同种子均值区间.csv',main)
    scan={
        '1_Simpson': 'Both splits and every seed reported separately; no pooled unequal-seed arm means used for primary contrasts.',
        '2_ecological': 'Image sampling uncertainty conditional on checkpoints; no individual-object causal inference.',
        '3_selection': 'Fixed existing best_stg2 selected on validation; selection uncertainty explicitly excluded.',
        '4_collider': 'No post-outcome sample filtering; all images resampled, not only large-object images.',
        '5_base_rate': 'Original large GT and image counts reported; COCO macro-category averaging retained.',
        '6_regression_to_mean': 'Worst seed not singled out as mechanism; all arms/seeds retained.',
        '7_survivorship': '40/40 archived models included; no new/dropout selection; seed2 baseline missing explicitly.',
        '8_look_elsewhere': 'All arm/seed contrasts exported; unadjusted intervals descriptive, no selected significance claims.',
        '9_forking': 'Exploratory protocol and code hash frozen before bootstrap; existing outcomes already inspected.',
        '10_causality': 'Neither interval overlap nor zero inclusion proves noise, equivalence or causal mechanism.',
        '11_reverse': 'No reverse-direction causal claim; path mechanism remains untested.'}
    dump(OUT/'汇总与核验.json',{'completed_utc':now(),'status':'completed_cpu_existing_predictions_only',
        'models':len(models),'replicates':2000,'main_comparisons':main,
        'point_parity_max_abs_error_raw':max(r['CSV_point_max_abs_error_raw'] for r in models),
        'duplicate_parity_max_abs_error_raw':max(r['official_duplicate_accumulation_max_abs_error_raw'] for r in models),
        'fallacy_scan_11_of_11':scan,'new_training':False,'new_model_inference':False,'remote_access':False})
    print(json.dumps({'completed_models':len(models),'main_comparisons':main},ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','summarize'])
    args=ap.parse_args()
    globals()[args.mode]()
