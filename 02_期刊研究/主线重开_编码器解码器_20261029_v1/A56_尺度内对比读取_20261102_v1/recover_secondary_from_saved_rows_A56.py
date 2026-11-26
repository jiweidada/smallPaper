"""Recover descriptive rows after JSON int32 failure; no evaluator/model calls."""
from pathlib import Path
import hashlib
import json
import math
import statistics

P = Path(__file__).resolve().parent


def load(p):
    return json.loads(p.read_text('utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    failure = load(P/'secondary_cached_failure_A56_v1.json')
    assert failure['COCO_evaluate_calls_entered'] == 3
    assert 'int32 is not JSON serializable' in failure['traceback']
    assert load(P/'fixed_endpoint_judgment_A56.json')['decision'] == 'RETIRED_EXACT_RECIPE_FIXED12'
    source = P/'secondary_cached_analysis_A56_v1/GT_rows.jsonl'
    rows = [json.loads(s) for s in source.read_text('utf-8').splitlines()]
    gt_path = P.parent/'A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B/epoch12_EMA_GT.json'
    gt = load(gt_path)
    annotations = {a['id']: a for a in gt['annotations']}
    images = {i['id']: i for i in gt['images']}
    assert len(rows) == len(annotations) == 38602 and len(images) == 548
    assert {r['GT_id'] for r in rows} == set(annotations)
    before = {str(p): sha(p) for p in [source, gt_path, P/'secondary_cached_failure_A56_v1.json']}
    groups = {n: [] for n in ['all','small','small_lt2','small_2to4','small_4to8','small_ge8']}
    per_image = {str(i): {'all': [], 'small': []} for i in images}
    per_class = {str(a['category_id']): [] for a in annotations.values()}
    for r in rows:
        a, im = annotations[r['GT_id']], images[r['image_id']]
        assert r['image_id'] == a['image_id'] and r['category_id'] == a['category_id']
        assert r['original_area'] == a['area']
        side = min(a['bbox'][2]*640/im['width'], a['bbox'][3]*640/im['height'])
        assert math.isclose(side, r['input_short_side'], rel_tol=0, abs_tol=1e-10)
        for arm in ['B','M','Q']:
            z = r['arms'][arm]
            assert isinstance(z['matched50'], bool) and isinstance(z['matched75'], bool)
            assert not z['matched75'] or z['matched50']
            e = z['signed_log_geomean_size_error_if_matched50']
            assert (e is not None) == z['matched50'] and (e is None or math.isfinite(e))
        groups['all'].append(r)
        per_image[str(r['image_id'])]['all'].append(r)
        per_class[str(r['category_id'])].append(r)
        if a['area'] <= 1024:
            groups['small'].append(r)
            groups['small_'+r['input_short_side_bin']].append(r)
            per_image[str(r['image_id'])]['small'].append(r)
    assert len(groups['small']) == 26444 and len(groups['small_4to8']) == 10200

    def describe(rs, arm):
        es = [r['arms'][arm]['signed_log_geomean_size_error_if_matched50']
              for r in rs if r['arms'][arm]['matched50']]
        n = len(rs)
        m50 = sum(r['arms'][arm]['matched50'] for r in rs)
        m75 = sum(r['arms'][arm]['matched75'] for r in rs)
        return {'GT': n, 'matched50': m50, 'matched75': m75,
                'recall50_percent': m50*100/n if n else None,
                'recall75_percent': m75*100/n if n else None,
                'size_error_TP50_count': len(es),
                'mean_absolute_log_geomean_size_error_on_own_TP50': statistics.fmean(map(abs, es)) if es else None,
                'mean_signed_log_geomean_size_error_on_own_TP50': statistics.fmean(es) if es else None}

    arms = {arm: {'groups': {n: describe(rs, arm) for n, rs in groups.items()},
                  'per_image': {i: {n: describe(rs, arm) for n, rs in d.items()} for i, d in per_image.items()},
                  'per_class_recalls_only': {c: describe(rs, arm) for c, rs in per_class.items()},
                  'per_class_AP': None, 'per_class_APs': None} for arm in ['B','M','Q']}
    common = {}
    for n, rs in groups.items():
        cohort = [r for r in rs if all(r['arms'][a]['matched50'] for a in arms)]
        common[n] = {'common_TP50_GT': len(cohort), 'arms': {a: {
            'mean_absolute_log_geomean_size_error': statistics.fmean(
                abs(r['arms'][a]['signed_log_geomean_size_error_if_matched50']) for r in cohort) if cohort else None
        } for a in arms}}
    old = load(P.parent/'A53_原版漏检与机制候选筛选_20261101_v1/error_audit_result_v4.json')
    assert all(arms['B']['groups']['all'][k] == old['all_GT'][k] for k in ['GT','matched50','matched75'])
    assert all(sha(Path(p)) == s for p, s in before.items())
    result = {'status': 'PARTIAL_SECONDARY_RECOVERED_FROM_ALREADY_SAVED_ROWS',
              'arms': arms, 'common_matched50_cohort': common,
              'unavailable': ['per-class COCO AP/APs were lost at serialization; no extra evaluation beyond frozen cap'],
              'source_files_sha256': before, 'recovery_script_sha256': sha(Path(__file__)),
              'new_COCO_evaluate_calls_by_recovery': 0, 'A56_total_cached_COCO_evaluate_calls': 3,
              'new_detector_forwards': 0, 'new_optimizer_updates': 0, 'primary_gates_changed': False,
              'limits': load(P/'secondary_analysis_scope_A56_FROZEN_v1.json')['limits']}
    with (P/'secondary_recovered_rows_A56_v1.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'small_4to8': {a: v['groups']['small_4to8'] for a, v in arms.items()},
                      'A56_total_cached_COCO_evaluate_calls': 3, 'new_detector_forwards': 0}))


if __name__ == '__main__':
    main()
