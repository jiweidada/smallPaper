"""One predeclared descriptive pass over saved B12 boxes, paired to saved B4/B8."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
from saved_B4_geometry_coverage_v1 import summarize

P = Path(__file__).resolve().parent
SCOPE = P / 'saved_B12_geometry_coverage_scope_FROZEN_v1.json'
OUT = P / 'saved_B12_geometry_coverage_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def paired(mask, old_iou, new_iou):
    old, new = old_iou[mask] >= .5, new_iou[mask] >= .5
    count = len(old)
    transitions = {'covered_both': int((old & new).sum()), 'gained_coverage': int((~old & new).sum()),
                   'lost_coverage': int((old & ~new).sum()), 'uncovered_both': int((~old & ~new).sum())}
    assert sum(transitions.values()) == count
    return {'GT': count, 'counts': transitions,
            'old_no_any_coverage_percent': 100 * int((~old).sum()) / count if count else None,
            'new_no_any_coverage_percent': 100 * int((~new).sum()) / count if count else None,
            'no_coverage_change_pp': 100 * (int((~new).sum()) - int((~old).sum())) / count if count else None}


def main():
    scope = read(SCOPE)
    assert not OUT.exists() and sha(Path(__file__)) == scope['script_sha256']
    for name, expected in scope['input_sha256'].items():
        assert sha(P / name) == expected, name
    gt, predictions = read(P / scope['GT_file']), read(P / scope['predictions_file'])
    assert len(gt['images']) == 548 and len(gt['annotations']) == 38602 and len(predictions) == 81505
    by_gt, by_pred = defaultdict(list), defaultdict(list)
    for item in gt['annotations']:
        by_gt[item['image_id']].append(item)
    for item in predictions:
        by_pred[item['image_id']].append(item)
    rows, pairs = [], 0
    for image in gt['images']:
        targets, proposals = by_gt[image['id']], by_pred[image['id']]
        gb = np.asarray([x['bbox'] for x in targets], dtype=np.float64).reshape(-1, 4)
        pb = np.asarray([x['bbox'] for x in proposals], dtype=np.float64).reshape(-1, 4)
        gl = np.asarray([x['category_id'] for x in targets]); pl = np.asarray([x['category_id'] for x in proposals])
        pairs += len(gb) * len(pb)
        if len(gb) and len(pb):
            lower = np.maximum(gb[:, None, :2], pb[None, :, :2])
            upper = np.minimum(gb[:, None, :2] + gb[:, None, 2:], pb[None, :, :2] + pb[None, :, 2:])
            sides = np.maximum(upper - lower, 0); intersection = sides[..., 0] * sides[..., 1]
            union = (gb[:, 2] * gb[:, 3])[:, None] + (pb[:, 2] * pb[:, 3])[None, :] - intersection
            iou = intersection / union
            best_any = iou.max(axis=1); best_same = np.where(gl[:, None] == pl[None, :], iou, 0).max(axis=1)
        else:
            best_any = np.zeros(len(gb)); best_same = np.zeros(len(gb))
        assert np.isfinite(best_any).all() and np.isfinite(best_same).all() and (best_same <= best_any + 1e-12).all()
        for i, target in enumerate(targets):
            x, y, w, h = target['bbox']; short = min(w * 640 / image['width'], h * 640 / image['height'])
            rows.append((target['id'], image['id'], target['category_id'], target['area'], short, best_any[i], best_same[i]))
    values = np.asarray(rows, dtype=np.float64); assert len(values) == 38602
    area, short, any_iou, same_iou = values[:, 3], values[:, 4], values[:, 5], values[:, 6]
    small = area <= 1024; very_tiny = small & (short <= 4)
    assert int(small.sum()) == 26444 and int(very_tiny.sum()) == 5361
    old = {}
    for label, name in [('B4', scope['B4_per_GT_file']), ('B8', scope['B8_per_GT_file'])]:
        with np.load(P / name) as arr:
            assert np.array_equal(arr['GT_id'], values[:, 0].astype(np.int64))
            assert np.array_equal(arr['image_id'], values[:, 1].astype(np.int64))
            assert np.array_equal(arr['category_id'], values[:, 2].astype(np.int64))
            assert np.array_equal(arr['original_area'], area) and np.array_equal(arr['input_short_side'], short)
            old[label] = arr['best_any_class_IoU'].copy()
    masks = {'all': np.ones(len(values), dtype=bool), 'small': small, 'very_tiny': very_tiny}
    groups = {name: summarize(mask, any_iou, same_iou) for name, mask in masks.items()}
    transitions = {name: paired(mask, old['B8'], any_iou) for name, mask in masks.items()}
    uncovered_all_three = {name: {'GT': int(mask.sum()),
        'count': int((mask & (old['B4'] < .5) & (old['B8'] < .5) & (any_iou < .5)).sum())}
        for name, mask in masks.items()}
    for value in uncovered_all_three.values():
        value['percent'] = 100 * value['count'] / value['GT'] if value['GT'] else None
    classes = {str(c): summarize(small & (values[:, 2] == c), any_iou, same_iou) for c in range(1, 11)}
    OUT.mkdir()
    save(OUT / 'run_once.json', {'scope_sha256': sha(SCOPE), 'new_model_forwards': 0, 'new_optimizer_updates': 0})
    np.savez_compressed(OUT / 'per_GT_coverage.npz', GT_id=values[:, 0].astype(np.int64),
                        image_id=values[:, 1].astype(np.int64), category_id=values[:, 2].astype(np.int64),
                        original_area=area, input_short_side=short,
                        best_any_class_IoU=any_iou, best_same_class_IoU=same_iou)
    result = {'status': 'COMPLETED_SAVED_B12_PAIRED_GT_GEOMETRY_COVERAGE',
              'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
              'scope_sha256': sha(SCOPE), 'input_sha256': scope['input_sha256'],
              'images': 548, 'predictions': len(predictions), 'GT': len(values),
              'GT_prediction_IoU_pairs': pairs, 'groups': groups,
              'B8_B12_paired_GT_transitions': transitions, 'uncovered_all_B4_B8_B12': uncovered_all_three,
              'small_by_class': classes, 'new_model_forwards': 0, 'new_optimizer_updates': 0,
              'new_COCO_evaluations': 0, 'new_image_reads': 0, 'causal_error_decomposition': False,
              'formal_TIDE_analysis': False, 'AP_or_AR_recomputation': False,
              'limitations': scope['limitations'], 'training_protocol_or_gates_changed': False}
    save(OUT / 'result.json', result)
    report = ('B12保存预测的几何覆盖与B8配对比较\n'
              '各GT独立取任意类别最佳IoU，同一预测可复用；这是乐观的候选覆盖，不是标准召回、漏检率、AP或TIDE。\n')
    for name, row in transitions.items():
        report += (f"\n{name}，{row['GT']}个GT：任意类别均无IoU≥0.5保留预测，"
                   f"B8 {row['old_no_any_coverage_percent']:.3f}%，B12 {row['new_no_any_coverage_percent']:.3f}%，"
                   f"变化 {row['no_coverage_change_pp']:+.3f}个百分点；"
                   f"B4/B8/B12均未覆盖 {uncovered_all_three[name]['count']}个。\n")
    report += ('\n这是同一基线继续训练的描述，不能归因于时间条件或证明候选Q有效。'
               '原有top300/NMS/ignore过滤后的未保留框不可见；原因可能涉及信息、候选框、置信度、后处理或解码。'
               '未新增模型前向、优化器更新、COCO重评或原图读取；M/Q对照与门槛不变。\n')
    with (OUT / '01_终点覆盖画像与解释边界.txt').open('x', encoding='utf-8', newline='\n') as target:
        target.write(report)
    print(json.dumps({'status': result['status'], 'small': transitions['small'],
                      'very_tiny': transitions['very_tiny'], 'all_three': uncovered_all_three,
                      'pairs': pairs, 'new_forwards': 0, 'new_updates': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
