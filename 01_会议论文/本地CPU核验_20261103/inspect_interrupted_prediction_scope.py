"""Summarize already-saved evaluation arrays and predictions, without running an evaluator."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import numpy as np

PAPER = Path(__file__).resolve().parent.parent
WORK = PAPER.parent
REMOTE = PAPER / '远程公平对照_20261103'
RUN = REMOTE / 'remote_evidence/interrupted_BASE_20261105/runs/BASE'
AUDIT = REMOTE / 'interrupted_base_saved_file_audit_20261105.json'
COCO = PAPER / '修订记录_20261103/第十一轮评价上限核查/sources/cocoeval.py'
FAST = PAPER / '修订记录_20261103/第十一轮评价上限核查/sources/faster_eval_api.py'
WRAPPER = PAPER / '修订记录_20261103/第十一轮评价上限核查/sources/coco_eval.py'
ANNOTATIONS = WORK / '05_代码与数据/原始论文数据/03_本轮实验记录_20260928/转换COCO标注原件_20260929/VisDrone2019-DET-val/annotations/val.json'
REPORT = REMOTE / 'interrupted_base_prediction_scope_20261105.json'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def per_image_top(path, image_ids):
    result = {i: float('-inf') for i in image_ids}
    preds = load(path)
    for p in preds:
        i = p['image_id']
        assert i in result
        result[i] = max(result[i], p['score'])
    assert len(preds) == 164400 and all(np.isfinite(v) for v in result.values())
    return result

assert not REPORT.exists()
audit = load(AUDIT)
assert audit['status'] == 'BEST_CHECKPOINT_BOUND_LAST_ARRAYS_ONLY'
assert sha(ANNOTATIONS) == audit['annotation_sha256']
assert sha(COCO) == '03ef7d1a4259ac4f2291d5aa6c164663edc98e72f99f73b1107dcada20888ed9'
assert sha(FAST) == '3712390832ee269228e2e989107734e317044f682bae40417c0596491485a963'
assert sha(WRAPPER) == '18f3c02252c2d721f09e88e7807f3d81dfd5d4a45b69564063e9c28344a11407'
ann = load(ANNOTATIONS)
categories = sorted(ann['categories'], key=lambda c: c['id'])
image_ids = {i['id'] for i in ann['images']}
assert len(categories) == 10 and len(image_ids) == 548
class_values = {}
image_max = {}
for label in ['best', 'latest']:
    row = audit['saved_file_results'][label]
    arrays_path = RUN / (label + '.evaluation_arrays.npz')
    predictions_path = RUN / (label + '.predictions.json')
    assert sha(arrays_path) == row['evaluation_arrays_sha256']
    assert sha(predictions_path) == row['prediction_sha256']
    with np.load(arrays_path, allow_pickle=False) as array:
        precision = array['precision']
        assert precision.shape == (10, 101, 10, 4, 3)
        values = []
        for k in range(10):
            v = precision[:, :, k, 0, 2]
            good = v[v > -1]
            assert good.size == 10 * 101
            values.append(float(np.mean(good)))
        assert abs(float(np.mean(values)) - row['metrics']['AP']) <= 1e-12
        class_values[label] = values
    image_max[label] = per_image_top(predictions_path, image_ids)
classes = [{'category_id': c['id'], 'category_name': c['name'],
            'best_AP': class_values['best'][i], 'latest_AP': class_values['latest'][i],
            'AP_difference_latest_minus_best': class_values['latest'][i] - class_values['best'][i]}
           for i, c in enumerate(categories)]
differences = [image_max['latest'][i] - image_max['best'][i] for i in image_ids]
report = {'status': 'SAVED_PREDICTION_AND_ARRAY_SCOPE_DESCRIBED',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'prior_audit_sha256': sha(AUDIT), 'annotation_sha256': sha(ANNOTATIONS),
          'category_order_evidence': {'cocoeval_sha256': sha(COCO), 'faster_eval_api_sha256': sha(FAST),
              'project_wrapper_sha256': sha(WRAPPER),
              'scope': 'The inspected evaluator initializes and sorts category IDs, and the project wrapper does not replace them; arrays themselves do not store category IDs.'},
          'actual_epoch_zero_based': {'best': 213, 'latest': 224},
          'per_category_saved_precision_summary': classes,
          'images': len(image_ids), 'images_with_lower_latest_top_score': sum(x < 0 for x in differences),
          'images_with_equal_latest_top_score': sum(x == 0 for x in differences),
          'images_with_higher_latest_top_score': sum(x > 0 for x in differences),
          'median_image_top_score': {k: float(np.median(list(v.values()))) for k, v in image_max.items()},
          'limits': ['This describes saved outputs from two epochs of one interrupted run, not independent trials.',
                     'Per-image maximum scores need not correspond to the same object or category across epochs.',
                     'The latest epoch has no corresponding saved checkpoint, and these patterns do not establish a cause.'],
          'model_inference_passes': 0, 'evaluator_reruns': 0, 'optimizer_updates': 0}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'],
                  'categories_with_lower_saved_AP': sum(x['AP_difference_latest_minus_best'] < 0 for x in classes),
                  'images_with_lower_latest_top_score': report['images_with_lower_latest_top_score'],
                  'median_image_top_score': report['median_image_top_score']}))
