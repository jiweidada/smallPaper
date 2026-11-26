"""Describe class heterogeneity from already verified endpoint precision arrays.

No subgroup is selected as a replacement endpoint or used to alter the gates.
"""
import collections
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
V = D / 'result_verification_v1'


def average_valid(x):
    valid = x[x >= 0]
    return float(valid.mean() * 100) if valid.size else None


def main():
    gate = json.loads((V / 'final_practical_gate_verification.json').read_text(encoding='utf-8'))
    assert gate['status'] == 'PASS'
    gt = json.loads((D / 'completed_results/B/arm_B/epoch12_EMA_GT.json').read_text(encoding='utf-8'))
    categories = sorted(gt['categories'], key=lambda x: x['id'])
    assert len(categories) == 10
    counts = collections.Counter(a['category_id'] for a in gt['annotations'])
    small_counts = collections.Counter(a['category_id'] for a in gt['annotations'] if a['area'] <= 1024)
    metrics = {}
    for arm in ['B', 'M', 'Q']:
        with np.load(V / (arm + '_epoch12_independent_metrics.npz')) as z:
            p = z['precision']
            assert p.shape == (10, 101, 10, 4)
            i75 = np.flatnonzero(np.isclose(z['iou_thresholds'], .75))
            assert len(i75) == 1
            metrics[arm] = [
                {'AP': average_valid(p[:, :, k, 0]), 'APs': average_valid(p[:, :, k, 1]),
                 'APs75': average_valid(p[i75[0], :, k, 1])}
                for k in range(10)]
            assert abs(average_valid(p[:, :, :, 1]) - gate['metrics'][arm]['APs']) < 1e-10
    rows = []
    for k, cat in enumerate(categories):
        row = {'category': cat, 'evaluated_GT': counts[cat['id']],
               'small_GT': small_counts[cat['id']],
               'metrics': {a: metrics[a][k] for a in ['B', 'M', 'Q']}}
        row['deltas_Q_minus_B_pp'] = {m: metrics['Q'][k][m] - metrics['B'][k][m]
                                      if metrics['Q'][k][m] is not None and metrics['B'][k][m] is not None else None
                                      for m in ['AP', 'APs', 'APs75']}
        row['deltas_Q_minus_M_pp'] = {m: metrics['Q'][k][m] - metrics['M'][k][m]
                                      if metrics['Q'][k][m] is not None and metrics['M'][k][m] is not None else None
                                      for m in ['AP', 'APs', 'APs75']}
        rows.append(row)
    output = {'status': 'DESCRIPTIVE_ONLY', 'rows': rows, 'primary_endpoint': 'APs',
              'unit_of_aggregation': 'COCO-style macro precision average across categories/IoU/recall; not a per-instance correctness probability',
              'primary_endpoint_replaced': False, 'additional_hypothesis_tests': 0,
              'new_model_forwards': 0, 'new_optimizer_updates': 0,
              'caution': 'Class-level signs describe heterogeneity only; no subgroup rescue or generalization conclusion.'}
    (V / 'per_class_descriptive_scope.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': output['status'], 'classes': len(rows),
                      'Q_minus_B_APs_positive_classes': sum(r['deltas_Q_minus_B_pp']['APs'] is not None and r['deltas_Q_minus_B_pp']['APs'] > 0 for r in rows),
                      'Q_minus_M_APs_positive_classes': sum(r['deltas_Q_minus_M_pp']['APs'] is not None and r['deltas_Q_minus_M_pp']['APs'] > 0 for r in rows),
                      'Q_minus_B_small_AP75_positive_classes': sum(r['deltas_Q_minus_B_pp']['APs75'] is not None and r['deltas_Q_minus_B_pp']['APs75'] > 0 for r in rows),
                      'Q_minus_M_small_AP75_positive_classes': sum(r['deltas_Q_minus_M_pp']['APs75'] is not None and r['deltas_Q_minus_M_pp']['APs75'] > 0 for r in rows)}))


if __name__ == '__main__':
    main()
