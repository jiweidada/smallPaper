"""Describe saved BASE logs and predictions; do not run inference or evaluation."""
from collections import Counter, defaultdict
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import csv
import gzip
import json
import math

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'remote_evidence/interrupted_BASE_20261105'
RUN = BASE / 'runs/BASE'
VERIFIED = ROOT / 'interrupted_base_download_verified_20261105.json'
REPORT = ROOT / 'interrupted_base_analysis_20261105.json'
PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def quantiles(items):
    a = sorted(items)
    return {str(q): a[int(q * (len(a) - 1))] for q in [0, .1, .5, .9, .99, 1]}

def prediction_description(path):
    data = load(path)
    scores, widths, heights, areas = [], [], [], []
    categories = Counter()
    per_image = Counter()
    image_max = defaultdict(lambda: -math.inf)
    nonfinite, nonpositive = 0, 0
    for item in data:
        score = item['score']
        x, y, w, h = item['bbox']
        vals = [score, x, y, w, h]
        if not all(math.isfinite(float(v)) for v in vals):
            nonfinite += 1
            continue
        scores.append(score)
        widths.append(w)
        heights.append(h)
        areas.append(w * h)
        nonpositive += w <= 0 or h <= 0
        categories[str(item['category_id'])] += 1
        per_image[str(item['image_id'])] += 1
        image_max[str(item['image_id'])] = max(image_max[str(item['image_id'])], score)
    assert len(data) == 164400 and len(per_image) == 548
    return {'file_sha256': sha(path), 'predictions': len(data), 'images': len(per_image),
            'per_image_count_min_max': [min(per_image.values()), max(per_image.values())],
            'category_counts': dict(sorted(categories.items(), key=lambda x: int(x[0]))),
            'score_quantiles': quantiles(scores), 'top_score_per_image_quantiles': quantiles(list(image_max.values())),
            'box_width_quantiles': quantiles(widths), 'box_height_quantiles': quantiles(heights),
            'box_area_quantiles': quantiles(areas), 'nonfinite_rows': nonfinite,
            'nonpositive_box_dimensions': nonpositive}

assert not REPORT.exists()
verified = load(VERIFIED)
assert verified['status'] == 'INCOMPLETE_BASE_ARTIFACTS_PRESERVED_HASH_MATCHED'
assert verified['protocol_sha256'] == PROTOCOL
for item in verified['files']:
    path = BASE.joinpath(*item['relative_path'].split('/'))
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
rows = [json.loads(line) for line in (RUN / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
receipt = load(RUN / 'receipt.json')
best = load(RUN / 'best.metrics.json')
assert len(rows) == receipt['completed_epochs'] == 225
assert all(row['epoch'] == i and row['optimizer_step_attempts'] == (i + 1) * 1617
           and row['protocol_sha256'] == PROTOCOL for i, row in enumerate(rows))
assert receipt['actual_loop_epoch'] == 225 and receipt['optimizer_step_attempts'] == 363926
assert best['epoch'] == receipt['best_epoch'] == max(range(len(rows)), key=lambda i: rows[i]['metrics']['AP'])
assert {k: v for k, v in best.items() if k != 'checkpoint_sha256'} == rows[best['epoch']]
assert sha(RUN / 'best.pt') == best['checkpoint_sha256']
for prefix, row in [('best', best), ('latest', rows[-1])]:
    assert sha(RUN / (prefix + '.predictions.json')) == row['prediction_sha256']
    assert sha(RUN / (prefix + '.evaluation_arrays.npz')) == row['evaluation_arrays_sha256']

lr_count, lr_last, truncated = 0, None, False
try:
    with gzip.open(RUN / 'executed_learning_rates.csv.gz', 'rt', encoding='utf-8', newline='') as f:
        for item in csv.DictReader(f):
            assert int(item['attempt']) == lr_count
            lr_count += 1
            lr_last = item
except EOFError:
    truncated = True

predictions = {prefix: prediction_description(RUN / (prefix + '.predictions.json'))
               for prefix in ['best', 'latest']}
report = {'status': 'INTERRUPTED_BASE_SAVED_FILES_DESCRIBED_NO_CAUSAL_CONCLUSION',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'verified_download_sha256': sha(VERIFIED), 'protocol_sha256': PROTOCOL,
          'completed_epochs': len(rows), 'actual_epoch_indices': [0, rows[-1]['epoch']],
          'last_reported_partial_epoch': receipt['actual_loop_epoch'],
          'last_reported_partial_batch': receipt['batch'],
          'best_saved': {'epoch_zero_based': best['epoch'], 'AP': best['metrics']['AP'],
                         'checkpoint_sha256': best['checkpoint_sha256']},
          'last_complete': {'epoch_zero_based': rows[-1]['epoch'], 'AP': rows[-1]['metrics']['AP'],
                            'ema_tensor_sha256': rows[-1]['ema_tensor_sha256']},
          'recent_epochs': [{'epoch_zero_based': r['epoch'], 'AP': r['metrics']['AP'],
                             'train_loss_mean': r['train_loss_mean'],
                             'first_executed_lrs': r['first_executed_lrs'],
                             'post_final_update_lrs': r['post_final_update_lrs']}
                            for r in rows[210:]],
          'saved_lr_stream': {'complete_rows_read': lr_count, 'last_complete_row': lr_last,
                              'gzip_missing_end_marker': truncated},
          'saved_predictions': predictions,
          'interpretation': ['The stored single-run validation AP decreases sharply after epoch 216; the cause is not established.',
                             'Logged training loss and per-epoch boundary learning rates alone do not identify a cause.',
                             'The last complete epoch has predictions and evaluator arrays but no saved matching last-epoch checkpoint in this interrupted run.',
                             'The selected best checkpoint is validation-selected and cannot replace the planned 320-epoch paired endpoint.'],
          'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'], 'best': report['best_saved'], 'last_complete': report['last_complete'],
                  'lr_rows': lr_count, 'lr_gzip_missing_end_marker': truncated,
                  'best_score_q50': predictions['best']['score_quantiles']['0.5'],
                  'latest_score_q50': predictions['latest']['score_quantiles']['0.5']}))
