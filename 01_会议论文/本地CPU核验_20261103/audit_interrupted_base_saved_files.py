"""Audit saved interrupted-run artifacts on CPU; no model or evaluator execution."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import math
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch

PAPER = Path(__file__).resolve().parent.parent
WORK = PAPER.parent
REMOTE = PAPER / '远程公平对照_20261103'
BASE = REMOTE / 'remote_evidence/interrupted_BASE_20261105'
RUN = BASE / 'runs/BASE'
ANNOTATIONS = WORK / '05_代码与数据/原始论文数据/03_本轮实验记录_20260928/转换COCO标注原件_20260929/VisDrone2019-DET-val/annotations/val.json'
REPORT = REMOTE / 'interrupted_base_saved_file_audit_20261105.json'
PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
ANNOTATION_SHA = '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def summaries(path):
    with np.load(path, allow_pickle=False) as arrays:
        p, r = arrays['precision'], arrays['recall']
        assert p.shape == (10, 101, 10, 4, 3) and r.shape == (10, 10, 4, 3)
        selected = [p[:, :, :, 0, 2], p[0, :, :, 0, 2], p[5, :, :, 0, 2],
                    p[:, :, :, 1, 2], p[:, :, :, 2, 2], p[:, :, :, 3, 2],
                    r[:, :, 0, 0], r[:, :, 0, 1], r[:, :, 0, 2],
                    r[:, :, 1, 2], r[:, :, 2, 2], r[:, :, 3, 2]]
        names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
        def mean_valid(a):
            values = a[a > -1]
            return float(np.mean(values)) if values.size else -1.0
        return dict(zip(names, map(mean_valid, selected)))

torch.set_num_threads(1)
assert not REPORT.exists()
verified = load(REMOTE / 'interrupted_base_download_verified_20261105.json')
assert verified['status'] == 'INCOMPLETE_BASE_ARTIFACTS_PRESERVED_HASH_MATCHED'
assert sha(BASE / 'execution_protocol.json') == PROTOCOL
assert sha(ANNOTATIONS) == ANNOTATION_SHA
assert sha(BASE / 'execution_configs/BASE.yml') == '1eed16436115ab62655c101e42f180e87fabb2f1a9e55d07b1d5241ce0070cdd'
rows = [json.loads(line) for line in (RUN / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
best, last = load(RUN / 'best.metrics.json'), rows[-1]
assert best['epoch'] == 213 and last['epoch'] == 224 and len(rows) == 225
assert sha(RUN / 'best.pt') == best['checkpoint_sha256']
assert best['protocol_sha256'] == last['protocol_sha256'] == PROTOCOL
assert best['config_sha256'] == last['config_sha256'] == sha(BASE / 'execution_configs/BASE.yml')
assert {k: v for k, v in best.items() if k != 'checkpoint_sha256'} == rows[best['epoch']]
for prefix, row in [('best', best), ('latest', last)]:
    assert sha(RUN / (prefix + '.predictions.json')) == row['prediction_sha256']
    assert sha(RUN / (prefix + '.evaluation_arrays.npz')) == row['evaluation_arrays_sha256']

ckpt = torch.load(RUN / 'best.pt', map_location='cpu', weights_only=True)
assert ckpt['arm'] == 'BASE' and ckpt['weight_source'] == 'ema.module'
assert ckpt['actual_loop_epoch'] == best['epoch'] and ckpt['last_epoch'] == best['last_epoch']
assert ckpt['optimizer_step_attempts'] == best['optimizer_step_attempts'] == (best['epoch'] + 1) * 1617
assert ckpt['protocol_sha256'] == PROTOCOL and ckpt['configuration_sha256'] == best['config_sha256']
assert ckpt['environment_sha256'] == sha(BASE / 'environment.current.json')
assert ckpt['ema']['updates'] == best['ema_updates']
h = sha256()
for key, value in sorted(ckpt['ema']['module'].items()):
    tensor = value.detach().cpu().contiguous()
    h.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
    h.update(tensor.numpy().tobytes())
assert h.hexdigest() == best['ema_tensor_sha256']

data = load(ANNOTATIONS)
image_ids = {x['id'] for x in data['images']}
categories = {x['id'] for x in data['categories']}
assert len(image_ids) == 548 and len(categories) == 10
results = {}
for prefix, row in [('best', best), ('latest', last)]:
    predictions = load(RUN / (prefix + '.predictions.json'))
    counts = dict.fromkeys(image_ids, 0)
    assert len(predictions) == 164400
    for prediction in predictions:
        assert prediction['image_id'] in image_ids and prediction['category_id'] in categories
        assert len(prediction['bbox']) == 4 and all(math.isfinite(v) for v in prediction['bbox'])
        assert math.isfinite(prediction['score'])
        counts[prediction['image_id']] += 1
    assert set(counts.values()) == {300}
    values = summaries(RUN / (prefix + '.evaluation_arrays.npz'))
    delta = {k: values[k] - row['metrics'][k] for k in values}
    assert max(abs(x) for x in delta.values()) <= 1e-12
    results[prefix] = {'actual_epoch_zero_based': row['epoch'],
                       'prediction_sha256': row['prediction_sha256'],
                       'evaluation_arrays_sha256': row['evaluation_arrays_sha256'],
                       'maximum_absolute_summary_difference': max(abs(x) for x in delta.values()),
                       'metrics': row['metrics'], 'matching_checkpoint_available': prefix == 'best'}
report = {'status': 'BEST_CHECKPOINT_BOUND_LAST_ARRAYS_ONLY',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'verified_download_sha256': sha(REMOTE / 'interrupted_base_download_verified_20261105.json'),
          'annotation_sha256': ANNOTATION_SHA, 'protocol_sha256': PROTOCOL,
          'best_checkpoint_sha256': best['checkpoint_sha256'], 'best_ema_tensor_sha256': h.hexdigest(),
          'image_count': 548, 'predictions_per_image': 300, 'array_summary_tolerance': 1e-12,
          'saved_file_results': results, 'python_packages': {'torch': torch.__version__, 'numpy': np.__version__},
          'scope': 'Saved-file integrity and summaries only. Best epoch 213 has a matching checkpoint; epoch 224 has saved predictions and arrays but no matching checkpoint.',
          'model_inference_passes': 0, 'optimizer_updates': 0, 'evaluator_reruns': 0,
          'completed_pair': False, 'submission_ready': False}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'],
                  'best_max_difference': results['best']['maximum_absolute_summary_difference'],
                  'last_max_difference': results['latest']['maximum_absolute_summary_difference'],
                  'best_AP': results['best']['metrics']['AP'], 'last_AP': results['latest']['metrics']['AP']}))
