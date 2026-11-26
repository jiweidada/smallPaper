"""Read-only completed checkpoint/log binding; CPU load, never GPU inference."""
import base64
import datetime
import hashlib
import json
import pathlib
import torch

account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
if account == 'A':
    source = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012/outputs/c_mean_full96_seed0')
    status_path = source.parent.parent/'jobs/c_mean_full96_seed0.json'
    status = json.loads(status_path.read_text())
    assert status['state'] in ['trained_pending_independent_eval', 'complete'] and status['returncode'] == 0
    arm, seed = 'C-mean', 0
else:
    source = pathlib.Path('/24085403014/experiments/c_mainline_seed1_full96_20261003_v1/outputs/p2_skip_seed1_full96_affine_epoch3_resume_v1')
    status = None
    arm, seed = 'C', 1
raw = (source/'log.txt').read_bytes()
rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
rows = [row for row in rows if 'test_coco_eval_bbox' in row]
assert rows and max(row['epoch'] for row in rows) == 95
checkpoint = source/'best_stg2.pth'
before = checkpoint.stat()
data = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert isinstance(data.get('ema'), dict) and 'module' in data['ema']
epoch = data['last_epoch']
selected = [row for row in rows if row['epoch'] == epoch]
assert len(selected) == 1
digest = hashlib.sha256()
with checkpoint.open('rb') as stream:
    for chunk in iter(lambda: stream.read(4*1024*1024), b''):
        digest.update(chunk)
after = checkpoint.stat()
assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
result = {'account': account, 'arm': arm, 'seed': seed, 'source': str(source),
          'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_status': status,
          'checkpoint': str(checkpoint), 'checkpoint_sha256': digest.hexdigest(),
          'checkpoint_epoch_zero_based': epoch, 'best_stat': data.get('best_stat'),
          'epochs_in_raw_log': [row['epoch'] for row in rows], 'selected_log_row': selected[0],
          'metrics_raw_0_to_1': selected[0]['test_coco_eval_bbox'],
          'log_sha256': hashlib.sha256(raw).hexdigest(), 'files': []}
for name in ['log.txt', 'args.json', 'round2_train_acceptance.json']:
    path = source/name
    if path.is_file():
        raw = path.read_bytes()
        result['files'].append({'name': name, 'remote': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                                'base64': base64.b64encode(raw).decode()})
print(json.dumps(result))
