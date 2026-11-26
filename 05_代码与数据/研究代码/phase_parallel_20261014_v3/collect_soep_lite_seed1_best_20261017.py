"""Read one completed job on CPU; bind actual best_stg2 to immutable raw evidence."""
import base64
import datetime
import hashlib
import json
import math
from pathlib import Path

import torch

torch.set_num_threads(1)
root = Path('/24085403014/experiments/phase_mechanism_20261012')
job = 'soep_lite_full96_seed1'
stage = root / 'artifacts/round2_20261014_v1'
pointer = json.loads((stage / 'current_execution.json').read_text())
queue = json.loads((stage / pointer['execution_directory'] / 'queue_state.json').read_text())
code = Path(queue['code_directory'])
status_path = root / 'jobs' / (job + '.json')
status = json.loads(status_path.read_text())
assert status['state'] in ['trained_pending_independent_eval', 'complete'], status['state']
assert status['returncode'] == 0
output = root / 'outputs' / job


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


raw = (output / 'log.txt').read_bytes()
rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
rows = [row for row in rows if 'test_coco_eval_bbox' in row]
assert [row['epoch'] for row in rows] == list(range(96))
checkpoint = output / 'best_stg2.pth'
before = checkpoint.stat()
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert isinstance(state.get('ema'), dict) and 'module' in state['ema']
epoch = state['last_epoch']
selected = [row for row in rows if row['epoch'] == epoch]
assert len(selected) == 1 and state['best_stat']['epoch'] == epoch
metrics = selected[0]['test_coco_eval_bbox']
assert len(metrics) == 12 and all(math.isfinite(x) for x in metrics)
assert abs(state['best_stat']['coco_eval_bbox'] - metrics[0]) < 1e-12
result = {
    'account': 'B', 'job': job, 'arm': 'SOEP-lite', 'seed': 1,
    'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'execution': pointer['execution_directory'], 'code_directory': str(code),
    'common_init_sha256': sha(root / 'artifacts/common_init_seed0.pth'),
    'source': str(output), 'job_status': status, 'checkpoint': str(checkpoint),
    'checkpoint_sha256': sha(checkpoint), 'checkpoint_bytes': before.st_size,
    'checkpoint_epoch_zero_based': epoch, 'best_round_one_based': epoch + 1,
    'best_stat': state['best_stat'], 'checkpoint_state_keys': list(state),
    'epochs_in_raw_log': [row['epoch'] for row in rows],
    'selected_log_row': selected[0], 'metrics_raw_0_to_1': metrics,
    'latest_log_row': rows[-1], 'log_sha256': hashlib.sha256(raw).hexdigest(),
    'selection': 'Actual AP-selected best_stg2 metadata, bound to training EMA validation log; independent inference not performed by this collection.',
    'files': [], 'code_hashes': {},
}
after = checkpoint.stat()
assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
for path in [output / 'log.txt', output / 'args.json',
             output / 'round2_train_acceptance.json', status_path]:
    content = path.read_bytes()
    result['files'].append({'name': path.name, 'remote': str(path),
                            'sha256': hashlib.sha256(content).hexdigest(),
                            'base64': base64.b64encode(content).decode('ascii')})
for relative in ['engine/backbone/hgnetv2_light_phase.py', 'engine/solver/det_solver.py',
                 'engine/solver/_solver.py', 'configs/experiments/phase/proto96.yml',
                 'configs/experiments/phase/soep_lite_full96.yml']:
    path = code / relative
    if path.is_file():
        result['code_hashes'][relative] = sha(path)
        if relative.endswith('det_solver.py') or relative.endswith('.yml'):
            content = path.read_bytes()
            result['files'].append({'name': 'source_' + path.name, 'remote': str(path),
                                    'sha256': hashlib.sha256(content).hexdigest(),
                                    'base64': base64.b64encode(content).decode('ascii')})
print(json.dumps(result, ensure_ascii=False))
