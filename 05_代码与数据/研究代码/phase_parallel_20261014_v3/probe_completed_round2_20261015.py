"""Bind all completed round-two best_stg2 files to raw log rows on CPU."""
import base64
import datetime
import hashlib
import json
import math
import pathlib
import torch

torch.set_num_threads(1)
account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
root = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A'
                    else '/24085403014/experiments/phase_mechanism_20261012')
stage_root = root/'artifacts/round2_20261014_v1'
execution = json.loads((stage_root/'current_execution.json').read_text())['execution_directory']
queue = json.loads((stage_root/execution/'queue_state.json').read_text())
code_directory = pathlib.Path(queue.get('code_directory', str(stage_root/execution/'code')))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


result = {'account': account, 'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'execution': execution, 'common_init_sha256': sha(root/'artifacts/common_init_seed0.pth'),
          'selection': 'actual best_stg2 metadata epoch, AP-selected training EMA validation; not independent inference',
          'runs': [], 'unfinished': []}
for spec in queue['planned_jobs']:
    arm, seed_text = spec.split(':')
    job = arm+'_seed'+seed_text
    status_path = root/'jobs'/(job+'.json')
    status = json.loads(status_path.read_text()) if status_path.is_file() else {'state': 'pending'}
    if status.get('state') not in ['trained_pending_independent_eval', 'complete']:
        result['unfinished'].append({'job': job, 'state': status.get('state')})
        continue
    assert status['returncode'] == 0, job
    output = root/'outputs'/job
    raw = (output/'log.txt').read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = [row for row in rows if 'test_coco_eval_bbox' in row]
    assert sorted(row['epoch'] for row in rows) == list(range(96)), job
    checkpoint = output/'best_stg2.pth'
    before = checkpoint.stat()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert isinstance(state.get('ema'), dict) and 'module' in state['ema'], job
    epoch = state['last_epoch']
    selected = [row for row in rows if row['epoch'] == epoch]
    assert len(selected) == 1 and state['best_stat']['epoch'] == epoch, job
    metrics = selected[0]['test_coco_eval_bbox']
    assert len(metrics) == 12 and all(math.isfinite(x) for x in metrics), job
    row = {'job': job, 'account': account, 'arm': arm, 'seed': int(seed_text),
           'source': str(output), 'job_status': status, 'checkpoint': str(checkpoint),
           'checkpoint_sha256': sha(checkpoint), 'checkpoint_epoch_zero_based': epoch,
           'best_round_one_based': epoch+1, 'best_stat': state['best_stat'],
           'checkpoint_state_keys': list(state), 'epochs_in_raw_log': [r['epoch'] for r in rows],
           'selected_log_row': selected[0], 'metrics_raw_0_to_1': metrics,
           'log_sha256': hashlib.sha256(raw).hexdigest(), 'files': []}
    after = checkpoint.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), job
    for path in [output/'log.txt', output/'args.json', output/'round2_train_acceptance.json', status_path]:
        content = path.read_bytes()
        row['files'].append({'name': path.name, 'remote': str(path),
                             'sha256': hashlib.sha256(content).hexdigest(),
                             'base64': base64.b64encode(content).decode('ascii')})
    row['code_hashes'] = {}
    for relative in ['engine/backbone/hgnetv2_light_phase.py', 'engine/solver/det_solver.py',
                     'engine/solver/_solver.py', 'configs/experiments/phase/proto96.yml',
                     'configs/experiments/phase/'+arm+'.yml']:
        path = code_directory/relative
        if path.is_file():
            row['code_hashes'][relative] = sha(path)
    result['runs'].append(row)
    del state
print(json.dumps(result, ensure_ascii=False))
