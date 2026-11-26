"""CPU-only checkpoint/runtime/recipe audit before authorized continuation."""
import datetime
import hashlib
import json
import pathlib
import sys
import torch

torch.set_num_threads(1)
account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A'
                     else '/24085403014/experiments/phase_mechanism_20261012')
stage = phase/'artifacts/round2_20261014_v1/execution_v3'
plan = json.loads((stage/'queue_plan.json').read_text())
old = json.loads((stage/'queue_state.json').read_text())
result = {'account': account, 'python': sys.executable, 'torch_version': torch.__version__,
          'cuda_build': torch.version.cuda, 'gpu_available': torch.cuda.is_available(),
          'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'common_init_sha256': hashlib.sha256((phase/'artifacts/common_init_seed0.pth').read_bytes()).hexdigest(),
          'resume_jobs': [], 'completed_jobs': old['training_completed_jobs'],
          'source_excerpts': {}, 'code_hashes': {}}
assert result['common_init_sha256'] == plan['common_init_sha256'] and result['gpu_available']
for job in old['active_jobs']:
    output = phase/'outputs'/job
    checkpoint = output/'last.pth'
    before = checkpoint.stat()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert all(k in state for k in ['model', 'ema', 'optimizer', 'scaler', 'last_epoch']), (job, list(state))
    assert state['optimizer']['state'] and state['ema']['module'] and state['scaler']
    values = [json.loads(line) for line in (output/'log.txt').read_text().splitlines() if line.strip()]
    values = [v for v in values if 'test_coco_eval_bbox' in v]
    epochs = [v['epoch'] for v in values]
    assert epochs == list(range(max(epochs)+1)), job
    assert state['last_epoch'] == max(epochs), (job, state['last_epoch'], max(epochs))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    after = checkpoint.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    args = json.loads((output/'args.json').read_text())
    assert args['epoches'] == 96 and args['seed'] == int(job.split('seed')[-1])
    result['resume_jobs'].append({'job': job, 'checkpoint': str(checkpoint), 'sha256': digest,
        'last_epoch_zero_based': state['last_epoch'], 'next_round_one_based': state['last_epoch']+2,
        'keys': list(state), 'optimizer_state_entries': len(state['optimizer']['state']),
        'ema_updates': state['ema'].get('updates'), 'scaler': state['scaler'],
        'optimizer_groups': [{'lr': v['lr'], 'initial_lr': v.get('initial_lr'), 'weight_decay': v.get('weight_decay'),
                              'parameter_count': len(v['params'])} for v in state['optimizer']['param_groups']],
        'args_sha256': hashlib.sha256((output/'args.json').read_bytes()).hexdigest(),
        'logged_epochs': epochs,
        'rng_state_keys': [k for k in state if 'rng' in k.lower() or 'random' in k.lower()]})
    del state
for rel, ranges in [('engine/solver/det_solver.py', [(285, 330), (650, 735), (870, 1038), (1470, 1535)]),
                    ('engine/solver/_solver.py', [(70, 136), (153, 209)]), ('train.py', [(30, 130)])]:
    path = stage/'code'/rel
    raw = path.read_bytes(); lines = raw.decode('utf-8').splitlines()
    result['source_excerpts'][rel] = [{'line': i+1, 'text': line} for i, line in enumerate(lines)
                                     if any(lo <= i+1 <= hi for lo, hi in ranges)]
    result['code_hashes'][rel] = hashlib.sha256(raw).hexdigest()
for rel in ['engine/backbone/hgnetv2_light_phase.py', 'configs/experiments/phase/proto96.yml']:
    result['code_hashes'][rel] = hashlib.sha256((stage/'code'/rel).read_bytes()).hexdigest()
print(json.dumps(result, ensure_ascii=False))
