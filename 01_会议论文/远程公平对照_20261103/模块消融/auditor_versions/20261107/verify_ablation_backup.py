"""Independently bind a saved pre-rollback ablation backup on the local CPU."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_tensor_schema(root, state, plan):
    import torch

    require(sha(root / 'inputs/initial.pt') == plan['probe_files']['initial.pt'], 'Initial checkpoint changed')
    initial = torch.load(root / 'inputs/initial.pt', map_location='cpu', weights_only=True)
    require(state['model'].keys() == state['ema']['module'].keys() == initial.keys(), 'Model state keys differ')
    require(all(state['model'][key].shape == state['ema']['module'][key].shape == value.shape
                and state['model'][key].dtype == state['ema']['module'][key].dtype == value.dtype
                for key, value in initial.items()), 'Model tensor shapes or dtypes differ')


def verify_rng_states(rng):
    import numpy as np
    import torch

    require(set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda_all',
                         'train_loader', 'train_sampler', 'validation_loader'}, 'Incomplete RNG state')
    random.Random().setstate(rng['python'])
    name, keys, position, has_gauss, cached = rng['numpy']
    np.random.RandomState().set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    for key in ('torch_cpu', 'train_loader', 'train_sampler', 'validation_loader'):
        torch.Generator(device='cpu').set_state(rng[key])
    require(len(rng['torch_cuda_all']) == 1
            and rng['torch_cuda_all'][0].dtype == torch.uint8
            and rng['torch_cuda_all'][0].numel() > 0, 'Missing CUDA RNG state')


def audit_evaluation(prediction_path, array_path, metrics):
    sys.path.insert(0, str(HERE.parent / '断点续训_20261105'))
    from verify_base_endpoint import summarize_arrays, validation_ids

    images, categories = validation_ids()
    counts = Counter()
    for prediction in read(prediction_path):
        require(prediction['image_id'] in images and prediction['category_id'] in categories,
                'Unknown image or category')
        require(len(prediction['bbox']) == 4 and all(math.isfinite(value) for value in prediction['bbox'])
                and math.isfinite(prediction['score']), 'Invalid prediction values')
        counts[prediction['image_id']] += 1
    require(counts == Counter(dict.fromkeys(images, 300)), 'Unexpected validation coverage')
    summaries = summarize_arrays(array_path)
    error = max(abs(value - metrics[name]) for name, value in summaries.items())
    require(error <= 1e-12, 'Saved arrays differ from the reported metrics')
    return summaries, len(images), sum(counts.values()), error


def verify(root, epoch, expected_protocol, backup_root=None):
    require(0 <= epoch < 296, 'Use separately saved evaluated checkpoints after the rollback boundary')
    root = root.resolve(strict=True)
    require(root.is_relative_to(HERE), 'Choose an arm inside the ablation bundle')
    require(sha(root / 'execution_protocol.json') == expected_protocol, 'Unexpected execution protocol')
    plan = read(root / 'execution_protocol.json')
    frozen = {item['path']: item['sha256'] for item in plan['frozen_files']}
    require(sha(root / 'backup_protocol.py') == frozen['backup_protocol.py'], 'Backup validator changed')
    spec = importlib.util.spec_from_file_location('audited_ablation_protocol', root / 'backup_protocol.py')
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    arm = plan['arm']
    config = read(root / 'execution_configs' / f'{arm}.yml')
    run_id = Path(config['output_dir']).parent.name
    backups = backup_root.resolve(strict=True) if backup_root is not None else root / 'backups'
    folder = backups / 'epochs' / f'epoch{epoch:04d}'
    payload = (folder / 'manifest.json').read_bytes()
    manifest = protocol.parse_manifest(payload, run_id, epoch)
    require((folder / 'ack.json').read_bytes() == protocol.ack_bytes(payload, manifest)
            and (folder / 'accepted.json').read_bytes() == protocol.accepted_bytes(payload, manifest),
            'Local backup and server acceptance differ')
    for item in manifest['files']:
        path = protocol.checked_local(folder / item['name'], folder)
        require(protocol.regular(path).st_size == item['bytes'] and sha(path) == item['sha256'],
                'Saved artifact differs at ' + item['name'])
    protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(folder / name))
    sys.path.insert(0, str(HERE.parent / '断点续训_20261105'))
    from verify_base_endpoint import tensor_sha
    import torch
    torch.set_num_threads(1)
    state = torch.load(folder / 'next_epoch.pt', map_location='cpu', weights_only=True)
    row = read(folder / 'epoch.metrics.json')
    verify_tensor_schema(root, state, plan)
    require(state['arm'] == arm and state['protocol_sha256'] == expected_protocol
            and state['configuration_sha256'] == plan['configuration_sha256']
            and state['environment_sha256'] == frozen['environment.current.json'], 'Checkpoint provenance differs')
    require(state['resume_format'] == 'fair_runner_post_policy_epoch_v1'
            and state['actual_loop_epoch'] == state['last_epoch'] == row['epoch'] == epoch
            and state['next_epoch'] == epoch + 1
            and state['optimizer_step_attempts'] == state['ema']['updates'] == (epoch + 1) * 1617,
            'Rolling state has inconsistent counters')
    require(state['latest_evaluated_row'] == row and state['best'] == manifest['best']
            and state['stage_one'] == manifest['stage_one'], 'Rolling state and evaluation manifest differ')
    require(tensor_sha(state['ema']['module']) == row['ema_tensor_sha256'], 'EMA tensor digest differs')
    verify_rng_states(state['rng_state'])
    summaries, image_count, prediction_count, error = audit_evaluation(
        folder / 'latest.predictions.json', folder / 'latest.evaluation_arrays.npz', row['metrics'])
    rows = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
    require([item['epoch'] for item in rows] == list(range(epoch + 1)) and rows[-1] == row,
            'Epoch prefix is incomplete')
    require(config['lrsheduler'] == 'flatcosine' and config['lr_gamma'] == 1.0, 'Unexpected schedule')
    base = [group.get('lr', config['optimizer']['lr']) for group in config['optimizer']['params']]
    base.append(config['optimizer']['lr'])
    require([group['initial_lr'] for group in state['optimizer']['param_groups']] == base,
            'Optimizer initial learning rates differ')
    preflight = read(root / 'inputs/preflight.json')
    require(all(abs(actual - rate / config['lr_warmup_scheduler']['warmup_duration']) <= 1e-15
                for actual, rate in zip(preflight['initial_executed_lrs'], base)),
            'Initial warmup constructor scaling differs')
    count = 0
    maximum_lr_error = 0.0
    with gzip.open(folder / 'executed_learning_rates.csv.gz', 'rt', newline='') as stream:
        records = csv.DictReader(stream)
        require(records.fieldnames == ['attempt', 'actual_epoch', 'batch', 'group0', 'group1', 'group2', 'group3'],
                'Unexpected learning-rate columns')
        for attempt, item in enumerate(records):
            index, batch = divmod(attempt, 1617)
            require([int(item[key]) for key in ('attempt', 'actual_epoch', 'batch')] == [attempt, index, batch],
                    'Learning-rate log order differs')
            expected = (preflight['initial_executed_lrs'] if attempt == 0 else
                        [rate * ((attempt - 1) / config['warmup_iter']) ** 2
                         if attempt - 1 < config['warmup_iter'] else rate for rate in base])
            actual = [float(item[f'group{group}']) for group in range(4)]
            if batch == 0:
                require(actual == rows[index]['first_executed_lrs'], 'Epoch learning-rate metadata differs')
            difference = max(abs(a - b) for a, b in zip(actual, expected))
            require(difference <= 1e-15, 'Executed learning rate differs')
            maximum_lr_error = max(maximum_lr_error, difference)
            count += 1
    require(count == (epoch + 1) * 1617, 'Learning rates omit part of the training prefix')
    post_update = [group['lr'] for group in state['optimizer']['param_groups']]
    expected_post = [rate * ((count - 1) / config['warmup_iter']) ** 2
                     if count - 1 < config['warmup_iter'] else rate for rate in base]
    require(post_update == row['post_final_update_lrs']
            and all(abs(actual - expected) <= 1e-15 for actual, expected in zip(post_update, expected_post)),
            'Saved post-update learning rates differ')
    first100 = [json.loads(line) for line in (folder / 'actual_first100.jsonl').read_text().splitlines()]
    require(first100 == read(root / 'inputs/first100.json'), 'Actual first 100 batches differ from preflight')
    return {'status': 'SAVED_ABLATION_BACKUP_STATE_AND_METRICS_VERIFIED', 'arm': arm,
            'round': epoch + 1, 'protocol_sha256': expected_protocol,
            'backup_root': str(backups),
            'rolling_checkpoint_sha256': manifest['rolling_sha256'], 'metrics': summaries,
            'validation_images': image_count, 'predictions': prediction_count,
            'rng_states_present': True, 'learning_rate_records': count,
            'maximum_learning_rate_error': maximum_lr_error, 'maximum_metric_error': error,
            'model_forwards': 0, 'evaluator_runs': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm-root', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--backup-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.arm_root, args.epoch, args.protocol_sha256, args.backup_root)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
