"""Audit an ablation prefix, native late restores, and separately saved endpoints."""

import argparse
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys

from verify_ablation_backup import (HERE, audit_evaluation, read, require, sha,
                                    verify_rng_states, verify_tensor_schema)

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / '断点续训_20261105'))
from summarize_completed_pair import BATCHES, EPOCHS, METRICS, endpoint_epochs, finite, validate_restores
from verify_base_endpoint import check_role, tensor_sha


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def validate_prefix_rates(path, rows, config, preflight, resets, stage_one):
    """Use the native LR audit rule with the observed prefix length as its limit."""
    require(config['lrsheduler'] == 'flatcosine' and config['lr_gamma'] == 1.0, 'Unexpected LR schedule')
    base = [group.get('lr', config['optimizer']['lr']) for group in config['optimizer']['params']]
    base.append(config['optimizer']['lr'])
    require(len(base) == preflight['optimizer_groups'] == 4, 'Optimizer group count differs')
    initial = [rate / config['lr_warmup_scheduler']['warmup_duration'] for rate in base]

    def compare(left, right, context):
        require(len(left) == len(right) and all(finite(value) for value in left), context)
        difference = max(abs(actual - expected) for actual, expected in zip(left, right))
        require(difference <= 1e-15, context)
        return difference

    compare(preflight['initial_executed_lrs'], initial, 'Initial constructor scaling differs')
    expected = initial
    count = 0
    maximum = 0.0
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        records = csv.DictReader(stream)
        require(records.fieldnames == ['attempt', 'actual_epoch', 'batch', 'group0', 'group1', 'group2', 'group3'],
                'Unexpected LR columns')
        for attempt, item in enumerate(records):
            epoch, batch = divmod(attempt, BATCHES)
            require(epoch < len(rows) and [int(item[key]) for key in ('attempt', 'actual_epoch', 'batch')]
                    == [attempt, epoch, batch], 'LR cursor differs')
            if batch == 0 and epoch in resets:
                expected = rows[stage_one]['post_final_update_lrs']
            actual = [float(item[f'group{index}']) for index in range(4)]
            maximum = max(maximum, compare(actual, expected, 'Executed LR differs'))
            if batch == 0:
                compare(actual, rows[epoch]['first_executed_lrs'], 'First executed LR differs')
            expected = [rate * (attempt / config['warmup_iter']) ** 2
                        if attempt < config['warmup_iter'] else rate for rate in base]
            if batch == BATCHES - 1:
                compare(expected, rows[epoch]['post_final_update_lrs'], 'Post-update LR differs')
            count += 1
    require(count == len(rows) * BATCHES, 'LR records omit part of the prefix')
    return {'recorded_attempts': count, 'maximum_absolute_difference': maximum, 'groups': 4}


def verify_endpoint(root, folder, role, row, plan, frozen, protocol_sha):
    import torch

    check_role(role, row)
    metadata = read(folder / f'{role}.metrics.json')
    require({key: value for key, value in metadata.items() if key != 'checkpoint_sha256'} == row,
            'Endpoint metrics differ from their epoch row')
    weight = folder / ('best.pt' if role == 'best' else 'terminal_epoch0319.pt')
    artifacts = {'checkpoint_sha256': weight, 'prediction_sha256': folder / f'{role}.predictions.json',
                 'evaluation_arrays_sha256': folder / f'{role}.evaluation_arrays.npz'}
    digests = {key: sha(path) for key, path in artifacts.items()}
    require(all(metadata[key] == value for key, value in digests.items()), 'Endpoint artifact digest differs')
    state = torch.load(weight, map_location='cpu', weights_only=True)
    require('resume_format' not in state and state['arm'] == plan['arm']
            and state['protocol_sha256'] == protocol_sha
            and state['configuration_sha256'] == plan['configuration_sha256']
            and state['environment_sha256'] == frozen['environment.current.json']
            and state['initialization_sha256'] == plan['probe_files']['initial.pt']
            and state['parent_protocol_sha256'] == plan['parent_protocol_sha256']
            and state['weight_source'] == 'ema.module', 'Endpoint provenance differs')
    verify_tensor_schema(root, state, plan)
    require(state['actual_loop_epoch'] == row['epoch'] and state['last_epoch'] == row['last_epoch']
            and state['optimizer_step_attempts'] == row['optimizer_step_attempts']
            and state['ema']['updates'] == row['ema_updates'], 'Endpoint counters differ')
    ema_sha = tensor_sha(state['ema']['module'])
    require(ema_sha == row['ema_tensor_sha256'], 'Endpoint EMA differs from the evaluated tensors')
    require([group['lr'] for group in state['optimizer']['param_groups']] == row['post_final_update_lrs'],
            'Endpoint LR state differs')
    metrics, image_count, predictions, error = audit_evaluation(
        artifacts['prediction_sha256'], artifacts['evaluation_arrays_sha256'], row['metrics'])
    return {'status': 'SAVED_ABLATION_ENDPOINT_VERIFIED', 'role': role, 'epoch_index': row['epoch'],
            'ema_tensor_sha256': ema_sha, 'model_tensor_sha256': tensor_sha(state['model']),
            **digests, 'metrics_sha256': sha(folder / f'{role}.metrics.json'), 'metrics': metrics,
            'validation_images': image_count, 'predictions': predictions, 'maximum_metric_error': error,
            'model_forwards': 0, 'evaluator_runs': 0}


def audit(root, backups, epoch, expected_protocol):
    require(0 <= epoch < EPOCHS, 'Expected an observed zero-based epoch index')
    root = root.resolve(strict=True)
    backups = backups.resolve(strict=True)
    require(root.is_relative_to(HERE) and sha(root / 'execution_protocol.json') == expected_protocol,
            'Unexpected arm or execution protocol')
    plan = read(root / 'execution_protocol.json')
    require(plan['authorized'] is True and plan['arm'] in ('NO_MPS', 'NO_MSCA') and plan['epochs'] == EPOCHS,
            'Unexpected experiment scope')
    frozen = {item['path']: item['sha256'] for item in plan['frozen_files']}
    for name in ('backup_protocol.py', 'train_runner.py', 'fair_helpers.py', 'ablation_model.py',
                 'inputs/preflight.json', 'inputs/first100.json', 'environment.current.json'):
        require(sha(root / name) == frozen[name], 'Frozen analysis input differs at ' + name)
    arm = plan['arm']
    config_path = root / 'execution_configs' / f'{arm}.yml'
    require(sha(config_path) == plan['configuration_sha256'] == frozen[f'execution_configs/{arm}.yml'],
            'Configuration differs')
    config = read(config_path)
    preflight = read(root / 'inputs/preflight.json')
    require(preflight['status'] == 'PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE'
            and preflight['optimizer_updates'] == 0, 'Unexpected preflight')
    module_spec = importlib.util.spec_from_file_location('lineage_backup_protocol', root / 'backup_protocol.py')
    protocol = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(protocol)
    folder = backups / 'epochs' / f'epoch{epoch:04d}'
    payload = (folder / 'manifest.json').read_bytes()
    run_id = Path(config['output_dir']).parent.name
    manifest = protocol.parse_manifest(payload, run_id, epoch)
    ack = protocol.ack_bytes(payload, manifest)
    require((folder / 'ack.json').read_bytes() == ack
            and (folder / 'accepted.json').read_bytes() == protocol.accepted_bytes(payload, manifest),
            'Backup acceptance differs')
    for item in manifest['files']:
        path = protocol.checked_local(folder / item['name'], folder)
        require(protocol.regular(path).st_size == item['bytes'] and sha(path) == item['sha256'],
                'Saved manifest artifact differs')
    protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(folder / name))
    rows = read_rows(folder / 'epochs.jsonl')
    require(len(rows) == epoch + 1 and rows[-1] == read(folder / 'epoch.metrics.json'), 'Incomplete epoch prefix')
    for index, row in enumerate(rows):
        require(row['epoch'] == index and row['optimizer_step_attempts'] == (index + 1) * BATCHES
                and row['protocol_sha256'] == expected_protocol
                and row['config_sha256'] == plan['configuration_sha256']
                and row['weight_source'] == 'ema.module', 'Epoch provenance or budget differs')
        require(finite(row['train_loss_mean']) and set(row['metrics']) == set(METRICS)
                and all(finite(value) and 0 <= value <= 1 for value in row['metrics'].values()),
                'Invalid loss or metrics')
    best = endpoint_epochs(rows)['best']
    stage_one = endpoint_epochs(rows[:296])['best']
    require(manifest['best']['epoch'] == best and manifest['stage_one']['epoch'] == stage_one,
            'Selected endpoint violates highest AP with earliest ties')
    if epoch >= 296:
        events = read_rows(folder / 'restore_events.jsonl')
        selected, resets = validate_restores(rows, events, manifest['stage_one']['checkpoint_sha256'])
        require(selected == stage_one, 'Restore selection differs')
    else:
        events, resets = [], set()
        require(all(row['last_epoch'] == row['epoch'] and row['ema_updates'] == row['optimizer_step_attempts']
                    for row in rows), 'Unexpected pre-stage-two state counters')
    rates = validate_prefix_rates(folder / 'executed_learning_rates.csv.gz', rows, config, preflight, resets, stage_one)
    require(read_rows(folder / 'actual_first100.jsonl') == read(root / 'inputs/first100.json'),
            'Actual first 100 batches differ')
    import torch
    torch.set_num_threads(1)
    rolling = torch.load(folder / 'next_epoch.pt', map_location='cpu', weights_only=True)
    require(rolling['resume_format'] == 'fair_runner_post_policy_epoch_v1'
            and rolling['arm'] == arm and rolling['protocol_sha256'] == expected_protocol
            and rolling['configuration_sha256'] == plan['configuration_sha256']
            and rolling['environment_sha256'] == frozen['environment.current.json']
            and rolling['initialization_sha256'] == plan['probe_files']['initial.pt']
            and rolling['parent_protocol_sha256'] == plan['parent_protocol_sha256']
            and rolling['weight_source'] == 'ema.module'
            and rolling['actual_loop_epoch'] == epoch and rolling['next_epoch'] == epoch + 1
            and rolling['optimizer_step_attempts'] == (epoch + 1) * BATCHES
            and rolling['latest_evaluated_row'] == rows[-1]
            and rolling['best'] == manifest['best'] and rolling['stage_one'] == manifest['stage_one'],
            'Rolling checkpoint binding differs')
    verify_tensor_schema(root, rolling, plan)
    verify_rng_states(rolling['rng_state'])
    post_restore = epoch >= 296 and epoch + 1 in resets
    rolling_row = rows[stage_one] if post_restore else rows[-1]
    require(rolling['last_epoch'] == rolling_row['last_epoch']
            and rolling['ema']['updates'] == rolling_row['ema_updates']
            and tensor_sha(rolling['ema']['module']) == rolling_row['ema_tensor_sha256']
            and [group['lr'] for group in rolling['optimizer']['param_groups']]
                == rolling_row['post_final_update_lrs'], 'Post-policy resume state differs')
    best_folder = folder / protocol.selected_folder(manifest['best'])
    endpoints = {'best': verify_endpoint(root, best_folder, 'best', rows[best], plan, frozen, expected_protocol)}
    stage_endpoint = endpoints['best']
    if stage_one != best:
        stage_folder = folder / protocol.selected_folder(manifest['stage_one'])
        endpoints['stage_one'] = verify_endpoint(root, stage_folder, 'best', rows[stage_one], plan, frozen, expected_protocol)
        stage_endpoint = endpoints['stage_one']
    if post_restore:
        require(tensor_sha(rolling['model']) == stage_endpoint['model_tensor_sha256'],
                'Restored model differs from the selected stage-one model')
    latest_metrics, image_count, predictions, error = audit_evaluation(
        folder / 'latest.predictions.json', folder / 'latest.evaluation_arrays.npz', rows[-1]['metrics'])
    status = 'VERIFIED_PARTIAL_ABLATION_LINEAGE'
    if epoch == EPOCHS - 1:
        endpoints['terminal'] = verify_endpoint(root, folder, 'terminal', rows[epoch], plan, frozen, expected_protocol)
        receipt = read(backups / 'completed_run_receipt.json')
        supervisor = read(backups / 'completed_supervisor.json')
        require(receipt['status'] == 'COMPLETED' and receipt['completed_epochs'] == EPOCHS
                and receipt['actual_loop_epoch'] == EPOCHS - 1 and receipt['next_epoch'] == EPOCHS
                and receipt['optimizer_step_attempts'] == EPOCHS * BATCHES
                and receipt['arm'] == arm and receipt['protocol_sha256'] == expected_protocol
                and receipt['configuration_sha256'] == receipt['config_sha256'] == plan['configuration_sha256']
                and receipt['environment_sha256'] == frozen['environment.current.json']
                and receipt['initialization_sha256'] == plan['probe_files']['initial.pt']
                and receipt['initialization_origin'] == preflight['initialization']
                and receipt['first_epoch_in_this_segment'] == 0
                and receipt['runner_sha256'] == frozen['train_runner.py']
                and receipt['best_epoch'] == best and receipt['best_AP'] == rows[best]['metrics']['AP']
                and receipt['latest_metrics'] == rows[-1]['metrics']
                and receipt['rolling_checkpoint_sha256'] == manifest['rolling_sha256']
                and receipt['local_backup_verified'] == {'epoch': epoch, 'manifest_sha256': protocol.digest(payload),
                                                          'ack_sha256': protocol.digest(ack)},
                'Final receipt does not bind this completed trajectory')
        require(supervisor['status'] == 'COMPLETED' and supervisor['exit_code'] == 0
                and supervisor['arm'] == arm and supervisor['protocol_sha256'] == expected_protocol
                and supervisor['final_receipt'] == receipt and supervisor['pid'] == receipt['pid']
                and supervisor['runner_sha256'] == receipt['runner_sha256'], 'Genuine successful worker exit is missing')
        if not post_restore:
            require(tensor_sha(rolling['model']) == endpoints['terminal']['model_tensor_sha256'],
                    'Unrestored final model differs from its actual evaluated terminal state')
        status = 'VERIFIED_ABLATION_LINEAGE_AND_SAVED_ENDPOINTS'
    return {'status': status, 'arm': arm, 'protocol_sha256': expected_protocol, 'completed_rounds': len(rows),
            'best_epoch_index': best, 'stage_one_best_epoch_index': stage_one, 'restore_event_count': len(events),
            'rolling_state_restored_after_evaluation': post_restore, 'learning_rates': rates, 'endpoints': endpoints,
            'latest_metrics': latest_metrics, 'validation_images': image_count, 'predictions': predictions,
            'maximum_latest_metric_error': error, 'backup_root': str(backups),
            'model_forwards': 0, 'evaluator_runs': 0, 'analysis_script_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm-root', type=Path, required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.arm_root, args.backup_root, args.epoch, args.protocol_sha256)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
