"""Audit existing completed-pair records with the standard library only.

No training, inference, evaluator execution, retries, or manuscript editing.
Run only after the four final saved-endpoint audits have succeeded. A running
queue returns NOT_READY without writing a result. This script is analysis code,
not part of the already frozen training driver.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys

PROTOCOL_SHA = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
EPOCHS, BATCHES = 320, 1617
METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    # A partial trailing row is a failure here, unlike the live status monitor.
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_epoch_rows(rows, config_sha):
    require(len(rows) == EPOCHS, 'Exactly 320 complete epoch rows are required')
    for epoch, row in enumerate(rows):
        require(row['epoch'] == epoch, f'Missing, reordered or duplicated epoch {epoch}')
        require(row['optimizer_step_attempts'] == (epoch + 1) * BATCHES, f'Update budget at epoch {epoch}')
        require(row['protocol_sha256'] == PROTOCOL_SHA and row['config_sha256'] == config_sha,
                f'Configuration binding at epoch {epoch}')
        require(row['weight_source'] == 'ema.module', f'Weight source at epoch {epoch}')
        require(finite(row['train_loss_mean']), f'Nonfinite mean loss at epoch {epoch}')
        require(set(row['metrics']) == set(METRICS), f'Incomplete metric vector at epoch {epoch}')
        require(all(finite(row['metrics'][key]) and 0 <= row['metrics'][key] <= 1 for key in METRICS),
                f'Invalid or unavailable metrics at epoch {epoch}')
        for field in ['first_executed_lrs', 'post_final_update_lrs']:
            require(len(row[field]) == 4 and all(finite(v) and v >= 0 for v in row[field]),
                    f'Invalid learning rates at epoch {epoch}')


def endpoint_epochs(rows):
    # max returns the earliest matching row, implementing the frozen tie rule.
    return {'terminal': EPOCHS - 1, 'best': max(range(len(rows)), key=lambda i: rows[i]['metrics']['AP'])}


def validate_restores(rows, events, stage_one_sha):
    stage_one_best = max(range(296), key=lambda i: rows[i]['metrics']['AP'])
    saved = rows[stage_one_best]
    expected_events, reset_before_epoch = [], {296}
    last_epoch, ema_updates, best_ap = -1, 0, -math.inf

    def restore(epoch, reason, attempts):
        nonlocal last_epoch, ema_updates
        expected_events.append({
            'actual_loop_epoch': epoch, 'reason': reason, 'checkpoint_sha256': stage_one_sha,
            'previous_last_epoch': last_epoch, 'restored_last_epoch': saved['last_epoch'],
            'monotonic_step_attempts': attempts, 'ema_updates': saved['ema_updates'],
        })
        last_epoch, ema_updates = saved['last_epoch'], saved['ema_updates']

    for epoch, row in enumerate(rows):
        if epoch == 296:
            restore(epoch, 'stage_two_entry', epoch * BATCHES)
        last_epoch += 1
        ema_updates += BATCHES
        require(row['last_epoch'] == last_epoch and row['ema_updates'] == ema_updates,
                f'Restored metadata versus actual epoch {epoch}')
        if row['metrics']['AP'] > best_ap:
            best_ap = row['metrics']['AP']
        elif epoch >= 296:
            restore(epoch, 'late_epoch_without_new_global_best', (epoch + 1) * BATCHES)
            reset_before_epoch.add(epoch + 1)
    require(events == expected_events, 'Missing, extra or inconsistent restore event')
    return stage_one_best, reset_before_epoch


def validate_learning_rates(path, rows, config, preflight, reset_before_epoch, stage_one_best):
    require(config['lrsheduler'] == 'flatcosine' and config['lr_gamma'] == 1.0,
            'This audit is specific to the frozen constant-after-warmup schedule')
    groups = config['optimizer']['params']
    base_lrs = [group.get('lr', config['optimizer']['lr']) for group in groups] + [config['optimizer']['lr']]
    require(len(base_lrs) == 4 and preflight['optimizer_groups'] == 4, 'Optimizer group count')
    initial = [v / config['lr_warmup_scheduler']['warmup_duration'] for v in base_lrs]
    tolerance = 1e-15

    def equal(left, right, context):
        require(len(left) == len(right) and all(finite(a) and abs(a - b) <= tolerance for a, b in zip(left, right)), context)

    equal(preflight['initial_executed_lrs'], initial, 'Initial constructor LR scaling')
    expected, count, max_error = initial, 0, 0.0
    warmup = config['warmup_iter']
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['attempt', 'actual_epoch', 'batch', 'group0', 'group1', 'group2', 'group3'],
                'Unexpected LR log schema')
        for attempt, item in enumerate(reader):
            epoch, batch = divmod(attempt, BATCHES)
            require(epoch < EPOCHS, 'Excess optimizer attempts')
            require([int(item[k]) for k in ['attempt', 'actual_epoch', 'batch']] == [attempt, epoch, batch],
                    f'LR attempt order at {attempt}')
            if batch == 0 and epoch in reset_before_epoch:
                expected = rows[stage_one_best]['post_final_update_lrs']
            actual = [float(item[f'group{i}']) for i in range(4)]
            equal(actual, expected, f'Executed LR at attempt {attempt}')
            max_error = max(max_error, *(abs(a - b) for a, b in zip(actual, expected)))
            if batch == 0:
                equal(actual, rows[epoch]['first_executed_lrs'], f'First executed LR at epoch {epoch}')
            # Independent expression of the frozen rule: update first, then
            # quadratic warmup; min_lr == base_lr makes all later branches constant.
            expected = [value * (attempt / warmup) ** 2 if attempt < warmup else value for value in base_lrs]
            if batch == BATCHES - 1:
                equal(expected, rows[epoch]['post_final_update_lrs'], f'Post-update LR at epoch {epoch}')
            count += 1
    require(count == EPOCHS * BATCHES, 'Incomplete executed LR log')
    return {'recorded_attempts': count, 'groups': 4, 'maximum_absolute_difference': max_error,
            'absolute_tolerance': tolerance}


def audit_pair(root):
    protocol_path = root / 'execution_protocol.json'
    require(sha(protocol_path) == PROTOCOL_SHA, 'Unexpected execution protocol')
    protocol = read(protocol_path)
    queue = read(root / 'queue.json')
    if queue['status'] != 'COMPLETED':
        return {'status': 'NOT_READY', 'queue_status': queue['status'], 'result_generated': False}
    require(queue['protocol_sha256'] == PROTOCOL_SHA, 'Queue protocol binding')
    require(queue['automatic_retry'] is False and queue['hard_timeout_seconds_per_arm'] == 604800,
            'Unexpected execution retry or timeout policy')
    require([run['arm'] for run in queue['runs']] == ['BASE', 'FULL'], 'Expected exactly one run per arm')
    for item in protocol['frozen_files']:
        require(sha(root / item['path']) == item['sha256'], f'Frozen file changed: {item["path"]}')

    report = {'status': 'PAIR_RECORDS_VERIFIED_PENDING_MANUSCRIPT_REVIEW',
              'protocol_sha256': PROTOCOL_SHA, 'queue_sha256': sha(root / 'queue.json'),
              'frozen_files_verified': len(protocol['frozen_files']), 'arms': {}, 'comparisons': {},
              'analysis_script_sha256': sha(Path(__file__)), 'analysis_utc': datetime.now(timezone.utc).isoformat(),
              'model_executions': 0, 'evaluator_reruns': 0, 'submission_ready': False}
    first_batches = []
    for run in queue['runs']:
        arm = run['arm']
        folder = root / 'runs' / arm
        receipt = read(folder / 'receipt.json')
        require(run['status'] == 'COMPLETED' and run['exit_code'] == 0, f'{arm} did not finish successfully')
        require(run['final_receipt'] == receipt, f'{arm} final receipt mismatch')
        require(receipt['status'] == 'COMPLETED' and receipt['completed_epochs'] == EPOCHS
                and receipt['actual_loop_epoch'] == EPOCHS - 1
                and receipt['optimizer_step_attempts'] == EPOCHS * BATCHES, f'{arm} incomplete budget')
        require(receipt['arm'] == arm and receipt['protocol_sha256'] == PROTOCOL_SHA, f'{arm} receipt binding')
        config_path = root / 'execution_configs' / f'{arm}.yml'
        config_sha = sha(config_path)
        for key, path in [('config_sha256', config_path), ('runner_sha256', root / 'fair_runner.py'),
                          ('environment_sha256', root / 'environment.current.json'),
                          ('inputs_receipt_sha256', root / 'cache_verification.json'),
                          ('initialization_sha256', root / 'preflight' / f'{arm}.initial.pt')]:
            require(receipt[key] == sha(path), f'{arm} {key}')
        rows = read_rows(folder / 'epochs.jsonl')
        validate_epoch_rows(rows, config_sha)
        actual_batches = read_rows(folder / 'actual_first100.jsonl')
        require(len(actual_batches) == 100 and actual_batches == read(root / 'preflight' / f'{arm}.first100.json'),
                f'{arm} actual first 100 batches')
        first_batches.append(actual_batches)
        events = read_rows(folder / 'restore_events.jsonl')
        stage_one_best, resets = validate_restores(rows, events, sha(folder / 'best_stg1.pt'))
        preflight = read(root / 'preflight' / f'{arm}.json')
        lr_result = validate_learning_rates(folder / 'executed_learning_rates.csv.gz', rows,
                                           read(config_path), preflight, resets, stage_one_best)
        selected = endpoint_epochs(rows)
        require(receipt['best_epoch'] == selected['best'] and receipt['best_AP'] == rows[selected['best']]['metrics']['AP'],
                f'{arm} best endpoint selection')
        endpoints = {}
        for endpoint, epoch in selected.items():
            metric_path = folder / f'{endpoint}.metrics.json'
            metrics = read(metric_path)
            require({k: v for k, v in metrics.items() if k != 'checkpoint_sha256'} == rows[epoch],
                    f'{arm}/{endpoint} mixed or wrong epoch row')
            audit_dir = root / 'evidence_audits' / f'{arm}_{endpoint}_epoch{epoch:04}'
            audit = read(audit_dir / 'receipt.json')
            require(audit['status'] == 'SAVED_ENDPOINT_BINDING_VERIFIED' and audit['arm'] == arm
                    and audit['endpoint'] == endpoint and audit['actual_epoch'] == epoch, f'{arm}/{endpoint} audit identity')
            require(audit['protocol_sha256'] == PROTOCOL_SHA and audit['metrics'] == rows[epoch]['metrics']
                    and audit['ema_tensor_sha256'] == rows[epoch]['ema_tensor_sha256'], f'{arm}/{endpoint} audit binding')
            require(audit['image_count'] == 548 and audit['prediction_count'] == 164400,
                    f'{arm}/{endpoint} validation coverage')
            require(all(audit[key] == 0 for key in ['model_inference_passes', 'optimizer_updates', 'evaluator_reruns']),
                    f'{arm}/{endpoint} unexpected audit execution scope')
            require(audit['metrics_recomputed_from_saved_arrays'] == audit['metrics']
                    or all(abs(audit['metrics_recomputed_from_saved_arrays'][k] - audit['metrics'][k]) <= 1e-12 for k in METRICS),
                    f'{arm}/{endpoint} saved-array summary')
            require(read(audit_dir / 'metrics.json') == metrics, f'{arm}/{endpoint} copied metric row')
            files = [('checkpoint.pt', 'best.pt' if endpoint == 'best' else 'terminal_epoch0319.pt', 'checkpoint_sha256'),
                     ('predictions.json', f'{endpoint}.predictions.json', 'prediction_sha256'),
                     ('evaluation_arrays.npz', f'{endpoint}.evaluation_arrays.npz', 'evaluation_arrays_sha256')]
            for copied, original, key in files:
                require(sha(audit_dir / copied) == sha(folder / original) == metrics[key] == audit[key],
                        f'{arm}/{endpoint} {key}')
            endpoints[endpoint] = {'actual_epoch': epoch, 'metrics': metrics['metrics'],
                                   'checkpoint_sha256': metrics['checkpoint_sha256'],
                                   'ema_tensor_sha256': metrics['ema_tensor_sha256'],
                                   'metrics_sha256': sha(metric_path), 'audit_receipt_sha256': sha(audit_dir / 'receipt.json')}
        report['arms'][arm] = {'trainable_parameters': preflight['trainable_parameters'],
                              'completed_epochs': len(rows), 'optimizer_step_attempts': receipt['optimizer_step_attempts'],
                              'stage_one_best_epoch': stage_one_best, 'restore_events': len(events),
                              'learning_rates': lr_result, 'endpoints': endpoints,
                              'record_hashes': {name: sha(folder / name) for name in ['receipt.json', 'epochs.jsonl',
                                  'restore_events.jsonl', 'actual_first100.jsonl', 'executed_learning_rates.csv.gz']}}
    require(first_batches[0] == first_batches[1], 'Cross-arm initial data mismatch')
    for endpoint in ['terminal', 'best']:
        base = report['arms']['BASE']['endpoints'][endpoint]
        full = report['arms']['FULL']['endpoints'][endpoint]
        report['comparisons'][endpoint] = {
            'role': 'primary' if endpoint == 'terminal' else 'secondary',
            'BASE_epoch': base['actual_epoch'], 'FULL_epoch': full['actual_epoch'],
            'FULL_minus_BASE': {k: full['metrics'][k] - base['metrics'][k] for k in METRICS},
            'FULL_minus_BASE_percentage_points': {k: 100 * (full['metrics'][k] - base['metrics'][k]) for k in METRICS}}
    report['conclusion_scope'] = ('One fixed-protocol pair, local converted COCO-style validation; total combination only. '
        'No isolated-module, multi-seed significance, independent-test, official VisDrone ranking or deployment-speed claim.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, help='Optional NEW JSON file inside the evidence root; never overwrites')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        report = audit_pair(root)
        if report['status'] == 'NOT_READY':
            print(json.dumps(report)); return 2
        if args.output:
            output = args.output.resolve()
            require(output.is_relative_to(root), 'Output must stay inside evidence root')
            with output.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, allow_nan=False))
        return 0
    except (AuditError, OSError, KeyError, ValueError, TypeError) as error:
        print(json.dumps({'status': 'FINAL_PAIR_AUDIT_FAILED', 'result_generated': False,
                          'error': f'{type(error).__name__}: {error}'})); return 1


if __name__ == '__main__':
    sys.exit(main())
