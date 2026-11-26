"""Audit the accepted FRESH trajectories and their saved evaluation endpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1')
DATA = HERE.parents[1] / '实验数据汇总_20261108/fresh_pair_20261112'
ARMS = ('BASE_FRESH', 'FULL_FRESH')
BACKUP_FIELDS = {'evaluated_checkpoint_sha256', 'evaluated_state_is_next_state', 'next_state_ema_tensor_sha256'}
sys.path[:0] = [str(PARENT), str(PARENT / '模块消融'), str(PARENT / '断点续训_20261105')]
from audit_ablation_lineage import validate_prefix_rates
from summarize_completed_pair import BATCHES, EPOCHS, METRICS, finite, read, read_rows, require, sha, validate_restores
from verify_ablation_backup import audit_evaluation


def artifact(path):
    return {'file': str(path), 'sha256': sha(path)}


def verify_payload(protocol, stage, manifest, name, digest):
    item = next(i for i in manifest['files'] if i['name'] == name)
    path = protocol.checked_local(stage / name, stage)
    require(protocol.regular(path).st_size == item['bytes'] and sha(path) == item['sha256'] == digest,
            'Saved artifact differs: ' + name)
    return {'file': str(path), 'sha256': digest}


def audit_endpoint(protocol, stage, manifest, validation, row, folder, role):
    prefix = folder + '/' if folder else ''
    metric = prefix + ('epoch.metrics.json' if role == 'latest' else role + '.metrics.json')
    metadata = read(stage / metric)
    weight_name = {'best': 'best.pt', 'latest': 'evaluated_epoch.pt', 'terminal': 'terminal_epoch0319.pt'}[role]
    weight_sha = row['evaluated_checkpoint_sha256'] if role == 'latest' else metadata['checkpoint_sha256']
    if role == 'latest':
        require(metadata == row, 'Latest endpoint differs from the accepted row')
    else:
        require({k: v for k, v in metadata.items() if k != 'checkpoint_sha256'} ==
                {k: v for k, v in row.items() if k not in BACKUP_FIELDS},
                'Saved endpoint differs from its exact evaluated row')
    weight_name = prefix + weight_name
    checked = validation['checked_payloads']
    entry = next(x for x in checked if x['file'] == weight_name)
    ema_evidence = next(x for x in checked if x['file'] == entry.get('same_payload_as', weight_name)
                        and x.get('complete_state_and_rng_fields_verified') is True)
    require(entry['file_sha256'] == ema_evidence['file_sha256'] == weight_sha
            and ema_evidence['epoch'] == row['epoch']
            and ema_evidence['ema_tensor_sha256'] == row['ema_tensor_sha256']
            and ema_evidence['complete_state_and_rng_fields_verified'] is True,
            'Weight lacks CPU verification of the evaluated EMA')
    predictions = verify_payload(protocol, stage, manifest, prefix + role + '.predictions.json', row['prediction_sha256'])
    arrays = verify_payload(protocol, stage, manifest, prefix + role + '.evaluation_arrays.npz', row['evaluation_arrays_sha256'])
    metrics, image_count, prediction_count, error = audit_evaluation(
        Path(predictions['file']), Path(arrays['file']), row['metrics'])
    return {'round': row['epoch'] + 1, 'ema_tensor_sha256': row['ema_tensor_sha256'],
            'checkpoint': verify_payload(protocol, stage, manifest, weight_name, weight_sha),
            'predictions': predictions, 'evaluation_arrays': arrays, 'metric_row': artifact(stage / metric),
            'metrics': metrics, 'validation_images': image_count, 'prediction_count': prediction_count,
            'maximum_metric_error_from_saved_arrays': error}


def verify_exit(archive, plan, rows, manifest, digest, binding):
    expected = binding['records']['supervisor.json']
    latest = read(archive / 'completed_run_receipt.json')
    supervisor = read(archive / 'completed_supervisor.json')
    require(latest['status'] == 'COMPLETED' and latest['actual_loop_epoch'] == EPOCHS - 1
            and latest['completed_epochs'] == latest['next_epoch'] == EPOCHS
            and latest['optimizer_step_attempts'] == EPOCHS * BATCHES
            and latest['latest_metrics'] == rows[-1]['metrics']
            and latest['best_epoch'] == manifest['best']['epoch']
            and latest['best_AP'] == rows[latest['best_epoch']]['metrics']['AP']
            and latest['rolling_checkpoint_sha256'] == manifest['rolling_sha256']
            and latest['local_backup_verified'] == {'epoch': EPOCHS - 1, 'manifest_sha256': digest,
                'ack_sha256': sha(archive / 'epochs/epoch0319/ack.json')}, 'Completed receipt differs from final accepted round')
    require(latest['arm'] == manifest['arm'] and latest['protocol_sha256'] == manifest['protocol_sha256']
            and latest['configuration_sha256'] == plan['configuration_sha256']
            and latest['initialization_sha256'] == plan['probe_files']['initial.pt']
            and latest['pid'] == expected['pid'] and latest['runner_sha256'] == expected['runner_sha256']
            and supervisor['status'] == 'COMPLETED' and supervisor['exit_code'] == 0
            and supervisor['final_receipt'] == latest, 'Completion does not bind the verified run')
    for key in ('arm', 'pid', 'start_ticks', 'supervisor_pid', 'supervisor_start_ticks', 'boot_id',
                'protocol_sha256', 'runner_sha256', 'command'):
        require(supervisor[key] == expected[key], 'Completed supervisor identity differs: ' + key)
    original_start = binding['records']['run/receipt.json']
    if original_start['first_epoch_in_this_segment']:
        for key in ('first_epoch_in_this_segment', 'resumed_from_checkpoint_sha256', 'recovery_authorization_sha256'):
            require(latest[key] == original_start[key], 'Continuation lineage differs: ' + key)
    return {name: artifact(archive / name) for name in ('completed_run_receipt.json', 'completed_supervisor.json')}


def verify_full_archive(control):
    sys.path.insert(0, str(control))
    try:
        backup = importlib.import_module('backup_monitor')
        require(Path(backup.__file__).resolve().parent == control.resolve(), 'Backup module loaded from another run')
        archive = control / 'backups'
        maximum = backup.MAX_BACKUP_BYTES
        for line in reversed((archive / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()):
            event = json.loads(line)
            if event['status'] == 'MONITOR_READY':
                maximum = event['storage_limit_bytes']
                break
        store = backup.BackupStore(archive, control, maximum=maximum)
        with backup.protect_local_lock(store.root):
            store.setup()
            cursor = store.verify_progress(control.name)
        require(cursor == EPOCHS, 'Complete verified backup prefix is missing')
        return {'accepted_rounds': cursor, 'verified_objects': len(store.verified),
                'archive_bytes': store.used, 'backup_limit_bytes': maximum}
    finally:
        sys.path.remove(str(control))
        sys.modules.pop('backup_monitor', None)
        sys.modules.pop('backup_protocol', None)


def audit_arm(arm, count, sources, binding):
    control = BACKUPS / arm
    archive = control / 'backups'
    module = importlib.util.spec_from_file_location('fresh_audit_' + arm, control / 'backup_protocol.py')
    protocol = importlib.util.module_from_spec(module)
    module.loader.exec_module(protocol)
    plan = read(control / 'execution_protocol.json')
    require(protocol.ARM == arm and plan['arm'] == arm and plan['epochs'] == EPOCHS, 'Unexpected experiment arm')
    for name in ('backup_monitor.py', 'backup_protocol.py', 'checkpoint_validator.py'):
        require(sha(control / name) == sha(HERE / 'driver' / name), 'Saved verification code differs: ' + name)
    require(count > 0 and sorted(sources) == list(range(1, count + 1)), 'Export is missing accepted rounds')
    stages = [archive / 'epochs' / f'epoch{i:04d}' for i in range(count)]
    rows = read_rows(stages[-1] / 'epochs.jsonl')
    require(len(rows) == count, 'Cumulative epoch history differs in length')
    validator_sha = sha(control / 'checkpoint_validator.py')
    for i, (stage, row) in enumerate(zip(stages, rows)):
        payload = (stage / 'manifest.json').read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        manifest = protocol.parse_manifest(payload, arm, i)
        require((stage / 'ack.json').read_bytes() == protocol.ack_bytes(payload, manifest)
                and (stage / 'accepted.json').read_bytes() == protocol.accepted_bytes(payload, manifest),
                'Backup acknowledgements differ at round ' + str(i + 1))
        protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(stage / name))
        validation = read(stage / 'payload_validation.json')
        require(validation['status'] == 'CHECKPOINT_PAYLOADS_VERIFIED_CPU_ONLY'
                and validation['manifest_sha256'] == digest and validation['epoch'] == i
                and validation['protocol_sha256'] == protocol.PROTOCOL and validation['validator_sha256'] == validator_sha
                and validation['model_forwards'] == validation['evaluator_runs'] == validation['optimizer_updates'] == 0,
                'Missing matching CPU validation for round ' + str(i + 1))
        source = sources[i + 1]
        row_path = stage / 'epoch.metrics.json'
        require(source['manifest_sha256'] == digest and source['sha256'] == sha(row_path)
                and row == read(row_path) and row['epoch'] == i and row['optimizer_step_attempts'] == (i + 1) * BATCHES
                and finite(row['train_loss_mean']) and set(row['metrics']) == set(METRICS)
                and all(finite(v) and 0 <= v <= 1 for v in row['metrics'].values()),
                'Cumulative epoch row differs from the verified export')
    last = stages[-1]
    best = max(range(count), key=lambda i: rows[i]['metrics']['AP'])
    stage_one = max(range(min(count, 296)), key=lambda i: rows[i]['metrics']['AP'])
    require(manifest['best']['epoch'] == best and manifest['stage_one']['epoch'] == stage_one,
            'Selected checkpoint differs from highest AP with earliest ties')
    if count > 296:
        events = read_rows(last / 'restore_events.jsonl')
        observed_stage_one, resets = validate_restores(rows, events, manifest['stage_one']['checkpoint_sha256'])
        require(observed_stage_one == stage_one, 'Stage-one selection differs')
    else:
        require(all(row['last_epoch'] == row['epoch'] and row['ema_updates'] == row['optimizer_step_attempts'] for row in rows),
                'Unexpected pre-stage-two counters')
        events, resets = [], set()
    frozen = {f['path']: f['sha256'] for f in plan['frozen_files']}
    base = arm.removesuffix('_FRESH')
    evidence = PARENT / 'remote_evidence'
    config_rel, preflight_rel = f'execution_configs/{base}.yml', f'preflight/{base}.json'
    require(all(sha(evidence / rel) == frozen['frozen_parent/' + rel] for rel in (config_rel, preflight_rel)),
            'Original frozen learning rate inputs differ')
    rate = validate_prefix_rates(last / 'executed_learning_rates.csv.gz', rows, read(evidence / config_rel),
                                 read(evidence / preflight_rel), resets, stage_one)
    first = evidence / f'preflight/{base}.first100.json'
    require(sha(first) == frozen['inputs/first100.json'] and read_rows(last / 'actual_first100.jsonl') == read(first),
            'Actual first 100 batches differ from the frozen input')
    endpoints = {
        'latest': audit_endpoint(protocol, last, manifest, validation, rows[-1], '', 'latest'),
        'best': audit_endpoint(protocol, last, manifest, validation, rows[best], protocol.selected_folder(manifest['best']), 'best'),
    }
    exit_evidence = None
    if count == EPOCHS:
        endpoints['terminal'] = audit_endpoint(protocol, last, manifest, validation, rows[-1], '', 'terminal')
        if binding is not None:
            exit_evidence = verify_exit(archive, plan, rows, manifest, digest, binding)
            exit_evidence['full_archive_verification'] = verify_full_archive(control)
    return {'arm': arm, 'accepted_rounds': count, 'last_manifest': artifact(last / 'manifest.json'),
            'learning_rates': rate, 'restore_event_count': len(events), 'endpoints': endpoints,
            'completed_run_evidence': exit_evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bindings', type=Path, help='Verified launch process identities required for final completion evidence.')
    args = parser.parse_args()
    report_path = DATA / '逐轮导出核对_FRESH.json'
    report = read(report_path)
    require(sha(DATA / '逐轮指标_FRESH.csv') == report['csv']['sha256'], 'CSV differs from verified export')
    bindings = {x['arm']: x for x in read(args.bindings)['remote']} if args.bindings else {}
    observations = [audit_arm(arm, report['accepted_rounds'][arm],
                              {row['round']: row for row in report['sources'] if row['arm'] == arm},
                              bindings.get(arm)) for arm in ARMS]
    complete = all(row['completed_run_evidence'] for row in observations)
    result = {'observed_local_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'COMPLETED_PAIR_EVIDENCE_VERIFIED' if complete else 'ACCEPTED_PREFIX_EVIDENCE_VERIFIED',
              'verified_csv_sha256': report['csv']['sha256'], 'verified_export': artifact(report_path),
              'audit_source': artifact(Path(__file__)),
              'run_identity_bindings': artifact(args.bindings.resolve()) if args.bindings else None,
              'arms': observations, 'model_forwards': 0, 'optimizer_updates': 0, 'evaluator_runs': 0}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    if complete:
        index_path = DATA / 'run_index.json'
        index = read(index_path)
        index.update(status=result['status'], final_endpoint_rows=2,
                     verified_completion=artifact(args.output.resolve()),
                     completed_arms={x['arm']: x['endpoints'] for x in observations})
        pending = index_path.with_suffix('.json.partial')
        with pending.open('x', encoding='utf-8') as stream:
            json.dump(index, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        pending.replace(index_path)
    print(json.dumps({'status': result['status'], 'accepted_rounds': report['accepted_rounds'],
                      'report': str(args.output.resolve())}, ensure_ascii=True))


if __name__ == '__main__':
    main()
