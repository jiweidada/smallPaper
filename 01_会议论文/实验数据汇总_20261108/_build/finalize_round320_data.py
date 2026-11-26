"""Append the independently audited NO_MSCA round-320 data and final endpoints."""

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PAPER = OUT.parent
RECOVERY = PAPER / '远程公平对照_20261103/模块消融/NO_MSCA/recovery30715_20261109'
AUDIT = RECOVERY / 'lineage_second_recovery_through_round320.json'
NATIVE = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
EVIDENCE = RECOVERY / 'completion_evidence_round320_20261111'
XLSX = Path('outputs/实验数据汇总_20261108/实验数据汇总.xlsx')
METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
LOCAL_DATE = '2026-11-11'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.round320.tmp')
    with temp.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    write_atomic(path, payload)


def csv_bytes(fields, rows, bom=True):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig' if bom else 'utf-8')


def percent_map(values):
    return {key: value * 100 for key, value in values.items()}


def main():
    assert not EVIDENCE.exists(), f'Final evidence package already exists: {EVIDENCE}'
    summary = read(OUT / '实验汇总.json')
    delivery = read(OUT / '交付清单.json')
    old_snapshot = OUT / 'verified_snapshots/NO_MSCA_round280_verified_20261110'
    old_snapshot_manifest = read(old_snapshot / 'snapshot_manifest.json')
    assert sha(old_snapshot / 'snapshot_manifest.json') == '883acf595c6fe7ebd7cd86a52e879b90d507c6052536f6d90b0aafe60cabe730'
    for item in old_snapshot_manifest['files']:
        path = old_snapshot / item['file']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['file']
    assert delivery['NO_MSCA_accepted_rounds'] == 280
    for item in delivery['files']:
        path = OUT / item['file']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['file']
    assert sha(OUT / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx') == '761abadabdfee496e82aa08c84efd506548c6415126a4e1ed7b988bc3e060182'

    audit = read(AUDIT)
    assert sha(AUDIT) == '66ba7a9b2f61c550222230aa0a661994f585abdc8d0d69cee9430ba727cb1734'
    assert audit['status'] == 'VERIFIED_ABLATION_LINEAGE_AND_SAVED_ENDPOINTS'
    assert audit['completed_rounds'] == 320 and audit['learning_rates']['recorded_attempts'] == 517440
    assert audit['learning_rates']['maximum_absolute_difference'] == 0
    assert audit['maximum_latest_metric_error'] == 0
    assert audit['validation_images'] == 548 and audit['predictions'] == 164400
    assert audit['model_forwards'] == audit['evaluator_runs'] == 0
    assert audit.get('remote_connections', audit.get('recovery', {}).get('remote_connections', 0)) == 0

    # Copy a small, hash-bound completion packet from the local native backup volume.
    copies = [
        ('monitor_status.json', NATIVE / 'monitor_status.json', 'Native backup monitor final status'),
        ('completed_run_receipt.json', NATIVE / 'completed_run_receipt.json', 'Final worker receipt'),
        ('completed_supervisor.json', NATIVE / 'completed_supervisor.json', 'Supervisor exit record'),
        ('monitor_events.jsonl', NATIVE / 'monitor_events.jsonl', 'Native backup monitor event history'),
        ('epoch0319/epochs.jsonl', NATIVE / 'epochs/epoch0319/epochs.jsonl', 'Cumulative accepted NO_MSCA epoch records'),
        ('epoch0319/manifest.json', NATIVE / 'epochs/epoch0319/manifest.json', 'Accepted round-320 backup manifest'),
        ('epoch0319/ack.json', NATIVE / 'epochs/epoch0319/ack.json', 'Round-320 backup acknowledgement'),
        ('epoch0319/accepted.json', NATIVE / 'epochs/epoch0319/accepted.json', 'Round-320 backup acceptance marker'),
        ('epoch0319/receipt.json', NATIVE / 'epochs/epoch0319/receipt.json', 'Accepted round-320 epoch receipt'),
        ('epoch0319/epoch.metrics.json', NATIVE / 'epochs/epoch0319/epoch.metrics.json', 'Latest accepted epoch metrics'),
        ('epoch0319/terminal.metrics.json', NATIVE / 'epochs/epoch0319/terminal.metrics.json', 'Saved terminal endpoint metrics'),
        ('epoch0319/restore_events.jsonl', NATIVE / 'epochs/epoch0319/restore_events.jsonl', 'Restore history through completion'),
        ('epoch0301/manifest.json', NATIVE / 'epochs/epoch0301/manifest.json', 'Accepted best-AP endpoint manifest'),
        ('epoch0301/ack.json', NATIVE / 'epochs/epoch0301/ack.json', 'Best-AP endpoint backup acknowledgement'),
        ('epoch0301/accepted.json', NATIVE / 'epochs/epoch0301/accepted.json', 'Best-AP endpoint acceptance marker'),
        ('epoch0301/epoch.metrics.json', NATIVE / 'epochs/epoch0301/epoch.metrics.json', 'Saved best-AP epoch metrics'),
    ]
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    copied = []
    for relative, source, purpose in copies:
        assert source.is_file(), source
        target = EVIDENCE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        copied.append({'file': relative.replace('\\', '/'), 'source_path': str(source),
                       'bytes': target.stat().st_size, 'sha256': sha(target), 'purpose': purpose})
    # Also pin the already validated independent endpoint audit in the packet.
    audit_copy = EVIDENCE / 'lineage_second_recovery_through_round320.json'
    shutil.copy2(AUDIT, audit_copy)
    assert sha(audit_copy) == sha(AUDIT)
    copied.append({'file': audit_copy.name, 'source_path': str(AUDIT), 'bytes': audit_copy.stat().st_size,
                   'sha256': sha(audit_copy), 'purpose': 'Independent saved-lineage and endpoint audit'})

    status = read(EVIDENCE / 'monitor_status.json')
    receipt = read(EVIDENCE / 'completed_run_receipt.json')
    supervisor = read(EVIDENCE / 'completed_supervisor.json')
    final_manifest = read(EVIDENCE / 'epoch0319/manifest.json')
    final_ack = read(EVIDENCE / 'epoch0319/ack.json')
    final_accepted = read(EVIDENCE / 'epoch0319/accepted.json')
    source_path = EVIDENCE / 'epoch0319/epochs.jsonl'
    raw_lines = source_path.read_bytes().splitlines()
    rows = [json.loads(line) for line in raw_lines]
    best = audit['endpoints']['best']
    terminal = audit['endpoints']['terminal']
    assert status['status'] == 'COMPLETED_AND_BACKED_UP' and status['epoch'] == 319
    assert status['supervisor_sha256'] == sha(EVIDENCE / 'completed_supervisor.json')
    assert receipt['status'] == 'COMPLETED' and receipt['actual_loop_epoch'] == 319
    assert receipt['completed_epochs'] == 320 and receipt['next_epoch'] == 320
    assert receipt['optimizer_step_attempts'] == 517440 and receipt['backup_epoch'] == 319
    assert receipt['rolling_checkpoint_sha256'] == status['rolling_checkpoint_sha256']
    assert receipt['local_backup_verified']['manifest_sha256'] == sha(EVIDENCE / 'epoch0319/manifest.json')
    assert receipt['local_backup_verified']['ack_sha256'] == sha(EVIDENCE / 'epoch0319/ack.json')
    assert supervisor['status'] == 'COMPLETED' and supervisor['exit_code'] == 0
    assert supervisor['pid'] == receipt['pid'] == 413 and supervisor['supervisor_pid'] == 412
    assert supervisor['final_receipt'] == receipt
    assert len(rows) == 320 and rows[-1]['metrics'] == audit['latest_metrics'] == terminal['metrics']
    assert best['epoch_index'] == max(range(320), key=lambda i: rows[i]['metrics']['AP']) == 301
    assert rows[best['epoch_index']]['metrics'] == best['metrics']
    assert len(raw_lines) == 320

    # Re-run the frozen native acceptance parser for both paper endpoints.
    spec = importlib.util.spec_from_file_location('epoch_export_round320', HERE / 'assemble_epoch_data.py')
    epochs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = epochs
    spec.loader.exec_module(epochs)
    native_rows, references, native_acceptance = epochs.accepted_stage('NO_MSCA', 319)
    assert native_rows == rows and native_acceptance['manifest_sha256'] == receipt['local_backup_verified']['manifest_sha256']
    best_rows, _, best_acceptance = epochs.accepted_stage('NO_MSCA', best['epoch_index'])
    assert best_rows[best['epoch_index']]['metrics'] == best['metrics']
    assert best_acceptance['included_last_epoch_index'] == best['epoch_index']
    for endpoint, index, endpoint_directory, checkpoint_name in (
            (terminal, 319, EVIDENCE / 'epoch0319', 'terminal_epoch0319.pt'),
            (best, best['epoch_index'], EVIDENCE / f"epoch{best['epoch_index']:04d}",
             f"selected/epoch{best['epoch_index']:04d}/best.pt")):
        row = rows[index]
        assert row['metrics'] == endpoint['metrics']
        for key in ('ema_tensor_sha256', 'prediction_sha256', 'evaluation_arrays_sha256'):
            assert row[key] == endpoint[key]
        endpoint_manifest = read(endpoint_directory / 'manifest.json')
        checkpoint = next(item for item in endpoint_manifest['files'] if item['name'] == checkpoint_name)
        assert checkpoint['sha256'] == endpoint['checkpoint_sha256']

    config_sha = receipt['configuration_sha256']
    param_count = receipt['initialization_origin']['trainable_parameters']
    assert param_count == 3699662
    recovery_folder = str(NATIVE / 'epochs')

    def endpoint_row(role, endpoint, index, source_pointer, local_folder):
        label = '固定320轮主端点' if role == 'primary' else '320轮轨迹内AP最佳次端点'
        return {
            'arm': 'NO_MSCA', 'endpoint_role': role, 'endpoint_label': label,
            'round_1based': index + 1, 'epoch_index_0based': index, 'seed': 0,
            'completed_rounds': 320, 'optimizer_step_attempts_at_evaluation': (index + 1) * 1617,
            'total_optimizer_step_attempts': 517440, 'trainable_parameters': param_count,
            'weight_source': 'EMA', 'checkpoint_sha256': endpoint['checkpoint_sha256'],
            'ema_tensor_sha256': endpoint['ema_tensor_sha256'], 'metrics_sha256': endpoint['metrics_sha256'],
            'prediction_sha256': endpoint['prediction_sha256'],
            'evaluation_arrays_sha256': endpoint['evaluation_arrays_sha256'],
            'protocol_sha256': audit['protocol_sha256'], 'configuration_sha256': config_sha,
            'metrics_fraction': endpoint['metrics'], 'metrics_percent': percent_map(endpoint['metrics']),
            'validation_images': 548, 'predictions': 164400,
            'source_id': 'NO_MSCA_SECOND_RECOVERY_R320', 'source_json_pointer': source_pointer,
            'local_artifact_folder': local_folder,
        }

    terminal_entry = endpoint_row('primary', terminal, 319, '/endpoints/terminal', str(NATIVE / 'epochs/epoch0319'))
    best_entry = endpoint_row('secondary', best, best['epoch_index'], '/endpoints/best',
                              str(NATIVE / f"epochs/epoch{best['epoch_index']:04d}"))

    # Maintain an immutable native completion packet before changing current summary outputs.
    binder = {
        'status': 'NO_MSCA_ROUND320_COMPLETION_AND_ENDPOINTS_HASH_BOUND',
        'observed_local_date': LOCAL_DATE,
        'independent_lineage_audit': {'file': str(AUDIT), 'sha256': sha(AUDIT), 'status': audit['status']},
        'native_monitor_status': {'file': 'monitor_status.json', 'sha256': sha(EVIDENCE / 'monitor_status.json'),
                                  'status': status['status'], 'epoch_index_0based': status['epoch']},
        'native_run_receipt': {'file': 'completed_run_receipt.json', 'sha256': sha(EVIDENCE / 'completed_run_receipt.json'),
                               'status': receipt['status'], 'completed_rounds': receipt['completed_epochs'],
                               'actual_loop_epoch': receipt['actual_loop_epoch'],
                               'optimizer_step_attempts': receipt['optimizer_step_attempts'],
                               'backup_manifest_sha256': receipt['local_backup_verified']['manifest_sha256'],
                               'backup_ack_sha256': receipt['local_backup_verified']['ack_sha256']},
        'native_supervisor': {'file': 'completed_supervisor.json', 'sha256': sha(EVIDENCE / 'completed_supervisor.json'),
                              'status': supervisor['status'], 'exit_code': supervisor['exit_code'],
                              'worker_pid': supervisor['pid'], 'supervisor_pid': supervisor['supervisor_pid']},
        'cumulative_epoch_source': {'file': 'epoch0319/epochs.jsonl', 'sha256': sha(source_path), 'records': len(rows)},
        'round320_native_acceptance': native_acceptance,
        'best_endpoint_native_acceptance': best_acceptance,
        'primary_endpoint': terminal_entry,
        'secondary_endpoint': best_entry,
        'endpoint_identity': 'Primary is accepted final epoch index319 before late rollback; secondary is highest AP in the same completed 320-round trajectory at index301. Both endpoint metrics are from single saved rows.',
        'recovery_limitations': [audit['recovery']['physical_compute_limitation'], audit['recovery']['verification_scope']],
        'validation_scope': {'images': 548, 'predictions_per_endpoint': 164400,
                             'maximum_metric_error': 0, 'model_forwards': 0, 'evaluator_runs': 0},
        'evidence_packet_files': copied,
    }
    write_json(RECOVERY / 'round320_completion_and_endpoints_20261111.json', binder)
    write_json(EVIDENCE / 'evidence_packet_manifest.json', {
        'status': 'HASH_VERIFIED_LOCAL_COPY_OF_NATIVE_COMPLETION_EVIDENCE',
        'source_backup_root': str(NATIVE), 'files': copied,
        'manifest_sha256_note': 'This packet manifest is intentionally excluded from its own file list.',
    })

    # Add the verified NO_MSCA results to the consolidated endpoint table.
    summary['completed_endpoints'].extend([terminal_entry, best_entry])
    for role, endpoint in (('primary', terminal), ('secondary', best)):
        reference = next(item for item in summary['completed_endpoints']
                         if item['arm'] == 'FULL' and item['endpoint_role'] == role)
        for metric in METRICS:
            summary['differences'].append({
                'endpoint_role': role, 'comparison': 'FULL-NO_MSCA', 'metric': metric,
                'FULL_fraction': reference['metrics_fraction'][metric],
                'reference_fraction': endpoint['metrics'][metric],
                'difference_percentage_points': (reference['metrics_fraction'][metric] - endpoint['metrics'][metric]) * 100,
            })

    progress = summary['NO_MSCA_progress']
    progress.update({
        'status': 'completed_320_rounds', 'completed_rounds': 320, 'epoch_index_0based': 319,
        'target_rounds': 320, 'remaining_rounds': 0, 'optimizer_step_attempts': 517440,
        'latest_metrics_fraction': terminal['metrics'], 'latest_metrics_percent': percent_map(terminal['metrics']),
        'rolling_checkpoint_sha256': receipt['rolling_checkpoint_sha256'],
        'latest_best_checkpoint_sha256': best['checkpoint_sha256'], 'ema_tensor_sha256': terminal['ema_tensor_sha256'],
        'source_ids': ['NO_MSCA_SECOND_RECOVERY_R320'], 'snapshot_cutoff': 'independently_verified_round320',
        'native_acceptance': native_acceptance,
        'update_since_previous_snapshot': {'previous_rounds': 280, 'new_verified_rounds': 40,
                                           'new_verified_updates': 64680},
        'current_remote_training_state': 'COMPLETED_AND_BACKED_UP',
        'latest_prediction_sha256': terminal['prediction_sha256'],
        'latest_evaluation_arrays_sha256': terminal['evaluation_arrays_sha256'],
        'primary_endpoint': terminal_entry,
        'secondary_endpoint_after_320_rounds': best_entry,
        'completion_evidence': {
            'binder_path': str(RECOVERY / 'round320_completion_and_endpoints_20261111.json'),
            'monitor_status_sha256': sha(EVIDENCE / 'monitor_status.json'),
            'run_receipt_sha256': sha(EVIDENCE / 'completed_run_receipt.json'),
            'supervisor_sha256': sha(EVIDENCE / 'completed_supervisor.json'),
            'supervisor_exit_code': 0, 'native_backup_epoch': 319,
        },
    })
    progress['recovery'].update({
        'new_verified_rounds': 156, 'new_verified_optimizer_updates': 252252,
        'cumulative_recovered_rounds_since29': 291,
        'cumulative_recovered_optimizer_updates_since29': 470547,
        'remaining_evaluations': 0, 'remaining_optimizer_update_attempts': 0,
        'retained_plus_new_rounds': 320, 'new_completed_rounds': 156,
        'new_optimizer_update_attempts': 252252,
    })
    progress['best_so_far_progress'] = {
        'role': 'AP_BEST_THROUGH_COMPLETED_320_ROUNDS', 'round_1based': best['epoch_index'] + 1,
        'epoch_index_0based': best['epoch_index'], 'cutoff_rounds': 320,
        'same_epoch_as_latest_progress': best['epoch_index'] == 319,
        'metrics_fraction': best['metrics'], 'metrics_percent': percent_map(best['metrics']),
        'checkpoint_sha256': best['checkpoint_sha256'], 'ema_tensor_sha256': best['ema_tensor_sha256'],
        'model_tensor_sha256': best['model_tensor_sha256'], 'prediction_sha256': best['prediction_sha256'],
        'evaluation_arrays_sha256': best['evaluation_arrays_sha256'], 'metrics_sha256': best['metrics_sha256'],
        'source_id': 'NO_MSCA_SECOND_RECOVERY_R320',
    }
    progress['connectivity'] = {
        'source_id': 'NO_MSCA_NATIVE_COMPLETION_R320', 'status': 'COMPLETED_AND_BACKED_UP',
        'remote_training_state_now': 'COMPLETED', 'native_accepted_round': 320,
        'worker_exit_code': 0, 'monitor_status_sha256': sha(EVIDENCE / 'monitor_status.json'),
        'run_receipt_sha256': sha(EVIDENCE / 'completed_run_receipt.json'),
        'supervisor_sha256': sha(EVIDENCE / 'completed_supervisor.json'),
        'scope': 'Native worker and supervisor completion evidence; no present SSH liveness claim.',
    }
    progress.setdefault('connectivity_history', []).append(dict(progress['connectivity']))
    progress.setdefault('observation_history', []).append(dict(progress['live_observation']))

    source_specs = [
        ('NO_MSCA_LINEAGE_R320', AUDIT, 'Independent audit of retained history and saved endpoints'),
        ('NO_MSCA_NATIVE_MONITOR_R320', EVIDENCE / 'monitor_status.json', 'Native monitor completion and backup status'),
        ('NO_MSCA_NATIVE_RECEIPT_R320', EVIDENCE / 'completed_run_receipt.json', 'Worker completion receipt'),
        ('NO_MSCA_NATIVE_SUPERVISOR_R320', EVIDENCE / 'completed_supervisor.json', 'Successful supervisor exit'),
        ('NO_MSCA_NATIVE_EPOCH_ROWS_R320', source_path, 'Cumulative accepted epoch records 0–319'),
        ('NO_MSCA_NATIVE_EPOCH_MANIFEST_R320', EVIDENCE / 'epoch0319/manifest.json', 'Round-320 checkpoint and artifact manifest'),
        ('NO_MSCA_NATIVE_BEST_MANIFEST_R302', EVIDENCE / 'epoch0301/manifest.json', 'Round-302 AP-best checkpoint and artifact manifest'),
    ]
    new_sources = []
    for source_id, path, note in source_specs:
        new_sources.append({'source_id': source_id, 'relative_to_remote_root': str(path), 'path': str(path),
                            'bytes': path.stat().st_size, 'sha256': sha(path), 'status_in_source': note})
    summary['input_sources'].extend(new_sources)
    summary['updated_local_date'] = LOCAL_DATE
    summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
    summary['limitations'] = [line for line in summary['limitations']
                              if not line.startswith('NO_MSCA本快照')
                              and not line.startswith('历史164轮快照')]
    summary['limitations'].extend([
        'NO_MSCA 已完成并核验320轮。其固定末轮主端点为320轮；AP最佳次端点为302轮，二者均使用同一548张本地COCO bbox验证集。',
        'NO_MSCA移除MSCA门控后训练参数量为3,699,662，较FULL少3,841,280；因此FULL与NO_MSCA的数值差同时伴随模型容量变化，应解释为条件消融结果。',
        'NO_MSCA从已接受的第29轮断点恢复，并在已接受第164轮后再次恢复。旧实例未接受的第165轮观察到1,401次更新；其他未保留物理工作量未知，不能声称等物理计算或无中断位级等价。',
        '末轮与AP最佳端点都在同一单次训练轨迹和验证集上选取；没有多种子方差或独立测试集证据。',
    ])
    summary['source_metadata_notes'].append(
        '逐轮CSV的前1240行按既有数据快照逐字节保留；其中NO_MSCA旧行的accepted_prefix_only标记表示其原入库阶段，最终完成状态见实验汇总及第320轮更新核查。')
    write_json(OUT / '实验汇总.json', summary)

    # Add the two verified endpoints and all twelve same-row metrics to the CSV tables.
    endpoint_path = OUT / '终点信息.csv'
    with endpoint_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); endpoint_fields = reader.fieldnames; endpoint_rows = list(reader)
    for entry in (terminal_entry, best_entry):
        endpoint_rows.append({key: entry.get(key, '') for key in endpoint_fields})
    write_atomic(endpoint_path, csv_bytes(endpoint_fields, endpoint_rows))

    metric_path = OUT / '终点指标.csv'
    with metric_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); metric_fields = reader.fieldnames; metric_rows = list(reader)
    for role, endpoint in (('primary', terminal), ('secondary', best)):
        entry = terminal_entry if role == 'primary' else best_entry
        for metric in METRICS:
            metric_rows.append({'arm': 'NO_MSCA', 'endpoint_role': role,
                                'round_1based': entry['round_1based'], 'epoch_index_0based': entry['epoch_index_0based'],
                                'metric': metric, 'value_fraction': endpoint['metrics'][metric],
                                'value_percent': endpoint['metrics'][metric] * 100,
                                'source_id': 'NO_MSCA_SECOND_RECOVERY_R320',
                                'source_json_pointer': f"/endpoints/{'terminal' if role == 'primary' else 'best'}/metrics/{metric}"})
    write_atomic(metric_path, csv_bytes(metric_fields, metric_rows))

    delta_path = OUT / 'FULL差值.csv'
    with delta_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); delta_fields = reader.fieldnames; delta_rows = list(reader)
    for role, endpoint in (('primary', terminal), ('secondary', best)):
        full = next(item for item in summary['completed_endpoints'] if item['arm'] == 'FULL' and item['endpoint_role'] == role)
        for metric in METRICS:
            delta_rows.append({'endpoint_role': role, 'comparison': 'FULL-NO_MSCA', 'metric': metric,
                               'FULL_fraction': full['metrics_fraction'][metric],
                               'reference_fraction': endpoint['metrics'][metric],
                               'difference_percentage_points': (full['metrics_fraction'][metric] - endpoint['metrics'][metric]) * 100})
    write_atomic(delta_path, csv_bytes(delta_fields, delta_rows))

    progress_path = OUT / 'NO_MSCA进度.csv'
    with progress_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); progress_fields = reader.fieldnames; progress_rows = list(reader)
    for row in progress_rows:
        metric = row['metric']
        row.update(status='completed_320_rounds', completed_rounds=320, target_rounds=320,
                   latest_value_fraction=terminal['metrics'][metric], latest_value_percent=terminal['metrics'][metric] * 100,
                   round320_primary_value=terminal['metrics'][metric],
                   round320_secondary_value=best['metrics'][metric],
                   source_ids='NO_MSCA_SECOND_RECOVERY_R320', remote_training_state_now='COMPLETED_AND_BACKED_UP',
                   native_backup_client_alive='COMPLETED_EXIT0_AND_FINAL_BACKUP_VERIFIED',
                   observation_scope='Metrics cover accepted rounds 1–320; terminal and AP-best endpoint are bound to separate saved rows.',
                   second_recovery_first_epoch_index=164)
    write_atomic(progress_path, csv_bytes(progress_fields, progress_rows))

    # Preserve every existing per-epoch byte and append only newly audited rounds 281–320.
    csv_path = OUT / '逐轮指标.csv'
    old_csv = csv_path.read_bytes()
    old_rows = list(csv.DictReader(io.StringIO(old_csv.decode('utf-8-sig'))))
    assert len(old_rows) == 1240 and old_rows[-1]['arm'] == 'NO_MSCA'
    assert int(old_rows[-1]['round_1based']) == 280
    fields = list(old_rows[0].keys())
    additional = []
    for index in range(280, 320):
        row = rows[index]
        raw_line = raw_lines[index]
        canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
        ref = references[index]
        item = {
            'arm': 'NO_MSCA', 'series_status': 'accepted_prefix_only', 'seed': 0,
            'round_1based': index + 1, 'actual_epoch_index_0based': row['epoch'],
            'last_epoch_state_counter': row['last_epoch'], 'optimizer_step_attempts': row['optimizer_step_attempts'],
            'ema_updates': row['ema_updates'], 'train_loss_mean': row['train_loss_mean'], **row['metrics'],
            'weight_source': row['weight_source'], 'evaluated_ema_tensor_sha256': row['ema_tensor_sha256'],
            'prediction_sha256': row['prediction_sha256'], 'evaluation_arrays_sha256': row['evaluation_arrays_sha256'],
            'first_executed_lrs': json.dumps(row['first_executed_lrs'], separators=(',', ':')),
            'post_final_update_lrs': json.dumps(row['post_final_update_lrs'], separators=(',', ':')),
            'seconds': row['seconds'], 'created_utc': row['created_utc'],
            'protocol_sha256': row['protocol_sha256'], 'configuration_sha256': row['config_sha256'],
            'source_id': 'NO_MSCA_NATIVE_EPOCH_ROWS_R320', 'source_sha256': sha(source_path),
            'source_path': str(source_path), 'source_locator': f'line {index + 1}',
            'source_record_index_0based': index, 'source_raw_record_sha256': digest(raw_line),
            'normalization': 'none', 'normalized_row_sha256': digest(canonical),
            'checkpoint_sha256_present_in_original_source': row.get('checkpoint_sha256', ''),
        }
        additional.append(item)
    assert len(additional) == 40
    tail = io.StringIO(newline='')
    csv.DictWriter(tail, fieldnames=fields, extrasaction='raise').writerows(additional)
    appended = old_csv + tail.getvalue().encode('utf-8')
    write_atomic(csv_path, appended)

    mapping_path = OUT / '逐轮来源与切片.json'
    mapping = read(mapping_path)
    mapping['sources'].extend([
        {'source_id': item['source_id'], 'path': item['path'], 'bytes': item['bytes'],
         'sha256': item['sha256'], 'expected_sha256': item['sha256']}
        for item in new_sources
    ])
    mapping['slices'].append({'arm': 'NO_MSCA', 'source_id': 'NO_MSCA_NATIVE_EPOCH_ROWS_R320',
                              'source_records_0based_half_open': [280, 320], 'rounds_included': [281, 320]})
    mapping['rows_per_arm']['NO_MSCA'] = 320
    mapping['accepted_stage_sources'].append(native_acceptance)
    mapping['excluded'] = [item for item in mapping['excluded'] if item.get('arm') != 'NO_MSCA']
    mapping['excluded'].append({'arm': 'NO_MSCA', 'rule':
        'Only accepted rounds 1–320 are included. The old expired-instance partial round165 branch and its unretained work are excluded; its known 1,401 update attempts and the unknown remaining physical work are disclosed in recovery limitations.'})
    mapping.update(updated_local_date=LOCAL_DATE, recovery_source_id='NO_MSCA_SECOND_RECOVERY_R320')
    write_json(mapping_path, mapping)

    # Record check results without changing the original per-row evidence prefix.
    for filename in ('输入核查.json', '逐轮核查.json'):
        report = read(OUT / filename)
        report.setdefault('checks', []).extend([
            {'check': 'Native worker and supervisor receipts prove completed round 320, exit code 0, and verified epoch0319 backup.', 'status': 'PASS'},
            {'check': 'Cumulative accepted NO_MSCA JSONL has 320 contiguous rows and matches the independent round-320 lineage audit.', 'status': 'PASS'},
            {'check': 'The 40 appended round281–320 CSV records preserve the original 1240-row prefix and match native metrics, counters, LR metadata, hashes and row digests.', 'status': 'PASS'},
            {'check': 'The round320 terminal endpoint and round302 highest-AP endpoint are separately identified from their saved rows.', 'status': 'PASS'},
        ])
        report['check_count'] = len(report['checks'])
        report['NO_MSCA_accepted_rounds'] = 320
        report.setdefault('new_source_sha256', {}).update({item['source_id']: item['sha256'] for item in new_sources})
        if filename == '逐轮核查.json':
            report['rows'] = 1280
            report['csv_sha256'] = sha(csv_path)
            report['NO_MSCA_latest_round'] = 320
            report['NO_MSCA_best_AP_round'] = best['epoch_index'] + 1
        write_json(OUT / filename, report)

    old_description = (OUT / '汇总说明.md').read_text(encoding='utf-8')
    first_heading_end = old_description.find('\n\n')
    history_marker = '\n2026-11-08 更新。'
    history_at = old_description.find(history_marker)
    if history_at < 0:
        history_at = old_description.find('\n2026-11-09 更新。')
    history = old_description[history_at:] if history_at >= 0 else ''
    base = next(x for x in summary['completed_endpoints'] if x['arm'] == 'BASE' and x['endpoint_role'] == 'primary')
    full = next(x for x in summary['completed_endpoints'] if x['arm'] == 'FULL' and x['endpoint_role'] == 'primary')
    no_mps = next(x for x in summary['completed_endpoints'] if x['arm'] == 'NO_MPS' and x['endpoint_role'] == 'primary')
    base_best = next(x for x in summary['completed_endpoints'] if x['arm'] == 'BASE' and x['endpoint_role'] == 'secondary')
    full_best = next(x for x in summary['completed_endpoints'] if x['arm'] == 'FULL' and x['endpoint_role'] == 'secondary')
    no_mps_best = next(x for x in summary['completed_endpoints'] if x['arm'] == 'NO_MPS' and x['endpoint_role'] == 'secondary')
    rows_for_summary = [(base, base_best), (full, full_best), (no_mps, no_mps_best), (terminal_entry, best_entry)]
    table = ['# 实验数据汇总', '', '本快照汇集 BASE、FULL、NO_MPS 和 NO_MSCA 各自320轮轨迹。主端点统一取实际第320轮晚期回滚前的 EMA 评估记录；次端点取同一轨迹中最高 AP 的单轮，并沿用该轮全部12项指标。', '',
             '| 配置 | 主端点轮次 | 主端点 AP | 主端点 APs | AP最佳轮次 | AP最佳值 | 训练参数量 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for primary, secondary in rows_for_summary:
        table.append(f"| {primary['arm']} | {primary['round_1based']} | {primary['metrics_percent']['AP']:.4f}% | {primary['metrics_percent']['APs']:.4f}% | {secondary['round_1based']} | {secondary['metrics_percent']['AP']:.4f}% | {primary['trainable_parameters']:,} |")
    full_no_mps_ap = (full['metrics_fraction']['AP'] - no_mps['metrics_fraction']['AP']) * 100
    full_no_mps_aps = (full['metrics_fraction']['APs'] - no_mps['metrics_fraction']['APs']) * 100
    full_no_mps_arl = (full['metrics_fraction']['ARl'] - no_mps['metrics_fraction']['ARl']) * 100
    full_no_msca_ap = (full['metrics_fraction']['AP'] - terminal['metrics']['AP']) * 100
    full_no_msca_aps = (full['metrics_fraction']['APs'] - terminal['metrics']['APs']) * 100
    full_no_msca_arl = (full['metrics_fraction']['ARl'] - terminal['metrics']['ARl']) * 100
    params_removed = full['trainable_parameters'] - param_count
    history_text = (
        f"\n\n主端点 FULL−BASE 的 AP 为{(full['metrics_fraction']['AP']-base['metrics_fraction']['AP'])*100:+.4f}个百分点；FULL−NO_MPS 的 AP/APs/ARl 为{full_no_mps_ap:+.4f}/{full_no_mps_aps:+.4f}/{full_no_mps_arl:+.4f}个百分点，其中 ARl 差值为负。FULL−NO_MSCA 末轮主端点 AP/APs/ARl 为{full_no_msca_ap:+.4f}/{full_no_msca_aps:+.4f}/{full_no_msca_arl:+.4f}个百分点。NO_MSCA 的AP最佳次端点位于第{best['epoch_index']+1}轮，AP为{best['metrics']['AP']*100:.4f}%；末轮与次端点不同，不能混为同一个权重。\n\n"
        f"NO_MSCA 移除 MSCA 门控后的训练参数量为{param_count:,}，较 FULL 少{params_removed:,}个。因容量发生变化，该数值是条件消融结果，不能单独归因为注意力机制的纯效应。FULL、NO_MPS、NO_MSCA间的训练初始化和结构差异按协议分别列明；BASE不代表两个模块联合移除。\n\n"
        "NO_MSCA 恢复自已接受的第29轮断点，并在第164轮后再次恢复。末轮第319索引/第320轮备份有本机接收凭证、累计逐轮记录与独立审计绑定。旧实例未接受的第165轮分支至少有1,401次弃用更新，后续未保留物理工作量未知；不声称完整物理计算量相等或与无中断运行位级等价。单种子、同一验证集选点及本地转换后的非官方 VisDrone COCO bbox 验证仍限制泛化和统计推断。\n\n"
        "逐轮 CSV 的既有1240行保持原始字节前缀。本次仅追加第281–320轮40条记录，来源为已接受的epoch0319累计日志；四组现各320行，共1280行。NO_MSCA旧行保留其入库时的accepted_prefix_only状态标记，最终完整状态以本快照的终点表、汇总JSON及第320轮核查记录为准。\n\n"
        "训练参数：6,471张训练图、548张验证图、640×640、batch4、FP32；maxDets 1/10/100，每图top-300。每组对应517,440次累计更新尝试。模型前向、评价器重跑和新训练均未用于本次数据整合。\n\n"
    )
    history = history.replace('NO_MSCA 的来源快照已独立核验至第 280 轮，进度 AP 为 21.9282%，训练参数量为 3,699,662。它单独收录于进度文件和工作表，320 轮主端点及最终 AP 最佳端点保持空缺。', '')
    write_atomic(OUT / '汇总说明.md', ('\n'.join(table) + history_text + history).encode('utf-8'))

    detail = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8')
    detail = detail.replace('共 1240 行', '共 1280 行').replace('共1240行', '共1280行')
    detail += ('\n\n2026-11-11 第320轮完成。NO_MSCA 1–320行均由独立通过的累计epoch0319日志核对；相对上一数据快照只追加281–320共40行，旧1240行保留逐字节前缀。四组现各320行，共1280行。主端点为NO_MSCA第320轮，AP最佳次端点为第302轮；两者的12指标和检查点哈希均分别绑定到原始单行。旧partial165分支没有并入接受轨迹。\n')
    write_atomic(OUT / '逐轮明细说明.md', detail.encode('utf-8'))

    # Add a round-320 pre-workbook report; final hash/visual checks are added after the workbook edit.
    write_json(OUT / '第320轮更新核查.json', {
        'status': 'ROUND320_TRAINING_AND_ENDPOINT_EVIDENCE_VERIFIED_DATA_WRITE_AWAITING_WORKBOOK_QA',
        'local_date': LOCAL_DATE, 'audit_sha256': sha(AUDIT),
        'cumulative_source_sha256': sha(source_path), 'native_monitor_sha256': sha(EVIDENCE / 'monitor_status.json'),
        'native_run_receipt_sha256': sha(EVIDENCE / 'completed_run_receipt.json'),
        'native_supervisor_sha256': sha(EVIDENCE / 'completed_supervisor.json'),
        'native_completed_rounds': 320, 'optimizer_step_attempts': 517440, 'supervisor_exit_code': 0,
        'NO_MSCA_terminal_round_1based': 320, 'NO_MSCA_best_AP_round_1based': best['epoch_index'] + 1,
        'NO_MSCA_terminal_metrics': terminal['metrics'], 'NO_MSCA_best_metrics': best['metrics'],
        'new_csv_rows_appended': 40, 'old_csv_exact_byte_prefix_rows': 1240,
        'per_epoch_rows_after_update': 1280, 'model_forwards': 0, 'evaluator_runs': 0,
        'workbook_and_snapshot_pending': True,
    })

    print(json.dumps({'status': 'ROUND320_DATA_SOURCES_BOUND_AND_ROWS_APPENDED',
                      'terminal_AP': terminal['metrics']['AP'], 'best_round': best['epoch_index'] + 1,
                      'best_AP': best['metrics']['AP'], 'new_rows': len(additional),
                      'rows_total': 1280, 'completion_packet': str(EVIDENCE)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
