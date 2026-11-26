"""Apply the saved, independently verified NO_MSCA round-30 record."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import assemble_epoch_data as epochs


OUT = Path(__file__).resolve().parent.parent
RECOVERY = OUT.parent / '远程公平对照_20261103/模块消融/NO_MSCA/recovery30219_r29_20261108'
SNAPSHOT = OUT / 'snapshots/NO_MSCA_round029_before_update_20261108'
XLSX = Path('outputs/实验数据汇总_20261108/实验数据汇总.xlsx')
NEW_SOURCES = {
    'NO_MSCA_RECOVERED_R30': 'lineage_recovered_through_round030.json',
    'NO_MSCA_NATIVE_R30': 'first_recovered_round030_native_acceptance.json',
    'NO_MSCA_LIVE_05': 'live_training_progress_05.json',
    'NO_MSCA_RECOVERY_ADDENDUM': 'recovery_addendum.json',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def replace_bytes(path, data):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def replace_json(path, value):
    replace_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def csv_bytes(fields, rows, bom=True):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig' if bom else 'utf-8')


def prepare():
    summary = read(OUT / '实验汇总.json')
    assert summary['NO_MSCA_progress']['completed_rounds'] == 29
    manifest = read(OUT / '交付清单.json')
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    snapshot_files = []
    for item in manifest['files']:
        source = OUT / item['file']
        assert sha(source) == item['sha256'], item['file']
        destination = SNAPSHOT / item['file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert sha(destination) == item['sha256']
        snapshot_files.append(item)
    shutil.copy2(OUT / '交付清单.json', SNAPSHOT / '交付清单.json')
    replace_json(SNAPSHOT / 'snapshot_manifest.json', {
        'status': 'NO_MSCA_ROUND29_SNAPSHOT_PRESERVED', 'local_date': '2026-11-08',
        'NO_MSCA_accepted_rounds': 29, 'per_epoch_rows': 989, 'files': snapshot_files,
        'old_manifest_sha256': sha(SNAPSHOT / '交付清单.json'),
    })
    cert = read(RECOVERY / NEW_SOURCES['NO_MSCA_RECOVERED_R30'])
    native = read(RECOVERY / NEW_SOURCES['NO_MSCA_NATIVE_R30'])
    live_wrapper = read(RECOVERY / NEW_SOURCES['NO_MSCA_LIVE_05'])
    assert cert['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and cert['completed_rounds'] == 30
    assert cert['learning_rates']['recorded_attempts'] == 48510
    assert cert['learning_rates']['maximum_absolute_difference'] == cert['maximum_latest_metric_error'] == 0
    assert cert['validation_images'] == 548 and cert['predictions'] == 164400
    assert cert['recovery']['new_completed_rounds'] == 1
    assert cert['recovery']['new_optimizer_update_attempts'] == 1617
    assert cert['recovery']['retained_plus_new_rounds'] == 30
    assert native['status'] == 'FIRST_RECOVERED_ROUND_NATIVE_BACKUP_ACCEPTED' and native['round'] == 30
    assert native['optimizer_step_attempts'] == 48510
    assert live_wrapper['exit_code'] == 0
    live = json.loads(live_wrapper['stdout'])
    live_receipt = live['records']['run/receipt.json']
    assert live_receipt['actual_loop_epoch'] == 30 and live_receipt['batch'] == 1
    assert live_receipt['completed_epochs'] == 30 and live_receipt['optimizer_step_attempts'] == 48511
    rows, references, accepted = epochs.accepted_stage('NO_MSCA', 29)
    assert accepted['manifest_sha256'] == native['manifest_sha256']
    assert accepted['rolling_checkpoint_sha256'] == native['rolling_sha256']
    assert rows[-1]['metrics'] == cert['latest_metrics'] == cert['endpoints']['best']['metrics']
    assert rows[-1]['ema_tensor_sha256'] == cert['endpoints']['best']['ema_tensor_sha256']
    new_sources = []
    for source_id, filename in NEW_SOURCES.items():
        path = RECOVERY / filename
        value = read(path)
        new_sources.append({
            'source_id': source_id, 'relative_to_remote_root': str(path.relative_to(epochs.REMOTE)).replace('\\', '/'),
            'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
            'status_in_source': value.get('status', 'SAVED_REMOTE_OBSERVATION'),
        })
    old_completed = json.loads(json.dumps(summary['completed_endpoints']))
    progress = summary['NO_MSCA_progress']
    best = cert['endpoints']['best']
    progress.update(
        completed_rounds=30, epoch_index_0based=29, remaining_rounds=290, optimizer_step_attempts=48510,
        latest_metrics_fraction=cert['latest_metrics'],
        latest_metrics_percent={metric: value * 100 for metric, value in cert['latest_metrics'].items()},
        rolling_checkpoint_sha256=native['rolling_sha256'], latest_best_checkpoint_sha256=best['checkpoint_sha256'],
        ema_tensor_sha256=best['ema_tensor_sha256'], source_ids=['NO_MSCA_RECOVERED_R30', 'NO_MSCA_NATIVE_R30'],
        recovery={
            'from_accepted_round': 29, 'new_verified_rounds': 1, 'new_verified_optimizer_updates': 1617,
            'original_checkpoint_sha256': cert['recovery']['prefix']['checkpoint_sha256'],
            'recovery_addendum_sha256': cert['recovery']['recovery_addendum_sha256'],
            'restored_state_sections': cert['recovery']['prefix']['anchor_state_sections_equal_selected_checkpoint'],
            'restored_rng_sections': ['python', 'numpy', 'torch_cpu', 'torch_cuda_all', 'train_loader',
                                     'train_sampler', 'validation_loader'],
            'target_gpu_uuid': cert['recovery']['target_gpu_uuid'], 'target_boot_id': cert['recovery']['target_boot_id'],
            'physical_compute_limitation': cert['recovery']['physical_compute_limitation'],
            'contract_live_flag': cert['recovery']['live_recovery_training_progress_verified'],
            'contract_live_flag_scope': 'Original 29-round contract subreport performs no remote live check. Top-level round30 saved-state audit verified one new round and 1617 new updates.',
        },
        live_observation={
            'source_id': 'NO_MSCA_LIVE_05', 'observed_server_utc': live['observed_server_utc'],
            'actual_epoch_index_0based': 30, 'round_1based': 31, 'batch': 1,
            'optimizer_step_attempts': 48511, 'accepted_rounds': 30,
            'scope': 'Saved running-process observation, separate from the 30 accepted metric rows.',
        },
    )
    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None
    summary['input_sources'].extend(new_sources)
    summary['updated_local_date'] = '2026-11-08'
    summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
    summary['limitations'][-1] = 'NO_MSCA本快照仅核验30轮，320轮终点为空，不参与最终端点差值。'
    summary['limitations'].append('NO_MSCA在第29轮后换机恢复，原始模型、EMA、优化器、warmup及全部保存RNG恢复已核验。失联后未保留的物理计算量未知。')
    assert summary['completed_endpoints'] == old_completed
    replace_json(OUT / '实验汇总.json', summary)
    progress_rows = []
    with (OUT / 'NO_MSCA进度.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        progress_fields = reader.fieldnames
        for item in reader:
            item.update(completed_rounds=30, latest_value_fraction=cert['latest_metrics'][item['metric']],
                        latest_value_percent=cert['latest_metrics'][item['metric']] * 100,
                        source_ids='NO_MSCA_RECOVERED_R30,NO_MSCA_NATIVE_R30')
            assert item['round320_primary_value'] == item['round320_secondary_value'] == ''
            progress_rows.append(item)
    replace_bytes(OUT / 'NO_MSCA进度.csv', csv_bytes(progress_fields, progress_rows))
    replace_bytes(OUT / '输入来源.csv', csv_bytes(list(summary['input_sources'][0]), summary['input_sources']))

    original_csv = (OUT / '逐轮指标.csv').read_bytes()
    with (OUT / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        old_rows = list(reader)
    assert len(old_rows) == 989
    assert int(old_rows[-1]['actual_epoch_index_0based']) == 28 and old_rows[-1]['arm'] == 'NO_MSCA'
    row, ref = rows[-1], references[-1]
    source_id = 'NO_MSCA_RECOVERED_R30_EPOCH_ROWS'
    canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    added = {
        'arm': 'NO_MSCA', 'series_status': 'accepted_prefix_only', 'seed': 0, 'round_1based': 30,
        'actual_epoch_index_0based': 29, 'last_epoch_state_counter': row['last_epoch'],
        'optimizer_step_attempts': row['optimizer_step_attempts'], 'ema_updates': row['ema_updates'],
        'train_loss_mean': row['train_loss_mean'], **row['metrics'], 'weight_source': row['weight_source'],
        'evaluated_ema_tensor_sha256': row['ema_tensor_sha256'], 'prediction_sha256': row['prediction_sha256'],
        'evaluation_arrays_sha256': row['evaluation_arrays_sha256'],
        'first_executed_lrs': json.dumps(row['first_executed_lrs'], separators=(',', ':')),
        'post_final_update_lrs': json.dumps(row['post_final_update_lrs'], separators=(',', ':')),
        'seconds': row['seconds'], 'created_utc': row['created_utc'], 'protocol_sha256': row['protocol_sha256'],
        'configuration_sha256': row['config_sha256'],
        **{key: value for key, value in ref.items() if key not in ('row', 'source_checkpoint_sha256')},
        'normalized_row_sha256': digest(canonical), 'checkpoint_sha256_present_in_original_source': '',
    }
    added['source_id'] = source_id
    stream = io.StringIO(newline='')
    csv.DictWriter(stream, fieldnames=fields).writerow(added)
    replace_bytes(OUT / '逐轮指标.csv', original_csv + stream.getvalue().encode('utf-8'))
    mapping = read(OUT / '逐轮来源与切片.json')
    new_epoch_source = next(item for item in epochs.SOURCES if item['source_id'] == 'NO_MSCA_ACCEPTED_EPOCH_ROWS')
    new_epoch_source['source_id'] = source_id
    mapping['sources'].append(new_epoch_source)
    mapping['sources'].extend(new_sources)
    mapping['slices'].append({'arm': 'NO_MSCA', 'source_id': source_id,
                              'source_records_0based_half_open': [29, 30], 'rounds_included': [30, 30]})
    mapping['rows_per_arm']['NO_MSCA'] = 30
    mapping['accepted_stage_sources'].append(accepted)
    mapping['excluded'][-1] = {'arm': 'NO_MSCA', 'rule': 'Live round31 updates are excluded from the accepted 30-round metric prefix.'}
    mapping['updated_local_date'] = '2026-11-08'
    mapping['recovery_source_id'] = 'NO_MSCA_RECOVERED_R30'
    replace_json(OUT / '逐轮来源与切片.json', mapping)
    for filename in ('输入核查.json', '逐轮核查.json'):
        report = read(OUT / filename)
        report['checks'].extend(epochs.CHECKS)
        report['checks'].append({'check': 'Recovered round30 independent certificate and native acceptance agree', 'status': 'PASS'})
        report['check_count'] = len(report['checks'])
        report['NO_MSCA_accepted_rounds'] = 30
        report['new_source_sha256'] = {item['source_id']: item['sha256'] for item in new_sources}
        if filename == '逐轮核查.json':
            report['rows'] = 990
            report['csv_sha256'] = sha(OUT / '逐轮指标.csv')
        replace_json(OUT / filename, report)
    text = (OUT / '汇总说明.md').read_text(encoding='utf-8')
    text = text.replace('NO_MSCA 的来源快照仅核验至第 29 轮，进度 AP 为 8.7082%',
                        'NO_MSCA 的来源快照已核验至第 30 轮，进度 AP 为 8.8572%')
    text += '\n2026-11-08 更新。NO_MSCA 从已接受的第 29 轮断点换机恢复，原始模型、EMA、优化器、criterion、postprocessor、warmup 和全部保存的 RNG 已核验恢复。新增第 30 轮及 1,617 次更新通过独立保存状态核验，累计 48,510 次。失联后未保留的物理计算量未知。第 31 轮的运行快照另作运行证据，进度指标仍截至已接受的第 30 轮。恢复子合约中的 live=false 仅表示该子报告未做远程实时检查。\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8')
    text = text.replace('共 989 行', '共 990 行').replace('前 29 行', '前 30 行')
    text = text.replace('NO_MSCA 使用已接受 epoch0028 备份中的累计日志，并核对清单、文件哈希和双端回执。第 30 轮进行中的批次不纳入本快照。',
                        'NO_MSCA 前 29 行保留 epoch0028 来源，第 30 行来自已接受 epoch0029 备份，并有换机恢复后的独立完整审计。第 31 轮进行中的批次不纳入本快照。')
    text += '\n2026-11-08 更新中，原 989 行的 CSV 字节前缀完整保留，只追加已核验的第 30 轮。第 29 轮快照保存在 snapshots/NO_MSCA_round029_before_update_20261108。换机恢复状态及物理计算量未知的限制见实验汇总 JSON。\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
    plan = {
        'status': 'DATA_UPDATED_WORKBOOK_PENDING', 'local_date': '2026-11-08',
        'snapshot': str(SNAPSHOT), 'xlsx_relative': str(XLSX),
        'old_xlsx_sha256': sha(SNAPSHOT / XLSX), 'new_NO_MSCA_metrics': cert['latest_metrics'],
        'new_source_sha256': {item['source_id']: item['sha256'] for item in new_sources},
        'protected_completed_endpoint_sha256': digest(json.dumps(old_completed, sort_keys=True).encode('utf-8')),
        'old_per_epoch_csv_sha256': digest(original_csv), 'new_per_epoch_csv_sha256': sha(OUT / '逐轮指标.csv'),
        'old_per_epoch_rows': 989, 'new_per_epoch_rows': 990,
    }
    replace_json(OUT / '_build/update_round30_plan.json', plan)
    print(json.dumps({'status': 'DATA_UPDATED_AND_ROUND29_SNAPSHOT_SAVED', 'per_epoch_rows': 990,
                      'NO_MSCA_accepted_rounds': 30, 'snapshot': str(SNAPSHOT)}, ensure_ascii=True))


if __name__ == '__main__':
    prepare()
