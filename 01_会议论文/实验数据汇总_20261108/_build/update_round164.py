"""Advance the saved NO_MSCA data cutoff to independent round164; no model/evaluator."""

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath

import assemble_epoch_data as epochs
from update_round30 import OUT, RECOVERY, XLSX, csv_bytes, read, replace_bytes, replace_json, sha

ROUNDS = 164
LOCAL_DATE = '2026-11-09'
CERTIFICATE = RECOVERY / 'lineage_recovered_through_round164.json'
LOSS_ROOT = RECOVERY.parent / 'connectivity_20261109/goal_monitor002_loss_20261109'
LOSS_PATH = LOSS_ROOT / 'access_loss_and_safe_round164.json'
LIVE_PATH = LOSS_ROOT / 'last_successful_remote_snapshot.json'
SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round160_verified_20261109'
FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round164_verified_20261109'


def main():
    summary = read(OUT / '实验汇总.json')
    assert summary['NO_MSCA_progress']['completed_rounds'] == 160
    previous_manifest = read(OUT / '交付清单.json')
    assert len(previous_manifest['files']) == 18
    assert sha(OUT / '交付清单.json') == sha(SNAPSHOT / '交付清单.json')
    for item in previous_manifest['files']:
        relative = PurePosixPath(item['file'].replace('\\', '/'))
        assert not relative.is_absolute() and '..' not in relative.parts
        for root in (OUT, SNAPSHOT):
            source = root.joinpath(*relative.parts)
            assert source.resolve().is_relative_to(root.resolve())
            assert source.stat().st_size == item['bytes'] and sha(source) == item['sha256']

    cert = read(CERTIFICATE)
    assert cert['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and cert['completed_rounds'] == ROUNDS
    assert cert['learning_rates']['recorded_attempts'] == ROUNDS * 1617 == 265188
    assert cert['learning_rates']['maximum_absolute_difference'] == cert['maximum_latest_metric_error'] == 0
    assert cert['validation_images'] == 548 and cert['predictions'] == 164400
    assert cert['recovery']['new_completed_rounds'] == 135
    assert cert['recovery']['new_optimizer_update_attempts'] == 218295
    assert cert['recovery']['retained_plus_new_rounds'] == ROUNDS
    assert cert['recovery']['recovery_addendum_sha256'] == summary['NO_MSCA_progress']['recovery']['recovery_addendum_sha256']
    assert cert['protocol_sha256'] == summary['NO_MSCA_progress']['protocol_sha256']
    loss = read(LOSS_PATH)
    assert loss['status'] == 'REMOTE_UNREACHABLE_LOCAL_BACKUP_CLIENT_FAILED_REMOTE_TRAINING_UNKNOWN'
    assert loss['last_native_accepted_round'] == loss['independent_state_and_lineage_round'] == ROUNDS
    assert loss['learning_rate_records'] == 265188
    assert loss['historical_last_observed_running_round'] == 165
    assert loss['backup_client']['alive'] is False and loss['observer']['alive'] is False
    assert loss['lineage_audit']['sha256'] == sha(CERTIFICATE)
    assert loss['remote_training_state_now'].startswith('UNKNOWN')
    live_wrapper = read(LIVE_PATH)
    assert live_wrapper['exit_code'] == 0
    live = json.loads(live_wrapper['stdout'])
    worker, supervisor = live['records']['run/receipt.json'], live['records']['supervisor.json']
    assert supervisor['pid'] == worker['pid'] == 1779501 and supervisor['supervisor_pid'] == 1779500
    assert supervisor['boot_id'] == live['boot_id'] == cert['recovery']['target_boot_id']
    assert supervisor['start_ticks'] == 18098010 and supervisor['supervisor_start_ticks'] == 18097864
    assert worker['actual_loop_epoch'] == worker['completed_epochs'] == ROUNDS
    assert worker['local_backup_verified']['epoch'] == 163
    rows, refs, accepted = epochs.accepted_stage('NO_MSCA', 163)
    assert len(rows) == ROUNDS and rows[-1]['metrics'] == cert['latest_metrics']
    assert rows[-1]['optimizer_step_attempts'] == 265188
    assert rows[-1]['ema_tensor_sha256'] == cert['endpoints']['best']['ema_tensor_sha256']
    assert accepted['rolling_checkpoint_sha256'] == worker['rolling_checkpoint_sha256']
    assert accepted['manifest_sha256'] == worker['local_backup_verified']['manifest_sha256']
    assert accepted['ack_sha256'] == worker['local_backup_verified']['ack_sha256']
    old_stage = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups/epochs/epoch0159')
    assert Path(refs[0]['source_path']).read_bytes().startswith((old_stage / 'epochs.jsonl').read_bytes())
    new_sources = []
    for source_id, source in [('NO_MSCA_RECOVERED_R164', CERTIFICATE), ('NO_MSCA_ACCESS_LOSS_R164', LOSS_PATH), ('NO_MSCA_HISTORICAL_LAST_REMOTE_R165', LIVE_PATH)]:
        value = read(source)
        new_sources.append({
            'source_id': source_id, 'relative_to_remote_root': str(source.relative_to(epochs.REMOTE)).replace('\\', '/'),
            'path': str(source), 'bytes': source.stat().st_size, 'sha256': sha(source),
            'status_in_source': value.get('status', 'SAVED_REMOTE_OBSERVATION'),
        })
    progress = summary['NO_MSCA_progress']
    best = cert['endpoints']['best']
    progress.update(
        completed_rounds=ROUNDS, epoch_index_0based=163, remaining_rounds=156,
        optimizer_step_attempts=265188, latest_metrics_fraction=cert['latest_metrics'],
        latest_metrics_percent={metric: value * 100 for metric, value in cert['latest_metrics'].items()},
        rolling_checkpoint_sha256=accepted['rolling_checkpoint_sha256'],
        latest_best_checkpoint_sha256=best['checkpoint_sha256'], ema_tensor_sha256=rows[-1]['ema_tensor_sha256'],
        source_ids=['NO_MSCA_RECOVERED_R164'], snapshot_cutoff='independently_verified_round164',
        native_acceptance=accepted,
        update_since_previous_snapshot={'previous_rounds': 160, 'new_verified_rounds': 4, 'new_verified_updates': 6468},
    )
    progress['recovery'].update(
        new_verified_rounds=135, new_verified_optimizer_updates=218295,
        physical_compute_limitation=cert['recovery']['physical_compute_limitation'],
        contract_live_flag=cert['recovery']['live_recovery_training_progress_verified'],
        contract_live_flag_scope='Original 29-round contract subreport performs no remote live check. The top-level round164 saved-state audit verifies 135 recovered rounds and 218295 recovered updates. Current remote training status is UNKNOWN.',
    )
    progress['live_observation'] = {
        'source_id': 'NO_MSCA_HISTORICAL_LAST_REMOTE_R165', 'observed_server_utc': live['observed_server_utc'],
        'actual_epoch_index_0based': worker['actual_loop_epoch'], 'round_1based': worker['actual_loop_epoch'] + 1,
        'batch': worker['batch'], 'optimizer_step_attempts': worker['optimizer_step_attempts'],
        'native_accepted_rounds_observed': worker['completed_epochs'], 'independent_metric_snapshot_rounds': ROUNDS,
        'scope': 'Historical last successful remote observation only; not current liveness. Remote state is now UNKNOWN. Independently audited metric data is cut off at round164.',
        'observation_kind': 'HISTORICAL_LAST_SUCCESSFUL_READ', 'current_remote_training_state': 'UNKNOWN',
    }
    progress['current_remote_training_state'] = 'UNKNOWN'
    progress['connectivity'] = {
        'source_id': 'NO_MSCA_ACCESS_LOSS_R164', 'observed_local_utc': loss['observed_local_utc'],
        'status': loss['status'], 'remote_training_state_now': 'UNKNOWN',
        'historical_last_observed_running_round': 165,
        'historical_last_successful_reader_server_utc': loss['historical_last_successful_reader_server_utc'],
        'historical_snapshot_file_mtime_local_utc': loss['historical_snapshot_file_mtime_local_utc'],
        'last_native_accepted_round': 164, 'independent_metric_snapshot_rounds': 164,
        'native_backup_client_alive': False, 'native_backup_client_exit_reason': loss['backup_client']['reason'],
        'read_only_observer_alive': False, 'access_failure_proves_terminal_training': False,
    }
    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None
    summary['input_sources'].extend(new_sources)
    summary['updated_local_date'] = LOCAL_DATE
    summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
    summary['limitations'] = [
        'NO_MSCA本快照仅独立核验164轮，320轮终点为空，不参与最终端点差值。'
        if line.startswith('NO_MSCA本快照') else line for line in summary['limitations']]
    summary['limitations'].append('当前远端训练状态UNKNOWN；失联前最后见第165轮训练，原生备份客户端已退出。指标严格截至已接受且独立核验第164轮，不证明远端已终止，也不证明仍在训练。')
    replace_json(OUT / '实验汇总.json', summary)
    with (OUT / 'NO_MSCA进度.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, progress_rows = reader.fieldnames, list(reader)
    fields = list(fields) + ['remote_training_state_now', 'historical_last_observed_running_round', 'native_backup_client_alive', 'observation_scope']
    for row in progress_rows:
        row.update(completed_rounds=ROUNDS, latest_value_fraction=cert['latest_metrics'][row['metric']],
                   latest_value_percent=cert['latest_metrics'][row['metric']] * 100, source_ids='NO_MSCA_RECOVERED_R164')
        row.update(remote_training_state_now='UNKNOWN', historical_last_observed_running_round=165,
                   native_backup_client_alive=False, observation_scope='Historical last read only; not current liveness; metrics cut off at independent round164')
        assert row['round320_primary_value'] == row['round320_secondary_value'] == ''
    replace_bytes(OUT / 'NO_MSCA进度.csv', csv_bytes(fields, progress_rows))
    replace_bytes(OUT / '输入来源.csv', csv_bytes(list(summary['input_sources'][0]), summary['input_sources']))
    old_csv = (OUT / '逐轮指标.csv').read_bytes()
    with (OUT / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, original_rows = reader.fieldnames, list(reader)
    assert len(original_rows) == 1120 and original_rows[-1]['arm'] == 'NO_MSCA'
    assert int(original_rows[-1]['round_1based']) == 160
    source_id = 'NO_MSCA_RECOVERED_R164_EPOCH_ROWS'
    additional = []
    for row, ref in zip(rows[160:164], refs[160:164]):
        canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
        item = {
            'arm': 'NO_MSCA', 'series_status': 'accepted_prefix_only', 'seed': 0, 'round_1based': row['epoch'] + 1,
            'actual_epoch_index_0based': row['epoch'], 'last_epoch_state_counter': row['last_epoch'],
            'optimizer_step_attempts': row['optimizer_step_attempts'], 'ema_updates': row['ema_updates'],
            'train_loss_mean': row['train_loss_mean'], **row['metrics'], 'weight_source': row['weight_source'],
            'evaluated_ema_tensor_sha256': row['ema_tensor_sha256'], 'prediction_sha256': row['prediction_sha256'],
            'evaluation_arrays_sha256': row['evaluation_arrays_sha256'],
            'first_executed_lrs': json.dumps(row['first_executed_lrs'], separators=(',', ':')),
            'post_final_update_lrs': json.dumps(row['post_final_update_lrs'], separators=(',', ':')),
            'seconds': row['seconds'], 'created_utc': row['created_utc'], 'protocol_sha256': row['protocol_sha256'],
            'configuration_sha256': row['config_sha256'],
            **{key: value for key, value in ref.items() if key not in ('row', 'source_checkpoint_sha256')},
            'normalized_row_sha256': epochs.digest(canonical), 'checkpoint_sha256_present_in_original_source': '',
        }
        item['source_id'] = source_id
        additional.append(item)
    assert len(additional) == 4
    stream = io.StringIO(newline='')
    csv.DictWriter(stream, fieldnames=fields).writerows(additional)
    replace_bytes(OUT / '逐轮指标.csv', old_csv + stream.getvalue().encode('utf-8'))
    mapping = read(OUT / '逐轮来源与切片.json')
    epoch_source = next(item for item in epochs.SOURCES if item['source_id'] == 'NO_MSCA_ACCEPTED_EPOCH_ROWS')
    epoch_source['source_id'] = source_id
    mapping['sources'].extend([epoch_source] + new_sources)
    mapping['slices'].append({'arm': 'NO_MSCA', 'source_id': source_id,
                              'source_records_0based_half_open': [160, 164], 'rounds_included': [161, 164]})
    mapping['rows_per_arm']['NO_MSCA'] = ROUNDS
    mapping['accepted_stage_sources'].append(accepted)
    mapping['excluded'][-1] = {'arm': 'NO_MSCA', 'rule':
        'Remote training status is now UNKNOWN and the native backup client has exited. The last successful historical observation saw round165; only accepted and independently audited rounds1-164 are exported. Round165 is excluded.'}
    mapping['updated_local_date'] = LOCAL_DATE
    mapping['recovery_source_id'] = 'NO_MSCA_RECOVERED_R164'
    replace_json(OUT / '逐轮来源与切片.json', mapping)
    for filename in ('输入核查.json', '逐轮核查.json'):
        report = read(OUT / filename)
        report['checks'].extend(epochs.CHECKS)
        report['checks'].append({'check': 'Round164 independent certificate cutoff retained; historical round165 excluded and current remote state UNKNOWN', 'status': 'PASS'})
        report['check_count'] = len(report['checks'])
        report['NO_MSCA_accepted_rounds'] = ROUNDS
        report.setdefault('new_source_sha256', {}).update({item['source_id']: item['sha256'] for item in new_sources})
        if filename == '逐轮核查.json':
            report['rows'] = 1124
            report['csv_sha256'] = sha(OUT / '逐轮指标.csv')
        replace_json(OUT / filename, report)
    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已独立核验至第 160 轮，进度 AP 为 16.4156%',
        'NO_MSCA 的来源快照已独立核验至第 164 轮，进度 AP 为 17.2070%')
    text += '\n2026-11-09 164轮更新。NO_MSCA 已接受且独立核验至第 164 轮，累计 265,188 次更新；第 29 轮换机后恢复段已核验 135 轮及 218,295 次更新。相对第 160 轮快照仅追加 4 行及 6,468 次更新。当前远端训练状态 UNKNOWN；失联前最后成功读取曾见第 165 轮训练，原生备份客户端因 600 秒重连窗口耗尽已自行退出，只读观察器也已退出。此历史现场不表示当前仍在训练，访问失败也不证明远端训练已终止。第 165 轮指标不纳入本汇总。原第 160 轮交付保持于 verified_snapshots/NO_MSCA_round160_verified_20261109。\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 1120 行', '共 1124 行').replace('前 160 行', '前 164 行')
    text = text.replace('第 141–160 行来自已接受 epoch0159 备份，并与恢复轨迹第 160 轮独立证书核对。第 161 轮及其后的指标不纳入本快照。',
                        '第 141–160 行来自已接受 epoch0159 备份，第 161–164 行来自已接受 epoch0163 备份，并与恢复轨迹第 164 轮独立证书核对。第 165 轮及其后的指标不纳入本快照。')
    text += '\n2026-11-09 164轮更新保留原 1120 行字节前缀，仅从已接受 epoch0163 的累计日志中追加第 161–164 轮。三组已完成数据各 320 行，NO_MSCA 164 行，总计 1124 行。当前远端状态 UNKNOWN，原生备份客户端已退出；最后见第 165 轮是历史观察，不作当前活跃状态或指标证据。全部指标仍为原始 0–1 比例。\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
    replace_json(OUT / '_build/update_round164_plan.json', {
        'status': 'DATA_UPDATED_WORKBOOK_PENDING', 'local_date': LOCAL_DATE,
        'snapshot': str(SNAPSHOT), 'xlsx_relative': str(XLSX), 'old_xlsx_sha256': sha(SNAPSHOT / XLSX),
        'target_rounds': ROUNDS, 'total_updates': 265188, 'previous_snapshot_rounds': 160,
        'update_tag': 'round164', 'new_NO_MSCA_metrics': cert['latest_metrics'],
        'progress_notes': [
            '第29轮换机：原始模型、EMA、优化器、warmup及保存RNG恢复已核验。',
            '恢复段135轮及218295次更新已核验，累计164轮及265188次。',
            '当前远端状态UNKNOWN；原生备份客户端及只读观察器均已退出。',
            '失联前最后见第165轮训练；只收录已接受且独立核验的164轮。',
            '失联后未保留的工作量未知，完整物理计算量无法确认。',
            '独立证书 NO_MSCA_RECOVERED_R164，12指标误差0，320轮终点空缺。',
        ],
        'new_source_sha256': {item['source_id']: item['sha256'] for item in new_sources},
    })
    print(json.dumps({'status': 'ROUND164_DATA_UPDATED', 'prior_snapshot': str(SNAPSHOT),
                      'NO_MSCA_rounds': ROUNDS, 'per_epoch_rows': 1124}, ensure_ascii=True))


if __name__ == '__main__':
    main()
