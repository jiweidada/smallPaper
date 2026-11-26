"""Advance only the NO_MSCA data snapshot to the saved round-41 certificate."""

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil

import assemble_epoch_data as epochs
from update_round30 import OUT, RECOVERY, XLSX, csv_bytes, read, replace_bytes, replace_json, sha


ROUNDS = 41
LOCAL_DATE = '2026-11-09'
CERTIFICATE = RECOVERY / 'lineage_recovered_through_round041.json'
LIVE_PATH = RECOVERY / 'live_goal_resume_request_20261109.json'
SNAPSHOT = OUT / 'snapshots/NO_MSCA_round030_before_update_20261109'


def main():
    summary = read(OUT / '实验汇总.json')
    assert summary['NO_MSCA_progress']['completed_rounds'] == 30
    previous_manifest = read(OUT / '交付清单.json')
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    for item in previous_manifest['files']:
        source = OUT / item['file']
        assert sha(source) == item['sha256'], item['file']
        target = SNAPSHOT / item['file']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == item['sha256']
    shutil.copy2(OUT / '交付清单.json', SNAPSHOT / '交付清单.json')
    replace_json(SNAPSHOT / 'snapshot_manifest.json', {
        'status': 'NO_MSCA_ROUND30_SNAPSHOT_PRESERVED', 'local_date': LOCAL_DATE,
        'NO_MSCA_independently_verified_rounds': 30, 'per_epoch_rows': 990,
        'files': previous_manifest['files'], 'old_manifest_sha256': sha(SNAPSHOT / '交付清单.json'),
    })
    cert = read(CERTIFICATE)
    assert cert['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and cert['completed_rounds'] == ROUNDS
    assert cert['learning_rates']['recorded_attempts'] == ROUNDS * 1617 == 66297
    assert cert['learning_rates']['maximum_absolute_difference'] == cert['maximum_latest_metric_error'] == 0
    assert cert['validation_images'] == 548 and cert['predictions'] == 164400
    assert cert['recovery']['new_completed_rounds'] == 12
    assert cert['recovery']['new_optimizer_update_attempts'] == 19404
    assert cert['recovery']['retained_plus_new_rounds'] == ROUNDS
    assert cert['recovery']['recovery_addendum_sha256'] == summary['NO_MSCA_progress']['recovery']['recovery_addendum_sha256']
    assert cert['protocol_sha256'] == summary['NO_MSCA_progress']['protocol_sha256']
    live_wrapper = read(LIVE_PATH)
    assert live_wrapper['exit_code'] == 0
    live = json.loads(live_wrapper['stdout'])
    worker = live['records']['run/receipt.json']
    supervisor = live['records']['supervisor.json']
    assert supervisor['pid'] == worker['pid'] == 1779501 and supervisor['supervisor_pid'] == 1779500
    assert supervisor['boot_id'] == live['boot_id'] == cert['recovery']['target_boot_id']
    assert supervisor['start_ticks'] == 18098010 and supervisor['supervisor_start_ticks'] == 18097864
    assert worker['actual_loop_epoch'] == 42 and worker['completed_epochs'] == 42
    assert worker['local_backup_verified']['epoch'] == 41
    rows, refs, accepted = epochs.accepted_stage('NO_MSCA', 40)
    assert len(rows) == ROUNDS and rows[-1]['metrics'] == cert['latest_metrics']
    assert rows[-1]['ema_tensor_sha256'] == cert['endpoints']['best']['ema_tensor_sha256']
    old_stage = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups/epochs/epoch0029')
    old_prefix = (old_stage / 'epochs.jsonl').read_bytes()
    assert (Path(refs[0]['source_path'])).read_bytes().startswith(old_prefix)
    new_sources = []
    for source_id, path in [('NO_MSCA_RECOVERED_R41', CERTIFICATE), ('NO_MSCA_LIVE_GOAL_20261109', LIVE_PATH)]:
        value = read(path)
        new_sources.append({
            'source_id': source_id, 'relative_to_remote_root': str(path.relative_to(epochs.REMOTE)).replace('\\', '/'),
            'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
            'status_in_source': value.get('status', 'SAVED_REMOTE_OBSERVATION'),
        })
    old_completed = summary['completed_endpoints']
    progress = summary['NO_MSCA_progress']
    best = cert['endpoints']['best']
    progress.update(
        completed_rounds=ROUNDS, epoch_index_0based=ROUNDS-1, remaining_rounds=320-ROUNDS,
        optimizer_step_attempts=ROUNDS*1617, latest_metrics_fraction=cert['latest_metrics'],
        latest_metrics_percent={metric: value * 100 for metric, value in cert['latest_metrics'].items()},
        rolling_checkpoint_sha256=accepted['rolling_checkpoint_sha256'],
        latest_best_checkpoint_sha256=best['checkpoint_sha256'], ema_tensor_sha256=best['ema_tensor_sha256'],
        source_ids=['NO_MSCA_RECOVERED_R41'], snapshot_cutoff='independently_verified_round41',
        native_acceptance=accepted,
        update_since_previous_snapshot={'previous_rounds': 30, 'new_verified_rounds': 11, 'new_verified_updates': 17787},
    )
    progress['recovery'].update(
        new_verified_rounds=12, new_verified_optimizer_updates=19404,
        physical_compute_limitation=cert['recovery']['physical_compute_limitation'],
        contract_live_flag=cert['recovery']['live_recovery_training_progress_verified'],
        contract_live_flag_scope='Original 29-round contract subreport performs no remote live check. The top-level round41 saved-state audit verifies 12 recovered rounds and 19404 recovered updates.',
    )
    progress['live_observation'] = {
        'source_id': 'NO_MSCA_LIVE_GOAL_20261109', 'observed_server_utc': live['observed_server_utc'],
        'actual_epoch_index_0based': worker['actual_loop_epoch'], 'round_1based': worker['actual_loop_epoch'] + 1,
        'batch': worker['batch'], 'optimizer_step_attempts': worker['optimizer_step_attempts'],
        'native_accepted_rounds_observed': worker['completed_epochs'], 'independent_metric_snapshot_rounds': ROUNDS,
        'scope': 'Saved running-process observation. Independently audited metric data is cut off at round41.',
    }
    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None
    summary['input_sources'].extend(new_sources)
    summary['updated_local_date'] = LOCAL_DATE
    summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
    summary['limitations'] = [
        'NO_MSCA本快照仅独立核验41轮，320轮终点为空，不参与最终端点差值。'
        if line.startswith('NO_MSCA本快照') else line for line in summary['limitations']]
    assert summary['completed_endpoints'] == old_completed
    replace_json(OUT / '实验汇总.json', summary)
    with (OUT / 'NO_MSCA进度.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, progress_rows = reader.fieldnames, list(reader)
    for row in progress_rows:
        row.update(completed_rounds=ROUNDS, latest_value_fraction=cert['latest_metrics'][row['metric']],
                   latest_value_percent=cert['latest_metrics'][row['metric']] * 100, source_ids='NO_MSCA_RECOVERED_R41')
        assert row['round320_primary_value'] == row['round320_secondary_value'] == ''
    replace_bytes(OUT / 'NO_MSCA进度.csv', csv_bytes(fields, progress_rows))
    replace_bytes(OUT / '输入来源.csv', csv_bytes(list(summary['input_sources'][0]), summary['input_sources']))
    old_csv = (OUT / '逐轮指标.csv').read_bytes()
    with (OUT / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, original_rows = reader.fieldnames, list(reader)
    assert len(original_rows) == 990 and original_rows[-1]['arm'] == 'NO_MSCA'
    assert int(original_rows[-1]['round_1based']) == 30
    source_id = 'NO_MSCA_RECOVERED_R41_EPOCH_ROWS'
    additional = []
    for row, ref in zip(rows[30:41], refs[30:41]):
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
    stream = io.StringIO(newline='')
    csv.DictWriter(stream, fieldnames=fields).writerows(additional)
    replace_bytes(OUT / '逐轮指标.csv', old_csv + stream.getvalue().encode('utf-8'))
    mapping = read(OUT / '逐轮来源与切片.json')
    epoch_source = next(item for item in epochs.SOURCES if item['source_id'] == 'NO_MSCA_ACCEPTED_EPOCH_ROWS')
    epoch_source['source_id'] = source_id
    mapping['sources'].extend([epoch_source] + new_sources)
    mapping['slices'].append({'arm': 'NO_MSCA', 'source_id': source_id,
                              'source_records_0based_half_open': [30, 41], 'rounds_included': [31, 41]})
    mapping['rows_per_arm']['NO_MSCA'] = ROUNDS
    mapping['accepted_stage_sources'].append(accepted)
    mapping['excluded'][-1] = {
        'arm': 'NO_MSCA', 'rule': 'The saved live observation reports 42 native accepted rounds and round43 in progress. Metric export stops at the independently audited round41 certificate.'}
    mapping['updated_local_date'] = LOCAL_DATE
    mapping['recovery_source_id'] = 'NO_MSCA_RECOVERED_R41'
    replace_json(OUT / '逐轮来源与切片.json', mapping)
    for filename in ('输入核查.json', '逐轮核查.json'):
        report = read(OUT / filename)
        report['checks'].extend(epochs.CHECKS)
        report['checks'].append({'check': 'Round41 independent certificate cutoff retained despite later native/live rounds', 'status': 'PASS'})
        report['check_count'] = len(report['checks'])
        report['NO_MSCA_accepted_rounds'] = ROUNDS
        report.setdefault('new_source_sha256', {}).update({item['source_id']: item['sha256'] for item in new_sources})
        if filename == '逐轮核查.json':
            report['rows'] = 1001
            report['csv_sha256'] = sha(OUT / '逐轮指标.csv')
        replace_json(OUT / filename, report)
    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已核验至第 30 轮，进度 AP 为 8.8572%',
        'NO_MSCA 的来源快照已独立核验至第 41 轮，进度 AP 为 10.4971%')
    text += '\n2026-11-09 更新。NO_MSCA 独立核验至第 41 轮，累计 66,297 次更新。第 29 轮换机恢复后已核验 12 轮及 19,404 次更新，本次相对第 30 轮快照新增 11 行。现场记录显示第 43 轮进行中、原生备份已接受 42 轮，本汇总仍以独立证书第 41 轮为数据截点。\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 990 行', '共 1001 行').replace('前 30 行', '前 41 行')
    text += '\n2026-11-09 更新保留原 990 行字节前缀，并从已接受 epoch0040 的累计日志中追加第 31–41 轮。三组已完成数据各 320 行，NO_MSCA 41 行。第 42 轮原生备份与第 43 轮现场进度另列证据，未进入本次指标数据。\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
    replace_json(OUT / '_build/update_round41_plan.json', {
        'status': 'DATA_UPDATED_WORKBOOK_PENDING', 'local_date': LOCAL_DATE,
        'snapshot': str(SNAPSHOT), 'xlsx_relative': str(XLSX), 'old_xlsx_sha256': sha(SNAPSHOT / XLSX),
        'target_rounds': ROUNDS, 'total_updates': ROUNDS * 1617, 'previous_snapshot_rounds': 30,
        'update_tag': 'round41', 'new_NO_MSCA_metrics': cert['latest_metrics'],
        'progress_notes': [
            '第29轮后换机，原始模型、EMA、优化器、warmup及保存RNG恢复已核验。',
            '恢复段累计核验12轮和19404次更新，数据快照共41轮。',
            '独立证书 NO_MSCA_RECOVERED_R41，全部12指标误差为0。',
            '现场第43轮进行，native备份42轮。指标截在独立核验41轮。',
            '失联后未保留的工作量未知，完整物理计算量无法确认。',
            '恢复子合约live=false仅说明该子报告不做远程实时检查。',
        ],
        'new_source_sha256': {item['source_id']: item['sha256'] for item in new_sources},
    })
    print(json.dumps({'status': 'ROUND41_DATA_UPDATED', 'snapshot': str(SNAPSHOT),
                      'NO_MSCA_rounds': ROUNDS, 'per_epoch_rows': 1001}, ensure_ascii=True))


if __name__ == '__main__':
    main()
