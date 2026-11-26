"""Advance the saved NO_MSCA data cutoff to independent round164; no model/evaluator."""

import csv
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath

import assemble_epoch_data as epochs
from update_round30 import OUT, RECOVERY, XLSX, csv_bytes, read, replace_bytes, replace_json, sha

ROUNDS = 165
LOCAL_DATE = '2026-11-09'
SECOND = RECOVERY.parent / 'recovery30715_20261109'
CERTIFICATE = SECOND / 'lineage_second_recovery_through_round165.json'
BINDING = SECOND / 'segment30715_audit_binding.json'
ADDENDUM = SECOND / 'recovery30715_addendum.json'
LIVE_PATH = SECOND / 'resume_monitor_20261109_001/milestone_round165_live.json'
IDENTITY = SECOND / 'resume_monitor_20261109_001/observer_identity.json'
SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round164_verified_20261109'
FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round165_verified_20261109'


def main():
    summary = read(OUT / '实验汇总.json')
    assert summary['NO_MSCA_progress']['completed_rounds'] == 164
    previous_manifest = read(OUT / '交付清单.json')
    assert len(previous_manifest['files']) == 19
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
    assert cert['learning_rates']['recorded_attempts'] == 266805 == ROUNDS * 1617
    assert cert['learning_rates']['maximum_absolute_difference'] == cert['maximum_latest_metric_error'] == 0
    assert cert['validation_images'] == 548 and cert['predictions'] == 164400
    assert cert['model_forwards'] == cert['evaluator_runs'] == 0
    recovery = cert['recovery']
    assert recovery['new_completed_rounds'] == 1 and recovery['new_optimizer_update_attempts'] == 1617
    assert recovery['retained_plus_new_rounds'] == ROUNDS and recovery['prefix']['retained_rounds'] == 164
    assert recovery['prefix']['full_saved_state_including_all_rng_equal'] is True
    assert recovery['archive']['archived_files'] == 16 and recovery['archive']['zip_bytes_hash_checked'] is True
    assert recovery['original29_recovery_addendum_sha256'] == summary['NO_MSCA_progress']['recovery']['recovery_addendum_sha256']
    assert recovery['binding_sha256'] == sha(BINDING) == '6de82b5b4bcbf8bbdbd3a61a2fb7ecbd70905b98da0eceefc55c7b451435a41e'
    assert recovery['recovery_addendum_sha256'] == sha(ADDENDUM) == '5b6fec3b7f4fe2f088cd0176b085fa6074731082307e6215d2ff46ce918a71c9'
    binding = read(BINDING)
    for name, item in binding['artifacts'].items():
        path = SECOND / item['relative_path']
        assert path.resolve().is_relative_to(SECOND.resolve()) and sha(path) == item['sha256'], name
    live_wrapper = read(LIVE_PATH)
    assert live_wrapper['exit_code'] == 0
    live = json.loads(live_wrapper['stdout'])
    worker, supervisor = live['records']['run/receipt.json'], live['records']['supervisor.json']
    assert supervisor['pid'] == worker['pid'] == 413 and supervisor['supervisor_pid'] == 412
    assert supervisor['start_ticks'] == 26597622 and supervisor['supervisor_start_ticks'] == 26597399
    assert supervisor['status'] == worker['status'] == 'RUNNING'
    assert supervisor['boot_id'] == live['boot_id'] == binding['target']['boot_id']
    assert live['gpu'].split(', ')[0] == binding['target']['gpu_uuid']
    assert worker['actual_loop_epoch'] == worker['completed_epochs'] == 165
    assert worker['optimizer_step_attempts'] == 266806 and worker['batch'] == 1
    assert worker['local_backup_verified']['epoch'] == 164
    assert worker['recovery_addendum_sha256'] == sha(ADDENDUM)
    assert worker['runner_sha256'] == read(ADDENDUM)['resume_runner_sha256']
    assert read(IDENTITY)['backup_client_pid'] == 8216
    rows, refs, accepted = epochs.accepted_stage('NO_MSCA', 164)
    assert len(rows) == ROUNDS and rows[-1]['metrics'] == cert['latest_metrics']
    assert rows[-1]['optimizer_step_attempts'] == 266805
    assert rows[-1]['ema_tensor_sha256'] == cert['endpoints']['best']['ema_tensor_sha256']
    assert accepted['rolling_checkpoint_sha256'] == worker['rolling_checkpoint_sha256']
    assert accepted['manifest_sha256'] == worker['local_backup_verified']['manifest_sha256']
    assert accepted['ack_sha256'] == worker['local_backup_verified']['ack_sha256']
    old_stage = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups/epochs/epoch0163')
    assert Path(refs[0]['source_path']).read_bytes().startswith((old_stage / 'epochs.jsonl').read_bytes())
    new_sources = []
    for source_id, source in [('NO_MSCA_SECOND_RECOVERY_R165', CERTIFICATE), ('NO_MSCA_SECOND_RECOVERY_BINDING', BINDING), ('NO_MSCA_SECOND_RECOVERY_ADDENDUM', ADDENDUM), ('NO_MSCA_SECOND_RECOVERY_HISTORY_MANIFEST', SECOND / 'prior_segment_archive_manifest.json'), ('NO_MSCA_SECOND_RECOVERY_HISTORY_ZIP', SECOND / 'prior_segment_archive.zip'), ('NO_MSCA_SECOND_RECOVERY_LIVE_R166', LIVE_PATH), ('NO_MSCA_SECOND_RECOVERY_OBSERVER_IDENTITY', IDENTITY)]:
        value = read(source) if source.suffix == '.json' else {}
        new_sources.append({
            'source_id': source_id, 'relative_to_remote_root': str(source.relative_to(epochs.REMOTE)).replace('\\', '/'),
            'path': str(source), 'bytes': source.stat().st_size, 'sha256': sha(source),
            'status_in_source': value.get('status', 'HASH_BOUND_PRIOR_SEGMENT_ARCHIVE' if source.suffix == '.zip' else 'SAVED_REMOTE_OBSERVATION'),
        })
    progress = summary['NO_MSCA_progress']
    best = cert['endpoints']['best']
    progress.update(
        completed_rounds=ROUNDS, epoch_index_0based=164, remaining_rounds=155,
        optimizer_step_attempts=266805, latest_metrics_fraction=cert['latest_metrics'],
        latest_metrics_percent={metric: value * 100 for metric, value in cert['latest_metrics'].items()},
        rolling_checkpoint_sha256=accepted['rolling_checkpoint_sha256'],
        latest_best_checkpoint_sha256=best['checkpoint_sha256'], ema_tensor_sha256=rows[-1]['ema_tensor_sha256'],
        source_ids=['NO_MSCA_SECOND_RECOVERY_R165'], snapshot_cutoff='independently_verified_round165',
        native_acceptance=accepted,
        update_since_previous_snapshot={'previous_rounds': 164, 'new_verified_rounds': 1, 'new_verified_updates': 1617},
    )
    previous_recovery = copy.deepcopy(progress['recovery'])
    previous_recovery['scope'] = 'Historical first migration from29 through accepted164; its live=false flag applies only to the original29 contract subreport.'
    progress.setdefault('recovery_history', []).append(previous_recovery)
    progress['recovery'] = copy.deepcopy(recovery)
    progress['recovery'].update(new_verified_rounds=1, new_verified_optimizer_updates=1617,
                                cumulative_recovered_rounds_since29=136, cumulative_recovered_optimizer_updates_since29=219912)
    progress.setdefault('observation_history', []).append(copy.deepcopy(progress['live_observation']))
    progress.setdefault('connectivity_history', []).append(copy.deepcopy(progress['connectivity']))
    progress['live_observation'] = {
        'source_id': 'NO_MSCA_SECOND_RECOVERY_LIVE_R166', 'observed_server_utc': live['observed_server_utc'],
        'actual_epoch_index_0based': 165, 'round_1based': 166, 'batch': worker['batch'],
        'optimizer_step_attempts': worker['optimizer_step_attempts'], 'native_accepted_rounds_observed': 165,
        'independent_metric_snapshot_rounds': ROUNDS, 'observation_kind': 'HASH_BOUND_SAVED_RUNNING_OBSERVATION',
        'current_remote_training_state': 'RUNNING', 'worker_pid': 413, 'supervisor_pid': 412,
        'scope': 'Saved process observation on authorized30715. Round166 was training at this instant; exported metrics stop at independently audited165.',
    }
    progress['current_remote_training_state'] = 'RUNNING'
    progress['connectivity'] = {
        'source_id': 'NO_MSCA_SECOND_RECOVERY_LIVE_R166', 'status': 'AUTHORIZED_PORT30715_CONTINUATION_RUNNING_AT_SAVED_OBSERVATION',
        'remote_training_state_now': 'RUNNING', 'remote_ssh_port': 30715, 'observed_server_utc': live['observed_server_utc'],
        'last_native_accepted_round': 165, 'independent_metric_snapshot_rounds': 165,
        'native_backup_client_pid': 8216, 'native_backup_client_verified_at_acceptance': True,
        'client_identity_source_id': 'NO_MSCA_SECOND_RECOVERY_OBSERVER_IDENTITY',
        'scope': 'Timestamped evidence, not an additional live connection by the data exporter.',
    }
    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None
    summary['input_sources'].extend(new_sources)
    summary['updated_local_date'] = LOCAL_DATE
    summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
    summary['limitations'] = [
        'NO_MSCA本快照仅独立核验165轮，320轮终点为空，不参与最终端点差值。'
        if line.startswith('NO_MSCA本快照') else line for line in summary['limitations']]
    summary['limitations'] = [
        '历史164轮快照时远端UNKNOWN且旧备份客户端已退出；该失联历史保留。现由已绑定30715运行快照支持续训状态，指标仅截至独立165轮。'
        if line.startswith('当前远端训练状态UNKNOWN') else line for line in summary['limitations']]
    summary['limitations'].append(recovery['physical_compute_limitation'])
    replace_json(OUT / '实验汇总.json', summary)
    with (OUT / 'NO_MSCA进度.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, progress_rows = reader.fieldnames, list(reader)
    fields = list(fields) + ['latest_observed_running_round', 'second_recovery_first_epoch_index']
    for row in progress_rows:
        row.update(completed_rounds=ROUNDS, latest_value_fraction=cert['latest_metrics'][row['metric']],
                   latest_value_percent=cert['latest_metrics'][row['metric']] * 100, source_ids='NO_MSCA_SECOND_RECOVERY_R165')
        row.update(remote_training_state_now='RUNNING', historical_last_observed_running_round=165,
                   native_backup_client_alive='VERIFIED_AT_ROUND165_ACCEPTANCE', latest_observed_running_round=166,
                   second_recovery_first_epoch_index=164,
                   observation_scope='Authorized30715 saved running observation; historical old165 partial work preserved; metrics cut off at independent165')
        assert row['round320_primary_value'] == row['round320_secondary_value'] == ''
    replace_bytes(OUT / 'NO_MSCA进度.csv', csv_bytes(fields, progress_rows))
    replace_bytes(OUT / '输入来源.csv', csv_bytes(list(summary['input_sources'][0]), summary['input_sources']))
    old_csv = (OUT / '逐轮指标.csv').read_bytes()
    with (OUT / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, original_rows = reader.fieldnames, list(reader)
    assert len(original_rows) == 1124 and original_rows[-1]['arm'] == 'NO_MSCA'
    assert int(original_rows[-1]['round_1based']) == 164
    source_id = 'NO_MSCA_SECOND_RECOVERY_R165_EPOCH_ROWS'
    additional = []
    for row, ref in zip(rows[164:165], refs[164:165]):
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
    assert len(additional) == 1
    stream = io.StringIO(newline='')
    csv.DictWriter(stream, fieldnames=fields).writerows(additional)
    replace_bytes(OUT / '逐轮指标.csv', old_csv + stream.getvalue().encode('utf-8'))
    mapping = read(OUT / '逐轮来源与切片.json')
    epoch_source = next(item for item in epochs.SOURCES if item['source_id'] == 'NO_MSCA_ACCEPTED_EPOCH_ROWS')
    epoch_source['source_id'] = source_id
    mapping['sources'].extend([epoch_source] + new_sources)
    mapping['slices'].append({'arm': 'NO_MSCA', 'source_id': source_id,
                              'source_records_0based_half_open': [164, 165], 'rounds_included': [165, 165]})
    mapping['rows_per_arm']['NO_MSCA'] = ROUNDS
    mapping['accepted_stage_sources'].append(accepted)
    mapping['excluded'][-1] = {'arm': 'NO_MSCA', 'rule':
        'Authorized30715 continuation is RUNNING in the saved observation at round166. Only accepted and independently audited rounds1-165 are exported; round166 and later are excluded. Old partial165 before instance expiry remains discarded historical work.'}
    mapping['updated_local_date'] = LOCAL_DATE
    mapping['recovery_source_id'] = 'NO_MSCA_SECOND_RECOVERY_R165'
    replace_json(OUT / '逐轮来源与切片.json', mapping)
    for filename in ('输入核查.json', '逐轮核查.json'):
        report = read(OUT / filename)
        report['checks'].extend(epochs.CHECKS)
        report['checks'].append({'check': 'Second recovery165 independent certificate retained; observed166 excluded; original1124 CSV bytes retained', 'status': 'PASS'})
        report['check_count'] = len(report['checks'])
        report['NO_MSCA_accepted_rounds'] = ROUNDS
        report.setdefault('new_source_sha256', {}).update({item['source_id']: item['sha256'] for item in new_sources})
        if filename == '逐轮核查.json':
            report['rows'] = 1125
            report['csv_sha256'] = sha(OUT / '逐轮指标.csv')
        replace_json(OUT / filename, report)
    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已独立核验至第 164 轮，进度 AP 为 17.2070%',
        'NO_MSCA 的来源快照已独立核验至第 165 轮，进度 AP 为 17.3667%')
    text += '\n2026-11-09 165轮更新。30715 已按原 seed0/batch4/FP32 配方从已接受164轮完整状态继续，原始模型、EMA、优化器、criterion、postprocessor、warmup 及全部保存RNG一致，完整科学循环AST未改。第二段恢复新增1轮及1,617次更新已独立核验，累计165轮/266,805次；最近已存现场见第166轮训练，指标严格止于165。原29→164来源及本次16文件历史ZIP保留，旧实例到期前未接受的第165轮部分工作与新接受165行严格区分；旧历史行不重写。已知旧partial165曾见1,401次额外更新，之后未保留工作量仍未知，早先29轮失联后的物理工作量也未知，不能声称完整物理计算相同或无中断位级等价。第164轮UNKNOWN仅为历史快照。NO_MSCA的320轮最终端点继续空缺。\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 1124 行', '共 1125 行').replace('前 164 行', '前 165 行')
    text = text.replace('第 161–164 行来自已接受 epoch0163 备份，并与恢复轨迹第 164 轮独立证书核对。第 165 轮及其后的指标不纳入本快照。',
                        '第 161–164 行来自已接受 epoch0163 备份；第 165 行来自第二段恢复后接受的 epoch0164 累计日志，并与独立165轮证书核对。第 166 轮及其后的指标不纳入本快照。')
    text += '\n2026-11-09 165轮更新仅在原 1124 行原字节后追加新接受且已独立核验的第165轮，旧行状态及来源仍代表各自历史快照，未因30715恢复而改写。三组已完成配置各320行，NO_MSCA165行，共1125行。新行来自第二次恢复后的epoch0164备份，不能与到期旧实例未接受的partial165混同。第166轮仅为已保存的运行观察。\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
    replace_json(OUT / '_build/update_round165_plan.json', {
        'status': 'DATA_UPDATED_WORKBOOK_PENDING', 'local_date': LOCAL_DATE,
        'snapshot': str(SNAPSHOT), 'xlsx_relative': str(XLSX), 'old_xlsx_sha256': sha(SNAPSHOT / XLSX),
        'target_rounds': ROUNDS, 'total_updates': 266805, 'previous_snapshot_rounds': 164,
        'update_tag': 'round165', 'new_NO_MSCA_metrics': cert['latest_metrics'],
        'progress_notes': [
            '30715续训在运行；已独立核验165轮，最近已存现场见第166轮。',
            '从164轮完整恢复原始模型、EMA、优化器、warmup及全部保存RNG。',
            '第二段新增1轮/1617次，累计165轮/266805次；旧29→164来源保留。',
            '旧partial165及16文件历史ZIP保留；其后未保留物理工作量未知。',
            '独立证书 NO_MSCA_SECOND_RECOVERY_R165，全部12指标误差0。',
            '单seed、验证集选点及非官方VisDrone限制保留；320轮终点空缺。',
        ],
        'new_source_sha256': {item['source_id']: item['sha256'] for item in new_sources},
    })
    print(json.dumps({'status': 'ROUND165_DATA_UPDATED', 'prior_snapshot': str(SNAPSHOT),
                      'NO_MSCA_rounds': ROUNDS, 'per_epoch_rows': 1125}, ensure_ascii=True))


if __name__ == '__main__':
    main()
