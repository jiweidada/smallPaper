"""Advance the saved NO_MSCA data cutoff to independent round280; no model/evaluator."""

import csv
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath

import assemble_epoch_data as epochs
from update_round30 import OUT, RECOVERY, XLSX, csv_bytes, read, replace_bytes, replace_json, sha

ROUNDS = 280
LOCAL_DATE = '2026-11-10'
SECOND = RECOVERY.parent / 'recovery30715_20261109'
CERTIFICATE = SECOND / 'lineage_second_recovery_through_round280.json'
BINDING = SECOND / 'segment30715_audit_binding.json'
ADDENDUM = SECOND / 'recovery30715_addendum.json'
LIVE_PATH = SECOND / 'resume_monitor_20261110_004/milestone_round280_live.json'
SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round240_verified_20261110'
FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round280_verified_20261110'


def main():
    summary = read(OUT / '实验汇总.json')
    assert summary['NO_MSCA_progress']['completed_rounds'] == 240
    previous_manifest = read(OUT / '交付清单.json')
    assert len(previous_manifest['files']) == 30
    assert sha(OUT / '交付清单.json') == sha(SNAPSHOT / '交付清单.json')
    for item in previous_manifest['files']:
        relative = PurePosixPath(item['file'].replace('\\', '/'))
        assert not relative.is_absolute() and '..' not in relative.parts
        for root in (OUT, SNAPSHOT):
            source = root.joinpath(*relative.parts)
            assert source.resolve().is_relative_to(root.resolve())
            assert source.stat().st_size == item['bytes'] and sha(source) == item['sha256']

    assert sha(CERTIFICATE) == 'c7b857af507b95f48dc6c7c789caf8aaef83ed71d6fe6050cddb685c579ccfa3'
    cert = read(CERTIFICATE)
    assert cert['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and cert['completed_rounds'] == ROUNDS
    assert cert['learning_rates']['recorded_attempts'] == 452760 == ROUNDS * 1617
    assert cert['learning_rates']['maximum_absolute_difference'] == cert['maximum_latest_metric_error'] == 0
    assert cert['learning_rates']['groups'] == 4 and cert['validation_images'] == 548 and cert['predictions'] == 164400
    assert cert['model_forwards'] == cert['evaluator_runs'] == 0
    recovery = cert['recovery']
    assert recovery['new_completed_rounds'] == 116 and recovery['new_optimizer_update_attempts'] == 187572
    assert recovery['retained_plus_new_rounds'] == ROUNDS and recovery['prefix']['retained_rounds'] == 164
    assert recovery['prefix']['full_saved_state_including_all_rng_equal'] is True
    assert recovery['archive']['archived_files'] == 16 and recovery['archive']['zip_bytes_hash_checked'] is True
    assert recovery['original29_recovery_addendum_sha256'] == summary['NO_MSCA_progress']['recovery']['original29_recovery_addendum_sha256']
    assert recovery['binding_sha256'] == sha(BINDING) == '6de82b5b4bcbf8bbdbd3a61a2fb7ecbd70905b98da0eceefc55c7b451435a41e'
    assert recovery['recovery_addendum_sha256'] == sha(ADDENDUM) == '5b6fec3b7f4fe2f088cd0176b085fa6074731082307e6215d2ff46ce918a71c9'
    binding = read(BINDING)
    for name, item in binding['artifacts'].items():
        path = SECOND / item['relative_path']
        assert path.resolve().is_relative_to(SECOND.resolve()) and sha(path) == item['sha256'], name
    live_wrapper = read(LIVE_PATH)
    assert live_wrapper['exit_code'] == 0
    assert sha(LIVE_PATH) == '5b8c48a536d63f25d7b2e802f11da9c5d34a95ff95e351c765d7f9eb7fab64b8'
    live = json.loads(live_wrapper['stdout'])
    worker, supervisor = live['records']['run/receipt.json'], live['records']['supervisor.json']
    assert supervisor['pid'] == worker['pid'] == 413 and supervisor['supervisor_pid'] == 412
    assert supervisor['start_ticks'] == 26597622 and supervisor['supervisor_start_ticks'] == 26597399
    assert supervisor['status'] == worker['status'] == 'RUNNING'
    assert supervisor['boot_id'] == live['boot_id'] == binding['target']['boot_id']
    assert live['gpu'].split(', ')[0] == binding['target']['gpu_uuid']
    assert worker['actual_loop_epoch'] == worker['completed_epochs'] == 280
    assert worker['optimizer_step_attempts'] == 452761 and worker['batch'] == 1
    assert worker['local_backup_verified']['epoch'] == 279
    assert worker['recovery_addendum_sha256'] == sha(ADDENDUM)
    assert worker['runner_sha256'] == read(ADDENDUM)['resume_runner_sha256']
    rows, refs, accepted = epochs.accepted_stage('NO_MSCA', 279)
    assert len(rows) == ROUNDS and rows[-1]['metrics'] == cert['latest_metrics']
    assert rows[-1]['optimizer_step_attempts'] == 452760
    best = cert['endpoints']['best']
    assert best['epoch_index'] == 279 and rows[279]['metrics'] == best['metrics']
    assert rows[279]['ema_tensor_sha256'] == best['ema_tensor_sha256']
    assert rows[279]['prediction_sha256'] == best['prediction_sha256']
    assert rows[279]['evaluation_arrays_sha256'] == best['evaluation_arrays_sha256']
    assert max(range(ROUNDS), key=lambda index: rows[index]['metrics']['AP']) == 279
    assert rows[-1]['metrics']['AP'] == 0.21928245309369632
    assert rows[-1]['metrics']['APs'] == 0.14481580486643042
    assert best['metrics']['AP'] == 0.21928245309369632
    assert rows[-1]['ema_tensor_sha256'] == best['ema_tensor_sha256']
    assert accepted['rolling_checkpoint_sha256'] == worker['rolling_checkpoint_sha256']
    assert accepted['manifest_sha256'] == worker['local_backup_verified']['manifest_sha256']
    assert accepted['ack_sha256'] == worker['local_backup_verified']['ack_sha256']
    old_stage = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups/epochs/epoch0239')
    assert Path(refs[0]['source_path']).read_bytes().startswith((old_stage / 'epochs.jsonl').read_bytes())
    new_sources = []
    for source_id, source in [('NO_MSCA_SECOND_RECOVERY_R280', CERTIFICATE), ('NO_MSCA_SECOND_RECOVERY_LIVE_R281', LIVE_PATH)]:
        value = read(source) if source.suffix == '.json' else {}
        new_sources.append({
            'source_id': source_id, 'relative_to_remote_root': str(source.relative_to(epochs.REMOTE)).replace('\\', '/'),
            'path': str(source), 'bytes': source.stat().st_size, 'sha256': sha(source),
            'status_in_source': value.get('status', 'HASH_BOUND_PRIOR_SEGMENT_ARCHIVE' if source.suffix == '.zip' else 'SAVED_REMOTE_OBSERVATION'),
        })
    progress = summary['NO_MSCA_progress']
    best = cert['endpoints']['best']
    progress.update(
        completed_rounds=ROUNDS, epoch_index_0based=279, remaining_rounds=40,
        optimizer_step_attempts=452760, latest_metrics_fraction=cert['latest_metrics'],
        latest_metrics_percent={metric: value * 100 for metric, value in cert['latest_metrics'].items()},
        rolling_checkpoint_sha256=accepted['rolling_checkpoint_sha256'],
        latest_best_checkpoint_sha256=best['checkpoint_sha256'], ema_tensor_sha256=rows[-1]['ema_tensor_sha256'],
        source_ids=['NO_MSCA_SECOND_RECOVERY_R280'], snapshot_cutoff='independently_verified_round280',
        native_acceptance=accepted,
        update_since_previous_snapshot={'previous_rounds': 240, 'new_verified_rounds': 40, 'new_verified_updates': 64680},
    )
    progress['best_so_far_progress'] = {
        'role': 'PROGRESS_AP_BEST_THROUGH_ROUND280',
        'round_1based': 280, 'epoch_index_0based': 279, 'cutoff_rounds': 280,
        'same_epoch_as_latest_progress': True,
        'metrics_fraction': best['metrics'], 'metrics_percent': {key: value * 100 for key, value in best['metrics'].items()},
        'checkpoint_sha256': best['checkpoint_sha256'], 'ema_tensor_sha256': best['ema_tensor_sha256'],
        'model_tensor_sha256': best['model_tensor_sha256'], 'prediction_sha256': best['prediction_sha256'],
        'evaluation_arrays_sha256': best['evaluation_arrays_sha256'], 'metrics_sha256': best['metrics_sha256'],
        'source_id': 'NO_MSCA_SECOND_RECOVERY_R280',
    }
    progress['latest_prediction_sha256'] = rows[-1]['prediction_sha256']
    progress['latest_evaluation_arrays_sha256'] = rows[-1]['evaluation_arrays_sha256']
    progress['recovery'] = copy.deepcopy(recovery)
    progress['recovery'].update(new_verified_rounds=116, new_verified_optimizer_updates=187572,
                                cumulative_recovered_rounds_since29=251, cumulative_recovered_optimizer_updates_since29=405867)
    progress.setdefault('observation_history', []).append(copy.deepcopy(progress['live_observation']))
    progress.setdefault('connectivity_history', []).append(copy.deepcopy(progress['connectivity']))
    progress['live_observation'] = {
        'source_id': 'NO_MSCA_SECOND_RECOVERY_LIVE_R281', 'observed_server_utc': live['observed_server_utc'],
        'actual_epoch_index_0based': 280, 'round_1based': 281, 'batch': worker['batch'],
        'optimizer_step_attempts': worker['optimizer_step_attempts'], 'native_accepted_rounds_observed': 280,
        'independent_metric_snapshot_rounds': ROUNDS, 'observation_kind': 'HASH_BOUND_IMMUTABLE_MILESTONE_OBSERVATION',
        'current_remote_training_state': 'RUNNING', 'worker_pid': 413, 'supervisor_pid': 412,
        'scope': 'The completed 004 window records round 281 at the milestone. The independent metrics cover the first 280 rounds.',
    }
    progress['current_remote_training_state'] = 'RUNNING'
    progress['connectivity'] = {
        'source_id': 'NO_MSCA_SECOND_RECOVERY_LIVE_R281', 'status': 'AUTHORIZED_PORT30715_CONTINUATION_RUNNING_AT_SAVED_OBSERVATION',
        'remote_training_state_now': 'RUNNING', 'remote_ssh_port': 30715, 'observed_server_utc': live['observed_server_utc'],
        'last_native_accepted_round': 280, 'independent_metric_snapshot_rounds': 280,
        'native_backup_client_pid': 8216, 'native_backup_client_verified_at_acceptance': True,
        'milestone_observer_window': '004', 'milestone_observer_target_round': 280,
        'milestone_observer_window_status': 'COMPLETED_AT280',
        'scope': 'Timestamped immutable evidence and fixed observer identity only; the data exporter makes no remote connection or ongoing liveness claim.',
    }
    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None
    summary['input_sources'].extend(new_sources)
    summary['updated_local_date'] = LOCAL_DATE
    summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
    summary['limitations'] = [
        'NO_MSCA本快照仅独立核验280轮，320轮终点为空，不参与最终端点差值。'
        if line.startswith('NO_MSCA本快照') else line for line in summary['limitations']]
    summary['limitations'] = [
        '历史164轮快照时远端UNKNOWN且旧备份客户端已退出；该失联历史保留。现由已绑定30715运行快照支持续训状态，本次指标仅截至独立280轮。'
        if line.startswith('当前远端训练状态UNKNOWN') else line for line in summary['limitations']]
    assert recovery['physical_compute_limitation'] in summary['limitations']
    replace_json(OUT / '实验汇总.json', summary)
    with (OUT / 'NO_MSCA进度.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, progress_rows = reader.fieldnames, list(reader)
    fields = list(fields)
    for row in progress_rows:
        row.update(completed_rounds=ROUNDS, latest_value_fraction=cert['latest_metrics'][row['metric']],
                   latest_value_percent=cert['latest_metrics'][row['metric']] * 100, source_ids='NO_MSCA_SECOND_RECOVERY_R280')
        row.update(remote_training_state_now='RUNNING', historical_last_observed_running_round=165,
                   native_backup_client_alive='VERIFIED_AT_ROUND280_ACCEPTANCE', latest_observed_running_round=281,
                   second_recovery_first_epoch_index=164,
                   observation_scope='Authorized30715 saved running observation; historical old165 partial work preserved; metrics cut off at independent280')
        assert row['round320_primary_value'] == row['round320_secondary_value'] == ''
    replace_bytes(OUT / 'NO_MSCA进度.csv', csv_bytes(fields, progress_rows))
    replace_bytes(OUT / '输入来源.csv', csv_bytes(list(summary['input_sources'][0]), summary['input_sources']))
    old_csv = (OUT / '逐轮指标.csv').read_bytes()
    with (OUT / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, original_rows = reader.fieldnames, list(reader)
    assert len(original_rows) == 1200 and original_rows[-1]['arm'] == 'NO_MSCA'
    assert int(original_rows[-1]['round_1based']) == 240
    source_id = 'NO_MSCA_SECOND_RECOVERY_R280_EPOCH_ROWS'
    additional = []
    for row, ref in zip(rows[240:280], refs[240:280]):
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
    assert len(additional) == 40
    stream = io.StringIO(newline='')
    csv.DictWriter(stream, fieldnames=fields).writerows(additional)
    replace_bytes(OUT / '逐轮指标.csv', old_csv + stream.getvalue().encode('utf-8'))
    mapping = read(OUT / '逐轮来源与切片.json')
    epoch_source = next(item for item in epochs.SOURCES if item['source_id'] == 'NO_MSCA_ACCEPTED_EPOCH_ROWS')
    epoch_source['source_id'] = source_id
    mapping['sources'].extend([epoch_source] + new_sources)
    mapping['slices'].append({'arm': 'NO_MSCA', 'source_id': source_id,
                              'source_records_0based_half_open': [240, 280], 'rounds_included': [241, 280]})
    mapping['rows_per_arm']['NO_MSCA'] = ROUNDS
    mapping['accepted_stage_sources'].append(accepted)
    mapping['excluded'][-1] = {'arm': 'NO_MSCA', 'rule':
        'The immutable 004 milestone records round 281. This export contains independently audited rounds 1–280. Round 280 supplies the latest progress and the AP best through that cutoff.'}
    mapping['updated_local_date'] = LOCAL_DATE
    mapping['recovery_source_id'] = 'NO_MSCA_SECOND_RECOVERY_R280'
    replace_json(OUT / '逐轮来源与切片.json', mapping)
    for filename in ('输入核查.json', '逐轮核查.json'):
        report = read(OUT / filename)
        report['checks'].extend(epochs.CHECKS)
        report['checks'].append({'check': 'Round280 certificate and fixed 004 milestone verified. The original 1200 rows retain their exact CSV bytes. Both progress roles refer to round 280.', 'status': 'PASS'})
        report['check_count'] = len(report['checks'])
        report['NO_MSCA_accepted_rounds'] = ROUNDS
        report.setdefault('new_source_sha256', {}).update({item['source_id']: item['sha256'] for item in new_sources})
        if filename == '逐轮核查.json':
            report['rows'] = 1240
            report['csv_sha256'] = sha(OUT / '逐轮指标.csv')
        replace_json(OUT / filename, report)
    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已独立核验至第 240 轮，进度 AP 为 21.0329%',
        'NO_MSCA 的来源快照已独立核验至第 280 轮，进度 AP 为 21.9282%')
    text += '\n2026-11-10 280轮更新。NO_MSCA 独立证据覆盖第1–280轮，累计452,760次更新。从已接受的epoch0279累计日志追加第241–280轮，共40行。第280轮 AP 为21.9282%，APs 为14.4816%。以第280轮为截止点的 AP 最佳也出现在该轮。最新进度与截至该轮的 AP 最佳各自保留角色，均对应这一评估记录。320轮主端点和次端点待核验。\n'
    text += '\n固定现场来源为已结束的004窗口里程碑。该文件于服务器UTC '+live['observed_server_utc']+' 记录同一30715进程在第281轮第1批。第164轮恢复后已核验116轮和187,572次更新。第29和164轮后的两次恢复及旧第165轮至少1,401次弃用更新均有原证据。额外物理工作量仍未知，无中断位级等同缺少证据。沿用单种子、同一验证集选点及非官方VisDrone口径。\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 1200 行', '共 1240 行').replace('前 240 行', '前 280 行')
    text = text.replace('第201–240行来自epoch0239累计日志，并与独立240轮证书核对。第241轮及之后指标不纳入本快照。',
                        '第201–240行来自epoch0239累计日志，第241–280行来自epoch0279累计日志。独立证书覆盖至第280轮。')
    text += '\n2026-11-10 280轮更新追加第241–280轮共40行，保留原1200行精确字节前缀和原字段格式。BASE、FULL、NO_MPS各320行，NO_MSCA 280行，合计1240行。14,880项原始指标及对应行、计数和哈希可逐项回溯。第280轮承担最新进度和截至该轮的AP最佳两个角色。\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
    replace_json(OUT / '_build/update_round280_plan.json', {
        'status': 'DATA_UPDATED_WORKBOOK_PENDING', 'local_date': LOCAL_DATE,
        'snapshot': str(SNAPSHOT), 'xlsx_relative': str(XLSX), 'old_xlsx_sha256': sha(SNAPSHOT / XLSX),
        'target_rounds': ROUNDS, 'total_updates': 452760, 'previous_snapshot_rounds': 240,
        'update_tag': 'round280', 'new_NO_MSCA_metrics': cert['latest_metrics'],
        'progress_notes': [
            '固定004现场记录第281轮第1批，独立指标覆盖至第280轮。',
            '第280轮 AP 21.9282%，APs 14.4816%。',
            '截至第280轮，AP最佳也出现在该轮。320轮端点待核验。',
            '从第164轮恢复后已核验116轮和187,572次更新。',
            '第29和164轮后两次恢复。旧第165轮至少有1,401次弃用更新。',
            '完整物理工作量未知。单种子、同一验证集选点及非官方 VisDrone 口径。',
        ],
        'new_source_sha256': {item['source_id']: item['sha256'] for item in new_sources},
    })
    print(json.dumps({'status': 'ROUND280_DATA_UPDATED', 'prior_snapshot': str(SNAPSHOT),
                      'NO_MSCA_rounds': ROUNDS, 'per_epoch_rows': 1240}, ensure_ascii=True))


if __name__ == '__main__':
    main()
