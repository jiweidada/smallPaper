"""Advance the saved NO_MSCA data cutoff to independent round240; no model/evaluator."""

import csv
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath

import assemble_epoch_data as epochs
from update_round30 import OUT, RECOVERY, XLSX, csv_bytes, read, replace_bytes, replace_json, sha

ROUNDS = 240
LOCAL_DATE = '2026-11-10'
SECOND = RECOVERY.parent / 'recovery30715_20261109'
CERTIFICATE = SECOND / 'lineage_second_recovery_through_round240.json'
BINDING = SECOND / 'segment30715_audit_binding.json'
ADDENDUM = SECOND / 'recovery30715_addendum.json'
LIVE_PATH = SECOND / 'resume_monitor_20261110_003/milestone_round240_live.json'
IDENTITY = SECOND / 'resume_monitor_20261110_003/observer_identity.json'
SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round200_verified_20261110'
FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round240_verified_20261110'


def main():
    summary = read(OUT / '实验汇总.json')
    assert summary['NO_MSCA_progress']['completed_rounds'] == 200
    previous_manifest = read(OUT / '交付清单.json')
    assert len(previous_manifest['files']) == 25
    assert sha(OUT / '交付清单.json') == sha(SNAPSHOT / '交付清单.json')
    for item in previous_manifest['files']:
        relative = PurePosixPath(item['file'].replace('\\', '/'))
        assert not relative.is_absolute() and '..' not in relative.parts
        for root in (OUT, SNAPSHOT):
            source = root.joinpath(*relative.parts)
            assert source.resolve().is_relative_to(root.resolve())
            assert source.stat().st_size == item['bytes'] and sha(source) == item['sha256']

    assert sha(CERTIFICATE) == 'b210f4fcaf5acb09fffbc8af8cb6f484c73b2a854136352227a087c04e132545'
    cert = read(CERTIFICATE)
    assert cert['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and cert['completed_rounds'] == ROUNDS
    assert cert['learning_rates']['recorded_attempts'] == 388080 == ROUNDS * 1617
    assert cert['learning_rates']['maximum_absolute_difference'] == cert['maximum_latest_metric_error'] == 0
    assert cert['learning_rates']['groups'] == 4 and cert['validation_images'] == 548 and cert['predictions'] == 164400
    assert cert['model_forwards'] == cert['evaluator_runs'] == 0
    recovery = cert['recovery']
    assert recovery['new_completed_rounds'] == 76 and recovery['new_optimizer_update_attempts'] == 122892
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
    assert sha(LIVE_PATH) == '809ab33296ee7e61be57f3f4a6967de1c7a1eeedd89ce1550ac030699a5baa1f'
    live = json.loads(live_wrapper['stdout'])
    worker, supervisor = live['records']['run/receipt.json'], live['records']['supervisor.json']
    assert supervisor['pid'] == worker['pid'] == 413 and supervisor['supervisor_pid'] == 412
    assert supervisor['start_ticks'] == 26597622 and supervisor['supervisor_start_ticks'] == 26597399
    assert supervisor['status'] == worker['status'] == 'RUNNING'
    assert supervisor['boot_id'] == live['boot_id'] == binding['target']['boot_id']
    assert live['gpu'].split(', ')[0] == binding['target']['gpu_uuid']
    assert worker['actual_loop_epoch'] == worker['completed_epochs'] == 240
    assert worker['optimizer_step_attempts'] == 388081 and worker['batch'] == 1
    assert worker['local_backup_verified']['epoch'] == 239
    assert worker['recovery_addendum_sha256'] == sha(ADDENDUM)
    assert worker['runner_sha256'] == read(ADDENDUM)['resume_runner_sha256']
    observer = read(IDENTITY)
    assert observer['backup_client_pid'] == 8216 and observer['observer_pid'] == 40816
    assert observer['observer_create_time'] == 1794312084.181138 and observer['target_round'] == 240
    assert observer['observer_source_sha256'] == '6530732ad870cb41e73822b8df07374fb71d0cc5045286089c777ac91f35867e'
    rows, refs, accepted = epochs.accepted_stage('NO_MSCA', 239)
    assert len(rows) == ROUNDS and rows[-1]['metrics'] == cert['latest_metrics']
    assert rows[-1]['optimizer_step_attempts'] == 388080
    best = cert['endpoints']['best']
    assert best['epoch_index'] == 238 and rows[238]['metrics'] == best['metrics']
    assert rows[238]['ema_tensor_sha256'] == best['ema_tensor_sha256']
    assert rows[238]['prediction_sha256'] == best['prediction_sha256']
    assert rows[238]['evaluation_arrays_sha256'] == best['evaluation_arrays_sha256']
    assert max(range(ROUNDS), key=lambda index: rows[index]['metrics']['AP']) == 238
    assert rows[-1]['metrics']['AP'] == 0.21032901073722304
    assert rows[-1]['metrics']['APs'] == 0.1393851562602276
    assert best['metrics']['AP'] == 0.21033699779236342
    assert rows[-1]['ema_tensor_sha256'] != best['ema_tensor_sha256']
    assert accepted['rolling_checkpoint_sha256'] == worker['rolling_checkpoint_sha256']
    assert accepted['manifest_sha256'] == worker['local_backup_verified']['manifest_sha256']
    assert accepted['ack_sha256'] == worker['local_backup_verified']['ack_sha256']
    old_stage = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups/epochs/epoch0199')
    assert Path(refs[0]['source_path']).read_bytes().startswith((old_stage / 'epochs.jsonl').read_bytes())
    new_sources = []
    for source_id, source in [('NO_MSCA_SECOND_RECOVERY_R240', CERTIFICATE), ('NO_MSCA_SECOND_RECOVERY_LIVE_R241', LIVE_PATH)]:
        value = read(source) if source.suffix == '.json' else {}
        new_sources.append({
            'source_id': source_id, 'relative_to_remote_root': str(source.relative_to(epochs.REMOTE)).replace('\\', '/'),
            'path': str(source), 'bytes': source.stat().st_size, 'sha256': sha(source),
            'status_in_source': value.get('status', 'HASH_BOUND_PRIOR_SEGMENT_ARCHIVE' if source.suffix == '.zip' else 'SAVED_REMOTE_OBSERVATION'),
        })
    progress = summary['NO_MSCA_progress']
    best = cert['endpoints']['best']
    progress.update(
        completed_rounds=ROUNDS, epoch_index_0based=239, remaining_rounds=80,
        optimizer_step_attempts=388080, latest_metrics_fraction=cert['latest_metrics'],
        latest_metrics_percent={metric: value * 100 for metric, value in cert['latest_metrics'].items()},
        rolling_checkpoint_sha256=accepted['rolling_checkpoint_sha256'],
        latest_best_checkpoint_sha256=best['checkpoint_sha256'], ema_tensor_sha256=rows[-1]['ema_tensor_sha256'],
        source_ids=['NO_MSCA_SECOND_RECOVERY_R240'], snapshot_cutoff='independently_verified_round240',
        native_acceptance=accepted,
        update_since_previous_snapshot={'previous_rounds': 200, 'new_verified_rounds': 40, 'new_verified_updates': 64680},
    )
    progress['best_so_far_progress'] = {
        'role': 'PROGRESS_ONLY_AP_BEST_WITHIN_FIRST240_NOT_A_FINAL_ENDPOINT',
        'round_1based': 239, 'epoch_index_0based': 238, 'cutoff_rounds': 240,
        'metrics_fraction': best['metrics'], 'metrics_percent': {key: value * 100 for key, value in best['metrics'].items()},
        'checkpoint_sha256': best['checkpoint_sha256'], 'ema_tensor_sha256': best['ema_tensor_sha256'],
        'model_tensor_sha256': best['model_tensor_sha256'], 'prediction_sha256': best['prediction_sha256'],
        'evaluation_arrays_sha256': best['evaluation_arrays_sha256'], 'metrics_sha256': best['metrics_sha256'],
        'source_id': 'NO_MSCA_SECOND_RECOVERY_R240',
    }
    progress['latest_prediction_sha256'] = rows[-1]['prediction_sha256']
    progress['latest_evaluation_arrays_sha256'] = rows[-1]['evaluation_arrays_sha256']
    progress['recovery'] = copy.deepcopy(recovery)
    progress['recovery'].update(new_verified_rounds=76, new_verified_optimizer_updates=122892,
                                cumulative_recovered_rounds_since29=211, cumulative_recovered_optimizer_updates_since29=341187)
    progress.setdefault('observation_history', []).append(copy.deepcopy(progress['live_observation']))
    progress.setdefault('connectivity_history', []).append(copy.deepcopy(progress['connectivity']))
    progress['live_observation'] = {
        'source_id': 'NO_MSCA_SECOND_RECOVERY_LIVE_R241', 'observed_server_utc': live['observed_server_utc'],
        'actual_epoch_index_0based': 240, 'round_1based': 241, 'batch': worker['batch'],
        'optimizer_step_attempts': worker['optimizer_step_attempts'], 'native_accepted_rounds_observed': 240,
        'independent_metric_snapshot_rounds': ROUNDS, 'observation_kind': 'HASH_BOUND_IMMUTABLE_MILESTONE_OBSERVATION',
        'current_remote_training_state': 'RUNNING', 'worker_pid': 413, 'supervisor_pid': 412,
        'scope': 'Immutable003 milestone240 process observation on30715. Round241 was training at this instant; metrics stop at independently audited240. No004 or mutable latest/status hash is permanent evidence.',
    }
    progress['current_remote_training_state'] = 'RUNNING'
    progress['connectivity'] = {
        'source_id': 'NO_MSCA_SECOND_RECOVERY_LIVE_R241', 'status': 'AUTHORIZED_PORT30715_CONTINUATION_RUNNING_AT_SAVED_OBSERVATION',
        'remote_training_state_now': 'RUNNING', 'remote_ssh_port': 30715, 'observed_server_utc': live['observed_server_utc'],
        'last_native_accepted_round': 240, 'independent_metric_snapshot_rounds': 240,
        'native_backup_client_pid': 8216, 'native_backup_client_verified_at_acceptance': True,
        'milestone_observer_identity_source_id': 'NO_MSCA_OBSERVER003_IDENTITY',
        'milestone_observer_pid': 40816, 'milestone_observer_target_round': 240,
        'milestone_observer_window_status': 'COMPLETED_AT240',
        'scope': 'Timestamped immutable evidence and fixed observer identity only; the data exporter makes no remote connection or ongoing liveness claim.',
    }
    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None
    summary['input_sources'].extend(new_sources)
    summary['updated_local_date'] = LOCAL_DATE
    summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
    summary['limitations'] = [
        'NO_MSCA本快照仅独立核验240轮，320轮终点为空，不参与最终端点差值。'
        if line.startswith('NO_MSCA本快照') else line for line in summary['limitations']]
    summary['limitations'] = [
        '历史164轮快照时远端UNKNOWN且旧备份客户端已退出；该失联历史保留。现由已绑定30715运行快照支持续训状态，本次指标仅截至独立240轮。'
        if line.startswith('当前远端训练状态UNKNOWN') else line for line in summary['limitations']]
    assert recovery['physical_compute_limitation'] in summary['limitations']
    replace_json(OUT / '实验汇总.json', summary)
    with (OUT / 'NO_MSCA进度.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, progress_rows = reader.fieldnames, list(reader)
    fields = list(fields)
    for row in progress_rows:
        row.update(completed_rounds=ROUNDS, latest_value_fraction=cert['latest_metrics'][row['metric']],
                   latest_value_percent=cert['latest_metrics'][row['metric']] * 100, source_ids='NO_MSCA_SECOND_RECOVERY_R240')
        row.update(remote_training_state_now='RUNNING', historical_last_observed_running_round=165,
                   native_backup_client_alive='VERIFIED_AT_ROUND240_ACCEPTANCE', latest_observed_running_round=241,
                   second_recovery_first_epoch_index=164,
                   observation_scope='Authorized30715 saved running observation; historical old165 partial work preserved; metrics cut off at independent240')
        assert row['round320_primary_value'] == row['round320_secondary_value'] == ''
    replace_bytes(OUT / 'NO_MSCA进度.csv', csv_bytes(fields, progress_rows))
    replace_bytes(OUT / '输入来源.csv', csv_bytes(list(summary['input_sources'][0]), summary['input_sources']))
    old_csv = (OUT / '逐轮指标.csv').read_bytes()
    with (OUT / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, original_rows = reader.fieldnames, list(reader)
    assert len(original_rows) == 1160 and original_rows[-1]['arm'] == 'NO_MSCA'
    assert int(original_rows[-1]['round_1based']) == 200
    source_id = 'NO_MSCA_SECOND_RECOVERY_R240_EPOCH_ROWS'
    additional = []
    for row, ref in zip(rows[200:240], refs[200:240]):
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
                              'source_records_0based_half_open': [200, 240], 'rounds_included': [201, 240]})
    mapping['rows_per_arm']['NO_MSCA'] = ROUNDS
    mapping['accepted_stage_sources'].append(accepted)
    mapping['excluded'][-1] = {'arm': 'NO_MSCA', 'rule':
        'Authorized30715 continuation is RUNNING in the immutable003 milestone240 observation at round241. Only accepted and independently audited rounds1-240 are exported; round241 and later are excluded. The stage-local best239 is progress only. Old discarded partial165 work remains historical.'}
    mapping['updated_local_date'] = LOCAL_DATE
    mapping['recovery_source_id'] = 'NO_MSCA_SECOND_RECOVERY_R240'
    replace_json(OUT / '逐轮来源与切片.json', mapping)
    for filename in ('输入核查.json', '逐轮核查.json'):
        report = read(OUT / filename)
        report['checks'].extend(epochs.CHECKS)
        report['checks'].append({'check': 'Second recovery240 independent certificate retained; observed241 excluded; original1160 CSV bytes retained; latest240 and best239 distinct', 'status': 'PASS'})
        report['check_count'] = len(report['checks'])
        report['NO_MSCA_accepted_rounds'] = ROUNDS
        report.setdefault('new_source_sha256', {}).update({item['source_id']: item['sha256'] for item in new_sources})
        if filename == '逐轮核查.json':
            report['rows'] = 1200
            report['csv_sha256'] = sha(OUT / '逐轮指标.csv')
        replace_json(OUT / filename, report)
    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已独立核验至第 200 轮，进度 AP 为 19.6500%',
        'NO_MSCA 的来源快照已独立核验至第 240 轮，进度 AP 为 21.0329%')
    text += '\n2026-11-10 240轮更新。NO_MSCA 独立证据截止第240轮，累计388,080次更新；相对200轮快照只追加201–240轮的40行及64,680次更新。第240轮AP为21.0329%、APs为13.9385%；截至该证据点的局部AP最佳在第239轮，AP为21.0337%。本页12指标和最新张量/预测/数组均对应第240轮；局部最佳权重与12指标另标进度，均不是最终320轮主/次端点。\n'
    text += '\n003已结束窗口的固定 milestone_round240_live.json 记录同一30715 worker413/supervisor412进入第241轮（服务器UTC '+live['observed_server_utc']+'）；该时点观察与独立指标截止240分开。新004与任何mutable latest/status均不作为本次永久来源。164轮恢复段累计76轮/122,892次更新；29/164两次恢复、旧partial165至少1,401次弃用更新及更多未知物理工作量、单seed/验证集选点/非官方VisDrone/非无中断位级等同限制全部保留。NO_MSCA的24个320轮终点单元格继续为空。\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 1160 行', '共 1200 行').replace('前 200 行', '前 240 行')
    text = text.replace('第166–200行来自epoch0199累计日志，并与独立200轮证书核对。第201轮及之后指标不纳入本快照。',
                        '第166–200行来自epoch0199累计日志，第201–240行来自epoch0239累计日志，并与独立240轮证书核对。第241轮及之后指标不纳入本快照。')
    text += '\n2026-11-10 240轮更新精确保留原1160行字节前缀及全部原字段/格式，只从接受的epoch0239追加201–240轮40行。BASE/FULL/NO_MPS各320行，NO_MSCA240行，总计1200行；14,400项原始0–1指标均逐行回溯。第239轮局部最佳与第240轮实际进度分开，241及之后排除。旧行状态仍代表各自历史快照。\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
    replace_json(OUT / '_build/update_round240_plan.json', {
        'status': 'DATA_UPDATED_WORKBOOK_PENDING', 'local_date': LOCAL_DATE,
        'snapshot': str(SNAPSHOT), 'xlsx_relative': str(XLSX), 'old_xlsx_sha256': sha(SNAPSHOT / XLSX),
        'target_rounds': ROUNDS, 'total_updates': 388080, 'previous_snapshot_rounds': 200,
        'update_tag': 'round240', 'new_NO_MSCA_metrics': cert['latest_metrics'],
        'progress_notes': [
            '30715续训运行；独立截止240轮，固定003现场见第241轮。',
            '本页为第240轮：AP21.0329%、APs13.9385%，均为进度。',
            '截至240轮局部最佳在第239轮，AP21.0337%；非320轮端点。',
            '164轮恢复段76轮/122892次更新；累计240轮/388080次。',
            '29/164两次恢复；旧partial165至少1401次，更多物理工作量未知。',
            '完整状态/RNG恢复已核验；单seed/验证集选点/非官方VisDrone限制保留。',
        ],
        'new_source_sha256': {item['source_id']: item['sha256'] for item in new_sources},
    })
    print(json.dumps({'status': 'ROUND240_DATA_UPDATED', 'prior_snapshot': str(SNAPSHOT),
                      'NO_MSCA_rounds': ROUNDS, 'per_epoch_rows': 1200}, ensure_ascii=True))


if __name__ == '__main__':
    main()
