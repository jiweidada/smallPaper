"""Prepare the certificate-bounded200 data update, preserving the complete165 delivery."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'update_round165.py').read_text(encoding='utf-8')
replacements = [
    ('independent round164', 'independent round200'),
    ('ROUNDS = 165', 'ROUNDS = 200'),
    ("LOCAL_DATE = '2026-11-09'", "LOCAL_DATE = '2026-11-10'"),
    ('lineage_second_recovery_through_round165.json', 'lineage_second_recovery_through_round200.json'),
    ('resume_monitor_20261109_001/milestone_round165_live.json', 'resume_monitor_20261109_002/milestone_round200_live.json'),
    ('resume_monitor_20261109_001/observer_identity.json', 'resume_monitor_20261110_003/observer_identity.json'),
    ("SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round164_verified_20261109'", "SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round165_verified_20261109'"),
    ("FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round165_verified_20261109'", "FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round200_verified_20261110'"),
    ("['completed_rounds'] == 164", "['completed_rounds'] == 165"),
    ("len(previous_manifest['files']) == 19", "len(previous_manifest['files']) == 20"),
    ("epochs.accepted_stage('NO_MSCA', 164)", "epochs.accepted_stage('NO_MSCA', 199)"),
    ("rows[-1]['optimizer_step_attempts'] == 266805", "rows[-1]['optimizer_step_attempts'] == 323400"),
    ("/epoch0163')", "/epoch0164')"),
    ('epoch_index_0based=164, remaining_rounds=155', 'epoch_index_0based=199, remaining_rounds=120'),
    ('optimizer_step_attempts=266805', 'optimizer_step_attempts=323400'),
    ("source_ids=['NO_MSCA_SECOND_RECOVERY_R165'], snapshot_cutoff='independently_verified_round165'", "source_ids=['NO_MSCA_SECOND_RECOVERY_R200'], snapshot_cutoff='independently_verified_round200'"),
    ("'previous_rounds': 164, 'new_verified_rounds': 1, 'new_verified_updates': 1617", "'previous_rounds': 165, 'new_verified_rounds': 35, 'new_verified_updates': 56595"),
    ("source_ids='NO_MSCA_SECOND_RECOVERY_R165'", "source_ids='NO_MSCA_SECOND_RECOVERY_R200'"),
    ("    fields = list(fields) + ['latest_observed_running_round', 'second_recovery_first_epoch_index']", "    fields = list(fields)"),
    ("native_backup_client_alive='VERIFIED_AT_ROUND165_ACCEPTANCE', latest_observed_running_round=166", "native_backup_client_alive='VERIFIED_AT_ROUND200_ACCEPTANCE', latest_observed_running_round=201"),
    ('metrics cut off at independent165', 'metrics cut off at independent200'),
    ('len(original_rows) == 1124', 'len(original_rows) == 1125'),
    ("['round_1based']) == 164", "['round_1based']) == 165"),
    ('NO_MSCA_SECOND_RECOVERY_R165_EPOCH_ROWS', 'NO_MSCA_SECOND_RECOVERY_R200_EPOCH_ROWS'),
    ('zip(rows[164:165], refs[164:165])', 'zip(rows[165:200], refs[165:200])'),
    ('len(additional) == 1', 'len(additional) == 35'),
    ("[164, 165], 'rounds_included': [165, 165]", "[165, 200], 'rounds_included': [166, 200]"),
    ('Authorized30715 continuation is RUNNING in the saved observation at round166. Only accepted and independently audited rounds1-165 are exported; round166 and later are excluded. Old partial165 before instance expiry remains discarded historical work.',
     'Authorized30715 continuation is RUNNING in the immutable milestone200 observation at round201. Only accepted and independently audited rounds1-200 are exported; round201 and later are excluded. The stage-local best199 is progress only. Old discarded partial165 work remains historical.'),
    ("mapping['recovery_source_id'] = 'NO_MSCA_SECOND_RECOVERY_R165'", "mapping['recovery_source_id'] = 'NO_MSCA_SECOND_RECOVERY_R200'"),
    ('Second recovery165 independent certificate retained; observed166 excluded; original1124 CSV bytes retained', 'Second recovery200 independent certificate retained; observed201 excluded; original1125 CSV bytes retained; latest200 and best199 distinct'),
    ("report['rows'] = 1125", "report['rows'] = 1160"),
    ('_build/update_round165_plan.json', '_build/update_round200_plan.json'),
    ("'total_updates': 266805, 'previous_snapshot_rounds': 164", "'total_updates': 323400, 'previous_snapshot_rounds': 165"),
    ("'update_tag': 'round165'", "'update_tag': 'round200'"),
    ('ROUND165_DATA_UPDATED', 'ROUND200_DATA_UPDATED'),
    ("'per_epoch_rows': 1125", "'per_epoch_rows': 1160"),
]
for old, new in replacements:
    assert text.count(old) >= 1, old
    text = text.replace(old, new)
start = text.index('    cert = read(CERTIFICATE)')
end = text.index('    rows, refs, accepted = ', start)
text = text[:start] + """    assert sha(CERTIFICATE) == 'c451348bb627f7be74fd9b56deaa36b7299742537e63aaed01505bed9dfaeb11'
    cert = read(CERTIFICATE)
    assert cert['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and cert['completed_rounds'] == ROUNDS
    assert cert['learning_rates']['recorded_attempts'] == 323400 == ROUNDS * 1617
    assert cert['learning_rates']['maximum_absolute_difference'] == cert['maximum_latest_metric_error'] == 0
    assert cert['learning_rates']['groups'] == 4 and cert['validation_images'] == 548 and cert['predictions'] == 164400
    assert cert['model_forwards'] == cert['evaluator_runs'] == 0
    recovery = cert['recovery']
    assert recovery['new_completed_rounds'] == 36 and recovery['new_optimizer_update_attempts'] == 58212
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
    live = json.loads(live_wrapper['stdout'])
    worker, supervisor = live['records']['run/receipt.json'], live['records']['supervisor.json']
    assert supervisor['pid'] == worker['pid'] == 413 and supervisor['supervisor_pid'] == 412
    assert supervisor['start_ticks'] == 26597622 and supervisor['supervisor_start_ticks'] == 26597399
    assert supervisor['status'] == worker['status'] == 'RUNNING'
    assert supervisor['boot_id'] == live['boot_id'] == binding['target']['boot_id']
    assert live['gpu'].split(', ')[0] == binding['target']['gpu_uuid']
    assert worker['actual_loop_epoch'] == worker['completed_epochs'] == 200
    assert worker['optimizer_step_attempts'] == 323501 and worker['batch'] == 101
    assert worker['local_backup_verified']['epoch'] == 199
    assert worker['recovery_addendum_sha256'] == sha(ADDENDUM)
    assert worker['runner_sha256'] == read(ADDENDUM)['resume_runner_sha256']
    observer = read(IDENTITY)
    assert observer['backup_client_pid'] == 8216 and observer['observer_pid'] == 40816
    assert observer['observer_create_time'] == 1794312084.181138 and observer['target_round'] == 240
    assert observer['observer_source_sha256'] == '6530732ad870cb41e73822b8df07374fb71d0cc5045286089c777ac91f35867e'
""" + text[end:]
old = "    assert rows[-1]['ema_tensor_sha256'] == cert['endpoints']['best']['ema_tensor_sha256']\n"
new = """    best = cert['endpoints']['best']
    assert best['epoch_index'] == 198 and rows[198]['metrics'] == best['metrics']
    assert rows[198]['ema_tensor_sha256'] == best['ema_tensor_sha256']
    assert rows[198]['prediction_sha256'] == best['prediction_sha256']
    assert rows[198]['evaluation_arrays_sha256'] == best['evaluation_arrays_sha256']
    assert max(range(ROUNDS), key=lambda index: rows[index]['metrics']['AP']) == 198
    assert rows[-1]['metrics']['AP'] == 0.19650041568758386
    assert rows[-1]['metrics']['APs'] == 0.13120383278867362
    assert best['metrics']['AP'] == 0.19665533181774233
    assert rows[-1]['ema_tensor_sha256'] != best['ema_tensor_sha256']
"""
assert text.count(old) == 1
text = text.replace(old, new)
start = text.index('    for source_id, source in [')
end = text.index(':\n', start)
text = text[:start] + "    for source_id, source in [('NO_MSCA_SECOND_RECOVERY_R200', CERTIFICATE), ('NO_MSCA_SECOND_RECOVERY_LIVE_R201', LIVE_PATH), ('NO_MSCA_OBSERVER003_IDENTITY', IDENTITY)]" + text[end:]
start = text.index('    previous_recovery = copy.deepcopy')
end = text.index("    assert progress['primary_endpoint']", start)
text = text[:start] + """    progress['best_so_far_progress'] = {
        'role': 'PROGRESS_ONLY_AP_BEST_WITHIN_FIRST200_NOT_A_FINAL_ENDPOINT',
        'round_1based': 199, 'epoch_index_0based': 198, 'cutoff_rounds': 200,
        'metrics_fraction': best['metrics'], 'metrics_percent': {key: value * 100 for key, value in best['metrics'].items()},
        'checkpoint_sha256': best['checkpoint_sha256'], 'ema_tensor_sha256': best['ema_tensor_sha256'],
        'model_tensor_sha256': best['model_tensor_sha256'], 'prediction_sha256': best['prediction_sha256'],
        'evaluation_arrays_sha256': best['evaluation_arrays_sha256'], 'metrics_sha256': best['metrics_sha256'],
        'source_id': 'NO_MSCA_SECOND_RECOVERY_R200',
    }
    progress['latest_prediction_sha256'] = rows[-1]['prediction_sha256']
    progress['latest_evaluation_arrays_sha256'] = rows[-1]['evaluation_arrays_sha256']
    progress['recovery'] = copy.deepcopy(recovery)
    progress['recovery'].update(new_verified_rounds=36, new_verified_optimizer_updates=58212,
                                cumulative_recovered_rounds_since29=171, cumulative_recovered_optimizer_updates_since29=276507)
    progress.setdefault('observation_history', []).append(copy.deepcopy(progress['live_observation']))
    progress.setdefault('connectivity_history', []).append(copy.deepcopy(progress['connectivity']))
    progress['live_observation'] = {
        'source_id': 'NO_MSCA_SECOND_RECOVERY_LIVE_R201', 'observed_server_utc': live['observed_server_utc'],
        'actual_epoch_index_0based': 200, 'round_1based': 201, 'batch': worker['batch'],
        'optimizer_step_attempts': worker['optimizer_step_attempts'], 'native_accepted_rounds_observed': 200,
        'independent_metric_snapshot_rounds': ROUNDS, 'observation_kind': 'HASH_BOUND_IMMUTABLE_MILESTONE_OBSERVATION',
        'current_remote_training_state': 'RUNNING', 'worker_pid': 413, 'supervisor_pid': 412,
        'scope': 'Immutable002 milestone200 process observation on30715. Round201 was training at this instant; metrics stop at independently audited200. Mutable003 latest/status hashes are not permanent evidence.',
    }
    progress['current_remote_training_state'] = 'RUNNING'
    progress['connectivity'] = {
        'source_id': 'NO_MSCA_SECOND_RECOVERY_LIVE_R201', 'status': 'AUTHORIZED_PORT30715_CONTINUATION_RUNNING_AT_SAVED_OBSERVATION',
        'remote_training_state_now': 'RUNNING', 'remote_ssh_port': 30715, 'observed_server_utc': live['observed_server_utc'],
        'last_native_accepted_round': 200, 'independent_metric_snapshot_rounds': 200,
        'native_backup_client_pid': 8216, 'native_backup_client_verified_at_acceptance': True,
        'next_observer_identity_source_id': 'NO_MSCA_OBSERVER003_IDENTITY',
        'next_observer_pid': 40816, 'next_observer_target_round': 240,
        'scope': 'Timestamped immutable evidence and fixed observer identity only; the data exporter makes no remote connection or ongoing liveness claim.',
    }
""" + text[end:]
text = text.replace('仅独立核验165轮', '仅独立核验200轮')
text = text.replace('指标仅截至独立165轮', '本次指标仅截至独立200轮')
text = text.replace("    summary['limitations'].append(recovery['physical_compute_limitation'])\n", "    assert recovery['physical_compute_limitation'] in summary['limitations']\n")
start = text.index("    text = (OUT / '汇总说明.md').read_text")
end = text.index("    replace_json(OUT / '_build/update_round200_plan.json'", start)
text = text[:start] + """    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已独立核验至第 165 轮，进度 AP 为 17.3667%',
        'NO_MSCA 的来源快照已独立核验至第 200 轮，进度 AP 为 19.6500%')
    text += '\\n2026-11-10 200轮更新。NO_MSCA 独立证据截止第200轮，累计323,400次更新；相对165轮快照只追加166–200轮的35行及56,595次更新。第200轮AP为19.6500%、APs为13.1204%；截至该证据截止点的局部AP最佳在第199轮，AP为19.6655%。本页12指标及最新张量/预测/数组对应第200轮，局部最佳权重和12指标另标为进度，二者都不是最终320轮主/次端点。\\n'
    text += '\\n30715同一worker413/supervisor412在已固定002 milestone200现场记录中进入第201轮（服务器UTC '+live['observed_server_utc']+'）；该时点观察与独立指标截止200分开。003只读观察器固定身份PID40816、目标240，动态status/latest_live不作为永久来源。第二次164轮恢复段已核验36轮/58,212次更新；29与164两次恢复、旧partial165至少1,401次弃用更新、其后未知物理工作量、单seed/验证集选点/非官方VisDrone及非无中断位级等同限制全部保留。NO_MSCA的24个320轮终点单元格继续为空。\\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 1125 行', '共 1160 行').replace('前 165 行', '前 200 行')
    text = text.replace('第 165 行来自第二段恢复后接受的 epoch0164 累计日志，并与独立165轮证书核对。第 166 轮及其后的指标不纳入本快照。',
                        '第 165 行来自第二段恢复后接受的 epoch0164 累计日志；第166–200行来自epoch0199累计日志，并与独立200轮证书核对。第201轮及之后指标不纳入本快照。')
    text += '\\n2026-11-10 200轮更新保留原1125行精确字节前缀及全部原字段/格式，只从已接受epoch0199追加第166–200轮，共35行。BASE/FULL/NO_MPS各320行，NO_MSCA200行，总计1160行；13,920项原始0–1指标均可逐行回溯。第199轮局部最佳与第200轮实际进度分开，201及以后不纳入。旧行状态仍表示各自历史快照。\\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
""" + text[end:]
start = text.index("        'progress_notes': [")
end = text.index("        'new_source_sha256':", start)
text = text[:start] + """        'progress_notes': [
            '30715续训运行；独立截止200轮，已固定现场见第201轮。',
            '本页为第200轮：AP19.6500%、APs13.1204%，均为进度。',
            '截至200轮局部最佳在第199轮，AP19.6655%；非320轮端点。',
            '164轮恢复段36轮/58212次更新；累计200轮/323400次。',
            '29/164两次恢复；旧partial165至少1401次，更多物理工作量未知。',
            '完整状态/RNG恢复已核验；单seed/验证集选点/非官方VisDrone限制保留。',
        ],
""" + text[end:]
target = root / 'update_round200.py'
compile(text, str(target), 'exec')
with target.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
workbook = (root / 'update_progress_workbook.mjs').read_text(encoding='utf-8')
old = '`${localDate}，第${rounds}轮已独立核验。320轮终点保持空缺。`'
assert workbook.count(old) == 1
workbook = workbook.replace(old, '`${localDate}，30715续训运行；独立截止${rounds}轮，现场第201轮。320轮终点空缺。`')
with (root / 'update_progress_workbook_round200.mjs').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(workbook)
render = (root / 'verify_round160_full_renders.mjs').read_text(encoding='utf-8').replace('round160', 'round200')
with (root / 'verify_round200_full_renders.mjs').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(render)
print('ROUND200_DATA_AND_WORKBOOK_SCRIPTS_PREPARED')
