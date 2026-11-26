"""Prepare a one-row data-only advance after the independently audited second recovery."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'update_round164.py').read_text(encoding='utf-8').replace('import csv\n', 'import csv\nimport copy\n')
start = text.index('ROUNDS = ')
end = text.index('\n\ndef main():', start)
text = text[:start] + """ROUNDS = 165
LOCAL_DATE = '2026-11-09'
SECOND = RECOVERY.parent / 'recovery30715_20261109'
CERTIFICATE = SECOND / 'lineage_second_recovery_through_round165.json'
BINDING = SECOND / 'segment30715_audit_binding.json'
ADDENDUM = SECOND / 'recovery30715_addendum.json'
LIVE_PATH = SECOND / 'resume_monitor_20261109_001/milestone_round165_live.json'
IDENTITY = SECOND / 'resume_monitor_20261109_001/observer_identity.json'
SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round164_verified_20261109'
FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round165_verified_20261109'
""" + text[end:]
text = text.replace("['completed_rounds'] == 160", "['completed_rounds'] == 164")
text = text.replace("len(previous_manifest['files']) == 18", "len(previous_manifest['files']) == 19")
start = text.index('    cert = read(CERTIFICATE)')
end = text.index('    rows, refs, accepted = ', start)
text = text[:start] + """    cert = read(CERTIFICATE)
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
""" + text[end:]
text = text.replace("epochs.accepted_stage('NO_MSCA', 163)", "epochs.accepted_stage('NO_MSCA', 164)")
text = text.replace("rows[-1]['optimizer_step_attempts'] == 265188", "rows[-1]['optimizer_step_attempts'] == 266805")
text = text.replace("/epoch0159')", "/epoch0163')")
old = "[('NO_MSCA_RECOVERED_R164', CERTIFICATE), ('NO_MSCA_ACCESS_LOSS_R164', LOSS_PATH), ('NO_MSCA_HISTORICAL_LAST_REMOTE_R165', LIVE_PATH)]"
new = "[('NO_MSCA_SECOND_RECOVERY_R165', CERTIFICATE), ('NO_MSCA_SECOND_RECOVERY_BINDING', BINDING), ('NO_MSCA_SECOND_RECOVERY_ADDENDUM', ADDENDUM), ('NO_MSCA_SECOND_RECOVERY_HISTORY_MANIFEST', SECOND / 'prior_segment_archive_manifest.json'), ('NO_MSCA_SECOND_RECOVERY_HISTORY_ZIP', SECOND / 'prior_segment_archive.zip'), ('NO_MSCA_SECOND_RECOVERY_LIVE_R166', LIVE_PATH), ('NO_MSCA_SECOND_RECOVERY_OBSERVER_IDENTITY', IDENTITY)]"
assert text.count(old) == 1
text = text.replace(old, new).replace('        value = read(source)\n', "        value = read(source) if source.suffix == '.json' else {}\n")
text = text.replace('epoch_index_0based=163, remaining_rounds=156', 'epoch_index_0based=164, remaining_rounds=155')
text = text.replace('optimizer_step_attempts=265188', 'optimizer_step_attempts=266805')
text = text.replace("source_ids=['NO_MSCA_RECOVERED_R164'], snapshot_cutoff='independently_verified_round164'", "source_ids=['NO_MSCA_SECOND_RECOVERY_R165'], snapshot_cutoff='independently_verified_round165'")
text = text.replace("'previous_rounds': 160, 'new_verified_rounds': 4, 'new_verified_updates': 6468", "'previous_rounds': 164, 'new_verified_rounds': 1, 'new_verified_updates': 1617")
start = text.index("    progress['recovery'].update(")
end = text.index("    assert progress['primary_endpoint']", start)
text = text[:start] + """    previous_recovery = copy.deepcopy(progress['recovery'])
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
""" + text[end:]
text = text.replace('仅独立核验164轮', '仅独立核验165轮')
old = "    summary['limitations'].append('当前远端训练状态UNKNOWN；失联前最后见第165轮训练，原生备份客户端已退出。指标严格截至已接受且独立核验第164轮，不证明远端已终止，也不证明仍在训练。')\n"
new = """    summary['limitations'] = [
        '历史164轮快照时远端UNKNOWN且旧备份客户端已退出；该失联历史保留。现由已绑定30715运行快照支持续训状态，指标仅截至独立165轮。'
        if line.startswith('当前远端训练状态UNKNOWN') else line for line in summary['limitations']]
    summary['limitations'].append(recovery['physical_compute_limitation'])
"""
assert text.count(old) == 1
text = text.replace(old, new)
text = text.replace("    fields = list(fields) + ['remote_training_state_now', 'historical_last_observed_running_round', 'native_backup_client_alive', 'observation_scope']", "    fields = list(fields) + ['latest_observed_running_round', 'second_recovery_first_epoch_index']")
text = text.replace("source_ids='NO_MSCA_RECOVERED_R164'", "source_ids='NO_MSCA_SECOND_RECOVERY_R165'")
old = """        row.update(remote_training_state_now='UNKNOWN', historical_last_observed_running_round=165,
                   native_backup_client_alive=False, observation_scope='Historical last read only; not current liveness; metrics cut off at independent round164')"""
new = """        row.update(remote_training_state_now='RUNNING', historical_last_observed_running_round=165,
                   native_backup_client_alive='VERIFIED_AT_ROUND165_ACCEPTANCE', latest_observed_running_round=166,
                   second_recovery_first_epoch_index=164,
                   observation_scope='Authorized30715 saved running observation; historical old165 partial work preserved; metrics cut off at independent165')"""
assert text.count(old) == 1
text = text.replace(old, new)
text = text.replace('len(original_rows) == 1120', 'len(original_rows) == 1124').replace("['round_1based']) == 160", "['round_1based']) == 164")
text = text.replace('NO_MSCA_RECOVERED_R164_EPOCH_ROWS', 'NO_MSCA_SECOND_RECOVERY_R165_EPOCH_ROWS')
text = text.replace('zip(rows[160:164], refs[160:164])', 'zip(rows[164:165], refs[164:165])').replace('len(additional) == 4', 'len(additional) == 1')
text = text.replace("[160, 164], 'rounds_included': [161, 164]", "[164, 165], 'rounds_included': [165, 165]")
text = text.replace('Remote training status is now UNKNOWN and the native backup client has exited. The last successful historical observation saw round165; only accepted and independently audited rounds1-164 are exported. Round165 is excluded.',
                        'Authorized30715 continuation is RUNNING in the saved observation at round166. Only accepted and independently audited rounds1-165 are exported; round166 and later are excluded. Old partial165 before instance expiry remains discarded historical work.')
text = text.replace("mapping['recovery_source_id'] = 'NO_MSCA_RECOVERED_R164'", "mapping['recovery_source_id'] = 'NO_MSCA_SECOND_RECOVERY_R165'")
text = text.replace('Round164 independent certificate cutoff retained; historical round165 excluded and current remote state UNKNOWN', 'Second recovery165 independent certificate retained; observed166 excluded; original1124 CSV bytes retained')
text = text.replace("report['rows'] = 1124", "report['rows'] = 1125")
start = text.index("    text = (OUT / '汇总说明.md').read_text")
end = text.index("    replace_json(OUT / '_build/update_round164_plan.json'", start)
text = text[:start] + """    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已独立核验至第 164 轮，进度 AP 为 17.2070%',
        'NO_MSCA 的来源快照已独立核验至第 165 轮，进度 AP 为 17.3667%')
    text += '\\n2026-11-09 165轮更新。30715 已按原 seed0/batch4/FP32 配方从已接受164轮完整状态继续，原始模型、EMA、优化器、criterion、postprocessor、warmup 及全部保存RNG一致，完整科学循环AST未改。第二段恢复新增1轮及1,617次更新已独立核验，累计165轮/266,805次；最近已存现场见第166轮训练，指标严格止于165。原29→164来源及本次16文件历史ZIP保留，旧实例到期前未接受的第165轮部分工作与新接受165行严格区分；旧历史行不重写。已知旧partial165曾见1,401次额外更新，之后未保留工作量仍未知，早先29轮失联后的物理工作量也未知，不能声称完整物理计算相同或无中断位级等价。第164轮UNKNOWN仅为历史快照。NO_MSCA的320轮最终端点继续空缺。\\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 1124 行', '共 1125 行').replace('前 164 行', '前 165 行')
    text = text.replace('第 161–164 行来自已接受 epoch0163 备份，并与恢复轨迹第 164 轮独立证书核对。第 165 轮及其后的指标不纳入本快照。',
                        '第 161–164 行来自已接受 epoch0163 备份；第 165 行来自第二段恢复后接受的 epoch0164 累计日志，并与独立165轮证书核对。第 166 轮及其后的指标不纳入本快照。')
    text += '\\n2026-11-09 165轮更新仅在原 1124 行原字节后追加新接受且已独立核验的第165轮，旧行状态及来源仍代表各自历史快照，未因30715恢复而改写。三组已完成配置各320行，NO_MSCA165行，共1125行。新行来自第二次恢复后的epoch0164备份，不能与到期旧实例未接受的partial165混同。第166轮仅为已保存的运行观察。\\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
""" + text[end:]
text = text.replace('_build/update_round164_plan.json', '_build/update_round165_plan.json')
text = text.replace("'total_updates': 265188, 'previous_snapshot_rounds': 160", "'total_updates': 266805, 'previous_snapshot_rounds': 164")
text = text.replace("'update_tag': 'round164'", "'update_tag': 'round165'")
start = text.index("        'progress_notes': [")
end = text.index("        'new_source_sha256':", start)
text = text[:start] + """        'progress_notes': [
            '30715续训在运行；已独立核验165轮，最近已存现场见第166轮。',
            '从164轮完整恢复原始模型、EMA、优化器、warmup及全部保存RNG。',
            '第二段新增1轮/1617次，累计165轮/266805次；旧29→164来源保留。',
            '旧partial165及16文件历史ZIP保留；其后未保留物理工作量未知。',
            '独立证书 NO_MSCA_SECOND_RECOVERY_R165，全部12指标误差0。',
            '单seed、验证集选点及非官方VisDrone限制保留；320轮终点空缺。',
        ],
""" + text[end:]
text = text.replace('ROUND164_DATA_UPDATED', 'ROUND165_DATA_UPDATED').replace("'per_epoch_rows': 1124", "'per_epoch_rows': 1125")
target = root / 'update_round165.py'
compile(text, str(target), 'exec')
with target.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
workbook = (root / 'update_progress_workbook.mjs').read_text(encoding='utf-8')
old = '`${localDate}，第${rounds}轮已独立核验。320轮终点保持空缺。`'
assert workbook.count(old) == 1
workbook = workbook.replace(old, '`${localDate}，30715续训运行；已核验${rounds}轮，现场第166轮。320轮终点空缺。`')
with (root / 'update_progress_workbook_round165.mjs').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(workbook)
render = (root / 'verify_round160_full_renders.mjs').read_text(encoding='utf-8').replace('round160', 'round165')
with (root / 'verify_round165_full_renders.mjs').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(render)
print('ROUND165_DATA_AND_WORKBOOK_SCRIPTS_PREPARED')
