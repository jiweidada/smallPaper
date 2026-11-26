"""Prepare a strictly local, certificate-bounded round164 update."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'update_round160.py').read_text(encoding='utf-8')
replacements = [
    ('cutoff to independent round160', 'cutoff to independent round164'),
    ('ROUNDS = 160', 'ROUNDS = 164'),
    ('lineage_recovered_through_round160.json', 'lineage_recovered_through_round164.json'),
    ("LIVE_PATH = RECOVERY / 'goal_monitor_20261109_001/milestone_round160_live.json'",
     "LOSS_ROOT = RECOVERY.parent / 'connectivity_20261109/goal_monitor002_loss_20261109'\nLOSS_PATH = LOSS_ROOT / 'access_loss_and_safe_round164.json'\nLIVE_PATH = LOSS_ROOT / 'last_successful_remote_snapshot.json'"),
    ("SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round140_verified_20261109'", "SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round160_verified_20261109'"),
    ("FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round160_verified_20261109'", "FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round164_verified_20261109'"),
    ("['completed_rounds'] == 140", "['completed_rounds'] == 160"),
    ("len(previous_manifest['files']) == 17", "len(previous_manifest['files']) == 18"),
    ('== 258720', '== 265188'),
    ("['new_completed_rounds'] == 131", "['new_completed_rounds'] == 135"),
    ("['new_optimizer_update_attempts'] == 211827", "['new_optimizer_update_attempts'] == 218295"),
    ("['epoch'] == 159", "['epoch'] == 163"),
    ("epochs.accepted_stage('NO_MSCA', 159)", "epochs.accepted_stage('NO_MSCA', 163)"),
    ("/epoch0139')", "/epoch0159')"),
    ("[('NO_MSCA_RECOVERED_R160', CERTIFICATE), ('NO_MSCA_LIVE_MILESTONE_R160', LIVE_PATH)]",
     "[('NO_MSCA_RECOVERED_R164', CERTIFICATE), ('NO_MSCA_ACCESS_LOSS_R164', LOSS_PATH), ('NO_MSCA_HISTORICAL_LAST_REMOTE_R165', LIVE_PATH)]"),
    ('epoch_index_0based=159, remaining_rounds=160', 'epoch_index_0based=163, remaining_rounds=156'),
    ('optimizer_step_attempts=258720', 'optimizer_step_attempts=265188'),
    ('NO_MSCA_RECOVERED_R160', 'NO_MSCA_RECOVERED_R164'),
    ('independently_verified_round160', 'independently_verified_round164'),
    ("'previous_rounds': 140, 'new_verified_rounds': 20, 'new_verified_updates': 32340", "'previous_rounds': 160, 'new_verified_rounds': 4, 'new_verified_updates': 6468"),
    ('new_verified_rounds=131, new_verified_optimizer_updates=211827', 'new_verified_rounds=135, new_verified_optimizer_updates=218295'),
    ('top-level round160 saved-state audit verifies 131 recovered rounds and 211827 recovered updates.', 'top-level round164 saved-state audit verifies 135 recovered rounds and 218295 recovered updates. Current remote training status is UNKNOWN.'),
    ("'source_id': 'NO_MSCA_LIVE_MILESTONE_R160'", "'source_id': 'NO_MSCA_HISTORICAL_LAST_REMOTE_R165'"),
    ("'scope': 'Saved running-process observation. Independently audited metric data is cut off at round160.'", "'scope': 'Historical last successful remote observation only; not current liveness. Remote state is now UNKNOWN. Independently audited metric data is cut off at round164.',\n        'observation_kind': 'HISTORICAL_LAST_SUCCESSFUL_READ', 'current_remote_training_state': 'UNKNOWN'"),
    ('仅独立核验160轮', '仅独立核验164轮'),
    ('len(original_rows) == 1100', 'len(original_rows) == 1120'),
    ("['round_1based']) == 140", "['round_1based']) == 160"),
    ('zip(rows[140:160], refs[140:160])', 'zip(rows[160:164], refs[160:164])'),
    ('len(additional) == 20', 'len(additional) == 4'),
    ("[140, 160], 'rounds_included': [141, 160]", "[160, 164], 'rounds_included': [161, 164]"),
    ('The saved live observation reports 160 native accepted rounds and round161 in progress. Metric export stops at the independently audited round160 certificate; subsequent live/native progress is excluded.',
     'Remote training status is now UNKNOWN and the native backup client has exited. The last successful historical observation saw round165; only accepted and independently audited rounds1-164 are exported. Round165 is excluded.'),
    ('Round160 independent certificate cutoff retained; live round161 excluded', 'Round164 independent certificate cutoff retained; historical round165 excluded and current remote state UNKNOWN'),
    ("report['rows'] = 1120", "report['rows'] = 1124"),
    ('_build/update_round160_plan.json', '_build/update_round164_plan.json'),
    ("'total_updates': 258720, 'previous_snapshot_rounds': 140", "'total_updates': 265188, 'previous_snapshot_rounds': 160"),
    ("'update_tag': 'round160'", "'update_tag': 'round164'"),
    ("'status': 'ROUND160_DATA_UPDATED'", "'status': 'ROUND164_DATA_UPDATED'"),
    ("'per_epoch_rows': 1120", "'per_epoch_rows': 1124"),
]
for old, new in replacements:
    assert text.count(old) >= 1, old
    text = text.replace(old, new)
needle = "    live_wrapper = read(LIVE_PATH)\n"
text = text.replace(needle, """    loss = read(LOSS_PATH)
    assert loss['status'] == 'REMOTE_UNREACHABLE_LOCAL_BACKUP_CLIENT_FAILED_REMOTE_TRAINING_UNKNOWN'
    assert loss['last_native_accepted_round'] == loss['independent_state_and_lineage_round'] == ROUNDS
    assert loss['learning_rate_records'] == 265188
    assert loss['historical_last_observed_running_round'] == 165
    assert loss['backup_client']['alive'] is False and loss['observer']['alive'] is False
    assert loss['lineage_audit']['sha256'] == sha(CERTIFICATE)
    assert loss['remote_training_state_now'].startswith('UNKNOWN')
""" + needle)
needle = "    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None\n"
text = text.replace(needle, """    progress['current_remote_training_state'] = 'UNKNOWN'
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
""" + needle)
needle = "    replace_json(OUT / '实验汇总.json', summary)\n"
text = text.replace(needle, "    summary['limitations'].append('当前远端训练状态UNKNOWN；失联前最后见第165轮训练，原生备份客户端已退出。指标严格截至已接受且独立核验第164轮，不证明远端已终止，也不证明仍在训练。')\n" + needle)
needle = "    for row in progress_rows:\n"
text = text.replace(needle, "    fields = list(fields) + ['remote_training_state_now', 'historical_last_observed_running_round', 'native_backup_client_alive', 'observation_scope']\n" + needle)
needle = "        assert row['round320_primary_value'] == row['round320_secondary_value'] == ''\n"
text = text.replace(needle, """        row.update(remote_training_state_now='UNKNOWN', historical_last_observed_running_round=165,
                   native_backup_client_alive=False, observation_scope='Historical last read only; not current liveness; metrics cut off at independent round164')
""" + needle)
start = text.index("    text = (OUT / '汇总说明.md').read_text")
end = text.index("    replace_json(OUT / '_build/update_round164_plan.json'", start)
text = text[:start] + """    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已独立核验至第 160 轮，进度 AP 为 16.4156%',
        'NO_MSCA 的来源快照已独立核验至第 164 轮，进度 AP 为 17.2070%')
    text += '\\n2026-11-09 164轮更新。NO_MSCA 已接受且独立核验至第 164 轮，累计 265,188 次更新；第 29 轮换机后恢复段已核验 135 轮及 218,295 次更新。相对第 160 轮快照仅追加 4 行及 6,468 次更新。当前远端训练状态 UNKNOWN；失联前最后成功读取曾见第 165 轮训练，原生备份客户端因 600 秒重连窗口耗尽已自行退出，只读观察器也已退出。此历史现场不表示当前仍在训练，访问失败也不证明远端训练已终止。第 165 轮指标不纳入本汇总。原第 160 轮交付保持于 verified_snapshots/NO_MSCA_round160_verified_20261109。\\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 1120 行', '共 1124 行').replace('前 160 行', '前 164 行')
    text = text.replace('第 141–160 行来自已接受 epoch0159 备份，并与恢复轨迹第 160 轮独立证书核对。第 161 轮及其后的指标不纳入本快照。',
                        '第 141–160 行来自已接受 epoch0159 备份，第 161–164 行来自已接受 epoch0163 备份，并与恢复轨迹第 164 轮独立证书核对。第 165 轮及其后的指标不纳入本快照。')
    text += '\\n2026-11-09 164轮更新保留原 1120 行字节前缀，仅从已接受 epoch0163 的累计日志中追加第 161–164 轮。三组已完成数据各 320 行，NO_MSCA 164 行，总计 1124 行。当前远端状态 UNKNOWN，原生备份客户端已退出；最后见第 165 轮是历史观察，不作当前活跃状态或指标证据。全部指标仍为原始 0–1 比例。\\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
""" + text[end:]
start = text.index("        'progress_notes': [")
end = text.index("        'new_source_sha256':", start)
text = text[:start] + """        'progress_notes': [
            '第29轮换机：原始模型、EMA、优化器、warmup及保存RNG恢复已核验。',
            '恢复段135轮及218295次更新已核验，累计164轮及265188次。',
            '当前远端状态UNKNOWN；原生备份客户端及只读观察器均已退出。',
            '失联前最后见第165轮训练；只收录已接受且独立核验的164轮。',
            '失联后未保留的工作量未知，完整物理计算量无法确认。',
            '独立证书 NO_MSCA_RECOVERED_R164，12指标误差0，320轮终点空缺。',
        ],
""" + text[end:]
target = root / 'update_round164.py'
compile(text, str(target), 'exec')
with target.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
workbook = (root / 'update_progress_workbook.mjs').read_text(encoding='utf-8')
old = '`${localDate}，第${rounds}轮已独立核验。320轮终点保持空缺。`'
assert workbook.count(old) == 1
workbook = workbook.replace(old, '`${localDate}，已独立核验第${rounds}轮；远端状态UNKNOWN，320轮终点空缺。`')
with (root / 'update_progress_workbook_round164.mjs').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(workbook)
print('ROUND164_UPDATER_AND_WORKBOOK_EDITOR_PREPARED')
