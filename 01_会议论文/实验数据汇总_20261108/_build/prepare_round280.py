"""Derive the bounded round280 export from the verified round240 workflow."""

from pathlib import Path


root = Path(__file__).resolve().parent


def substitute(source, replacements):
    for old, new in replacements:
        assert old in source, old
        source = source.replace(old, new)
    return source


def save(name, source):
    target = root / name
    if target.suffix == '.py':
        compile(source, str(target), 'exec')
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(source)


update = (root / 'update_round240.py').read_text(encoding='utf-8')
update = substitute(update, [
    ('independent round240', 'independent round280'),
    ('ROUNDS = 240', 'ROUNDS = 280'),
    ('lineage_second_recovery_through_round240.json', 'lineage_second_recovery_through_round280.json'),
    ('resume_monitor_20261110_003/milestone_round240_live.json', 'resume_monitor_20261110_004/milestone_round280_live.json'),
    ("IDENTITY = SECOND / 'resume_monitor_20261110_003/observer_identity.json'\n", ''),
    ("SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round200_verified_20261110'", "SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round240_verified_20261110'"),
    ("FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round240_verified_20261110'", "FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round280_verified_20261110'"),
    ("['completed_rounds'] == 200", "['completed_rounds'] == 240"),
    ("len(previous_manifest['files']) == 25", "len(previous_manifest['files']) == 30"),
    ('b210f4fcaf5acb09fffbc8af8cb6f484c73b2a854136352227a087c04e132545', 'c7b857af507b95f48dc6c7c789caf8aaef83ed71d6fe6050cddb685c579ccfa3'),
    ('809ab33296ee7e61be57f3f4a6967de1c7a1eeedd89ce1550ac030699a5baa1f', '5b8c48a536d63f25d7b2e802f11da9c5d34a95ff95e351c765d7f9eb7fab64b8'),
    ("recorded_attempts'] == 388080", "recorded_attempts'] == 452760"),
    ("['new_completed_rounds'] == 76", "['new_completed_rounds'] == 116"),
    ("['new_optimizer_update_attempts'] == 122892", "['new_optimizer_update_attempts'] == 187572"),
    ("worker['actual_loop_epoch'] == worker['completed_epochs'] == 240", "worker['actual_loop_epoch'] == worker['completed_epochs'] == 280"),
    ("worker['optimizer_step_attempts'] == 388081", "worker['optimizer_step_attempts'] == 452761"),
    ("worker['local_backup_verified']['epoch'] == 239", "worker['local_backup_verified']['epoch'] == 279"),
    ("epochs.accepted_stage('NO_MSCA', 239)", "epochs.accepted_stage('NO_MSCA', 279)"),
    ("rows[-1]['optimizer_step_attempts'] == 388080", "rows[-1]['optimizer_step_attempts'] == 452760"),
    ("best['epoch_index'] == 238", "best['epoch_index'] == 279"),
    ('rows[238]', 'rows[279]'),
    ("rows[index]['metrics']['AP']) == 238", "rows[index]['metrics']['AP']) == 279"),
    ('0.21032901073722304', '0.21928245309369632'),
    ('0.1393851562602276', '0.14481580486643042'),
    ('0.21033699779236342', '0.21928245309369632'),
    ("rows[-1]['ema_tensor_sha256'] != best['ema_tensor_sha256']", "rows[-1]['ema_tensor_sha256'] == best['ema_tensor_sha256']"),
    ("/epoch0199')", "/epoch0239')"),
    ('epoch_index_0based=239, remaining_rounds=80', 'epoch_index_0based=279, remaining_rounds=40'),
    ('optimizer_step_attempts=388080', 'optimizer_step_attempts=452760'),
    ('NO_MSCA_SECOND_RECOVERY_R240', 'NO_MSCA_SECOND_RECOVERY_R280'),
    ('NO_MSCA_SECOND_RECOVERY_LIVE_R241', 'NO_MSCA_SECOND_RECOVERY_LIVE_R281'),
    ('independently_verified_round240', 'independently_verified_round280'),
    ("'previous_rounds': 200, 'new_verified_rounds': 40, 'new_verified_updates': 64680", "'previous_rounds': 240, 'new_verified_rounds': 40, 'new_verified_updates': 64680"),
    ('PROGRESS_ONLY_AP_BEST_WITHIN_FIRST240_NOT_A_FINAL_ENDPOINT', 'PROGRESS_AP_BEST_THROUGH_ROUND280'),
    ("'round_1based': 239, 'epoch_index_0based': 238, 'cutoff_rounds': 240,", "'round_1based': 280, 'epoch_index_0based': 279, 'cutoff_rounds': 280,\n        'same_epoch_as_latest_progress': True,"),
    ('new_verified_rounds=76, new_verified_optimizer_updates=122892', 'new_verified_rounds=116, new_verified_optimizer_updates=187572'),
    ('cumulative_recovered_rounds_since29=211, cumulative_recovered_optimizer_updates_since29=341187', 'cumulative_recovered_rounds_since29=251, cumulative_recovered_optimizer_updates_since29=405867'),
    ("'actual_epoch_index_0based': 240, 'round_1based': 241", "'actual_epoch_index_0based': 280, 'round_1based': 281"),
    ("'native_accepted_rounds_observed': 240", "'native_accepted_rounds_observed': 280"),
    ('Immutable003 milestone240 process observation on30715. Round241 was training at this instant; metrics stop at independently audited240. No004 or mutable latest/status hash is permanent evidence.',
     'The completed 004 window records round 281 at the milestone. The independent metrics cover the first 280 rounds.'),
    ("'last_native_accepted_round': 240, 'independent_metric_snapshot_rounds': 240", "'last_native_accepted_round': 280, 'independent_metric_snapshot_rounds': 280"),
    ("'milestone_observer_identity_source_id': 'NO_MSCA_OBSERVER003_IDENTITY',\n        'milestone_observer_pid': 40816, 'milestone_observer_target_round': 240,\n        'milestone_observer_window_status': 'COMPLETED_AT240',", "'milestone_observer_window': '004', 'milestone_observer_target_round': 280,\n        'milestone_observer_window_status': 'COMPLETED_AT280',"),
    ('仅独立核验240轮', '仅独立核验280轮'),
    ('本次指标仅截至独立240轮', '本次指标仅截至独立280轮'),
    ('VERIFIED_AT_ROUND240_ACCEPTANCE', 'VERIFIED_AT_ROUND280_ACCEPTANCE'),
    ('latest_observed_running_round=241', 'latest_observed_running_round=281'),
    ('metrics cut off at independent240', 'metrics cut off at independent280'),
    ('len(original_rows) == 1160', 'len(original_rows) == 1200'),
    ("['round_1based']) == 200", "['round_1based']) == 240"),
    ('zip(rows[200:240], refs[200:240])', 'zip(rows[240:280], refs[240:280])'),
    ("[200, 240], 'rounds_included': [201, 240]", "[240, 280], 'rounds_included': [241, 280]"),
    ('Authorized30715 continuation is RUNNING in the immutable003 milestone240 observation at round241. Only accepted and independently audited rounds1-240 are exported; round241 and later are excluded. The stage-local best239 is progress only. Old discarded partial165 work remains historical.',
     'The immutable 004 milestone records round 281. This export contains independently audited rounds 1–280. Round 280 supplies the latest progress and the AP best through that cutoff.'),
    ('Second recovery240 independent certificate retained; observed241 excluded; original1160 CSV bytes retained; latest240 and best239 distinct', 'Round280 certificate and fixed 004 milestone verified. The original 1200 rows retain their exact CSV bytes. Both progress roles refer to round 280.'),
    ("report['rows'] = 1200", "report['rows'] = 1240"),
    ('_build/update_round240_plan.json', '_build/update_round280_plan.json'),
    ("'total_updates': 388080, 'previous_snapshot_rounds': 200", "'total_updates': 452760, 'previous_snapshot_rounds': 240"),
    ("'update_tag': 'round240'", "'update_tag': 'round280'"),
    ('ROUND240_DATA_UPDATED', 'ROUND280_DATA_UPDATED'),
    ("'per_epoch_rows': 1200", "'per_epoch_rows': 1240"),
])
start = update.index('    observer = read(IDENTITY)')
end = update.index('    rows, refs, accepted =', start)
update = update[:start] + update[end:]
start = update.index("    text = (OUT / '汇总说明.md').read_text")
end = update.index("    replace_json(OUT / '_build/update_round280_plan.json'", start)
update = update[:start] + """    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已独立核验至第 240 轮，进度 AP 为 21.0329%',
        'NO_MSCA 的来源快照已独立核验至第 280 轮，进度 AP 为 21.9282%')
    text += '\\n2026-11-10 280轮更新。NO_MSCA 独立证据覆盖第1–280轮，累计452,760次更新。从已接受的epoch0279累计日志追加第241–280轮，共40行。第280轮 AP 为21.9282%，APs 为14.4816%。以第280轮为截止点的 AP 最佳也出现在该轮。最新进度与截至该轮的 AP 最佳各自保留角色，均对应这一评估记录。320轮主端点和次端点待核验。\\n'
    text += '\\n固定现场来源为已结束的004窗口里程碑。该文件于服务器UTC '+live['observed_server_utc']+' 记录同一30715进程在第281轮第1批。第164轮恢复后已核验116轮和187,572次更新。第29和164轮后的两次恢复及旧第165轮至少1,401次弃用更新均有原证据。额外物理工作量仍未知，无中断位级等同缺少证据。沿用单种子、同一验证集选点及非官方VisDrone口径。\\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 1200 行', '共 1240 行').replace('前 240 行', '前 280 行')
    text = text.replace('第201–240行来自epoch0239累计日志，并与独立240轮证书核对。第241轮及之后指标不纳入本快照。',
                        '第201–240行来自epoch0239累计日志，第241–280行来自epoch0279累计日志。独立证书覆盖至第280轮。')
    text += '\\n2026-11-10 280轮更新追加第241–280轮共40行，保留原1200行精确字节前缀和原字段格式。BASE、FULL、NO_MPS各320行，NO_MSCA 280行，合计1240行。14,880项原始指标及对应行、计数和哈希可逐项回溯。第280轮承担最新进度和截至该轮的AP最佳两个角色。\\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
""" + update[end:]
start = update.index("        'progress_notes': [")
end = update.index("        'new_source_sha256':", start)
update = update[:start] + """        'progress_notes': [
            '固定004现场记录第281轮第1批，独立指标覆盖至第280轮。',
            '第280轮 AP 21.9282%，APs 14.4816%。',
            '截至第280轮，AP最佳也出现在该轮。320轮端点待核验。',
            '从第164轮恢复后已核验116轮和187,572次更新。',
            '第29和164轮后两次恢复。旧第165轮至少有1,401次弃用更新。',
            '完整物理工作量未知。单种子、同一验证集选点及非官方 VisDrone 口径。',
        ],
""" + update[end:]
save('update_round280.py', update)

verify = (root / 'verify_round240_update.py').read_text(encoding='utf-8')
verify = substitute(verify, [
    ('round-240 delivery', 'round-280 delivery'),
    ('from update_round240 import', 'from update_round280 import'),
    ("== 240 and progress['optimizer_step_attempts'] == 388080", "== 280 and progress['optimizer_step_attempts'] == 452760"),
    ("best_progress['round_1based'] == 239 and best_progress['epoch_index_0based'] == 238", "best_progress['round_1based'] == 280 and best_progress['epoch_index_0based'] == 279"),
    ('PROGRESS_ONLY_AP_BEST_WITHIN_FIRST240_NOT_A_FINAL_ENDPOINT', 'PROGRESS_AP_BEST_THROUGH_ROUND280'),
    ('0.21032901073722304', '0.21928245309369632'),
    ('0.1393851562602276', '0.14481580486643042'),
    ('0.21033699779236342', '0.21928245309369632'),
    ("assert progress['ema_tensor_sha256'] != best_progress['ema_tensor_sha256']", "assert best_progress['same_epoch_as_latest_progress'] is True\n    assert progress['ema_tensor_sha256'] == best_progress['ema_tensor_sha256']"),
    ("['new_verified_rounds'] == 76", "['new_verified_rounds'] == 116"),
    ("['new_verified_optimizer_updates'] == 122892", "['new_verified_optimizer_updates'] == 187572"),
    ("['round_1based'] == 241", "['round_1based'] == 281"),
    ("['native_accepted_rounds_observed'] == 240", "['native_accepted_rounds_observed'] == 280"),
    ("['independent_metric_snapshot_rounds'] == 240", "['independent_metric_snapshot_rounds'] == 280"),
    ("['cumulative_recovered_rounds_since29'] == 211", "['cumulative_recovered_rounds_since29'] == 251"),
    ("['cumulative_recovered_optimizer_updates_since29'] == 341187", "['cumulative_recovered_optimizer_updates_since29'] == 405867"),
    ("'previous_rounds': 200, 'new_verified_rounds': 40, 'new_verified_updates': 64680", "'previous_rounds': 240, 'new_verified_rounds': 40, 'new_verified_updates': 64680"),
    ("int(row['completed_rounds']) == 240", "int(row['completed_rounds']) == 280"),
    ('VERIFIED_AT_ROUND240_ACCEPTANCE', 'VERIFIED_AT_ROUND280_ACCEPTANCE'),
    ("row['latest_observed_running_round'] == '241'", "row['latest_observed_running_round'] == '281'"),
    ('len(rows) == 1200', 'len(rows) == 1240'),
    ("'NO_MSCA': 240}", "'NO_MSCA': 280}"),
    ('range(201, 241)', 'range(241, 281)'),
    ('NO_MSCA_SECOND_RECOVERY_R240_EPOCH_ROWS', 'NO_MSCA_SECOND_RECOVERY_R280_EPOCH_ROWS'),
    ("['optimizer_step_attempts']) == 388080", "['optimizer_step_attempts']) == 452760"),
    ('max(range(240)', 'max(range(280)'),
    ("float(arm_rows[index]['AP'])) == 238", "float(arm_rows[index]['AP'])) == 279"),
    ('arm_rows[238]', 'arm_rows[279]'),
    ("and 'observer_status.json' not in item['path']", "and 'observer_status.json' not in item['path'] and '_005/' not in item['path'].replace('\\\\', '/')"),
    ('_build/round240_workbook_checks.json', '_build/round280_workbook_checks.json'),
    ("assert '30715' in cell_text('A3') and '241' in cell_text('A3') and '240' in cell_text('A3')", "assert '第281轮' in cell_text('A3') and '第280轮' in cell_text('A3')"),
    ("assert '独立截止240轮' in cell_text('A28')", "assert '004' in cell_text('A28') and '第281轮第1批' in cell_text('A28')"),
    ("'第240轮' in cell_text('A29') and '21.0329%'", "'第280轮' in cell_text('A29') and '21.9282%'"),
    ("assert '第239轮' in cell_text('A30') and '21.0337%' in cell_text('A30') and '非320轮端点' in cell_text('A30')", "assert '第280轮' in cell_text('A30') and 'AP最佳也出现在该轮' in cell_text('A30') and '320轮端点待核验' in cell_text('A30')"),
    ('NO_MSCA_accepted_rounds=240', 'NO_MSCA_accepted_rounds=280'),
    ('prior_round200_verified_snapshot=str(SNAPSHOT)', 'prior_round240_verified_snapshot=str(SNAPSHOT)'),
    ('ROUND240_CERTIFICATE_CUTOFF_UPDATE_VERIFIED', 'ROUND280_CERTIFICATE_CUTOFF_UPDATE_VERIFIED'),
    ("'NO_MSCA_independently_verified_rounds': 240", "'NO_MSCA_independently_verified_rounds': 280"),
    ("'NO_MSCA_cumulative_optimizer_updates': 388080", "'NO_MSCA_cumulative_optimizer_updates': 452760"),
    ("'second_recovery_segment_verified_rounds': 76, 'second_recovery_segment_verified_updates': 122892", "'second_recovery_segment_verified_rounds': 116, 'second_recovery_segment_verified_updates': 187572"),
    ("'per_epoch_csv_rows': 1200, 'raw_metric_values_rechecked': 14400", "'per_epoch_csv_rows': 1240, 'raw_metric_values_rechecked': 14880"),
    ('old_1160_csv_rows_preserved_as_exact_byte_prefix', 'old_1200_csv_rows_preserved_as_exact_byte_prefix'),
    ("'last_native_accepted_round': 240, 'latest_saved_observed_running_round': 241", "'last_native_accepted_round': 280, 'latest_saved_observed_running_round': 281"),
    ('native_backup_verified_at240_acceptance', 'native_backup_verified_at280_acceptance'),
    ('metric_cutoff_remains_round240', 'metric_cutoff_remains_round280'),
    ('第240轮更新核查.json', '第280轮更新核查.json'),
    ('round240_progress_after.png', 'round280_progress_after.png'),
    ('round240_endpoints_full_after.png', 'round280_endpoints_full_after.png'),
    ('round240_sources_full_after.png', 'round280_sources_full_after.png'),
    ("report['latest_round240_and_local_best_round239_distinct'] = True", "report['latest_and_prefix_ap_best_both_round280'] = True"),
    ("manifest['NO_MSCA_accepted_rounds'] = 240", "manifest['NO_MSCA_accepted_rounds'] = 280"),
    ('independent_round240_certificate', 'independent_round280_certificate'),
    ("manifest['previous_round200_verified_snapshot'] = str(SNAPSHOT)", "manifest['previous_round240_verified_snapshot'] = str(SNAPSHOT)"),
    ('len(entries) == 31', 'len(entries) == 36'),
    ("for item in entries}) == 31", "for item in entries}) == 36"),
    ('ROUND240_CURRENT_DELIVERY_BYTE_VERIFIED', 'ROUND280_CURRENT_DELIVERY_BYTE_VERIFIED'),
    ("'per_epoch_csv_rows': 1200", "'per_epoch_csv_rows': 1240"),
    ('original round200 manifest', 'original round240 manifest'),
    ('_build/prior_round200_verification.json', '_build/prior_round240_verification.json'),
    ('_build/round240_snapshot_verification.json', '_build/round280_snapshot_verification.json'),
    ('prior_round200_snapshot_unchanged', 'prior_round240_snapshot_unchanged'),
])
save('verify_round280_update.py', verify)

workbook = (root / 'update_progress_workbook.mjs').read_text(encoding='utf-8')
workbook = substitute(workbook, [
    ('`${localDate}，第${rounds}轮已独立核验。320轮终点保持空缺。`',
     '`${localDate}，指标截止第${rounds}轮，固定现场记录第281轮。320轮终点待核验。`'),
])
save('update_progress_workbook_round280.mjs', workbook)
save('verify_round280_full_renders.mjs', (root / 'verify_round160_full_renders.mjs').read_text(encoding='utf-8').replace('round160', 'round280'))
print('ROUND280_SCRIPTS_PREPARED')
