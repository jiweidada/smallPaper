"""Prepare complete200 data, cache, render and byte-identical snapshot checks."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'verify_round165_update.py').read_text(encoding='utf-8')
replacements = [
    ('round-165 delivery', 'round-200 delivery'),
    ('from update_round165 import', 'from update_round200 import'),
    ("== 165 and progress['optimizer_step_attempts'] == 266805", "== 200 and progress['optimizer_step_attempts'] == 323400"),
    ("['new_verified_rounds'] == 1", "['new_verified_rounds'] == 36"),
    ("['new_verified_optimizer_updates'] == 1617", "['new_verified_optimizer_updates'] == 58212"),
    ("['round_1based'] == 166", "['round_1based'] == 201"),
    ("['native_accepted_rounds_observed'] == 165", "['native_accepted_rounds_observed'] == 200"),
    ("['independent_metric_snapshot_rounds'] == 165", "['independent_metric_snapshot_rounds'] == 200"),
    ("'previous_rounds': 164, 'new_verified_rounds': 1, 'new_verified_updates': 1617", "'previous_rounds': 165, 'new_verified_rounds': 35, 'new_verified_updates': 56595"),
    ("int(row['completed_rounds']) == 165", "int(row['completed_rounds']) == 200"),
    ('VERIFIED_AT_ROUND165_ACCEPTANCE', 'VERIFIED_AT_ROUND200_ACCEPTANCE'),
    ("row['latest_observed_running_round'] == '166'", "row['latest_observed_running_round'] == '201'"),
    ('len(rows) == 1125', 'len(rows) == 1160'),
    ("'NO_MSCA': 165}", "'NO_MSCA': 200}"),
    ('rows[-1:]', 'rows[-35:]'),
    ('range(165, 166)', 'range(166, 201)'),
    ('NO_MSCA_SECOND_RECOVERY_R165_EPOCH_ROWS', 'NO_MSCA_SECOND_RECOVERY_R200_EPOCH_ROWS'),
    ("['optimizer_step_attempts']) == 266805", "['optimizer_step_attempts']) == 323400"),
    ('_build/round165_workbook_checks.json', '_build/round200_workbook_checks.json'),
    ('NO_MSCA_accepted_rounds=165', 'NO_MSCA_accepted_rounds=200'),
    ('prior_round164_verified_snapshot=str(SNAPSHOT)', 'prior_round165_verified_snapshot=str(SNAPSHOT)'),
    ('ROUND165_CERTIFICATE_CUTOFF_UPDATE_VERIFIED', 'ROUND200_CERTIFICATE_CUTOFF_UPDATE_VERIFIED'),
    ("'NO_MSCA_independently_verified_rounds': 165", "'NO_MSCA_independently_verified_rounds': 200"),
    ("'NO_MSCA_cumulative_optimizer_updates': 266805", "'NO_MSCA_cumulative_optimizer_updates': 323400"),
    ("'second_recovery_segment_verified_rounds': 1, 'second_recovery_segment_verified_updates': 1617", "'second_recovery_segment_verified_rounds': 36, 'second_recovery_segment_verified_updates': 58212"),
    ("'added_since_prior_snapshot_rounds': 1, 'added_since_prior_snapshot_updates': 1617", "'added_since_prior_snapshot_rounds': 35, 'added_since_prior_snapshot_updates': 56595"),
    ("'per_epoch_csv_rows': 1125, 'raw_metric_values_rechecked': 13500", "'per_epoch_csv_rows': 1160, 'raw_metric_values_rechecked': 13920"),
    ('old_1124_csv_rows_preserved_as_exact_byte_prefix', 'old_1125_csv_rows_preserved_as_exact_byte_prefix'),
    ("summary['input_sources'][-7:]", "summary['input_sources'][-3:]"),
    ("'last_native_accepted_round': 165, 'latest_saved_observed_running_round': 166", "'last_native_accepted_round': 200, 'latest_saved_observed_running_round': 201"),
    ('native_backup_verified_at165_acceptance', 'native_backup_verified_at200_acceptance'),
    ('metric_cutoff_remains_round165', 'metric_cutoff_remains_round200'),
    ('第165轮更新核查.json', '第200轮更新核查.json'),
    ("manifest['NO_MSCA_accepted_rounds'] = 165", "manifest['NO_MSCA_accepted_rounds'] = 200"),
    ('independent_round165_certificate', 'independent_round200_certificate'),
    ("manifest['previous_round164_verified_snapshot'] = str(SNAPSHOT)", "manifest['previous_round165_verified_snapshot'] = str(SNAPSHOT)"),
    ('len(entries) == 21', 'len(entries) == 26'),
    ("for item in entries}) == 21", "for item in entries}) == 26"),
    ('ROUND165_CURRENT_DELIVERY_BYTE_VERIFIED', 'ROUND200_CURRENT_DELIVERY_BYTE_VERIFIED'),
    ("'per_epoch_csv_rows': 1125", "'per_epoch_csv_rows': 1160"),
    ('original round164 manifest', 'original round165 manifest'),
    ('_build/prior_round164_verification.json', '_build/prior_round165_verification.json'),
    ('_build/round165_snapshot_verification.json', '_build/round200_snapshot_verification.json'),
    ('prior_round164_snapshot_unchanged', 'prior_round165_snapshot_unchanged'),
]
for old, new in replacements:
    assert text.count(old) >= 1, old
    text = text.replace(old, new)
text = text.replace("== 'HASH_BOUND_SAVED_RUNNING_OBSERVATION'", "== 'HASH_BOUND_IMMUTABLE_MILESTONE_OBSERVATION'")
start = text.index("    historical = dict(progress['recovery_history'][-1])")
end = text.index("    assert progress['recovery']['prefix']", start)
text = text[:start] + "    assert progress['recovery_history'] == before['NO_MSCA_progress']['recovery_history']\n" + text[end:]
text = text.replace("['cumulative_recovered_rounds_since29'] == 136", "['cumulative_recovered_rounds_since29'] == 171")
text = text.replace("['cumulative_recovered_optimizer_updates_since29'] == 219912", "['cumulative_recovered_optimizer_updates_since29'] == 276507")
needle = "    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None\n"
text = text.replace(needle, needle + """    best_progress = progress['best_so_far_progress']
    assert best_progress['round_1based'] == 199 and best_progress['epoch_index_0based'] == 198
    assert best_progress['role'] == 'PROGRESS_ONLY_AP_BEST_WITHIN_FIRST200_NOT_A_FINAL_ENDPOINT'
    assert best_progress['metrics_fraction']['AP'] == 0.19665533181774233
    assert progress['latest_metrics_fraction']['AP'] == 0.19650041568758386
    assert progress['latest_metrics_fraction']['APs'] == 0.13120383278867362
    assert progress['ema_tensor_sha256'] != best_progress['ema_tensor_sha256']
""")
needle = "    assert rows[-1]['arm'] == 'NO_MSCA' and int(rows[-1]['optimizer_step_attempts']) == 323400\n"
text = text.replace(needle, needle + """    arm_rows = [row for row in rows if row['arm'] == 'NO_MSCA']
    assert max(range(200), key=lambda index: float(arm_rows[index]['AP'])) == 198
    assert float(arm_rows[198]['AP']) == best_progress['metrics_fraction']['AP']
    assert arm_rows[198]['evaluated_ema_tensor_sha256'] == best_progress['ema_tensor_sha256']
    assert arm_rows[-1]['prediction_sha256'] == progress['latest_prediction_sha256']
    assert arm_rows[-1]['evaluation_arrays_sha256'] == progress['latest_evaluation_arrays_sha256']
    assert all('latest_live.json' not in item['path'] and 'observer_status.json' not in item['path']
               for item in summary['input_sources'][-3:])
""")
start = text.index("        assert '30715' in cell_text('A3')")
end = text.index("        for row_num, metric", start)
text = text[:start] + """        assert '30715' in cell_text('A3') and '201' in cell_text('A3') and '200' in cell_text('A3')
        assert '独立截止200轮' in cell_text('A28')
        assert '第200轮' in cell_text('A29') and '19.6500%' in cell_text('A29')
        assert '第199轮' in cell_text('A30') and '19.6655%' in cell_text('A30') and '非320轮端点' in cell_text('A30')
""" + text[end:]
needle = "    replace_json(OUT / '第200轮更新核查.json', report)\n"
text = text.replace(needle, """    visual_files = ['_build/round200_workbook_checks.json',
                    '_build/previews/round200_progress_after.png',
                    '_build/previews/round200_endpoints_full_after.png',
                    '_build/previews/round200_sources_full_after.png']
    report['visual_and_workbook_verification_files'] = [
        {'file': name, 'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)} for name in visual_files]
    report['latest_round200_and_local_best_round199_distinct'] = True
    report['local_best_progress_only_not_final_endpoint'] = True
    report['mutable_observer_latest_or_status_used_as_permanent_source'] = False
""" + needle)
text = text.replace("filenames = [item['file'] for item in manifest['files']] + ['第200轮更新核查.json']",
                    "filenames = [item['file'] for item in manifest['files']] + ['第200轮更新核查.json'] + visual_files")
target = root / 'verify_round200_update.py'
compile(text, str(target), 'exec')
with target.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
print('ROUND200_VERIFIER_CREATED_AND_COMPILED')
