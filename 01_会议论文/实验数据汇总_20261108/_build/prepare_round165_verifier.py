"""Create a one-row second-recovery delivery verifier; leave old verification files intact."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'verify_round164_update.py').read_text(encoding='utf-8')
replacements = [
    ('round-164 delivery', 'round-165 delivery'),
    ('from update_round164 import', 'from update_round165 import'),
    ("== 164 and progress['optimizer_step_attempts'] == 265188", "== 165 and progress['optimizer_step_attempts'] == 266805"),
    ("['new_verified_rounds'] == 135", "['new_verified_rounds'] == 1"),
    ("['new_verified_optimizer_updates'] == 218295", "['new_verified_optimizer_updates'] == 1617"),
    ("['round_1based'] == 165", "['round_1based'] == 166"),
    ("['native_accepted_rounds_observed'] == 164", "['native_accepted_rounds_observed'] == 165"),
    ("['independent_metric_snapshot_rounds'] == 164", "['independent_metric_snapshot_rounds'] == 165"),
    ("'previous_rounds': 160, 'new_verified_rounds': 4, 'new_verified_updates': 6468", "'previous_rounds': 164, 'new_verified_rounds': 1, 'new_verified_updates': 1617"),
    ("int(row['completed_rounds']) == 164", "int(row['completed_rounds']) == 165"),
    ("row['remote_training_state_now'] == 'UNKNOWN'", "row['remote_training_state_now'] == 'RUNNING'"),
    ("row['native_backup_client_alive'] == 'False'", "row['native_backup_client_alive'] == 'VERIFIED_AT_ROUND165_ACCEPTANCE'"),
    ('len(rows) == 1124', 'len(rows) == 1125'),
    ("'NO_MSCA': 164}", "'NO_MSCA': 165}"),
    ('rows[-4:]', 'rows[-1:]'),
    ('range(161, 165)', 'range(165, 166)'),
    ('NO_MSCA_RECOVERED_R164_EPOCH_ROWS', 'NO_MSCA_SECOND_RECOVERY_R165_EPOCH_ROWS'),
    ("['optimizer_step_attempts']) == 265188", "['optimizer_step_attempts']) == 266805"),
    ('_build/round164_workbook_checks.json', '_build/round165_workbook_checks.json'),
    ('NO_MSCA_accepted_rounds=164', 'NO_MSCA_accepted_rounds=165'),
    ('prior_round160_verified_snapshot=str(SNAPSHOT)', 'prior_round164_verified_snapshot=str(SNAPSHOT)'),
    ('ROUND164_CERTIFICATE_CUTOFF_UPDATE_VERIFIED', 'ROUND165_CERTIFICATE_CUTOFF_UPDATE_VERIFIED'),
    ("'NO_MSCA_independently_verified_rounds': 164", "'NO_MSCA_independently_verified_rounds': 165"),
    ("'NO_MSCA_cumulative_optimizer_updates': 265188", "'NO_MSCA_cumulative_optimizer_updates': 266805"),
    ("'recovery_segment_verified_rounds': 135, 'recovery_segment_verified_updates': 218295", "'second_recovery_segment_verified_rounds': 1, 'second_recovery_segment_verified_updates': 1617,\n        'first_recovery_preserved_through_round': 164, 'old_partial165_preserved_in_16_file_archive': True"),
    ("'added_since_prior_snapshot_rounds': 4, 'added_since_prior_snapshot_updates': 6468", "'added_since_prior_snapshot_rounds': 1, 'added_since_prior_snapshot_updates': 1617"),
    ("'per_epoch_csv_rows': 1124, 'raw_metric_values_rechecked': 13488", "'per_epoch_csv_rows': 1125, 'raw_metric_values_rechecked': 13500"),
    ('old_1120_csv_rows_preserved_as_exact_byte_prefix', 'old_1124_csv_rows_preserved_as_exact_byte_prefix'),
    ("summary['input_sources'][-3:]", "summary['input_sources'][-7:]"),
    ("'last_native_accepted_round': 164, 'historical_last_observed_running_round': 165,\n        'current_remote_training_state': 'UNKNOWN', 'native_backup_client_alive': False,\n        'historical_running_observation_is_current_liveness': False", "'last_native_accepted_round': 165, 'latest_saved_observed_running_round': 166,\n        'current_remote_training_state': 'RUNNING', 'native_backup_verified_at165_acceptance': True,\n        'old_connectivity_unknown_record_preserved_as_history': True"),
    ('metric_cutoff_remains_round164', 'metric_cutoff_remains_round165'),
    ('第164轮更新核查.json', '第165轮更新核查.json'),
    ("manifest['NO_MSCA_accepted_rounds'] = 164", "manifest['NO_MSCA_accepted_rounds'] = 165"),
    ('independent_round164_certificate', 'independent_round165_certificate'),
    ("manifest['previous_round160_verified_snapshot'] = str(SNAPSHOT)", "manifest['previous_round164_verified_snapshot'] = str(SNAPSHOT)"),
    ('len(entries) == 20', 'len(entries) == 21'),
    ("for item in entries}) == 20", "for item in entries}) == 21"),
    ('ROUND164_CURRENT_DELIVERY_BYTE_VERIFIED', 'ROUND165_CURRENT_DELIVERY_BYTE_VERIFIED'),
    ("'per_epoch_csv_rows': 1124", "'per_epoch_csv_rows': 1125"),
    ('original round160 manifest', 'original round164 manifest'),
    ('_build/prior_round160_verification.json', '_build/prior_round164_verification.json'),
    ('_build/round164_snapshot_verification.json', '_build/round165_snapshot_verification.json'),
    ('prior_round160_snapshot_unchanged', 'prior_round164_snapshot_unchanged'),
]
for old, new in replacements:
    assert text.count(old) >= 1, old
    text = text.replace(old, new)
start = text.index("    assert progress['current_remote_training_state']")
end = text.index("    assert progress['update_since_previous_snapshot']", start)
text = text[:start] + """    assert progress['current_remote_training_state'] == 'RUNNING'
    assert progress['live_observation']['observation_kind'] == 'HASH_BOUND_SAVED_RUNNING_OBSERVATION'
    assert progress['live_observation']['worker_pid'] == 413 and progress['live_observation']['supervisor_pid'] == 412
    assert progress['connectivity']['remote_ssh_port'] == 30715
    assert progress['connectivity']['native_backup_client_verified_at_acceptance'] is True
    assert progress['connectivity']['native_backup_client_pid'] == 8216
    assert progress['connectivity_history'][-1] == before['NO_MSCA_progress']['connectivity']
    assert progress['observation_history'][-1] == before['NO_MSCA_progress']['live_observation']
    historical = dict(progress['recovery_history'][-1])
    historical.pop('scope')
    assert historical == before['NO_MSCA_progress']['recovery']
    assert progress['recovery']['prefix']['full_saved_state_including_all_rng_equal'] is True
    assert progress['recovery']['archive']['archived_files'] == 16
    assert progress['recovery']['binding_sha256'] == '6de82b5b4bcbf8bbdbd3a61a2fb7ecbd70905b98da0eceefc55c7b451435a41e'
    assert progress['recovery']['cumulative_recovered_rounds_since29'] == 136
    assert progress['recovery']['cumulative_recovered_optimizer_updates_since29'] == 219912
""" + text[end:]
text = text.replace("        assert row['historical_last_observed_running_round'] == '165'\n", "        assert row['historical_last_observed_running_round'] == '165' and row['latest_observed_running_round'] == '166'\n")
start = text.index("        assert 'UNKNOWN' in cell_text('A3')")
end = text.index("        for row_num, metric", start)
text = text[:start] + """        assert '30715' in cell_text('A3') and '166' in cell_text('A3')
        assert '续训在运行' in cell_text('A28') and '165轮' in cell_text('A28')
        assert '16文件历史ZIP' in cell_text('A31')
""" + text[end:]
target = root / 'verify_round165_update.py'
compile(text, str(target), 'exec')
with target.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
print('ROUND165_VERIFIER_CREATED_AND_COMPILED')
