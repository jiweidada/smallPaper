"""Create the bounded240 data/worksheet/render/snapshot verifier."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'verify_round200_update.py').read_text(encoding='utf-8')
replacements = [
    ('round-200 delivery', 'round-240 delivery'),
    ('from update_round200 import', 'from update_round240 import'),
    ("== 200 and progress['optimizer_step_attempts'] == 323400", "== 240 and progress['optimizer_step_attempts'] == 388080"),
    ("best_progress['round_1based'] == 199 and best_progress['epoch_index_0based'] == 198", "best_progress['round_1based'] == 239 and best_progress['epoch_index_0based'] == 238"),
    ('PROGRESS_ONLY_AP_BEST_WITHIN_FIRST200_NOT_A_FINAL_ENDPOINT', 'PROGRESS_ONLY_AP_BEST_WITHIN_FIRST240_NOT_A_FINAL_ENDPOINT'),
    ('0.19650041568758386', '0.21032901073722304'),
    ('0.13120383278867362', '0.1393851562602276'),
    ('0.19665533181774233', '0.21033699779236342'),
    ("['new_verified_rounds'] == 36", "['new_verified_rounds'] == 76"),
    ("['new_verified_optimizer_updates'] == 58212", "['new_verified_optimizer_updates'] == 122892"),
    ("['round_1based'] == 201", "['round_1based'] == 241"),
    ("['native_accepted_rounds_observed'] == 200", "['native_accepted_rounds_observed'] == 240"),
    ("['independent_metric_snapshot_rounds'] == 200", "['independent_metric_snapshot_rounds'] == 240"),
    ("['cumulative_recovered_rounds_since29'] == 171", "['cumulative_recovered_rounds_since29'] == 211"),
    ("['cumulative_recovered_optimizer_updates_since29'] == 276507", "['cumulative_recovered_optimizer_updates_since29'] == 341187"),
    ("'previous_rounds': 165, 'new_verified_rounds': 35, 'new_verified_updates': 56595", "'previous_rounds': 200, 'new_verified_rounds': 40, 'new_verified_updates': 64680"),
    ("int(row['completed_rounds']) == 200", "int(row['completed_rounds']) == 240"),
    ('VERIFIED_AT_ROUND200_ACCEPTANCE', 'VERIFIED_AT_ROUND240_ACCEPTANCE'),
    ("row['latest_observed_running_round'] == '201'", "row['latest_observed_running_round'] == '241'"),
    ('len(rows) == 1160', 'len(rows) == 1200'),
    ("'NO_MSCA': 200}", "'NO_MSCA': 240}"),
    ('rows[-35:]', 'rows[-40:]'),
    ('range(166, 201)', 'range(201, 241)'),
    ('NO_MSCA_SECOND_RECOVERY_R200_EPOCH_ROWS', 'NO_MSCA_SECOND_RECOVERY_R240_EPOCH_ROWS'),
    ("['optimizer_step_attempts']) == 323400", "['optimizer_step_attempts']) == 388080"),
    ('max(range(200)', 'max(range(240)'),
    ("float(arm_rows[index]['AP'])) == 198", "float(arm_rows[index]['AP'])) == 238"),
    ('arm_rows[198]', 'arm_rows[238]'),
    ("summary['input_sources'][-3:]", "summary['input_sources'][-2:]"),
    ('_build/round200_workbook_checks.json', '_build/round240_workbook_checks.json'),
    ("'201' in cell_text('A3') and '200' in cell_text('A3')", "'241' in cell_text('A3') and '240' in cell_text('A3')"),
    ('独立截止200轮', '独立截止240轮'),
    ("'第200轮' in cell_text('A29') and '19.6500%'", "'第240轮' in cell_text('A29') and '21.0329%'"),
    ("'第199轮' in cell_text('A30') and '19.6655%'", "'第239轮' in cell_text('A30') and '21.0337%'"),
    ('NO_MSCA_accepted_rounds=200', 'NO_MSCA_accepted_rounds=240'),
    ('prior_round165_verified_snapshot=str(SNAPSHOT)', 'prior_round200_verified_snapshot=str(SNAPSHOT)'),
    ('ROUND200_CERTIFICATE_CUTOFF_UPDATE_VERIFIED', 'ROUND240_CERTIFICATE_CUTOFF_UPDATE_VERIFIED'),
    ("'NO_MSCA_independently_verified_rounds': 200", "'NO_MSCA_independently_verified_rounds': 240"),
    ("'NO_MSCA_cumulative_optimizer_updates': 323400", "'NO_MSCA_cumulative_optimizer_updates': 388080"),
    ("'second_recovery_segment_verified_rounds': 36, 'second_recovery_segment_verified_updates': 58212", "'second_recovery_segment_verified_rounds': 76, 'second_recovery_segment_verified_updates': 122892"),
    ("'added_since_prior_snapshot_rounds': 35, 'added_since_prior_snapshot_updates': 56595", "'added_since_prior_snapshot_rounds': 40, 'added_since_prior_snapshot_updates': 64680"),
    ("'per_epoch_csv_rows': 1160, 'raw_metric_values_rechecked': 13920", "'per_epoch_csv_rows': 1200, 'raw_metric_values_rechecked': 14400"),
    ('old_1125_csv_rows_preserved_as_exact_byte_prefix', 'old_1160_csv_rows_preserved_as_exact_byte_prefix'),
    ("'last_native_accepted_round': 200, 'latest_saved_observed_running_round': 201", "'last_native_accepted_round': 240, 'latest_saved_observed_running_round': 241"),
    ('native_backup_verified_at200_acceptance', 'native_backup_verified_at240_acceptance'),
    ('metric_cutoff_remains_round200', 'metric_cutoff_remains_round240'),
    ('第200轮更新核查.json', '第240轮更新核查.json'),
    ('round200_progress_after.png', 'round240_progress_after.png'),
    ('round200_endpoints_full_after.png', 'round240_endpoints_full_after.png'),
    ('round200_sources_full_after.png', 'round240_sources_full_after.png'),
    ('latest_round200_and_local_best_round199_distinct', 'latest_round240_and_local_best_round239_distinct'),
    ("manifest['NO_MSCA_accepted_rounds'] = 200", "manifest['NO_MSCA_accepted_rounds'] = 240"),
    ('independent_round200_certificate', 'independent_round240_certificate'),
    ("manifest['previous_round165_verified_snapshot'] = str(SNAPSHOT)", "manifest['previous_round200_verified_snapshot'] = str(SNAPSHOT)"),
    ('len(entries) == 26', 'len(entries) == 31'),
    ("for item in entries}) == 26", "for item in entries}) == 31"),
    ('ROUND200_CURRENT_DELIVERY_BYTE_VERIFIED', 'ROUND240_CURRENT_DELIVERY_BYTE_VERIFIED'),
    ("'per_epoch_csv_rows': 1160", "'per_epoch_csv_rows': 1200"),
    ('original round165 manifest', 'original round200 manifest'),
    ('_build/prior_round165_verification.json', '_build/prior_round200_verification.json'),
    ('_build/round200_snapshot_verification.json', '_build/round240_snapshot_verification.json'),
    ('prior_round165_snapshot_unchanged', 'prior_round200_snapshot_unchanged'),
]
for old, new in replacements:
    assert text.count(old) >= 1, old
    text = text.replace(old, new)
target = root / 'verify_round240_update.py'
compile(text, str(target), 'exec')
with target.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
print('ROUND240_VERIFIER_CREATED_AND_COMPILED')
