"""Preserve prior verification scripts and derive the bounded round164 checks."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'verify_round160_update.py').read_text(encoding='utf-8')
replacements = [
    ('round-160 delivery', 'round-164 delivery'),
    ('from update_round160 import', 'from update_round164 import'),
    ("== 160 and progress['optimizer_step_attempts'] == 258720", "== 164 and progress['optimizer_step_attempts'] == 265188"),
    ("['new_verified_rounds'] == 131", "['new_verified_rounds'] == 135"),
    ("['new_verified_optimizer_updates'] == 211827", "['new_verified_optimizer_updates'] == 218295"),
    ("['round_1based'] == 161", "['round_1based'] == 165"),
    ("['native_accepted_rounds_observed'] == 160", "['native_accepted_rounds_observed'] == 164"),
    ("['independent_metric_snapshot_rounds'] == 160", "['independent_metric_snapshot_rounds'] == 164"),
    ("'previous_rounds': 140, 'new_verified_rounds': 20, 'new_verified_updates': 32340", "'previous_rounds': 160, 'new_verified_rounds': 4, 'new_verified_updates': 6468"),
    ("int(row['completed_rounds']) == 160", "int(row['completed_rounds']) == 164"),
    ('len(rows) == 1120', 'len(rows) == 1124'),
    ("'NO_MSCA': 160}", "'NO_MSCA': 164}"),
    ('rows[-20:]', 'rows[-4:]'),
    ('range(141, 161)', 'range(161, 165)'),
    ('NO_MSCA_RECOVERED_R160_EPOCH_ROWS', 'NO_MSCA_RECOVERED_R164_EPOCH_ROWS'),
    ("['optimizer_step_attempts']) == 258720", "['optimizer_step_attempts']) == 265188"),
    ('_build/round160_workbook_checks.json', '_build/round164_workbook_checks.json'),
    ('NO_MSCA_accepted_rounds=160', 'NO_MSCA_accepted_rounds=164'),
    ('prior_round140_verified_snapshot=str(SNAPSHOT)', 'prior_round160_verified_snapshot=str(SNAPSHOT)'),
    ('ROUND160_CERTIFICATE_CUTOFF_UPDATE_VERIFIED', 'ROUND164_CERTIFICATE_CUTOFF_UPDATE_VERIFIED'),
    ("'NO_MSCA_independently_verified_rounds': 160", "'NO_MSCA_independently_verified_rounds': 164"),
    ("'NO_MSCA_cumulative_optimizer_updates': 258720", "'NO_MSCA_cumulative_optimizer_updates': 265188"),
    ("'recovery_segment_verified_rounds': 131, 'recovery_segment_verified_updates': 211827", "'recovery_segment_verified_rounds': 135, 'recovery_segment_verified_updates': 218295"),
    ("'added_since_prior_snapshot_rounds': 20, 'added_since_prior_snapshot_updates': 32340", "'added_since_prior_snapshot_rounds': 4, 'added_since_prior_snapshot_updates': 6468"),
    ("'per_epoch_csv_rows': 1120, 'raw_metric_values_rechecked': 13440", "'per_epoch_csv_rows': 1124, 'raw_metric_values_rechecked': 13488"),
    ('old_1100_csv_rows_preserved_as_exact_byte_prefix', 'old_1120_csv_rows_preserved_as_exact_byte_prefix'),
    ("summary['input_sources'][-2:]", "summary['input_sources'][-3:]"),
    ("'live_observed_native_rounds': 160, 'live_observed_running_round': 161", "'last_native_accepted_round': 164, 'historical_last_observed_running_round': 165,\n        'current_remote_training_state': 'UNKNOWN', 'native_backup_client_alive': False,\n        'historical_running_observation_is_current_liveness': False"),
    ('metric_cutoff_remains_round160', 'metric_cutoff_remains_round164'),
    ('第160轮更新核查.json', '第164轮更新核查.json'),
    ("manifest['NO_MSCA_accepted_rounds'] = 160", "manifest['NO_MSCA_accepted_rounds'] = 164"),
    ('independent_round160_certificate', 'independent_round164_certificate'),
    ("manifest['previous_round140_verified_snapshot'] = str(SNAPSHOT)", "manifest['previous_round160_verified_snapshot'] = str(SNAPSHOT)"),
    ('len(entries) == 19', 'len(entries) == 20'),
    ("for item in entries}) == 19", "for item in entries}) == 20"),
    ('ROUND160_CURRENT_DELIVERY_BYTE_VERIFIED', 'ROUND164_CURRENT_DELIVERY_BYTE_VERIFIED'),
    ("'per_epoch_csv_rows': 1120", "'per_epoch_csv_rows': 1124"),
    ('original round140 manifest', 'original round160 manifest'),
    ('_build/prior_round140_verification.json', '_build/prior_round160_verification.json'),
    ('_build/round160_snapshot_verification.json', '_build/round164_snapshot_verification.json'),
    ('prior_round140_snapshot_unchanged', 'prior_round160_snapshot_unchanged'),
]
for old, new in replacements:
    assert text.count(old) >= 1, old
    text = text.replace(old, new)
needle = "    assert progress['live_observation']['independent_metric_snapshot_rounds'] == 164\n"
text = text.replace(needle, needle + """    assert progress['current_remote_training_state'] == 'UNKNOWN'
    assert progress['live_observation']['observation_kind'] == 'HISTORICAL_LAST_SUCCESSFUL_READ'
    assert progress['live_observation']['current_remote_training_state'] == 'UNKNOWN'
    assert progress['connectivity']['remote_training_state_now'] == 'UNKNOWN'
    assert progress['connectivity']['native_backup_client_alive'] is False
    assert progress['connectivity']['read_only_observer_alive'] is False
    assert progress['connectivity']['access_failure_proves_terminal_training'] is False
    assert progress['connectivity']['last_native_accepted_round'] == 164
""")
needle = "        assert int(row['completed_rounds']) == 164\n"
text = text.replace(needle, needle + """        assert row['remote_training_state_now'] == 'UNKNOWN'
        assert row['native_backup_client_alive'] == 'False'
        assert row['historical_last_observed_running_round'] == '165'
""")
needle = "    assert 'matched 0 entries' in wb_check['error_scan']\n"
text = text.replace(needle, needle + """    assert len(wb_check['full_protected_render_comparison']) == 2
    assert all(item['before_sha256'] == item['after_sha256'] and item['identical']
               for item in wb_check['full_protected_render_comparison'])
""")
needle = "        cells = {cell.attrib['r']: cell for cell in partial.findall('.//x:c', ns)}\n"
text = text.replace(needle, needle + """        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        def cell_text(address):
            cell = cells[address]
            if cell.attrib.get('t') == 's':
                return shared[int(cell.find('x:v', ns).text)]
            if cell.attrib.get('t') == 'str':
                return cell.find('x:v', ns).text or ''
            return ''.join(node.text or '' for node in cell.findall('.//x:t', ns))
        assert 'UNKNOWN' in cell_text('A3')
        assert '已退出' in cell_text('A30') and 'UNKNOWN' in cell_text('A30')
        assert '失联前最后见第165轮' in cell_text('A31')
""")
needle = "    assert sha(SNAPSHOT / '交付清单.json') == initial['delivery_manifest_sha256']\n"
text = text.replace(needle, needle + "    assert sha(SNAPSHOT / 'snapshot_manifest.json') == initial['snapshot_manifest_sha256']\n")
target = root / 'verify_round164_update.py'
compile(text, str(target), 'exec')
with target.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
render = (root / 'verify_round160_full_renders.mjs').read_text(encoding='utf-8').replace('round160', 'round164')
with (root / 'verify_round164_full_renders.mjs').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(render)
print('ROUND164_VERIFIER_AND_FULL_RENDER_CHECK_PREPARED')
