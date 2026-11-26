"""Create a round160-specific verifier from the preserved round140 verifier."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'verify_round140_update.py').read_text(encoding='utf-8')
replacements = [
    ('round-140 delivery', 'round-160 delivery'),
    ('from update_round140 import OUT, SNAPSHOT, XLSX, LOCAL_DATE',
     'from update_round160 import OUT, SNAPSHOT, XLSX, LOCAL_DATE, FINAL_SNAPSHOT'),
    ("== 140 and progress['optimizer_step_attempts'] == 226380", "== 160 and progress['optimizer_step_attempts'] == 258720"),
    ("['new_verified_rounds'] == 111", "['new_verified_rounds'] == 131"),
    ("['new_verified_optimizer_updates'] == 179487", "['new_verified_optimizer_updates'] == 211827"),
    ("['round_1based'] == 142", "['round_1based'] == 161"),
    ("['native_accepted_rounds_observed'] == 141", "['native_accepted_rounds_observed'] == 160"),
    ("['independent_metric_snapshot_rounds'] == 140", "['independent_metric_snapshot_rounds'] == 160"),
    ("'previous_rounds': 41, 'new_verified_rounds': 99, 'new_verified_updates': 160083",
     "'previous_rounds': 140, 'new_verified_rounds': 20, 'new_verified_updates': 32340"),
    ("int(row['completed_rounds']) == 140", "int(row['completed_rounds']) == 160"),
    ('assert len(rows) == 1100', 'assert len(rows) == 1120'),
    ("'NO_MSCA': 140}", "'NO_MSCA': 160}"),
    ('_build/round140_workbook_checks.json', '_build/round160_workbook_checks.json'),
    ('NO_MSCA_accepted_rounds=140', 'NO_MSCA_accepted_rounds=160'),
    ('prior_round41_snapshot=str(SNAPSHOT)', 'prior_round140_verified_snapshot=str(SNAPSHOT)'),
    ('ROUND140_CERTIFICATE_CUTOFF_UPDATE_VERIFIED', 'ROUND160_CERTIFICATE_CUTOFF_UPDATE_VERIFIED'),
    ("'NO_MSCA_independently_verified_rounds': 140", "'NO_MSCA_independently_verified_rounds': 160"),
    ("'NO_MSCA_cumulative_optimizer_updates': 226380", "'NO_MSCA_cumulative_optimizer_updates': 258720"),
    ("'recovery_segment_verified_rounds': 111, 'recovery_segment_verified_updates': 179487",
     "'recovery_segment_verified_rounds': 131, 'recovery_segment_verified_updates': 211827"),
    ("'added_since_prior_snapshot_rounds': 99, 'added_since_prior_snapshot_updates': 160083",
     "'added_since_prior_snapshot_rounds': 20, 'added_since_prior_snapshot_updates': 32340"),
    ("'per_epoch_csv_rows': 1100, 'raw_metric_values_rechecked': 13200",
     "'per_epoch_csv_rows': 1120, 'raw_metric_values_rechecked': 13440"),
    ('old_1001_csv_rows_preserved_as_exact_byte_prefix', 'old_1100_csv_rows_preserved_as_exact_byte_prefix'),
    ("'live_observed_native_rounds': 141, 'live_observed_running_round': 142",
     "'live_observed_native_rounds': 160, 'live_observed_running_round': 161"),
    ('metric_cutoff_remains_round140', 'metric_cutoff_remains_round160'),
    ('第140轮更新核查.json', '第160轮更新核查.json'),
    ("manifest['NO_MSCA_accepted_rounds'] = 140", "manifest['NO_MSCA_accepted_rounds'] = 160"),
    ('independent_round140_certificate', 'independent_round160_certificate'),
    ("manifest['previous_round41_snapshot'] = str(SNAPSHOT)", "manifest['previous_round140_verified_snapshot'] = str(SNAPSHOT)"),
]
for old, new in replacements:
    assert text.count(old) >= 1, old
    text = text.replace(old, new)
text = text.replace('from pathlib import Path', 'from pathlib import Path, PurePosixPath\nimport shutil')
needle = "    assert wb_check['xlsx_sha256'] == sha(OUT / XLSX)\n"
text = text.replace(needle, needle + "    assert 'matched 0 entries' in wb_check['error_scan']\n")
needle = "        assert row['evaluation_arrays_sha256'] == original['evaluation_arrays_sha256']\n"
text = text.replace(needle, needle + """        assert int(row['last_epoch_state_counter']) == original['last_epoch']
        assert row['weight_source'] == original['weight_source']
        assert row['protocol_sha256'] == original['protocol_sha256']
        assert row['configuration_sha256'] == original['config_sha256']
        assert json.loads(row['first_executed_lrs']) == original['first_executed_lrs']
        assert json.loads(row['post_final_update_lrs']) == original['post_final_update_lrs']
        assert float(row['seconds']) == original['seconds']
        assert row['created_utc'] == original['created_utc']
    assert [int(row['round_1based']) for row in rows[-20:]] == list(range(141, 161))
    assert all(row['source_id'] == 'NO_MSCA_RECOVERED_R160_EPOCH_ROWS' for row in rows[-20:])
    assert rows[-1]['arm'] == 'NO_MSCA' and int(rows[-1]['optimizer_step_attempts']) == 258720
    assert all(item['status'] == 'PASS' for item in read(OUT / '输入核查.json')['checks'])
    assert all(item['status'] == 'PASS' for item in read(OUT / '逐轮核查.json')['checks'])
""")
needle = "    replace_json(OUT / '交付清单.json', manifest)\n"
text = text.replace(needle, "    manifest['verified_snapshot_destination'] = str(FINAL_SNAPSHOT)\n" + needle + """    # Freeze all current delivery bytes, with paths resolvable from this manifest's parent.
    FINAL_SNAPSHOT.mkdir(parents=True, exist_ok=False)
    entries = []
    for name in filenames + ['交付清单.json']:
        relative = PurePosixPath(name.replace('\\\\', '/'))
        assert not relative.is_absolute() and '..' not in relative.parts
        source = OUT.joinpath(*relative.parts)
        target = FINAL_SNAPSHOT.joinpath(*relative.parts)
        assert source.resolve().is_relative_to(OUT.resolve())
        assert target.resolve().is_relative_to(FINAL_SNAPSHOT.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert source.read_bytes() == target.read_bytes()
        entries.append({'file': relative.as_posix(), 'bytes': source.stat().st_size, 'sha256': sha(source)})
    assert len(entries) == 19 and len({item['file'] for item in entries}) == 19
    snapshot_manifest_path = FINAL_SNAPSHOT / 'snapshot_manifest.json'
    replace_json(snapshot_manifest_path, {
        'status': 'ROUND160_CURRENT_DELIVERY_BYTE_VERIFIED', 'local_date': LOCAL_DATE,
        'NO_MSCA_independently_verified_rounds': 160, 'per_epoch_csv_rows': 1120,
        'path_resolution_base': 'snapshot_manifest.json parent directory',
        'source_delivery_manifest_sha256': sha(OUT / '交付清单.json'),
        'prior_verified_snapshot': str(SNAPSHOT), 'files': entries,
    })
    fixed = read(snapshot_manifest_path)
    for entry in fixed['files']:
        relative = PurePosixPath(entry['file'])
        saved_path = snapshot_manifest_path.parent.joinpath(*relative.parts)
        assert saved_path.resolve().is_relative_to(snapshot_manifest_path.parent.resolve())
        assert saved_path.stat().st_size == entry['bytes'] and sha(saved_path) == entry['sha256']
        assert saved_path.read_bytes() == OUT.joinpath(*relative.parts).read_bytes()
    # Re-read the original round140 manifest and files: no prior snapshot/index was modified.
    previous_manifest = read(SNAPSHOT / '交付清单.json')
    initial = read(OUT / '_build/prior_round140_verification.json')
    assert sha(SNAPSHOT / '交付清单.json') == initial['delivery_manifest_sha256']
    for entry in previous_manifest['files']:
        assert sha(SNAPSHOT / entry['file']) == entry['sha256']
    replace_json(OUT / '_build/round160_snapshot_verification.json', {
        'status': 'VERIFIED', 'files': len(entries), 'byte_identical_to_current': True,
        'relative_paths_resolve': True, 'prior_round140_snapshot_unchanged': True,
        'snapshot_manifest': str(snapshot_manifest_path), 'snapshot_manifest_sha256': sha(snapshot_manifest_path),
        'delivery_manifest_sha256': sha(OUT / '交付清单.json'),
        'xlsx_sha256': sha(OUT / XLSX), 'per_epoch_csv_sha256': sha(OUT / '逐轮指标.csv'),
    })
""")
target = root / 'verify_round160_update.py'
with target.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
compile(text, str(target), 'exec')
print('ROUND160_VERIFIER_CREATED_AND_COMPILED')
