"""Verify the certificate-cutoff round-240 delivery without evaluating any model."""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import xml.etree.ElementTree as ET
import zipfile

from update_round240 import OUT, SNAPSHOT, XLSX, LOCAL_DATE, FINAL_SNAPSHOT
from update_round30 import read, replace_json, sha


def main():
    summary, before = read(OUT / '实验汇总.json'), read(SNAPSHOT / '实验汇总.json')
    assert summary['completed_endpoints'] == before['completed_endpoints']
    assert summary['differences'] == before['differences']
    assert summary['recipe_and_evaluation'] == before['recipe_and_evaluation']
    assert summary['input_sources'][:len(before['input_sources'])] == before['input_sources']
    protected_files = ('终点指标.csv', '终点信息.csv', 'FULL差值.csv')
    for name in protected_files:
        assert sha(OUT / name) == sha(SNAPSHOT / name)
    assert (OUT / '输入来源.csv').read_bytes().startswith((SNAPSHOT / '输入来源.csv').read_bytes())
    for item in summary['input_sources']:
        assert sha(Path(item['path'])) == item['sha256']
    progress = summary['NO_MSCA_progress']
    assert progress['completed_rounds'] == 240 and progress['optimizer_step_attempts'] == 388080
    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None
    best_progress = progress['best_so_far_progress']
    assert best_progress['round_1based'] == 239 and best_progress['epoch_index_0based'] == 238
    assert best_progress['role'] == 'PROGRESS_ONLY_AP_BEST_WITHIN_FIRST240_NOT_A_FINAL_ENDPOINT'
    assert best_progress['metrics_fraction']['AP'] == 0.21033699779236342
    assert progress['latest_metrics_fraction']['AP'] == 0.21032901073722304
    assert progress['latest_metrics_fraction']['APs'] == 0.1393851562602276
    assert progress['ema_tensor_sha256'] != best_progress['ema_tensor_sha256']
    assert progress['recovery']['new_verified_rounds'] == 76
    assert progress['recovery']['new_verified_optimizer_updates'] == 122892
    assert progress['live_observation']['round_1based'] == 241
    assert progress['live_observation']['native_accepted_rounds_observed'] == 240
    assert progress['live_observation']['independent_metric_snapshot_rounds'] == 240
    assert progress['current_remote_training_state'] == 'RUNNING'
    assert progress['live_observation']['observation_kind'] == 'HASH_BOUND_IMMUTABLE_MILESTONE_OBSERVATION'
    assert progress['live_observation']['worker_pid'] == 413 and progress['live_observation']['supervisor_pid'] == 412
    assert progress['connectivity']['remote_ssh_port'] == 30715
    assert progress['connectivity']['native_backup_client_verified_at_acceptance'] is True
    assert progress['connectivity']['native_backup_client_pid'] == 8216
    assert progress['connectivity_history'][-1] == before['NO_MSCA_progress']['connectivity']
    assert progress['observation_history'][-1] == before['NO_MSCA_progress']['live_observation']
    assert progress['recovery_history'] == before['NO_MSCA_progress']['recovery_history']
    assert progress['recovery']['prefix']['full_saved_state_including_all_rng_equal'] is True
    assert progress['recovery']['archive']['archived_files'] == 16
    assert progress['recovery']['binding_sha256'] == '6de82b5b4bcbf8bbdbd3a61a2fb7ecbd70905b98da0eceefc55c7b451435a41e'
    assert progress['recovery']['cumulative_recovered_rounds_since29'] == 211
    assert progress['recovery']['cumulative_recovered_optimizer_updates_since29'] == 341187
    assert progress['update_since_previous_snapshot'] == {
        'previous_rounds': 200, 'new_verified_rounds': 40, 'new_verified_updates': 64680}
    with (OUT / 'NO_MSCA进度.csv').open(encoding='utf-8-sig', newline='') as stream:
        progress_rows = list(csv.DictReader(stream))
    assert len(progress_rows) == 12
    for row in progress_rows:
        assert int(row['completed_rounds']) == 240
        assert row['remote_training_state_now'] == 'RUNNING'
        assert row['native_backup_client_alive'] == 'VERIFIED_AT_ROUND240_ACCEPTANCE'
        assert row['historical_last_observed_running_round'] == '165' and row['latest_observed_running_round'] == '241'
        assert float(row['latest_value_fraction']) == progress['latest_metrics_fraction'][row['metric']]
        assert row['round320_primary_value'] == row['round320_secondary_value'] == ''
    with (OUT / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1200
    assert dict(Counter(row['arm'] for row in rows)) == {'BASE': 320, 'FULL': 320, 'NO_MPS': 320, 'NO_MSCA': 240}
    assert (OUT / '逐轮指标.csv').read_bytes().startswith((SNAPSHOT / '逐轮指标.csv').read_bytes())
    mapping = read(OUT / '逐轮来源与切片.json')
    sources, cache, seen = {item['source_id']: item for item in mapping['sources']}, {}, set()
    for row in rows:
        key = row['arm'], int(row['actual_epoch_index_0based'])
        assert key not in seen
        seen.add(key)
        source_id = row['source_id']
        if source_id not in cache:
            payload = Path(row['source_path']).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == row['source_sha256'] == sources[source_id]['sha256']
            cache[source_id] = payload
        payload = cache[source_id]
        if row['source_locator'] == 'JSON root':
            raw = payload
            original = json.loads(payload)
            assert original['checkpoint_sha256'] == row['checkpoint_sha256_present_in_original_source']
            original = {key: value for key, value in original.items() if key != 'checkpoint_sha256'}
        else:
            index = int(row['source_locator'].split()[1]) - 1
            raw = payload.splitlines()[index]
            original = json.loads(raw)
        assert hashlib.sha256(raw).hexdigest() == row['source_raw_record_sha256']
        canonical = json.dumps(original, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                               allow_nan=False).encode('utf-8')
        assert hashlib.sha256(canonical).hexdigest() == row['normalized_row_sha256']
        assert int(row['round_1based']) == original['epoch'] + 1
        assert int(row['optimizer_step_attempts']) == original['optimizer_step_attempts']
        assert int(row['ema_updates']) == original['ema_updates']
        assert float(row['train_loss_mean']) == original['train_loss_mean']
        for metric in summary['metric_order']:
            assert float(row[metric]) == original['metrics'][metric]
        assert row['evaluated_ema_tensor_sha256'] == original['ema_tensor_sha256']
        assert row['prediction_sha256'] == original['prediction_sha256']
        assert row['evaluation_arrays_sha256'] == original['evaluation_arrays_sha256']
        assert int(row['last_epoch_state_counter']) == original['last_epoch']
        assert row['weight_source'] == original['weight_source']
        assert row['protocol_sha256'] == original['protocol_sha256']
        assert row['configuration_sha256'] == original['config_sha256']
        assert json.loads(row['first_executed_lrs']) == original['first_executed_lrs']
        assert json.loads(row['post_final_update_lrs']) == original['post_final_update_lrs']
        assert float(row['seconds']) == original['seconds']
        assert row['created_utc'] == original['created_utc']
    assert [int(row['round_1based']) for row in rows[-40:]] == list(range(201, 241))
    assert all(row['source_id'] == 'NO_MSCA_SECOND_RECOVERY_R240_EPOCH_ROWS' for row in rows[-40:])
    assert rows[-1]['arm'] == 'NO_MSCA' and int(rows[-1]['optimizer_step_attempts']) == 388080
    arm_rows = [row for row in rows if row['arm'] == 'NO_MSCA']
    assert max(range(240), key=lambda index: float(arm_rows[index]['AP'])) == 238
    assert float(arm_rows[238]['AP']) == best_progress['metrics_fraction']['AP']
    assert arm_rows[238]['evaluated_ema_tensor_sha256'] == best_progress['ema_tensor_sha256']
    assert arm_rows[-1]['prediction_sha256'] == progress['latest_prediction_sha256']
    assert arm_rows[-1]['evaluation_arrays_sha256'] == progress['latest_evaluation_arrays_sha256']
    assert all('latest_live.json' not in item['path'] and 'observer_status.json' not in item['path']
               for item in summary['input_sources'][-2:])
    assert all(item['status'] == 'PASS' for item in read(OUT / '输入核查.json')['checks'])
    assert all(item['status'] == 'PASS' for item in read(OUT / '逐轮核查.json')['checks'])
    wb_check = read(OUT / '_build/round240_workbook_checks.json')
    assert wb_check['protected_values_formulas_and_sources_identical'] and wb_check['protected_sheet_renders_identical']
    assert wb_check['xlsx_sha256'] == sha(OUT / XLSX)
    assert 'matched 0 entries' in wb_check['error_scan']
    assert len(wb_check['full_protected_render_comparison']) == 2
    assert all(item['before_sha256'] == item['after_sha256'] and item['identical']
               for item in wb_check['full_protected_render_comparison'])
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(OUT / XLSX) as archive:
        completed = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        partial = ET.fromstring(archive.read('xl/worksheets/sheet2.xml'))
        cells = {cell.attrib['r']: cell for cell in completed.findall('.//x:c', ns)}
        for role, start in [('primary', 6), ('secondary', 23)]:
            for offset, metric in enumerate(summary['metric_order']):
                row_num = start + offset
                for column, arm in [('B', 'BASE'), ('C', 'FULL'), ('D', 'NO_MPS')]:
                    endpoint = next(item for item in summary['completed_endpoints']
                                    if item['arm'] == arm and item['endpoint_role'] == role)
                    assert abs(float(cells[f'{column}{row_num}'].find('x:v', ns).text) -
                               endpoint['metrics_fraction'][metric]) < 1e-15
                for column, comparison in [('E', 'FULL-BASE'), ('F', 'FULL-NO_MPS')]:
                    cell = cells[f'{column}{row_num}']
                    assert cell.find('x:f', ns) is not None
                    expected = next(item for item in summary['differences'] if item['endpoint_role'] == role
                                    and item['metric'] == metric and item['comparison'] == comparison)['difference_percentage_points']
                    assert abs(float(cell.find('x:v', ns).text) - expected) < 1e-12
        assert len(completed.findall('.//x:f', ns)) == 48
        cells = {cell.attrib['r']: cell for cell in partial.findall('.//x:c', ns)}
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        def cell_text(address):
            cell = cells[address]
            if cell.attrib.get('t') == 's':
                return shared[int(cell.find('x:v', ns).text)]
            if cell.attrib.get('t') == 'str':
                return cell.find('x:v', ns).text or ''
            return ''.join(node.text or '' for node in cell.findall('.//x:t', ns))
        assert '30715' in cell_text('A3') and '241' in cell_text('A3') and '240' in cell_text('A3')
        assert '独立截止240轮' in cell_text('A28')
        assert '第240轮' in cell_text('A29') and '21.0329%' in cell_text('A29')
        assert '第239轮' in cell_text('A30') and '21.0337%' in cell_text('A30') and '非320轮端点' in cell_text('A30')
        for row_num, metric in enumerate(summary['metric_order'], 6):
            assert abs(float(cells[f'B{row_num}'].find('x:v', ns).text) - progress['latest_metrics_fraction'][metric]) < 1e-15
            for column in ('C', 'D'):
                cell = cells.get(f'{column}{row_num}')
                assert cell is None or (cell.find('x:v', ns) is None and cell.find('x:f', ns) is None)
    export = read(OUT / '导出核查.json')
    export.update(
        xlsx_sha256=sha(OUT / XLSX), NO_MSCA_accepted_rounds=240, source_files_unchanged=len(summary['input_sources']),
        completed_sheet_values_formulas_sources_preserved=True, completed_sheet_render_hashes_identical=True,
        update_local_date=LOCAL_DATE, prior_round200_verified_snapshot=str(SNAPSHOT),
    )
    replace_json(OUT / '导出核查.json', export)
    report = {
        'status': 'ROUND240_CERTIFICATE_CUTOFF_UPDATE_VERIFIED', 'local_date': LOCAL_DATE,
        'NO_MSCA_independently_verified_rounds': 240, 'NO_MSCA_cumulative_optimizer_updates': 388080,
        'second_recovery_segment_verified_rounds': 76, 'second_recovery_segment_verified_updates': 122892,
        'first_recovery_preserved_through_round': 164, 'old_partial165_preserved_in_16_file_archive': True,
        'added_since_prior_snapshot_rounds': 40, 'added_since_prior_snapshot_updates': 64680,
        'per_epoch_csv_rows': 1200, 'raw_metric_values_rechecked': 14400,
        'old_1160_csv_rows_preserved_as_exact_byte_prefix': True,
        'completed_endpoint_JSON_and_CSV_unchanged': True,
        'completed_workbook_values_formulas_sources_and_renders_preserved': True,
        'NO_MSCA_final_endpoint_blank_cells': 24,
        'xlsx_sha256': sha(OUT / XLSX), 'per_epoch_csv_sha256': sha(OUT / '逐轮指标.csv'),
        'new_evidence': {item['source_id']: item['sha256'] for item in summary['input_sources'][-2:]},
        'prior_snapshot': str(SNAPSHOT), 'physical_compute_limitation': progress['recovery']['physical_compute_limitation'],
        'last_native_accepted_round': 240, 'latest_saved_observed_running_round': 241,
        'current_remote_training_state': 'RUNNING', 'native_backup_verified_at240_acceptance': True,
        'old_connectivity_unknown_record_preserved_as_history': True,
        'metric_cutoff_remains_round240': True, 'model_forwards': 0, 'evaluator_runs': 0, 'remote_connections': 0,
    }
    visual_files = ['_build/round240_workbook_checks.json',
                    '_build/previews/round240_progress_after.png',
                    '_build/previews/round240_endpoints_full_after.png',
                    '_build/previews/round240_sources_full_after.png']
    report['visual_and_workbook_verification_files'] = [
        {'file': name, 'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)} for name in visual_files]
    report['latest_round240_and_local_best_round239_distinct'] = True
    report['local_best_progress_only_not_final_endpoint'] = True
    report['mutable_observer_latest_or_status_used_as_permanent_source'] = False
    replace_json(OUT / '第240轮更新核查.json', report)
    manifest = read(SNAPSHOT / '交付清单.json')
    filenames = [item['file'] for item in manifest['files']] + ['第240轮更新核查.json'] + visual_files
    manifest['files'] = [{'file': name, 'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)}
                         for name in filenames]
    manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['updated_local_date'] = LOCAL_DATE
    manifest['NO_MSCA_accepted_rounds'] = 240
    manifest['data_cutoff_basis'] = 'independent_round240_certificate'
    manifest['previous_round200_verified_snapshot'] = str(SNAPSHOT)
    manifest['verified_snapshot_destination'] = str(FINAL_SNAPSHOT)
    replace_json(OUT / '交付清单.json', manifest)
    # Freeze all current delivery bytes, with paths resolvable from this manifest's parent.
    FINAL_SNAPSHOT.mkdir(parents=True, exist_ok=False)
    entries = []
    for name in filenames + ['交付清单.json']:
        relative = PurePosixPath(name.replace('\\', '/'))
        assert not relative.is_absolute() and '..' not in relative.parts
        source = OUT.joinpath(*relative.parts)
        target = FINAL_SNAPSHOT.joinpath(*relative.parts)
        assert source.resolve().is_relative_to(OUT.resolve())
        assert target.resolve().is_relative_to(FINAL_SNAPSHOT.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert source.read_bytes() == target.read_bytes()
        entries.append({'file': relative.as_posix(), 'bytes': source.stat().st_size, 'sha256': sha(source)})
    assert len(entries) == 31 and len({item['file'] for item in entries}) == 31
    snapshot_manifest_path = FINAL_SNAPSHOT / 'snapshot_manifest.json'
    replace_json(snapshot_manifest_path, {
        'status': 'ROUND240_CURRENT_DELIVERY_BYTE_VERIFIED', 'local_date': LOCAL_DATE,
        'NO_MSCA_independently_verified_rounds': 240, 'per_epoch_csv_rows': 1200,
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
    # Re-read the original round200 manifest and files: no prior snapshot/index was modified.
    previous_manifest = read(SNAPSHOT / '交付清单.json')
    initial = read(OUT / '_build/prior_round200_verification.json')
    assert sha(SNAPSHOT / '交付清单.json') == initial['delivery_manifest_sha256']
    assert sha(SNAPSHOT / 'snapshot_manifest.json') == initial['snapshot_manifest_sha256']
    for entry in previous_manifest['files']:
        assert sha(SNAPSHOT / entry['file']) == entry['sha256']
    replace_json(OUT / '_build/round240_snapshot_verification.json', {
        'status': 'VERIFIED', 'files': len(entries), 'byte_identical_to_current': True,
        'relative_paths_resolve': True, 'prior_round200_snapshot_unchanged': True,
        'snapshot_manifest': str(snapshot_manifest_path), 'snapshot_manifest_sha256': sha(snapshot_manifest_path),
        'delivery_manifest_sha256': sha(OUT / '交付清单.json'),
        'xlsx_sha256': sha(OUT / XLSX), 'per_epoch_csv_sha256': sha(OUT / '逐轮指标.csv'),
    })
    print(json.dumps({key: value for key, value in report.items() if key != 'new_evidence'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
