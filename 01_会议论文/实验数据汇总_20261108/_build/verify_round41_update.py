"""Verify the certificate-cutoff round-41 delivery without evaluating any model."""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from update_round41 import OUT, SNAPSHOT, XLSX, LOCAL_DATE
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
    assert progress['completed_rounds'] == 41 and progress['optimizer_step_attempts'] == 66297
    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None
    assert progress['recovery']['new_verified_rounds'] == 12
    assert progress['recovery']['new_verified_optimizer_updates'] == 19404
    assert progress['live_observation']['round_1based'] == 43
    assert progress['live_observation']['native_accepted_rounds_observed'] == 42
    assert progress['live_observation']['independent_metric_snapshot_rounds'] == 41
    assert progress['update_since_previous_snapshot'] == {
        'previous_rounds': 30, 'new_verified_rounds': 11, 'new_verified_updates': 17787}
    with (OUT / 'NO_MSCA进度.csv').open(encoding='utf-8-sig', newline='') as stream:
        progress_rows = list(csv.DictReader(stream))
    assert len(progress_rows) == 12
    for row in progress_rows:
        assert int(row['completed_rounds']) == 41
        assert float(row['latest_value_fraction']) == progress['latest_metrics_fraction'][row['metric']]
        assert row['round320_primary_value'] == row['round320_secondary_value'] == ''
    with (OUT / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1001
    assert dict(Counter(row['arm'] for row in rows)) == {'BASE': 320, 'FULL': 320, 'NO_MPS': 320, 'NO_MSCA': 41}
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
    wb_check = read(OUT / '_build/round41_workbook_checks.json')
    assert wb_check['protected_values_formulas_and_sources_identical'] and wb_check['protected_sheet_renders_identical']
    assert wb_check['xlsx_sha256'] == sha(OUT / XLSX)
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
        for row_num, metric in enumerate(summary['metric_order'], 6):
            assert abs(float(cells[f'B{row_num}'].find('x:v', ns).text) - progress['latest_metrics_fraction'][metric]) < 1e-15
            for column in ('C', 'D'):
                cell = cells.get(f'{column}{row_num}')
                assert cell is None or (cell.find('x:v', ns) is None and cell.find('x:f', ns) is None)
    export = read(OUT / '导出核查.json')
    export.update(
        xlsx_sha256=sha(OUT / XLSX), NO_MSCA_accepted_rounds=41, source_files_unchanged=len(summary['input_sources']),
        completed_sheet_values_formulas_sources_preserved=True, completed_sheet_render_hashes_identical=True,
        update_local_date=LOCAL_DATE, prior_round30_snapshot=str(SNAPSHOT),
    )
    replace_json(OUT / '导出核查.json', export)
    report = {
        'status': 'ROUND41_CERTIFICATE_CUTOFF_UPDATE_VERIFIED', 'local_date': LOCAL_DATE,
        'NO_MSCA_independently_verified_rounds': 41, 'NO_MSCA_cumulative_optimizer_updates': 66297,
        'recovery_segment_verified_rounds': 12, 'recovery_segment_verified_updates': 19404,
        'added_since_prior_snapshot_rounds': 11, 'added_since_prior_snapshot_updates': 17787,
        'per_epoch_csv_rows': 1001, 'raw_metric_values_rechecked': 12012,
        'old_990_csv_rows_preserved_as_exact_byte_prefix': True,
        'completed_endpoint_JSON_and_CSV_unchanged': True,
        'completed_workbook_values_formulas_sources_and_renders_preserved': True,
        'NO_MSCA_final_endpoint_blank_cells': 24,
        'xlsx_sha256': sha(OUT / XLSX), 'per_epoch_csv_sha256': sha(OUT / '逐轮指标.csv'),
        'new_evidence': {item['source_id']: item['sha256'] for item in summary['input_sources'][-2:]},
        'prior_snapshot': str(SNAPSHOT), 'physical_compute_limitation': progress['recovery']['physical_compute_limitation'],
        'live_observed_native_rounds': 42, 'live_observed_running_round': 43,
        'metric_cutoff_remains_round41': True, 'model_forwards': 0, 'evaluator_runs': 0, 'remote_connections': 0,
    }
    replace_json(OUT / '第41轮更新核查.json', report)
    manifest = read(SNAPSHOT / '交付清单.json')
    filenames = [item['file'] for item in manifest['files']] + ['第41轮更新核查.json']
    manifest['files'] = [{'file': name, 'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)}
                         for name in filenames]
    manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['updated_local_date'] = LOCAL_DATE
    manifest['NO_MSCA_accepted_rounds'] = 41
    manifest['data_cutoff_basis'] = 'independent_round41_certificate'
    manifest['previous_round30_snapshot'] = str(SNAPSHOT)
    replace_json(OUT / '交付清单.json', manifest)
    print(json.dumps({key: value for key, value in report.items() if key != 'new_evidence'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
