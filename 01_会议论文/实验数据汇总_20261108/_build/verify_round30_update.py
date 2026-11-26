"""Verify the round-30 data update and refresh the deliverable hashes."""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from update_round30 import OUT, RECOVERY, SNAPSHOT, XLSX, replace_bytes, replace_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = read(OUT / '实验汇总.json')
    previous = read(SNAPSHOT / '实验汇总.json')
    assert summary['completed_endpoints'] == previous['completed_endpoints']
    assert summary['differences'] == previous['differences']
    assert summary['recipe_and_evaluation'] == previous['recipe_and_evaluation']
    assert summary['input_sources'][:9] == previous['input_sources']
    for name in ('终点指标.csv', '终点信息.csv', 'FULL差值.csv'):
        assert sha(OUT / name) == sha(SNAPSHOT / name)
    assert (OUT / '输入来源.csv').read_bytes().startswith((SNAPSHOT / '输入来源.csv').read_bytes())
    for source in summary['input_sources']:
        assert sha(Path(source['path'])) == source['sha256']
    progress = summary['NO_MSCA_progress']
    assert progress['completed_rounds'] == 30 and progress['optimizer_step_attempts'] == 48510
    assert progress['primary_endpoint'] is None and progress['secondary_endpoint_after_320_rounds'] is None
    assert progress['recovery']['new_verified_rounds'] == 1
    assert progress['recovery']['new_verified_optimizer_updates'] == 1617
    assert progress['recovery']['contract_live_flag'] is False
    assert progress['live_observation']['round_1based'] == 31 and progress['live_observation']['accepted_rounds'] == 30
    with (OUT / 'NO_MSCA进度.csv').open(encoding='utf-8-sig', newline='') as stream:
        progress_rows = list(csv.DictReader(stream))
    assert len(progress_rows) == 12
    for row in progress_rows:
        assert int(row['completed_rounds']) == 30
        assert float(row['latest_value_fraction']) == progress['latest_metrics_fraction'][row['metric']]
        assert row['round320_primary_value'] == row['round320_secondary_value'] == ''
    with (OUT / '逐轮指标.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 990 and dict(Counter(row['arm'] for row in rows)) == {
        'BASE': 320, 'FULL': 320, 'NO_MPS': 320, 'NO_MSCA': 30}
    assert (OUT / '逐轮指标.csv').read_bytes().startswith((SNAPSHOT / '逐轮指标.csv').read_bytes())
    mapping = read(OUT / '逐轮来源与切片.json')
    sources = {item['source_id']: item for item in mapping['sources']}
    source_payloads = {}
    for row in rows:
        source_id = row['source_id']
        if source_id not in source_payloads:
            payload = Path(row['source_path']).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == row['source_sha256'] == sources[source_id]['sha256']
            source_payloads[source_id] = payload
        payload = source_payloads[source_id]
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
        for field in ('prediction_sha256', 'evaluation_arrays_sha256'):
            assert row[field] == original[field]
        assert row['evaluated_ema_tensor_sha256'] == original['ema_tensor_sha256']
    workbook_checks = read(OUT / '_build/progress_workbook_checks.json')
    assert workbook_checks['protected_values_formulas_and_sources_identical']
    assert workbook_checks['protected_sheet_renders_identical']
    assert workbook_checks['xlsx_sha256'] == sha(OUT / XLSX)
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(OUT / XLSX) as archive:
        primary = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        progress_xml = ET.fromstring(archive.read('xl/worksheets/sheet2.xml'))
        cells = {cell.attrib['r']: cell for cell in primary.findall('.//x:c', ns)}
        for role, start in [('primary', 6), ('secondary', 23)]:
            for metric_index, metric in enumerate(summary['metric_order']):
                index = start + metric_index
                for column, arm in [('B', 'BASE'), ('C', 'FULL'), ('D', 'NO_MPS')]:
                    endpoint = next(record for record in summary['completed_endpoints']
                                    if record['arm'] == arm and record['endpoint_role'] == role)
                    assert abs(float(cells[column + str(index)].find('x:v', ns).text) -
                               endpoint['metrics_fraction'][metric]) < 1e-15
                for column, comparison in [('E', 'FULL-BASE'), ('F', 'FULL-NO_MPS')]:
                    cell = cells[column + str(index)]
                    assert cell.find('x:f', ns) is not None
                    expected = next(record for record in summary['differences']
                                    if record['endpoint_role'] == role and record['metric'] == metric
                                    and record['comparison'] == comparison)['difference_percentage_points']
                    assert abs(float(cell.find('x:v', ns).text) - expected) < 1e-12
        assert len(primary.findall('.//x:f', ns)) == 48
        progress_cells = {cell.attrib['r']: cell for cell in progress_xml.findall('.//x:c', ns)}
        for index, metric in enumerate(summary['metric_order'], 6):
            assert abs(float(progress_cells[f'B{index}'].find('x:v', ns).text) -
                       progress['latest_metrics_fraction'][metric]) < 1e-15
            for column in ('C', 'D'):
                cell = progress_cells.get(f'{column}{index}')
                assert cell is None or (cell.find('x:v', ns) is None and cell.find('x:f', ns) is None)
    assert sha(SNAPSHOT / XLSX) == '104f84f68b394a4bfcf9b8a688f8ebd5a2a0b6d04f29dffe842f1cac76c7f3f7'
    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        '本次整理核对了 9 份来源报告或协议', '初始汇总核对了 9 份来源报告或协议')
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    export_check = read(OUT / '导出核查.json')
    export_check.update(
        xlsx_sha256=sha(OUT / XLSX), NO_MSCA_accepted_rounds=30, source_files_unchanged=len(summary['input_sources']),
        NO_MSCA_progress_metric_cells_verified=12, completed_sheet_values_formulas_sources_preserved=True,
        completed_sheet_render_hashes_identical=True, update_local_date='2026-11-08',
        prior_round29_snapshot=str(SNAPSHOT),
    )
    replace_json(OUT / '导出核查.json', export_check)
    report = {
        'status': 'ROUND30_UPDATE_VERIFIED', 'local_date': '2026-11-08',
        'NO_MSCA_accepted_rounds': 30, 'NO_MSCA_cumulative_optimizer_updates': 48510,
        'recovery_new_verified_rounds': 1, 'recovery_new_verified_updates': 1617,
        'per_epoch_csv_rows': 990, 'raw_metric_values_rechecked': 11880,
        'old_989_csv_rows_preserved_as_exact_byte_prefix': True,
        'completed_endpoint_JSON_and_CSV_unchanged': True,
        'completed_workbook_values_formulas_sources_and_renders_preserved': True,
        'NO_MSCA_320_round_endpoint_cells_blank': 24,
        'xlsx_sha256': sha(OUT / XLSX), 'per_epoch_csv_sha256': sha(OUT / '逐轮指标.csv'),
        'source_sha256': {item['source_id']: item['sha256'] for item in summary['input_sources']},
        'prior_snapshot': str(SNAPSHOT),
        'contract_live_false_scope': progress['recovery']['contract_live_flag_scope'],
        'physical_compute_limitation': progress['recovery']['physical_compute_limitation'],
        'model_forwards': 0, 'evaluator_runs': 0, 'remote_connections': 0,
    }
    replace_json(OUT / '第30轮更新核查.json', report)
    manifest = read(SNAPSHOT / '交付清单.json')
    filenames = [item['file'] for item in manifest['files']]
    filenames.append('第30轮更新核查.json')
    manifest['files'] = [{'file': name, 'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)}
                         for name in filenames]
    manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['updated_local_date'] = '2026-11-08'
    manifest['NO_MSCA_accepted_rounds'] = 30
    manifest['previous_round29_snapshot'] = str(SNAPSHOT)
    replace_json(OUT / '交付清单.json', manifest)
    print(json.dumps({key: value for key, value in report.items() if key != 'source_sha256'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
