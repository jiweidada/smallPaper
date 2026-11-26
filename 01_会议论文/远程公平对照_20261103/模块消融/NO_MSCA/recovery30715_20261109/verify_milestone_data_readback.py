"""Independent read-only CSV/XLSX/source verification; no ML execution."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import zipfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[3]
DATA = PAPER / '实验数据汇总_20261108'
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
read = lambda p: json.loads(p.read_text('utf-8-sig'))
digest = lambda p: sha256(p.read_bytes()).hexdigest()
METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']

p = argparse.ArgumentParser()
p.add_argument('--round', type=int, required=True)
p.add_argument('--old-snapshot', type=Path, required=True)
p.add_argument('--old-snapshot-sha256', required=True)
p.add_argument('--new-snapshot', type=Path, required=True)
p.add_argument('--new-snapshot-sha256', required=True)
p.add_argument('--audit-sha256', required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
assert 0 < args.round < 320 and not args.output.exists()
OLD, NEW = args.old_snapshot.resolve(), args.new_snapshot.resolve()
assert OLD.is_relative_to(DATA.resolve()) and NEW.is_relative_to(DATA.resolve())
assert digest(OLD / 'snapshot_manifest.json') == args.old_snapshot_sha256
assert digest(NEW / 'snapshot_manifest.json') == args.new_snapshot_sha256
audit_path = HERE / f'lineage_second_recovery_through_round{args.round}.json'
assert digest(audit_path) == args.audit_sha256
audit = read(audit_path)
assert audit['completed_rounds'] == args.round and audit['maximum_latest_metric_error'] == 0

def verify_entries(root, entries):
    names = set()
    for item in entries:
        relative = PurePosixPath(item['file'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert item['file'] not in names
        names.add(item['file'])
        target = (root / str(relative)).resolve()
        assert target.is_relative_to(root.resolve())
        assert target.stat().st_size == item['bytes'], target
        assert digest(target) == item['sha256'], target
    return len(names)

old_manifest, new_manifest = [read(s / 'snapshot_manifest.json') for s in [OLD, NEW]]
old_count = verify_entries(OLD, old_manifest['files'])
new_count = verify_entries(NEW, new_manifest['files'])
delivery = read(NEW / '交付清单.json')
assert delivery['NO_MSCA_accepted_rounds'] == args.round
delivery_count = verify_entries(DATA, delivery['files'])
assert (DATA / '交付清单.json').read_bytes() == (NEW / '交付清单.json').read_bytes()
assert set(i['file'] for i in new_manifest['files']) == set(i['file'] for i in delivery['files']) | {'交付清单.json'}
for item in delivery['files']:
    assert (DATA / item['file']).read_bytes() == (NEW / item['file']).read_bytes()

old_raw, new_raw = [(s / '逐轮指标.csv').read_bytes() for s in [OLD, NEW]]
assert new_raw.startswith(old_raw)
old_rows = list(csv.DictReader(io.StringIO(old_raw.decode('utf-8-sig'))))
rows = list(csv.DictReader(io.StringIO(new_raw.decode('utf-8-sig'))))
old_round = len(old_rows) - 960
assert 0 < old_round < args.round and len(rows) == 960 + args.round
assert Counter(r['arm'] for r in rows) == {'BASE': 320, 'FULL': 320, 'NO_MPS': 320, 'NO_MSCA': args.round}
suffix = rows[len(old_rows):]
assert [int(r['round_1based']) for r in suffix] == list(range(old_round + 1, args.round + 1))
source = BACKUPS / f'epochs/epoch{args.round - 1:04d}/epochs.jsonl'
raw_records = source.read_bytes().splitlines()
native = [json.loads(b) for b in raw_records]
assert len(native) == args.round
for row in rows:
    if row['arm'] == 'NO_MSCA':
        record = native[int(row['round_1based']) - 1]
        for metric in METRICS:
            assert float(row[metric]) == record['metrics'][metric]
for row in suffix:
    i = int(row['round_1based']) - 1
    record = native[i]
    assert row['arm'] == 'NO_MSCA' and row['seed'] == '0' and row['series_status'] == 'accepted_prefix_only'
    for col, key in [('actual_epoch_index_0based', 'epoch'), ('last_epoch_state_counter', 'last_epoch'),
                     ('optimizer_step_attempts', 'optimizer_step_attempts'), ('ema_updates', 'ema_updates')]:
        assert int(row[col]) == record[key]
    for col in ['train_loss_mean', 'seconds']:
        assert float(row[col]) == record[col]
    for col, key in [('weight_source', 'weight_source'), ('evaluated_ema_tensor_sha256', 'ema_tensor_sha256'),
                     ('prediction_sha256', 'prediction_sha256'), ('evaluation_arrays_sha256', 'evaluation_arrays_sha256'),
                     ('created_utc', 'created_utc'), ('protocol_sha256', 'protocol_sha256'), ('configuration_sha256', 'config_sha256')]:
        assert row[col] == record[key]
    for col in ['first_executed_lrs', 'post_final_update_lrs']:
        assert json.loads(row[col]) == record[col]
    assert Path(row['source_path']).resolve() == source.resolve()
    assert row['source_sha256'] == digest(source)
    assert row['source_locator'] == f'line {i + 1}' and int(row['source_record_index_0based']) == i
    assert row['source_raw_record_sha256'] == sha256(raw_records[i]).hexdigest()
    assert row['normalization'] == 'none'
    canonical = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    assert row['normalized_row_sha256'] == sha256(canonical).hexdigest()
    assert row['checkpoint_sha256_present_in_original_source'] == record.get('checkpoint_sha256', '')
assert {m: float(rows[-1][m]) for m in METRICS} == audit['latest_metrics']
best_row = next(r for r in rows if r['arm'] == 'NO_MSCA' and int(r['round_1based']) == audit['best_epoch_index'] + 1)
assert {m: float(best_row[m]) for m in METRICS} == audit['endpoints']['best']['metrics']
for col, key in [('evaluated_ema_tensor_sha256', 'ema_tensor_sha256'), ('prediction_sha256', 'prediction_sha256'),
                 ('evaluation_arrays_sha256', 'evaluation_arrays_sha256')]:
    assert best_row[col] == audit['endpoints']['best'][key]

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
RELNS = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
def workbook(path):
    result, formulas = {}, {}
    with zipfile.ZipFile(path) as z:
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(c.itertext()) for c in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        for sh in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS):
            target = rels[sh.attrib[RELNS]]
            name = posixpath.normpath(target.lstrip('/') if target.startswith('/') else posixpath.join('xl', target))
            assert name.startswith('xl/') and '..' not in PurePosixPath(name).parts
            cells = {}
            for c in ET.fromstring(z.read(name)).findall('.//s:c', NS):
                value = c.findtext('s:v', None, NS)
                if c.attrib.get('t') == 's' and value is not None:
                    value = shared[int(value)]
                elif c.attrib.get('t') == 'inlineStr':
                    value = ''.join(c.find('s:is', NS).itertext())
                f = c.findtext('s:f', None, NS)
                cells[c.attrib['r']] = {'value': value, 'formula': f, 'type': c.attrib.get('t')}
                if f is not None:
                    formulas[(sh.attrib['name'], c.attrib['r'])] = f
            result[sh.attrib['name']] = cells
    return result, formulas

xlsx = Path('outputs/实验数据汇总_20261108/实验数据汇总.xlsx')
old_sheets, old_formulas = workbook(OLD / xlsx)
sheets, formulas = workbook(NEW / xlsx)
assert len(formulas) == 48 and formulas == old_formulas
for protected in ['终点对比', '来源与权重']:
    assert sheets[protected] == old_sheets[protected]
progress = sheets['NO_MSCA进度']
for row, metric in enumerate(METRICS, 6):
    assert progress[f'A{row}']['value'] == metric
    assert float(progress[f'B{row}']['value']) == audit['latest_metrics'][metric]
    for col in ['C', 'D']:
        cell = progress[f'{col}{row}']
        assert cell['value'] is None and cell['formula'] is None
assert int(progress['B20']['value']) == args.round
assert int(progress['B21']['value']) == 320
assert int(progress['B22']['value']) == args.round * 1617
assert int(progress['B23']['value']) == 3699662 and int(progress['B24']['value']) == 0
assert str(audit['best_epoch_index'] + 1) in progress['A30']['value']
negative = sheets['终点对比']['F17']
assert negative['formula'] == '(C17-D17)*100'
assert float(negative['value']) == -1.6869030784671835

report = {
    'status': 'INDEPENDENT_PARENT_DATA_READBACK_PASSED_PENDING_PARENT_VISUAL_QA',
    'observed_utc': datetime.now(timezone.utc).isoformat(),
    'cutoff_round': args.round, 'partial_best_round': audit['best_epoch_index'] + 1,
    'delivery_payload_files_byte_verified': delivery_count,
    'immutable_snapshot_payload_files_byte_verified': new_count,
    'old_snapshot_payload_files_byte_verified_unchanged': old_count,
    'per_epoch_csv_rows': len(rows), 'old_csv_exact_byte_prefix_preserved_rows': len(old_rows),
    'new_rows_against_native_JSONL': len(suffix), 'new_raw_metric_values_compared': len(suffix) * 12,
    'all_NO_MSCA_metric_values_compared_to_cumulative_source': args.round * 12,
    'new_native_loss_counters_LR_metadata_tensor_prediction_array_and_record_hashes_equal': True,
    'progress_XLSX_metric_cells_verified': 12, 'maximum_workbook_metric_cell_error': 0,
    'final_NO_MSCA_cells_C6_D17_remain_empty': 24, 'formula_cells_preserved': 48,
    'negative_FULL_minus_NO_MPS_ARl_preserved': True,
    'protected_endpoint_and_source_sheet_cell_maps_equal': True,
    'audit_sha256': digest(audit_path), 'native_cumulative_source_sha256': digest(source),
    'verification_script_sha256': digest(Path(__file__)),
    'latest_and_partial_best_rounds_are_distinct': args.round != audit['best_epoch_index'] + 1,
    'permanent_snapshot_manifest': str(NEW / 'snapshot_manifest.json'),
    'permanent_snapshot_manifest_sha256': digest(NEW / 'snapshot_manifest.json'),
    'delivery_manifest_sha256': digest(NEW / '交付清单.json'),
    'workbook_sha256': digest(NEW / xlsx), 'CSV_sha256': digest(NEW / '逐轮指标.csv'),
    'parent_visual_QA_performed_by_this_script': False,
    'model_forwards': 0, 'evaluator_runs': 0, 'training_actions': 0,
}
with args.output.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(report, ensure_ascii=True))
