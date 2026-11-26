"""Export the exact accepted per-epoch records without executing a model or evaluator."""

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent.parent
PAPER = OUT.parent
REMOTE = PAPER / '远程公平对照_20261103'
CURVES = PAPER / '修订记录_20261103/第三十三轮实验主线重组_20261108'
METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
NO_MSCA_CUTOFF_EPOCH = 28
CHECKS = []
SOURCES = []
RAW_FILES = {}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(4 * 1024**2):
            hasher.update(data)
    return hasher.hexdigest()


def check(label, condition):
    if not condition:
        raise RuntimeError(label)
    CHECKS.append({'check': label, 'status': 'PASS'})


def source(source_id, path, expected=None):
    payload = path.read_bytes()
    actual = digest(payload)
    if expected is not None:
        check(source_id + ' expected SHA-256', actual == expected)
    SOURCES.append({'source_id': source_id, 'path': str(path), 'bytes': len(payload),
                    'sha256': actual, 'expected_sha256': expected})
    RAW_FILES[source_id] = payload
    return SOURCES[-1]


def jsonl_source(source_id, path, expected=None):
    info = source(source_id, path, expected)
    lines = RAW_FILES[source_id].splitlines()
    rows = [json.loads(line) for line in lines]
    return rows, [dict(row=row, source_id=source_id, source_sha256=info['sha256'], source_path=str(path),
                      source_locator=f'line {index + 1}', source_record_index_0based=index,
                      source_raw_record_sha256=digest(line), normalization='none')
                  for index, (line, row) in enumerate(zip(lines, rows))]


def accepted_stage(arm, epoch):
    root = REMOTE / '模块消融' / arm
    plan_path = root / 'execution_protocol.json'
    plan = read(plan_path)
    frozen = {item['path']: item['sha256'] for item in plan['frozen_files']}
    protocol_path = root / 'backup_protocol.py'
    check(arm + ' frozen backup protocol', sha(protocol_path) == frozen['backup_protocol.py'])
    spec = importlib.util.spec_from_file_location('_epoch_export_' + arm, protocol_path)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    base = Path('D:/MPSMA_DEIM_experiment_backups/20261107') / arm / 'backups'
    observed = sorted(int(path.name[5:]) for path in (base / 'epochs').iterdir()
                      if path.is_dir() and path.name.startswith('epoch') and path.name[5:].isdigit()
                      and (path / 'accepted.json').is_file())
    check(arm + ' accepted prefix has no gap', observed == list(range(observed[-1] + 1)) and observed[-1] >= epoch)
    folder = base / 'epochs' / f'epoch{epoch:04d}'
    data = (folder / 'manifest.json').read_bytes()
    run_id = Path(read(root / 'execution_configs' / f'{arm}.yml')['output_dir']).parent.name
    manifest = protocol.parse_manifest(data, run_id, epoch)
    check(arm + ' accepted manifest protocol', manifest['protocol_sha256'] == sha(plan_path))
    check(arm + ' accepted acknowledgement', (folder / 'ack.json').read_bytes() == protocol.ack_bytes(data, manifest)
          and (folder / 'accepted.json').read_bytes() == protocol.accepted_bytes(data, manifest))
    for item in manifest['files']:
        path = protocol.checked_local(folder / item['name'], folder)
        check(arm + ' accepted artifact ' + item['name'],
              protocol.regular(path).st_size == item['bytes'] and sha(path) == item['sha256'])
    protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(folder / name))
    expected = next(item['sha256'] for item in manifest['files'] if item['name'] == 'epochs.jsonl')
    rows, refs = jsonl_source(arm + '_ACCEPTED_EPOCH_ROWS', folder / 'epochs.jsonl', expected)
    check(arm + ' cumulative accepted row count', len(rows) == epoch + 1
          and rows[-1] == read(folder / 'epoch.metrics.json'))
    return rows, refs, {'arm': arm, 'included_last_epoch_index': epoch, 'included_rounds': epoch + 1,
                        'highest_accepted_epoch_observed': observed[-1],
                        'manifest_path': str(folder / 'manifest.json'), 'manifest_sha256': digest(data),
                        'ack_sha256': sha(folder / 'ack.json'), 'accepted_sha256': sha(folder / 'accepted.json'),
                        'rolling_checkpoint_sha256': manifest['rolling_sha256']}


def write_json(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    summary = read(OUT / '实验汇总.json')
    provenance_path = CURVES / 'learning_curve_provenance.json'
    source('PAIRED_CURVE_PROVENANCE', provenance_path)
    provenance = read(provenance_path)
    curve_path = CURVES / 'paired_learning_curve_rows.json'
    source('PAIRED_CURVE_ROWS', curve_path)
    curves = read(curve_path)
    pair_path = REMOTE / '断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
    source('VERIFIED_PAIR_REPORT', pair_path, provenance['pair_report_sha256'])
    check('provenance bound to supplied pair summary', provenance['pair_report_sha256'] ==
          next(row['sha256'] for row in summary['input_sources'] if row['source_id'] == 'BASE_FULL'))
    entries = list(provenance['source_sha256'].items())
    check('exact five source classes', len(entries) == 5 and entries[0][0].endswith('/runs/BASE/epochs.jsonl')
          and entries[1][0].endswith('/backup_epoch0307/epochs.jsonl')
          and entries[2][0].endswith('/BASE_30166_r308/terminal_evidence/run/epochs.jsonl')
          and entries[3][0].endswith('/FULL_best_epoch0000/metrics.json')
          and entries[4][0].endswith('/FULL_30848_20261106/backups/epochs/epoch0319/epochs.jsonl'))
    inputs = {}
    for source_id, (relative, expected) in zip(
            ['BASE_ORIGINAL', 'BASE_SAVED_THROUGH_308', 'BASE_RECOVERY_309_320', 'FULL_ORIGINAL_ROUND1', 'FULL_CONTINUATION'],
            entries):
        path = PAPER / relative
        if path.suffix == '.jsonl':
            inputs[source_id] = jsonl_source(source_id, path, expected)
        else:
            info = source(source_id, path, expected)
            original = json.loads(RAW_FILES[source_id])
            check('FULL first round has only one extra checkpoint field', set(original) - set(curves['FULL'][0]) == {'checkpoint_sha256'})
            normalized = {key: value for key, value in original.items() if key != 'checkpoint_sha256'}
            inputs[source_id] = ([normalized], [{
                'row': normalized, 'source_id': source_id, 'source_sha256': info['sha256'], 'source_path': str(path),
                'source_locator': 'JSON root', 'source_record_index_0based': 0,
                'source_raw_record_sha256': digest(RAW_FILES[source_id]),
                'normalization': 'remove checkpoint_sha256 only for curve-row equality',
                'source_checkpoint_sha256': original['checkpoint_sha256'],
            }])
    original, original_refs = inputs['BASE_ORIGINAL']
    prefix, prefix_refs = inputs['BASE_SAVED_THROUGH_308']
    suffix, suffix_refs = inputs['BASE_RECOVERY_309_320']
    check('BASE original slice and excluded eleven rows', len(original) == 225
          and [row['epoch'] for row in original] == list(range(225)) and len(original[214:]) == 11)
    check('BASE saved continuation slice', [row['epoch'] for row in prefix] == list(range(214, 308)))
    check('BASE final recovery slice', [row['epoch'] for row in suffix] == list(range(308, 320)))
    joined = {
        'BASE': original[:214] + prefix + suffix,
        'FULL': inputs['FULL_ORIGINAL_ROUND1'][0] + inputs['FULL_CONTINUATION'][0],
    }
    refs = {
        'BASE': original_refs[:214] + prefix_refs + suffix_refs,
        'FULL': inputs['FULL_ORIGINAL_ROUND1'][1] + inputs['FULL_CONTINUATION'][1],
    }
    for arm in ('BASE', 'FULL'):
        check(arm + ' reconstructed equals recorded paired curves', joined[arm] == curves[arm] and len(joined[arm]) == 320)
    stages = []
    for arm, last_epoch in [('NO_MPS', 319), ('NO_MSCA', NO_MSCA_CUTOFF_EPOCH)]:
        joined[arm], refs[arm], stage = accepted_stage(arm, last_epoch)
        stages.append(stage)
    check('NO_MSCA frozen snapshot cutoff', len(joined['NO_MSCA']) == 29)
    export_rows = []
    for arm in ('BASE', 'FULL', 'NO_MPS', 'NO_MSCA'):
        for index, (row, ref) in enumerate(zip(joined[arm], refs[arm])):
            check(f'{arm} epoch {index} cursor and metric schema', row['epoch'] == index
                  and row['optimizer_step_attempts'] == (index + 1) * 1617
                  and row['weight_source'] == 'ema.module' and set(row['metrics']) == set(METRICS))
            canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
            export_rows.append({
                'arm': arm, 'series_status': 'accepted_prefix_only' if arm == 'NO_MSCA' else 'completed_320_rounds',
                'seed': 0, 'round_1based': index + 1, 'actual_epoch_index_0based': row['epoch'],
                'last_epoch_state_counter': row['last_epoch'], 'optimizer_step_attempts': row['optimizer_step_attempts'],
                'ema_updates': row['ema_updates'], 'train_loss_mean': row['train_loss_mean'],
                **row['metrics'], 'weight_source': row['weight_source'],
                'evaluated_ema_tensor_sha256': row['ema_tensor_sha256'],
                'prediction_sha256': row['prediction_sha256'],
                'evaluation_arrays_sha256': row['evaluation_arrays_sha256'],
                'first_executed_lrs': json.dumps(row['first_executed_lrs'], separators=(',', ':')),
                'post_final_update_lrs': json.dumps(row['post_final_update_lrs'], separators=(',', ':')),
                'seconds': row['seconds'], 'created_utc': row['created_utc'],
                'protocol_sha256': row['protocol_sha256'], 'configuration_sha256': row['config_sha256'],
                **{key: value for key, value in ref.items() if key not in ('row', 'source_checkpoint_sha256')},
                'normalized_row_sha256': digest(canonical),
                'checkpoint_sha256_present_in_original_source': ref.get('source_checkpoint_sha256', ''),
            })
    check('complete export row count', len(export_rows) == 989)
    for endpoint in summary['completed_endpoints']:
        row = joined[endpoint['arm']][endpoint['epoch_index_0based']]
        check(endpoint['arm'] + '/' + endpoint['endpoint_role'] + ' matches endpoint summary',
              row['metrics'] == endpoint['metrics_fraction'] and row['ema_tensor_sha256'] == endpoint['ema_tensor_sha256']
              and row['prediction_sha256'] == endpoint['prediction_sha256']
              and row['evaluation_arrays_sha256'] == endpoint['evaluation_arrays_sha256'])
    for info in SOURCES:
        check(info['source_id'] + ' unchanged while exporting', sha(Path(info['path'])) == info['sha256'])
    fields = list(export_rows[0])
    with (OUT / '逐轮指标.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(export_rows)
    mapping = {
        'status': 'EXACT_JOIN_AND_ACCEPTED_PREFIX_VERIFIED', 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'metric_units': {key: 'raw fraction 0–1' for key in METRICS},
        'unit_conversion': 'metric percent = raw fraction * 100; percentage-point difference = (raw A - raw B) * 100',
        'source_locator_definition': 'JSONL line positions are 1-based, record indices are 0-based; FULL first round is the JSON root.',
        'source_raw_record_hash_definition': 'JSONL hashes exclude line terminators. JSON-root hashes cover the complete original file.',
        'normalized_row_hash_definition': 'SHA-256 of sorted-key compact UTF-8 JSON after the explicitly recorded normalization.',
        'slices': [
            {'arm': 'BASE', 'source_id': 'BASE_ORIGINAL', 'source_records_0based_half_open': [0, 214], 'rounds_included': [1, 214]},
            {'arm': 'BASE', 'source_id': 'BASE_SAVED_THROUGH_308', 'source_records_0based_half_open': [0, 94], 'rounds_included': [215, 308]},
            {'arm': 'BASE', 'source_id': 'BASE_RECOVERY_309_320', 'source_records_0based_half_open': [0, 12], 'rounds_included': [309, 320]},
            {'arm': 'FULL', 'source_id': 'FULL_ORIGINAL_ROUND1', 'source_records_0based_half_open': [0, 1], 'rounds_included': [1, 1]},
            {'arm': 'FULL', 'source_id': 'FULL_CONTINUATION', 'source_records_0based_half_open': [0, 319], 'rounds_included': [2, 320]},
            {'arm': 'NO_MPS', 'source_id': 'NO_MPS_ACCEPTED_EPOCH_ROWS', 'source_records_0based_half_open': [0, 320], 'rounds_included': [1, 320]},
            {'arm': 'NO_MSCA', 'source_id': 'NO_MSCA_ACCEPTED_EPOCH_ROWS', 'source_records_0based_half_open': [0, 29], 'rounds_included': [1, 29]},
        ],
        'excluded': [
            {'arm': 'BASE', 'source_id': 'BASE_ORIGINAL', 'source_records_0based_half_open': [214, 225],
             'count': 11, 'reason': 'Original rows after the round-214 checkpoint belong to the abandoned trajectory.'},
            {'arm': 'BASE', 'rule': 'Only the epoch0307 saved first-continuation prefix and the completed r308 recovery suffix are included. Other earlier-segment tails are excluded.'},
            {'arm': 'NO_MSCA', 'rule': 'Live or in-flight round-30 updates are excluded until an accepted backup is selected for a later snapshot.'},
        ],
        'rows_per_arm': {arm: len(rows) for arm, rows in joined.items()}, 'accepted_stage_sources': stages,
        'sources': SOURCES, 'limitations': summary['limitations'],
    }
    write_json('逐轮来源与切片.json', mapping)
    write_json('逐轮核查.json', {'status': 'PASSED', 'checks': CHECKS, 'check_count': len(CHECKS),
               'rows': len(export_rows), 'csv_sha256': sha(OUT / '逐轮指标.csv'),
               'paired_curve_join_exact': True, 'model_forwards': 0, 'evaluator_runs': 0,
               'training_updates': 0, 'remote_connections': 0})
    print(json.dumps({'status': 'PER_EPOCH_EXPORT_COMPLETE', 'rows': len(export_rows),
                      'rows_per_arm': mapping['rows_per_arm'], 'checks': len(CHECKS)}))


if __name__ == '__main__':
    main()
