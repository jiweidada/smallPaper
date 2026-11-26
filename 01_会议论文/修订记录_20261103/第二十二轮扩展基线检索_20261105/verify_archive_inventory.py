"""Independently re-enumerate accessible args and verify the saved inventory."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = HERE / 'archived_args_inventory.json'
OUT = HERE / 'independent_inventory_verification.json'
PRUNED = ['!**/runtime*/**', '!**/local_deps/**', '!**/build_tools/**',
          '!**/yolo_smoke_runtime_20261029', '!**/cpu_smoke_runtime_20261029',
          '!**/native_import_deps', '!**/isolated_runtime', '!**/analysis_tools',
          '!**/aps_official_eval_deps', '!**/c10_eval_test_deps']


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    report = json.loads(AUDIT.read_text(encoding='utf-8'))
    assert report['status'] == 'ACCESSIBLE_ARCHIVED_ARGS_INVENTORIED'
    rg = shutil.which('rg')
    args = [rg, '--files', '04_实验记录', '05_代码与数据', '90_历史与工具', '-g', 'args.json']
    for exclusion in PRUNED:
        args.extend(['-g', exclusion])
    scan = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding='utf-8', errors='replace', check=False)
    assert scan.returncode == 0 and not scan.stderr
    observed = {(ROOT / line.strip()).resolve() for line in scan.stdout.splitlines() if line.strip()}
    records = {Path(row['args_path']).resolve(): row for row in report['records']}
    assert observed == set(records) and len(observed) == report['files_returned']
    epoch_counts, checksums = Counter(), 0
    for path, row in records.items():
        assert digest(path) == row['args_sha256']
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        cfg = data.get('yaml_cfg', data)
        epoch_count = data.get('epoches', cfg.get('epoches', '<missing>'))
        epoch_counts[str(epoch_count)] += 1
        assert epoch_count == row['selected_config']['epochs_configured']
        if row.get('adjacent_log_present'):
            assert digest(Path(row['adjacent_log']['path'])) == row['adjacent_log']['sha256']
            checksums += 1
    assert dict(epoch_counts) == report['configured_epoch_counts']
    assert all(k in {'12', '24', '96', '132'} for k in epoch_counts)
    assert not any(row.get('coarse_320_batch4_fp32') for row in records.values())
    assert not any(row.get('adjacent_log', {}).get('contains_all_0_319') for row in records.values())
    selected = report['reference_N4']['selected']
    assert selected['epochs_configured'] == 320 and selected['train_batch'] == 4 and selected['use_amp'] is False
    examples = {}
    suffixes = {'visdrone_132': '正式132轮基线模型下载_20261008_v1/seed1/run/args.json',
                'visdrone_96': 'archive_review_inputs/outputs/O_batch8_seed1_full96/args.json',
                'uavdt_96': '20261019T013140Z_all_metrics/A/outputs/uavdt_baseline_full96_seed0/args.json'}
    for key, suffix in suffixes.items():
        hits = [row for path, row in records.items() if path.as_posix().endswith(suffix)]
        assert len(hits) == 1
        row = hits[0]
        examples[key] = {'args_path': row['args_path'], 'args_sha256': row['args_sha256'],
                         'config': row['selected_config'], 'log': row.get('adjacent_log')}
    assert examples['visdrone_132']['config']['epochs_configured'] == 132
    assert examples['visdrone_132']['config']['encoder_hidden_dim'] == 256
    assert examples['visdrone_132']['log']['unique_epochs'] == 132
    assert examples['visdrone_96']['config']['epochs_configured'] == 96
    assert examples['visdrone_96']['config']['train_batch'] == 8
    assert 'UAVDT' in examples['uavdt_96']['config']['train_ann']
    out = {'status': 'PRUNED_RESCAN_AND_CHECKSUMS_MATCHED', 'original_inventory_sha256': digest(AUDIT),
           'rg_exit_code': scan.returncode, 'rg_error_line_count': len(scan.stderr.splitlines()),
           'additional_pruned_dependency_directories': PRUNED[3:], 'args_files_verified': len(records),
           'adjacent_logs_hashed': checksums, 'epoch_counts': dict(epoch_counts),
           'coarse_320_batch4_fp32_count': 0, 'adjacent_logs_with_all_0_319': 0,
           'examples': examples, 'historical_data_identity_proven': False,
           'limits': ['The pruned dependency directories, hidden files, archives and other filenames are outside the scan.',
                      'No match in this scope does not prove that no comparable experiment exists elsewhere.',
                      'Saved configurations and logs alone do not authenticate historical source or data bytes.',
                      'No model, evaluation or training was run.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': out['status'], 'args_files_verified': len(records), 'adjacent_logs_hashed': checksums,
                      'epoch_counts': dict(epoch_counts), 'rg_errors': 0, 'sha256': digest(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
