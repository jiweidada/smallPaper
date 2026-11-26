"""Inspect bounded local records and transfer archives for lost FULL outputs.

Only file hashes, archive member names and saved JSON metadata are read. No
archive is extracted, and no model, training, prediction or evaluator is run.
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import re
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
ROOT = PAPER.parent
REMOTE = PAPER / '远程公平对照_20261103'
OUT = HERE / 'full_local_backup_audit.json'
STATS = HERE / 'external_record_search_stats.txt'
ERRORS = HERE / 'external_record_search_stderr.txt'
PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
EMA127 = '17df8ef8e5cdea764027320c7c769bc59ce7a7a53bc10da3e0fe225772d6b257'


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def main():
    assert not any(path.exists() for path in [OUT, STATS, ERRORS])
    receipt_path = REMOTE / 'early_full_evidence_download_audit.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    assert receipt['status'] == 'EARLY_FULL_SAVED_EVIDENCE_VERIFIED_AFTER_DOWNLOAD'
    assert receipt['actual_epoch'] == 0 and receipt['protocol_sha256'] == PROTOCOL
    early_root = REMOTE / 'secondary_evidence/evidence_audits/FULL_best_epoch0000'
    early_files = {}
    for name, expected in receipt['files'].items():
        path = early_root / name
        assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256']
        early_files[name] = ref(path)
    snapshot_path = REMOTE / 'parallel_evidence/FULL_before_connection_loss_20261104.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8-sig'))
    observation = json.loads(snapshot['stdout'])
    full = observation['arms']['FULL']
    assert full['complete_epoch_rows'] == full['receipt']['completed_epochs'] == 128
    latest = full['latest_complete_epoch']
    assert latest['epoch'] == 127 and latest['ema_tensor_sha256'] == EMA127
    assert latest['protocol_sha256'] == PROTOCOL
    assert not any((early_root / name).exists() for name in ['epoch0127.pt', 'checkpoint_epoch0127.pt'])

    archives = []
    for name in ['input_bundle.zip', 'secondary_frozen_inputs.tar.gz']:
        path = REMOTE / name
        if name.endswith('.zip'):
            with ZipFile(path) as archive:
                members = archive.namelist()
        else:
            with tarfile.open(path, 'r:gz') as archive:
                members = [entry.name for entry in archive]
        hits = [value for value in members if any(pattern in value.lower().replace('\\', '/')
                for pattern in ['runs/full/', 'full_best_epoch', 'epoch0127', 'secondary_queue.json'])]
        archives.append({'archive': ref(path), 'members': len(members), 'member_names': members,
                         'full_training_output_name_matches': hits,
                         'full_named_input_members': [value for value in members if 'full' in value.lower()]})
    expected = json.loads((REMOTE / 'secondary_frozen_inputs_receipt.json').read_text(encoding='utf-8-sig'))
    assert archives[1]['archive']['sha256'] == expected['sha256'] and archives[1]['members'] == expected['files']
    assert not any(archive['full_training_output_name_matches'] for archive in archives)

    globs = ['*.json', '*.jsonl', '*.txt', '*.md', '!**/.git', '!**/.codex', '!**/.agents',
             '!**/runtime*/**', '!**/local_deps/**', '!**/build_tools/**', '!**/yolo_smoke_runtime_20261029',
             '!**/cpu_smoke_runtime_20261029', '!**/native_import_deps', '!**/isolated_runtime',
             '!**/analysis_tools', '!**/aps_official_eval_deps', '!**/c10_eval_test_deps']
    rg = shutil.which('rg')
    cmd = [rg, '-l', '--stats', '-F', '--hidden', '--no-ignore', '--max-filesize', '32M',
           '-e', PROTOCOL, '-e', EMA127, '-e', 'mpsma_fair_20261103_v1',
           '04_实验记录', '05_代码与数据', '90_历史与工具']
    for glob in globs:
        cmd.extend(['-g', glob])
    search = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace', check=False)
    assert search.returncode == 1 and not search.stderr
    matches = re.search(r'(?m)^(\d+) files contained matches$', search.stdout)
    searched = re.search(r'(?m)^(\d+) files searched$', search.stdout)
    assert matches and searched and int(matches.group(1)) == 0
    STATS.write_text(search.stdout, encoding='utf-8')
    ERRORS.write_text(search.stderr, encoding='utf-8')
    output = {'status': 'BOUNDED_LOCAL_FULL_BACKUP_SEARCH_COMPLETE', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'early_epoch0_receipt': ref(receipt_path), 'early_epoch0_files': early_files,
              'latest_snapshot': ref(snapshot_path), 'latest_confirmed_epoch_zero_based': latest['epoch'],
              'latest_confirmed_ema_tensor_sha256': EMA127, 'latest_snapshot_is_metadata_only': True,
              'archives': archives, 'external_search': {'roots': ['04_实验记录', '05_代码与数据', '90_历史与工具'],
                  'extensions': ['json', 'jsonl', 'txt', 'md'], 'max_file_size': '32M', 'globs': globs,
                  'patterns': ['protocol_sha256', 'epoch127_ema_tensor_sha256', 'experiment_directory_name'],
                  'files_searched': int(searched.group(1)), 'files_with_matches': int(matches.group(1)),
                  'exit_code': search.returncode, 'stats': ref(STATS), 'stderr': ref(ERRORS)},
              'limits': ['The two archives were searched by member names, not by interpreting file contents or extracting them.',
                         'External text search excludes other file types, files over 32M, excluded directories, and archive contents.',
                         'A missing local match does not prove the original remote instance or an external backup cannot be recovered.',
                         'The snapshot does not supply the epoch-127 weights or validate the preselected epoch-319 endpoint.',
                         'No model, prediction, evaluator or training was run.'],
              'model_instantiations': 0, 'model_forwards': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': output['status'], 'first_epoch_files_verified': len(early_files),
                      'latest_snapshot_epoch': latest['epoch'], 'archives_checked': len(archives),
                      'external_text_files_searched': int(searched.group(1)), 'external_matches': 0,
                      'sha256': digest(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
