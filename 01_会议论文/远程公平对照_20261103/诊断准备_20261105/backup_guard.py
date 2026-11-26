"""Verify local copies of diagnostic artifacts before issuing a backup acknowledgement.

This file does not contact a server, construct a model, evaluate or train. It is
one preparatory component, not a complete diagnostic runner or authorization.
"""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import argparse
import json
import os
import re
import shutil

HERE = Path(__file__).resolve().parent
REMOTE_WORK = HERE.parent
WORKSPACE = HERE.parents[2]
PLAN = REMOTE_WORK / '基线骤降最小诊断方案_待授权_20261105.txt'
STORAGE_PLAN = REMOTE_WORK / '诊断存储与中断保护补充_待授权_20261105.txt'
PLAN_SHA = 'fc8804407fcdbd9dd8210cd7f2f3c34ae503189c049659cbccf45a0023eff4fa'
STORAGE_PLAN_SHA = '7dfb4c22fd467154713d9cf64451ecfe970358122b70b4197110b809b2ff4008'
CHECKPOINT_SHA = '33bb8012aa6b43a505b2f1a554189ed6abeeb4c39dc5cd6f20e83726fb922f65'
RUN_ID = 'mpsma_diag_base_apdrop_20261105_v1'
REMOTE_RUN = '/25085400120/mpsma_fair_20261103_v1/diagnostics/' + RUN_ID
MAX_FILE_BYTES = 2 * 1024**3
START_EPOCH, FINAL_EPOCH, BATCHES = 213, 224, 1617
EVAL_FILES = {'predictions.json', 'evaluation_arrays.npz', 'metrics.json', 'phase_record.json'}
EPOCH_FILES = EVAL_FILES | {'checkpoint.pt', 'learning_rates.csv.gz', 'gradient_summary.json'}

class BackupError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise BackupError(message)

def digest_bytes(data):
    return sha256(data).hexdigest()

def digest_file(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def check_plans():
    require(digest_file(PLAN) == PLAN_SHA, 'Diagnostic plan changed; review before backup')
    require(digest_file(STORAGE_PLAN) == STORAGE_PLAN_SHA, 'Storage plan changed; review before backup')

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackupError('Duplicate JSON key: ' + key)
        result[key] = value
    return result

def parse_manifest(data):
    require(isinstance(data, bytes) and 0 < len(data) <= 1024 * 1024, 'Invalid manifest size')
    try:
        obj = json.loads(data.decode('utf-8'), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BackupError('Invalid UTF-8 JSON manifest') from error
    require(isinstance(obj, dict), 'Manifest must be an object')
    require(set(obj) == {'schema', 'status', 'run_id', 'plan_sha256', 'storage_plan_sha256',
                         'input_checkpoint_sha256', 'nonce', 'phase', 'epoch_index',
                         'optimizer_updates', 'validation_passes', 'files'}, 'Unexpected manifest fields')
    require(obj.get('schema') == 1 and type(obj.get('schema')) is int, 'Unexpected schema')
    require(obj.get('status') == 'READY_FOR_BACKUP' and obj.get('run_id') == RUN_ID, 'Unexpected run or status')
    require(obj.get('plan_sha256') == PLAN_SHA and obj.get('storage_plan_sha256') == STORAGE_PLAN_SHA,
            'Unreviewed plan')
    require(obj.get('input_checkpoint_sha256') == CHECKPOINT_SHA, 'Wrong input checkpoint')
    nonce = obj.get('nonce')
    require(isinstance(nonce, str) and re.fullmatch(r'[0-9a-f]{32}', nonce) is not None, 'Invalid nonce')
    phase = obj.get('phase')
    epoch, updates, validations = (obj.get(k) for k in ['epoch_index', 'optimizer_updates', 'validation_passes'])
    require(all(type(x) is int for x in [epoch, updates, validations]), 'Counters must be integers')
    if phase == 'initial_ema':
        require((epoch, updates, validations) == (START_EPOCH, 0, 1), 'Invalid initial EMA counters')
        expected = EVAL_FILES
    elif phase == 'initial_raw':
        require((epoch, updates, validations) == (START_EPOCH, 0, 2), 'Invalid initial raw counters')
        expected = EVAL_FILES
    elif isinstance(phase, str) and re.fullmatch(r'epoch_\d{4}', phase):
        require(START_EPOCH + 1 <= epoch <= FINAL_EPOCH and phase == f'epoch_{epoch:04d}', 'Invalid epoch')
        require(updates == (epoch - START_EPOCH) * BATCHES and validations == 2 + epoch - START_EPOCH,
                'Invalid epoch counters')
        expected = EPOCH_FILES
    elif phase == 'terminal_raw':
        require(START_EPOCH + 1 <= epoch <= FINAL_EPOCH, 'Invalid terminal epoch')
        require(updates == (epoch - START_EPOCH) * BATCHES and validations == 3 + epoch - START_EPOCH,
                'Invalid terminal counters')
        expected = EVAL_FILES
    else:
        raise BackupError('Unexpected phase')
    require(0 <= updates <= 11 * BATCHES and 1 <= validations <= 14, 'Budget exceeded')
    files = obj.get('files')
    require(isinstance(files, list) and len(files) == len(expected), 'Wrong file count')
    names = []
    for item in files:
        require(isinstance(item, dict), 'Invalid file item')
        require(set(item) == {'name', 'bytes', 'sha256'}, 'Unexpected file fields')
        name, size, digest = item.get('name'), item.get('bytes'), item.get('sha256')
        require(isinstance(name, str) and name in expected, 'Unexpected filename')
        require(type(size) is int and 0 <= size <= MAX_FILE_BYTES, 'Invalid file size')
        require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest) is not None, 'Invalid digest')
        names.append(name)
    require(set(names) == expected and len({n.casefold() for n in names}) == len(names), 'Duplicate or missing file')
    return obj

def checked_backup_root(path):
    path = Path(path)
    require('..' not in path.parts, 'Parent traversal in backup root')
    lexical = path.absolute()
    workspace = WORKSPACE.resolve()
    require(lexical.is_relative_to(workspace), 'Backup root must be inside the workspace')
    current = workspace
    for part in lexical.relative_to(workspace).parts:
        current = current / part
        require(not current.is_symlink() and not getattr(current, 'is_junction', lambda: False)(),
                'Symlink or junction in backup path')
    resolved = path.resolve()
    require(resolved.is_relative_to(workspace), 'Resolved backup root outside the workspace')
    return resolved

def predecessor(obj):
    phase, epoch = obj['phase'], obj['epoch_index']
    if phase == 'initial_ema': return None
    if phase == 'initial_raw': return 'initial_ema'
    if phase == 'terminal_raw': return f'epoch_{epoch:04d}'
    return 'initial_raw' if epoch == START_EPOCH + 1 else f'epoch_{epoch - 1:04d}'

def read_ack_for_previous(run_root, phase):
    stage = run_root / phase
    require(not stage.is_symlink() and not getattr(stage, 'is_junction', lambda: False)(), 'Unsafe previous stage')
    path = stage / 'ack.json'
    require(path.is_file() and not path.is_symlink(), 'Previous verified backup missing')
    manifest_path = stage / 'manifest.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'Previous manifest missing')
    try:
        ack = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise BackupError('Previous backup acknowledgement unreadable') from error
    data = manifest_path.read_bytes()
    prior = parse_manifest(data)
    require(ack.get('status') == 'LOCAL_BACKUP_VERIFIED' and ack.get('run_id') == RUN_ID
            and ack.get('phase') == phase and prior['phase'] == phase, 'Previous acknowledgement invalid')
    require(ack.get('manifest_sha256') == digest_bytes(data) and ack.get('nonce') == prior['nonce'],
            'Previous acknowledgement does not bind its manifest')
    require(all(ack.get(k) == prior[k] for k in ['epoch_index', 'optimizer_updates', 'validation_passes']),
            'Previous acknowledgement counters differ')
    expected = {'manifest.json', 'ack.json'} | {i['name'] for i in prior['files']}
    require({p.name for p in stage.iterdir()} == expected, 'Previous backup has missing or unexpected entries')
    ack_files = ack.get('files')
    require(isinstance(ack_files, list) and len(ack_files) == len(prior['files']), 'Previous file receipt invalid')
    by_name = {item.get('name'): item for item in ack_files if isinstance(item, dict)}
    require(len(by_name) == len(ack_files), 'Duplicate previous file receipt')
    for item in prior['files']:
        actual = stage / item['name']
        require(actual.is_file() and not actual.is_symlink() and actual.stat().st_size == item['bytes'],
                'Previous backup file missing or size changed')
        require(digest_file(actual) == item['sha256'] and by_name.get(item['name']) == item,
                'Previous backup file or receipt changed')

def prepare(manifest_path, backup_root):
    check_plans()
    manifest_path = Path(manifest_path)
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'Manifest must be a regular file')
    require(0 < manifest_path.stat().st_size <= 1024 * 1024, 'Invalid manifest size')
    data = manifest_path.read_bytes()
    obj = parse_manifest(data)
    root = checked_backup_root(backup_root)
    # Check real free space at every boundary, allowing headroom beyond this
    # phase's artifacts.  Never delete earlier evidence to make space.
    existing_parent = root
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    require(shutil.disk_usage(existing_parent).free > sum(item['bytes'] for item in obj['files']) + 512 * 1024**2,
            'Insufficient local space for the next verified copy')
    run_root = root / RUN_ID
    require(not run_root.is_symlink() and not getattr(run_root, 'is_junction', lambda: False)(), 'Unsafe run root')
    require(run_root.resolve().is_relative_to(root), 'Run root outside backup root')
    prior = predecessor(obj)
    if prior: read_ack_for_previous(run_root, prior)
    stage = run_root / obj['phase']
    require(stage.resolve().is_relative_to(root), 'Stage outside backup root')
    stage.mkdir(parents=True, exist_ok=False)
    with (stage / 'manifest.json').open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return {'status': 'LOCAL_BACKUP_STAGE_PREPARED', 'run_id': RUN_ID, 'phase': obj['phase'],
            'manifest_sha256': digest_bytes(data), 'local_stage': str(stage),
            'downloads': [{'remote': REMOTE_RUN + '/phases/' + obj['phase'] + '/' + item['name'],
                           'local': str(stage / item['name']), 'bytes': item['bytes'], 'sha256': item['sha256']}
                          for item in obj['files']]}

def verify(manifest_path, backup_root):
    check_plans()
    manifest_path = Path(manifest_path)
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'Manifest must be a regular file')
    require(0 < manifest_path.stat().st_size <= 1024 * 1024, 'Invalid manifest size')
    data = manifest_path.read_bytes()
    obj = parse_manifest(data)
    root = checked_backup_root(backup_root)
    run_root = root / RUN_ID
    require(not run_root.is_symlink() and not getattr(run_root, 'is_junction', lambda: False)(), 'Unsafe run root')
    stage = run_root / obj['phase']
    require(stage.is_dir() and not stage.is_symlink() and not getattr(stage, 'is_junction', lambda: False)()
            and stage.resolve().is_relative_to(root), 'Stage missing or unsafe')
    prior = predecessor(obj)
    if prior: read_ack_for_previous(run_root, prior)
    require((stage / 'manifest.json').is_file() and not (stage / 'manifest.json').is_symlink(), 'Staged manifest missing')
    require((stage / 'manifest.json').read_bytes() == data, 'Manifest changed after preparation')
    require(not (stage / 'ack.json').exists(), 'Acknowledgement already exists')
    expected = {'manifest.json'} | {i['name'] for i in obj['files']}
    require({p.name for p in stage.iterdir()} == expected, 'Missing or unexpected staged entries')
    verified = []
    for item in obj['files']:
        path = stage / item['name']
        require(path.is_file() and not path.is_symlink(), 'Backup file missing or a symlink')
        before = path.stat()
        require(before.st_size == item['bytes'], 'Backup size mismatch: ' + item['name'])
        digest = digest_file(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Backup changed while read')
        require(digest == item['sha256'], 'Backup digest mismatch: ' + item['name'])
        verified.append({'name': item['name'], 'bytes': after.st_size, 'sha256': digest})
    ack = {'schema': 1, 'status': 'LOCAL_BACKUP_VERIFIED', 'run_id': RUN_ID, 'phase': obj['phase'],
           'epoch_index': obj['epoch_index'], 'optimizer_updates': obj['optimizer_updates'],
           'validation_passes': obj['validation_passes'], 'nonce': obj['nonce'],
           'manifest_sha256': digest_bytes(data), 'files': verified,
           'verified_local_utc': datetime.now(timezone.utc).isoformat()}
    with (stage / 'ack.json').open('x', encoding='utf-8') as f:
        json.dump(ack, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    return {'status': ack['status'], 'run_id': RUN_ID, 'phase': obj['phase'],
            'manifest_sha256': ack['manifest_sha256'], 'local_ack': str(stage / 'ack.json')}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--backup-root', required=True)
    args = parser.parse_args()
    result = prepare(args.manifest, args.backup_root) if args.mode == 'prepare' else verify(args.manifest, args.backup_root)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
