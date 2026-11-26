"""Inert file-integrity guard for a future two-arm experiment.

This module does not import a model, connect to a server, or start training. It
verifies bytes and metadata, not the semantic contents of checkpoint tensors or
random state. A future runner and transport must integrate and test it before
any launch.
"""
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import shutil
import stat

EPOCHS = 320
BATCHES = 1617
MAX_FILE_BYTES = 2_000_000_000
MAX_MANIFEST_BYTES = 1024 * 1024
BASE_FILES = {'evaluated.pt', 'predictions.json', 'evaluation_arrays.npz',
              'learning_rates.csv.gz', 'phase_record.json'}
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z')
NONCE = re.compile(r'[0-9a-f]{32}\Z')


class BackupError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BackupError(message)


def digest_bytes(data):
    return sha256(data).hexdigest()


def digest_file(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse_json(data):
    require(isinstance(data, bytes) and 0 < len(data) <= MAX_MANIFEST_BYTES, 'Invalid JSON size')
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(BackupError('Non-finite JSON number')))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError('Invalid UTF-8 JSON') from exc


def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def regular_file(path):
    path = Path(path)
    require(not path.is_symlink() and not getattr(path, 'is_junction', lambda: False)(), 'Linked file is not allowed')
    try:
        info = path.stat()
    except OSError as exc:
        raise BackupError('Missing file: ' + str(path)) from exc
    require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_FILE_BYTES, 'Invalid file: ' + str(path))
    return info


def expected_files(restored):
    require(type(restored) is bool, 'Restore flag must be boolean')
    return BASE_FILES | ({'continuation.pt'} if restored else set())


def validate_identity(run_id, arm, protocol_sha256):
    require(isinstance(run_id, str) and RUN_ID.fullmatch(run_id), 'Invalid run id')
    require(arm in ('BASE', 'FULL'), 'Invalid arm')
    require(isinstance(protocol_sha256, str) and HEX64.fullmatch(protocol_sha256), 'Invalid protocol digest')


def parse_manifest(data, run_id, arm, protocol_sha256):
    validate_identity(run_id, arm, protocol_sha256)
    m = parse_json(data)
    keys = {'schema', 'status', 'run_id', 'arm', 'protocol_sha256', 'epoch_index',
            'optimizer_updates', 'validation_passes', 'restore_after_evaluation', 'nonce', 'files'}
    require(isinstance(m, dict) and set(m) == keys, 'Unexpected manifest fields')
    require(type(m['schema']) is int and m['schema'] == 1 and m['status'] == 'EPOCH_ARCHIVE_READY', 'Invalid manifest version')
    require((m['run_id'], m['arm'], m['protocol_sha256']) == (run_id, arm, protocol_sha256), 'Manifest identity mismatch')
    epoch = m['epoch_index']
    require(type(epoch) is int and 0 <= epoch < EPOCHS, 'Invalid epoch')
    require(type(m['optimizer_updates']) is int and m['optimizer_updates'] == (epoch + 1) * BATCHES,
            'Unexpected optimizer update count')
    require(type(m['validation_passes']) is int and m['validation_passes'] == epoch + 1,
            'Unexpected validation count')
    restored = m['restore_after_evaluation']
    names = expected_files(restored)
    require(not restored or epoch >= 296, 'Post-validation restore outside configured range')
    require(isinstance(m['nonce'], str) and NONCE.fullmatch(m['nonce']), 'Invalid nonce')
    require(isinstance(m['files'], list) and len(m['files']) == len(names), 'Wrong file count')
    seen = set()
    for item in m['files']:
        require(isinstance(item, dict) and set(item) == {'name', 'bytes', 'sha256'}, 'Invalid file entry')
        name = item['name']
        require(isinstance(name, str) and name in names and name not in seen, 'Unexpected or duplicate filename')
        require(type(item['bytes']) is int and 0 < item['bytes'] <= MAX_FILE_BYTES, 'Invalid file size')
        require(isinstance(item['sha256'], str) and HEX64.fullmatch(item['sha256']), 'Invalid file digest')
        seen.add(name)
    require(seen == names, 'Missing file')
    return m


def build_manifest(stage, run_id, arm, protocol_sha256, epoch, restored, nonce):
    """Build bytes for a completed stage; does not create or publish files."""
    validate_identity(run_id, arm, protocol_sha256)
    require(type(epoch) is int and 0 <= epoch < EPOCHS, 'Invalid epoch')
    require(isinstance(nonce, str) and NONCE.fullmatch(nonce), 'Invalid nonce')
    names = expected_files(restored)
    stage = Path(stage)
    require(stage.is_dir() and not stage.is_symlink() and not getattr(stage, 'is_junction', lambda: False)(),
            'Invalid stage directory')
    require({p.name for p in stage.iterdir()} == names, 'Unexpected stage contents')
    files = []
    for name in sorted(names):
        path = stage / name
        before = regular_file(path)
        sha = digest_file(path)
        after = regular_file(path)
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed while hashing')
        files.append({'name': name, 'bytes': after.st_size, 'sha256': sha})
    m = {'schema': 1, 'status': 'EPOCH_ARCHIVE_READY', 'run_id': run_id, 'arm': arm,
         'protocol_sha256': protocol_sha256, 'epoch_index': epoch,
         'optimizer_updates': (epoch + 1) * BATCHES, 'validation_passes': epoch + 1,
         'restore_after_evaluation': restored, 'nonce': nonce, 'files': files}
    data = encode_json(m)
    parse_manifest(data, run_id, arm, protocol_sha256)
    check_phase_record(stage, m)
    return data


def check_phase_record(stage, manifest):
    path = Path(stage) / 'phase_record.json'
    before = regular_file(path)
    data = path.read_bytes()
    after = regular_file(path)
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Phase record changed while read')
    files = {item['name']: item['sha256'] for item in manifest['files']}
    require(digest_bytes(data) == files['phase_record.json'], 'Phase record digest mismatch')
    phase = parse_json(data)
    require(isinstance(phase, dict), 'Invalid phase record')
    for key in ('run_id', 'arm', 'protocol_sha256', 'epoch_index', 'optimizer_updates',
                'validation_passes', 'restore_after_evaluation'):
        require(phase.get(key) == manifest[key] and type(phase.get(key)) is type(manifest[key]),
                'Phase record mismatch: ' + key)
    require(phase.get('evaluated_checkpoint_sha256') == files['evaluated.pt'], 'Evaluation checkpoint binding mismatch')
    require(phase.get('prediction_sha256') == files['predictions.json'], 'Prediction binding mismatch')
    require(phase.get('evaluation_arrays_sha256') == files['evaluation_arrays.npz'], 'Array binding mismatch')
    require(phase.get('continuation_checkpoint_sha256') == files.get('continuation.pt'),
            'Continuation checkpoint binding mismatch')
    require(isinstance(phase.get('evaluated_tensor_sha256'), str) and HEX64.fullmatch(phase['evaluated_tensor_sha256']),
            'Missing evaluated tensor digest')


def checked_root(root, allowed_root):
    root, allowed_root = Path(root).absolute(), Path(allowed_root).resolve(strict=True)
    require(root.is_relative_to(allowed_root), 'Backup root outside allowed workspace')
    cursor = allowed_root
    for part in root.relative_to(allowed_root).parts:
        cursor = cursor / part
        require(not cursor.is_symlink() and not getattr(cursor, 'is_junction', lambda: False)(), 'Linked backup path')
    require(root.resolve().is_relative_to(allowed_root), 'Resolved backup root outside workspace')
    return root


def stage_path(root, manifest):
    root = Path(root)
    stage = root
    for part in (manifest['run_id'], manifest['arm'], 'epoch_%04d' % manifest['epoch_index']):
        stage = stage / part
        require(not stage.is_symlink() and not getattr(stage, 'is_junction', lambda: False)(), 'Linked stage path')
    require(stage.resolve().is_relative_to(root.resolve()), 'Stage outside backup root')
    return stage


def validate_ack(data, manifest_data, manifest):
    ack = parse_json(data)
    keys = {'schema', 'status', 'run_id', 'arm', 'protocol_sha256', 'epoch_index',
            'optimizer_updates', 'validation_passes', 'nonce', 'manifest_sha256', 'files'}
    require(isinstance(ack, dict) and set(ack) == keys, 'Unexpected acknowledgement fields')
    require(type(ack['schema']) is int and ack['schema'] == 1 and ack['status'] == 'LOCAL_BACKUP_VERIFIED',
            'Invalid acknowledgement')
    for key in ('run_id', 'arm', 'protocol_sha256', 'epoch_index', 'optimizer_updates', 'validation_passes', 'nonce'):
        require(ack[key] == manifest[key] and type(ack[key]) is type(manifest[key]), 'Acknowledgement mismatch: ' + key)
    require(ack['manifest_sha256'] == digest_bytes(manifest_data), 'Acknowledgement manifest mismatch')
    require(ack['files'] == manifest['files'], 'Acknowledgement file list mismatch')
    return ack


def check_files(stage, manifest, include_ack):
    stage = Path(stage)
    require(stage.is_dir() and not stage.is_symlink() and not getattr(stage, 'is_junction', lambda: False)(),
            'Missing or linked stage')
    names = {x['name'] for x in manifest['files']} | {'manifest.json'} | ({'ack.json'} if include_ack else set())
    require({p.name for p in stage.iterdir()} == names, 'Missing or unexpected staged files')
    for item in manifest['files']:
        path = stage / item['name']
        before = regular_file(path)
        require(before.st_size == item['bytes'], 'File size mismatch: ' + item['name'])
        actual = digest_file(path)
        after = regular_file(path)
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed during verification')
        require(actual == item['sha256'], 'File digest mismatch: ' + item['name'])
    check_phase_record(stage, manifest)


def check_previous(root, manifest):
    if manifest['epoch_index'] == 0:
        return
    previous = dict(manifest, epoch_index=manifest['epoch_index'] - 1)
    stage = stage_path(root, previous)
    manifest_path, ack_path = stage / 'manifest.json', stage / 'ack.json'
    regular_file(manifest_path)
    regular_file(ack_path)
    data = manifest_path.read_bytes()
    prior = parse_manifest(data, manifest['run_id'], manifest['arm'], manifest['protocol_sha256'])
    require(prior['epoch_index'] == previous['epoch_index'], 'Previous epoch mismatch')
    ack_data = ack_path.read_bytes()
    validate_ack(ack_data, data, prior)
    check_files(stage, prior, include_ack=True)
    require(manifest_path.read_bytes() == data and ack_path.read_bytes() == ack_data,
            'Previous manifest or acknowledgement changed during verification')


def prepare(manifest_data, root, allowed_root, run_id, arm, protocol_sha256, reserve_bytes=512 * 1024**2):
    """Create a fresh local stage after validating its predecessor and free space."""
    manifest = parse_manifest(manifest_data, run_id, arm, protocol_sha256)
    root = checked_root(root, allowed_root)
    require(type(reserve_bytes) is int and reserve_bytes >= 0, 'Invalid reserve')
    require(shutil.disk_usage(Path(allowed_root).resolve()).free > reserve_bytes + sum(x['bytes'] for x in manifest['files']),
            'Insufficient local space')
    check_previous(root, manifest)
    stage = stage_path(root, manifest)
    require(stage.resolve().is_relative_to(Path(allowed_root).resolve()), 'Stage outside workspace')
    stage.mkdir(parents=True, exist_ok=False)
    with (stage / 'manifest.json').open('xb') as stream:
        stream.write(manifest_data)
        stream.flush()
        os.fsync(stream.fileno())
    return stage


def verify(manifest_data, root, allowed_root, run_id, arm, protocol_sha256):
    """Verify copied bytes and create an acknowledgement only after full validation."""
    manifest = parse_manifest(manifest_data, run_id, arm, protocol_sha256)
    root = checked_root(root, allowed_root)
    stage = stage_path(root, manifest)
    require(stage.resolve().is_relative_to(Path(allowed_root).resolve()), 'Stage outside workspace')
    check_previous(root, manifest)
    regular_file(stage / 'manifest.json')
    require((stage / 'manifest.json').read_bytes() == manifest_data, 'Staged manifest changed')
    require(not (stage / 'ack.json').exists(), 'Acknowledgement already exists')
    check_files(stage, manifest, include_ack=False)
    require((stage / 'manifest.json').read_bytes() == manifest_data, 'Staged manifest changed during verification')
    ack = {'schema': 1, 'status': 'LOCAL_BACKUP_VERIFIED',
           **{k: manifest[k] for k in ('run_id', 'arm', 'protocol_sha256', 'epoch_index',
                                       'optimizer_updates', 'validation_passes', 'nonce')},
           'manifest_sha256': digest_bytes(manifest_data), 'files': manifest['files']}
    data = encode_json(ack)
    validate_ack(data, manifest_data, manifest)
    with (stage / 'ack.json').open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return data
