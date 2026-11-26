"""Shared bounds and integrity helpers for the explicitly authorized diagnostic."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import math
import os
import secrets
import time

RUN_ID = 'mpsma_diag_base_apdrop_20261105_v1'
ORIGINAL = Path('/25085400120/mpsma_fair_20261103_v1')
ROOT = ORIGINAL / 'diagnostics' / RUN_ID
CACHE = Path('/tmp') / (RUN_ID + '_data')
PLAN_SHA = 'fc8804407fcdbd9dd8210cd7f2f3c34ae503189c049659cbccf45a0023eff4fa'
STORAGE_SHA = '7dfb4c22fd467154713d9cf64451ecfe970358122b70b4197110b809b2ff4008'
AUTH_SHA = 'aa7698e43d3a7b09a24c0787c86610aa8ce87d8b8590c303ff72c375b1f4a107'
PROTOCOL_SHA = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
DATA_SHA = 'fdde141711867fa1bc2de3a1e6342cca257eb056e2a1700795eb04068aa90266'
DATA_MANIFEST_SHA = 'ba463a04fe688d5c3005cb3cf9ca6e13aa699b0b11016e15e7fa9b76b52e2a56'
BASE_CONFIG_SHA = '1eed16436115ab62655c101e42f180e87fabb2f1a9e55d07b1d5241ce0070cdd'
CHECKPOINT_SHA = '33bb8012aa6b43a505b2f1a554189ed6abeeb4c39dc5cd6f20e83726fb922f65'
START_EPOCH, FINAL_EPOCH, BATCHES = 213, 224, 1617
MAX_UPDATES, MAX_VALIDATIONS, BACKUP_TIMEOUT = 17787, 14, 600
EVAL_FILES = {'predictions.json', 'evaluation_arrays.npz', 'metrics.json', 'phase_record.json'}
EPOCH_FILES = EVAL_FILES | {'checkpoint.pt', 'learning_rates.csv.gz', 'gradient_summary.json'}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def now():
    return datetime.now(timezone.utc).isoformat()

def fsync_file(path):
    with Path(path).open('rb+') as f:
        f.flush()
        os.fsync(f.fileno())

def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def atomic_json(path, value, *, new=False):
    path = Path(path)
    require(path.parent.is_dir(), 'Missing parent directory: ' + str(path.parent))
    if new:
        require(not path.exists() and not path.is_symlink(), 'Output already exists: ' + str(path))
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)

def auth_check():
    require(ROOT.resolve() == Path(__file__).resolve().parent, 'Script not in the approved isolated directory')
    receipt = ROOT / 'authorization.json'
    require(sha(receipt) == AUTH_SHA, 'Authorization receipt hash mismatch')
    auth = read(receipt)
    require(auth['status'] == 'EXPLICITLY_AUTHORIZED_ONE_BOUNDED_DIAGNOSTIC', 'Authorization absent')
    for key, expected in [('run_id', RUN_ID), ('plan_sha256', PLAN_SHA), ('storage_supplement_sha256', STORAGE_SHA),
                          ('input_checkpoint_sha256', CHECKPOINT_SHA), ('host_port', 30166), ('gpus', 1),
                          ('batch_size', 4), ('seed', 0), ('max_training_epochs', 11),
                          ('max_optimizer_updates', MAX_UPDATES), ('max_full_validation_passes', MAX_VALIDATIONS),
                          ('images_per_validation', 548), ('backup_timeout_seconds', BACKUP_TIMEOUT)]:
        require(auth[key] == expected, 'Authorization field mismatch: ' + key)
    require(auth['fp32'] is True and auth['automatic_retries'] is False, 'Unexpected authorization mode')
    require(auth['hyperparameter_tuning_authorized'] is False and auth['full_fair_pair_or_additional_seeds_authorized'] is False,
            'Unexpected authorized scope')
    require(sha(ROOT / 'plan.txt') == PLAN_SHA and sha(ROOT / 'storage_plan.txt') == STORAGE_SHA,
            'Reviewed plan bytes changed')
    return auth

def package_check():
    package = read(ROOT / 'code_manifest.json')
    require(package['run_id'] == RUN_ID and package['authorization_sha256'] == AUTH_SHA, 'Code manifest mismatch')
    for item in package['files']:
        name = item['name']
        require(name in {'diagnostic_contract.py', 'remote_setup.py', 'diagnostic_worker.py',
                         'diagnostic_controller.py', 'diagnostic_status.py'}, 'Unexpected code file')
        require(sha(ROOT / name) == item['sha256'], 'Code changed: ' + name)
    require({x['name'] for x in package['files']} == {'diagnostic_contract.py', 'remote_setup.py',
            'diagnostic_worker.py', 'diagnostic_controller.py', 'diagnostic_status.py'}, 'Code file missing')
    return package

def expected_counters(phase, epoch):
    if phase == 'initial_ema':
        require(epoch == START_EPOCH, 'Wrong initial epoch')
        return 0, 1
    if phase == 'initial_raw':
        require(epoch == START_EPOCH, 'Wrong initial epoch')
        return 0, 2
    if phase == 'terminal_raw':
        require(START_EPOCH < epoch <= FINAL_EPOCH, 'Wrong terminal epoch')
        n = epoch - START_EPOCH
        return n * BATCHES, n + 3
    require(phase == 'epoch_%04d' % epoch and START_EPOCH < epoch <= FINAL_EPOCH, 'Wrong training epoch')
    n = epoch - START_EPOCH
    return n * BATCHES, n + 2

def make_backup_manifest(phase, epoch):
    stage = ROOT / 'phases' / phase
    updates, validations = expected_counters(phase, epoch)
    require(updates <= MAX_UPDATES and validations <= MAX_VALIDATIONS, 'Budget exceeded')
    expected = EPOCH_FILES if phase.startswith('epoch_') else EVAL_FILES
    require({p.name for p in stage.iterdir()} == expected, 'Unexpected phase files before manifest')
    files = []
    for name in sorted(expected):
        path = stage / name
        require(path.is_file() and not path.is_symlink(), 'Phase file missing: ' + name)
        fsync_file(path)
        st = path.stat()
        require(st.st_size <= 2 * 1024**3, 'Phase file exceeds allowed size')
        files.append({'name': name, 'bytes': st.st_size, 'sha256': sha(path)})
    fsync_dir(stage)
    manifest = {'schema': 1, 'status': 'READY_FOR_BACKUP', 'run_id': RUN_ID,
                'plan_sha256': PLAN_SHA, 'storage_plan_sha256': STORAGE_SHA,
                'input_checkpoint_sha256': CHECKPOINT_SHA, 'nonce': secrets.token_hex(16),
                'phase': phase, 'epoch_index': epoch, 'optimizer_updates': updates,
                'validation_passes': validations, 'files': files}
    atomic_json(stage / 'manifest.json', manifest, new=True)
    return manifest

def await_backup(phase, manifest):
    stage = ROOT / 'phases' / phase
    manifest_hash = sha(stage / 'manifest.json')
    start = time.monotonic()
    # The local monitor uploads the complete acknowledgement first, then a
    # zero-byte marker in a separate SFTP operation.  Never read a file while
    # SFTP may still be writing it.
    ack_path = stage / 'ack.uploading'
    marker = stage / 'ack.ready'
    while not marker.exists():
        if time.monotonic() - start >= BACKUP_TIMEOUT:
            raise TimeoutError('Local backup acknowledgement not received within 600 seconds: ' + phase)
        time.sleep(2)
    require(marker.is_file() and not marker.is_symlink() and marker.stat().st_size == 0,
            'Invalid acknowledgement marker')
    require(ack_path.is_file() and not ack_path.is_symlink(), 'Invalid acknowledgement file')
    ack = read(ack_path)
    require(ack.get('status') == 'LOCAL_BACKUP_VERIFIED' and ack.get('run_id') == RUN_ID,
            'Invalid backup acknowledgement')
    require(ack.get('manifest_sha256') == manifest_hash and ack.get('nonce') == manifest['nonce'],
            'Acknowledgement does not bind phase manifest')
    for key in ['phase', 'epoch_index', 'optimizer_updates', 'validation_passes']:
        require(ack.get(key) == manifest[key], 'Acknowledgement counter mismatch: ' + key)
    require(sorted(ack.get('files', []), key=lambda x: x['name']) ==
            sorted(manifest['files'], key=lambda x: x['name']), 'Acknowledged file list mismatch')
    for item in manifest['files']:
        path = stage / item['name']
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Remote phase artifact changed before acknowledgement')
    return {'manifest_sha256': manifest_hash, 'ack_sha256': sha(ack_path),
            'acknowledgement_wait_seconds': time.monotonic() - start}

def finite_metrics(metrics):
    names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
    require(set(metrics) == set(names) and all(math.isfinite(float(metrics[k])) for k in names), 'Invalid metrics')
    return metrics
