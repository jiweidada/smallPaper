"""Closed manifest and acknowledgement protocol for the protected FULL continuation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import time


_PLAN_PATH = Path(__file__).resolve().with_name('execution_protocol.json')
_PLAN = json.loads(_PLAN_PATH.read_text(encoding='utf-8'))
SCHEMA = 'module_ablation_backup_v1'
PROTOCOL = hashlib.sha256(_PLAN_PATH.read_bytes()).hexdigest()
CONFIG = _PLAN['configuration_sha256']
ARM = _PLAN['arm']
FINAL_EPOCH = 319
BATCHES = 1617
METRICS = frozenset(('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl'))
PROBE_FILES = _PLAN['probe_files']
BEST_FILES = ('best.pt', 'best.metrics.json', 'best.predictions.json', 'best.evaluation_arrays.npz')
TERMINAL_FILES = ('terminal_epoch0319.pt', 'terminal.metrics.json', 'terminal.predictions.json',
                  'terminal.evaluation_arrays.npz')
MAX_FILE_BYTES = 192 * 1024**2
MAX_METADATA_BYTES = 128 * 1024
GATE_TIMEOUT_SECONDS = 1800
SAFE_NAME = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,94}\Z')
HEX_SHA = re.compile(r'[0-9a-f]{64}\Z')


class BackupError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BackupError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def unique_pairs(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), 'Duplicate JSON field')
    return result


def decode(data):
    require(isinstance(data, bytes) and 0 < len(data) <= MAX_METADATA_BYTES, 'Invalid metadata length')
    return json.loads(data, object_pairs_hook=unique_pairs)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        hasher = hashlib.sha256()
        while block := stream.read(4 * 1024**2):
            hasher.update(block)
    return hasher.hexdigest()


def sync_dir(directory):
    if os.name == 'posix':
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def durable_replace(source, destination):
    if os.name == 'nt':
        import ctypes

        move = ctypes.WinDLL('kernel32', use_last_error=True).MoveFileExW
        move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move.restype = ctypes.c_int
        if not move(str(source), str(destination), 0x1 | 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.replace(source, destination)
        sync_dir(destination.parent)


def write_atomic(path, data):
    temporary = path.with_name(path.name + '.partial')
    # Exclusive creation avoids following a link if an interrupted write left a name behind.
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    durable_replace(temporary, path)


def is_link(path):
    return path.is_symlink() or getattr(path, 'is_junction', lambda: False)()


def checked_local(path, root):
    require(path.is_relative_to(root), f'Path outside backup scope: {path}')
    cursor = path
    while True:
        require(not is_link(cursor), f'Linked backup path: {cursor}')
        if cursor == root:
            return path
        cursor = cursor.parent


def regular(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not is_link(path), f'Expected regular file: {path}')
    return info


def local_metadata(path):
    size = regular(path).st_size
    require(0 < size <= MAX_METADATA_BYTES, f'Invalid metadata size: {path}')
    data = path.read_bytes()
    require(len(data) == size, f'Metadata changed during read: {path}')
    return data


def phase_name(epoch):
    require(type(epoch) is int and -1 <= epoch <= FINAL_EPOCH, 'Invalid backup epoch')
    return 'probe' if epoch == -1 else f'epoch{epoch:04d}'


def selected_folder(ref):
    return f"selected/epoch{ref['epoch']:04d}"


def identity(run_id, epoch):
    return {'schema': SCHEMA, 'arm': ARM, 'run_id': run_id, 'epoch': epoch,
            'protocol_sha256': PROTOCOL, 'initialization_checkpoint_sha256': PROBE_FILES['initial.pt']}


def safe_relative(value):
    require(type(value) is str, 'Invalid relative path')
    path = PurePosixPath(value)
    require(bool(path.parts) and str(path) == value and all(SAFE_NAME.fullmatch(part) for part in path.parts),
            'Invalid relative path')
    return path


def valid_ref(ref, max_epoch):
    require(isinstance(ref, dict) and set(ref) == {'epoch', 'checkpoint_sha256', 'metrics_sha256'}
            and type(ref['epoch']) is int and 0 <= ref['epoch'] <= max_epoch
            and all(type(ref[key]) is str and HEX_SHA.fullmatch(ref[key])
                    for key in ('checkpoint_sha256', 'metrics_sha256')), 'Invalid selected checkpoint reference')


def expected_sources(manifest):
    epoch = manifest['epoch']
    if epoch == -1:
        anchor = safe_relative(manifest['anchor'])
        require(anchor.parts[0] not in ('run', 'backup_gate'), 'Invalid probe source')
        return {name: str(anchor / name) for name in PROBE_FILES}
    best, stage_one = manifest['best'], manifest['stage_one']
    valid_ref(best, epoch)
    valid_ref(stage_one, min(epoch, 295))
    require(stage_one['epoch'] <= best['epoch'] and (epoch >= 296 or stage_one == best),
            'Invalid stage-one checkpoint ordering')
    sources = {
        'next_epoch.pt': 'run/next_epoch.pt',
        'epoch.metrics.json': f'backup_gate/{phase_name(epoch)}/epoch.metrics.json',
        'receipt.json': 'run/receipt.json',
    }
    for name in ('predictions.json', 'evaluation_arrays.npz'):
        sources[f'latest.{name}'] = (f'run/{selected_folder(best)}/best.{name}' if epoch == 0
                                     else f'run/latest.{name}')
    for ref in (best, stage_one):
        for name in BEST_FILES:
            relative = f'{selected_folder(ref)}/{name}'
            sources[relative] = f'run/{relative}'
    if epoch >= 0:
        sources['actual_first100.jsonl'] = 'run/actual_first100.jsonl'
        for name in ('epochs.jsonl', 'executed_learning_rates.csv.gz'):
            sources[name] = f'run/{name}'
    if epoch >= 296:
        sources['restore_events.jsonl'] = 'run/restore_events.jsonl'
    if epoch == FINAL_EPOCH:
        sources.update({name: f'run/{name}' for name in TERMINAL_FILES})
    return sources


def parse_manifest(data, run_id, epoch):
    require(type(run_id) is str and SAFE_NAME.fullmatch(run_id), 'Invalid run id')
    phase_name(epoch)
    manifest = decode(data)
    expected = identity(run_id, epoch)
    other = {'anchor'} if epoch == -1 else {'next_epoch', 'optimizer_step_attempts', 'rolling_sha256', 'best', 'stage_one'}
    require(isinstance(manifest, dict) and set(manifest) == set(expected) | other | {'files', 'nonce'}
            and all(manifest[key] == value and type(manifest[key]) is type(value) for key, value in expected.items()),
            'Backup manifest identity mismatch')
    require(type(manifest['nonce']) is str and re.fullmatch(r'[0-9a-f]{32}', manifest['nonce']), 'Invalid backup nonce')
    if epoch >= 0:
        require(type(manifest['next_epoch']) is int and manifest['next_epoch'] == epoch + 1
                and type(manifest['optimizer_step_attempts']) is int
                and manifest['optimizer_step_attempts'] == (epoch + 1) * BATCHES
                and type(manifest['rolling_sha256']) is str and HEX_SHA.fullmatch(manifest['rolling_sha256']),
                'Invalid rolling checkpoint cursor')
    sources = expected_sources(manifest)
    files = manifest['files']
    require(isinstance(files, list) and len(files) == len(sources), 'Unexpected manifest file count')
    by_name = {}
    for item in files:
        require(isinstance(item, dict) and set(item) == {'name', 'source', 'bytes', 'sha256'}, 'Invalid file manifest')
        name = item['name']
        require(type(name) is str and name in sources and name not in by_name and item['source'] == sources[name],
                'Unexpected, duplicate or unsafe backup source')
        require(type(item['bytes']) is int and 0 < item['bytes'] <= MAX_FILE_BYTES
                and type(item['sha256']) is str and HEX_SHA.fullmatch(item['sha256']), 'Invalid backup file size or hash')
        by_name[name] = item['sha256']
    if epoch == -1:
        require(by_name == PROBE_FILES, 'Transport probe differs from the preserved FULL anchor')
    else:
        require(by_name['next_epoch.pt'] == manifest['rolling_sha256'], 'Rolling checkpoint hash mismatch')
        for ref in (manifest['best'], manifest['stage_one']):
            folder = selected_folder(ref)
            require(by_name[f'{folder}/best.pt'] == ref['checkpoint_sha256']
                    and by_name[f'{folder}/best.metrics.json'] == ref['metrics_sha256'], 'Unbound selected evidence')
    return manifest


def check_metric_files(manifest, read):
    by_name = {item['name']: item['sha256'] for item in manifest['files']}
    epoch = manifest['epoch']

    def checked_row(name, index, predictions, arrays, checkpoint=None):
        row = decode(read(name))
        require(type(row['epoch']) is int and row['epoch'] == index and row['protocol_sha256'] == PROTOCOL
                and row['config_sha256'] == CONFIG and row['optimizer_step_attempts'] == (index + 1) * BATCHES
                and row['weight_source'] == 'ema.module', f'Wrong evaluated boundary: {name}')
        metrics = row['metrics']
        require(isinstance(metrics, dict) and set(metrics) == METRICS
                and all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values()),
                f'Invalid COCO metrics: {name}')
        require(row['prediction_sha256'] == by_name[predictions] and row['evaluation_arrays_sha256'] == by_name[arrays],
                f'Unbound evaluation output: {name}')
        if checkpoint is not None:
            require(row['checkpoint_sha256'] == by_name[checkpoint], f'Unbound evaluated checkpoint: {name}')
        return row

    if epoch == -1:
        report = decode(read('preflight.json'))
        require(report['arm'] == ARM and report['status'] == 'PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE'
                and report['optimizer_updates'] == 0
                and report['initial_file_sha256'] == by_name['initial.pt'], 'Unbound preflight transport probe')
        return
    checked_row('epoch.metrics.json', epoch, 'latest.predictions.json', 'latest.evaluation_arrays.npz')
    receipt = decode(read('receipt.json'))
    require(receipt['arm'] == ARM and receipt['protocol_sha256'] == PROTOCOL
            and receipt['phase'] == 'awaiting_local_backup' and receipt['backup_epoch'] == epoch
            and receipt['rolling_checkpoint_sha256'] == manifest['rolling_sha256']
            and receipt['optimizer_step_attempts'] == manifest['optimizer_step_attempts'],
            'Receipt differs from the protected rolling checkpoint')
    for ref in (manifest['best'], manifest['stage_one']):
        folder = selected_folder(ref)
        checked_row(f'{folder}/best.metrics.json', ref['epoch'], f'{folder}/best.predictions.json',
                    f'{folder}/best.evaluation_arrays.npz', f'{folder}/best.pt')
    if epoch == FINAL_EPOCH:
        checked_row('terminal.metrics.json', epoch, 'terminal.predictions.json',
                    'terminal.evaluation_arrays.npz', 'terminal_epoch0319.pt')
        for name in ('predictions.json', 'evaluation_arrays.npz'):
            require(by_name[f'terminal.{name}'] == by_name[f'latest.{name}'], 'Terminal and final evaluation differ')


def ack_bytes(manifest_data, manifest):
    return encode({**identity(manifest['run_id'], manifest['epoch']), 'status': 'LOCAL_BACKUP_VERIFIED',
                   'nonce': manifest['nonce'], 'manifest_sha256': digest(manifest_data)})


def accepted_bytes(manifest_data, manifest):
    ack = ack_bytes(manifest_data, manifest)
    return encode({**identity(manifest['run_id'], manifest['epoch']), 'status': 'REMOTE_BACKUP_ACK_ACCEPTED',
                   'manifest_sha256': digest(manifest_data), 'ack_sha256': digest(ack)})


class EpochGate:
    def __init__(self, output):
        output = Path(output)
        require(output.is_absolute() and output.name == 'run' and SAFE_NAME.fullmatch(output.parent.name),
                'Protected output must be the run subdirectory of an absolute run root')
        self.output = output
        self.here = output.parent
        self.run_id = self.here.name
        self.base = self.here / 'backup_gate'

    def stage(self, epoch):
        return checked_local(self.base / phase_name(epoch), self.here)

    def source(self, item):
        return checked_local(self.here / item['source'], self.here)

    def publish(self, epoch, extra):
        stage = self.stage(epoch)
        manifest = {**identity(self.run_id, epoch), **extra, 'nonce': secrets.token_hex(16)}
        files = []
        for name, source in sorted(expected_sources(manifest).items()):
            item = {'name': name, 'source': source}
            path = self.source(item)
            size = regular(path).st_size
            require(0 < size <= MAX_FILE_BYTES, f'Invalid backup source size: {path}')
            files.append({**item, 'bytes': size, 'sha256': file_sha(path)})
        manifest['files'] = files
        payload = encode(manifest)
        parse_manifest(payload, self.run_id, epoch)
        paths = {item['name']: self.source(item) for item in files}
        check_metric_files(manifest, lambda name: local_metadata(paths[name]))
        write_atomic(stage / 'manifest.json', payload)
        return payload, manifest

    def publish_probe(self, anchor):
        anchor = checked_local(Path(anchor).absolute(), self.here)
        rel = anchor.relative_to(self.here).as_posix()
        safe_relative(rel)
        self.stage(-1).mkdir(parents=True, exist_ok=False)
        sync_dir(self.base)
        sync_dir(self.here)
        payload, _ = self.publish(-1, {'anchor': rel})
        return digest(payload)

    def verify_ack(self, payload, manifest, timeout):
        stage = self.stage(manifest['epoch'])
        expected_ack = ack_bytes(payload, manifest)
        ack_path = stage / 'ack.json'
        deadline = time.monotonic() + timeout
        while not ack_path.exists():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Local backup acknowledgement missing for {phase_name(manifest['epoch'])}")
            time.sleep(min(2, remaining))
        require(timeout == 0 or time.monotonic() <= deadline, 'Backup acknowledgement arrived after its deadline')
        require(local_metadata(ack_path) == expected_ack, 'Backup acknowledgement does not bind this manifest')
        for item in manifest['files']:
            path = self.source(item)
            require(regular(path).st_size == item['bytes'] and file_sha(path) == item['sha256'],
                    f'Backup source changed while waiting: {path}')
        require(local_metadata(stage / 'manifest.json') == payload and local_metadata(ack_path) == expected_ack,
                'Backup manifest or acknowledgement changed during validation')
        require(timeout == 0 or time.monotonic() <= deadline, 'Backup verification exceeded its deadline')
        accepted = accepted_bytes(payload, manifest)
        path = stage / 'accepted.json'
        if path.exists():
            require(local_metadata(path) == accepted, 'Existing server acceptance differs')
        else:
            write_atomic(path, accepted)
        return {'epoch': manifest['epoch'], 'manifest_sha256': digest(payload), 'ack_sha256': digest(expected_ack)}

    def verify_probe(self):
        payload = local_metadata(self.stage(-1) / 'manifest.json')
        return self.verify_ack(payload, parse_manifest(payload, self.run_id, -1), 0)

    def require_probe(self):
        stage = self.stage(-1)
        payload = local_metadata(stage / 'manifest.json')
        manifest = parse_manifest(payload, self.run_id, -1)
        require(local_metadata(stage / 'ack.json') == ack_bytes(payload, manifest)
                and local_metadata(stage / 'accepted.json') == accepted_bytes(payload, manifest),
                'Missing accepted transport proof')

    def protect(self, epoch, rolling_sha, best, stage_one, evaluated):
        stage = self.stage(epoch)
        stage.mkdir(exist_ok=False)
        sync_dir(self.base)
        write_atomic(stage / 'epoch.metrics.json', encode(evaluated))
        payload, manifest = self.publish(epoch, {
            'next_epoch': epoch + 1, 'optimizer_step_attempts': (epoch + 1) * BATCHES,
            'rolling_sha256': rolling_sha, 'best': best, 'stage_one': stage_one,
        })
        return self.verify_ack(payload, manifest, GATE_TIMEOUT_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['publish-probe', 'verify-probe'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--anchor', type=Path, help='directory below the run root containing the frozen best files')
    args = parser.parse_args()
    gate = EpochGate(args.output.absolute())
    if args.mode == 'publish-probe':
        require(args.anchor is not None, 'Publishing the proof requires --anchor')
        print(json.dumps({'status': 'TRANSPORT_PROBE_PUBLISHED', 'manifest_sha256': gate.publish_probe(args.anchor)}))
    else:
        print(json.dumps({'status': 'TRANSPORT_PROBE_VERIFIED', **gate.verify_probe()}))


if __name__ == '__main__':
    main()
