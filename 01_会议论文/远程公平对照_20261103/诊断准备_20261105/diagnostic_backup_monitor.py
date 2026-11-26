"""Local one-shot backup monitor; credentials remain only in process memory."""
from pathlib import Path
from datetime import datetime, timezone
import getpass
import json
import os
import shutil
import stat
import sys
import time
import traceback
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import paramiko
import backup_guard as g

ROOT = g.REMOTE_RUN
BACKUP = g.REMOTE_WORK / 'diagnostic_backups_20261105'
INCOMING = BACKUP / '_incoming'
EVENTS = BACKUP / 'monitor_events.jsonl'
PHASES = ['initial_ema', 'initial_raw'] + ['epoch_%04d' % e for e in range(214, 225)] + ['terminal_raw']

def stamp():
    return datetime.now(timezone.utc).isoformat()

def emit(status, **details):
    event = {'status': status, 'utc': stamp(), **details}
    with EVENTS.open('a', encoding='utf-8') as out:
        out.write(json.dumps(event, ensure_ascii=False) + '\n')
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(event, ensure_ascii=False), flush=True)

def read_remote(sftp, path, limit=1024 * 1024):
    info = sftp.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise RuntimeError('Unexpected remote file type or size: ' + path)
    with sftp.open(path, 'rb') as f:
        # State files are atomically replaced.  Pin their opened inode before
        # checking length so a normal state update cannot look like corruption.
        opened = f.stat()
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > limit:
            raise RuntimeError('Unexpected opened remote file: ' + path)
        data = f.read(limit + 1)
    if len(data) != opened.st_size or len(data) > limit:
        raise RuntimeError('Remote file changed while read: ' + path)
    return data

def optional(sftp, path):
    try:
        return read_remote(sftp, path)
    except FileNotFoundError:
        return None

def copy_one(sftp, item):
    path = Path(item['local'])
    info = sftp.lstat(item['remote'])
    if not stat.S_ISREG(info.st_mode) or info.st_size != item['bytes']:
        raise RuntimeError('Unexpected remote backup file: ' + item['remote'])
    with sftp.open(item['remote'], 'rb') as src, path.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if path.stat().st_size != item['bytes']:
        raise RuntimeError('Incomplete local backup: ' + str(path))

def exclusive_upload(sftp, path, data):
    # Paramiko's x flag uses SFTP EXCL; the marker is created only after all
    # acknowledgement bytes have been sent and the preceding file was closed.
    with sftp.open(path, 'wx') as out:
        if data:
            out.write(data)
            out.flush()

def verify_controller_identity(sftp, state):
    pid, ticks = state.get('controller_pid'), state.get('controller_start_ticks')
    if not isinstance(pid, int) or pid <= 0:
        raise RuntimeError('Controller identity is missing')
    try:
        # Procfs reports size zero for its virtual files, so do not apply the
        # ordinary on-disk size check used for experiment artifacts.
        with sftp.open('/proc/%d/stat' % pid, 'rb') as proc:
            data = proc.read(65536).decode()
    except FileNotFoundError:
        return False
    parts = data.rsplit(')', 1)[1].split()
    return parts[0] != 'Z' and len(parts) > 19 and parts[19] == str(ticks)

def backup_phase(sftp, phase, data):
    obj = g.parse_manifest(data)
    if obj['phase'] != phase:
        raise RuntimeError('Phase does not match manifest')
    incoming = INCOMING / (phase + '.json')
    with incoming.open('xb') as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    prepared = g.prepare(incoming, BACKUP)
    emit('BACKUP_STARTED', phase=phase, epoch_index=obj['epoch_index'], files=len(obj['files']))
    for item in prepared['downloads']:
        copy_one(sftp, item)
    verified = g.verify(incoming, BACKUP)
    ack_bytes = Path(verified['local_ack']).read_bytes()
    stage = ROOT + '/phases/' + phase
    exclusive_upload(sftp, stage + '/ack.uploading', ack_bytes)
    exclusive_upload(sftp, stage + '/ack.ready', b'')
    emit('BACKUP_VERIFIED_AND_ACKNOWLEDGED', phase=phase, epoch_index=obj['epoch_index'],
         optimizer_updates=obj['optimizer_updates'], validation_passes=obj['validation_passes'],
         manifest_sha256=verified['manifest_sha256'])

def main():
    g.check_plans()
    receipt = json.loads((Path(__file__).resolve().parent / 'local_preflight.json').read_text(encoding='utf-8'))
    if receipt.get('status') != 'LOCAL_PREFLIGHT_PASSED_NO_MODEL_EXECUTION':
        raise RuntimeError('Local preflight is missing')
    for item in receipt['local_files']:
        if g.digest_file(Path(__file__).resolve().parent / item['name']) != item['sha256']:
            raise RuntimeError('Local backup package changed: ' + item['name'])
    root = g.checked_backup_root(BACKUP)
    if root.exists() or root.is_symlink():
        raise RuntimeError('Backup root already exists; do not reuse a one-shot diagnostic')
    if shutil.disk_usage(root.parent).free < 5 * 1024**3:
        raise RuntimeError('Less than 5 GiB local free space')
    root.mkdir(exist_ok=False)
    INCOMING.mkdir(exist_ok=False)
    with EVENTS.open('x', encoding='utf-8'):
        pass
    password = getpass.getpass('SSH password (not saved): ')
    if not password:
        raise RuntimeError('No password supplied')
    client = paramiko.SSHClient()
    client.load_host_keys(str(g.REMOTE_WORK / 'known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('172.17.120.160', port=30166, username='root', password=password,
                   allow_agent=False, look_for_keys=False, timeout=15, auth_timeout=15)
    password = None
    client.get_transport().set_keepalive(20)
    sftp = client.open_sftp()
    sftp.get_channel().settimeout(45)
    ready = json.loads(read_remote(sftp, ROOT + '/setup_ready.json'))
    if ready.get('status') != 'SETUP_READY_NO_MODEL_EXECUTION':
        raise RuntimeError('Remote setup is not ready')
    seen = set()
    launched = False
    started = time.monotonic()
    emit('MONITOR_READY', backup_root=str(root), automatic_training_retries=False)
    while True:
        raw_state = optional(sftp, ROOT + '/controller_state.json')
        if raw_state is not None:
            state = json.loads(raw_state)
            launched = True
            status = state.get('status', '')
            if status.startswith('STOPPED') or status == 'DIAGNOSTIC_COMPLETE_NOT_A_FAIR_PAIR':
                emit('CONTROLLER_FINISHED', controller_status=status,
                     optimizer_updates=state.get('optimizer_updates'),
                     completed_validations=state.get('completed_validations'))
                break
            if not verify_controller_identity(sftp, state):
                emit('CONTROLLER_NOT_RUNNING', controller_status=status)
                break
        elif not launched and time.monotonic() - started > 900:
            emit('MONITOR_STOPPED_NO_CONTROLLER')
            break
        for phase in PHASES:
            if phase in seen:
                continue
            data = optional(sftp, ROOT + '/phases/' + phase + '/manifest.json')
            if data is not None:
                backup_phase(sftp, phase, data)
                seen.add(phase)
        time.sleep(5)
    sftp.close()
    client.close()

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if EVENTS.is_file():
            emit('MONITOR_FAILED_NO_AUTOMATIC_RETRY', failure=traceback.format_exc())
        else:
            print(json.dumps({'status': 'MONITOR_FAILED_BEFORE_START', 'failure': traceback.format_exc()}), flush=True)
        raise
