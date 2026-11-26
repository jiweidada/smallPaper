"""Pull and verify the FULL backup stream through pinned, password-authenticated SFTP."""

import argparse
import errno
import getpass
import hashlib
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import socket
import stat
import sys
import time
import warnings

import backup_protocol as protocol


LOCAL_ROOT = Path(__file__).resolve().parent
MAX_BACKUP_BYTES = 100_000_000_000
MIN_FREE_BYTES = 32 * 1024**3
WINDOW_BYTES = 1024**2
MAX_OFFLINE_SECONDS = 600
POLL_SECONDS = 2


def require(condition, message):
    protocol.require(condition, message)


def remote_info(sftp, path):
    try:
        return sftp.lstat(str(path))
    except OSError as error:
        if error.errno == errno.ENOENT:
            return None
        raise


def check_remote_dirs(sftp, directory):
    for path in (*reversed(directory.parents), directory):
        info = remote_info(sftp, path)
        require(info is not None and stat.S_ISDIR(info.st_mode), f'Expected remote directory: {path}')


def remote_metadata(sftp, path, stable=True):
    info = remote_info(sftp, path)
    if info is None:
        return None
    require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= protocol.MAX_METADATA_BYTES,
            f'Invalid remote metadata: {path}')
    check_remote_dirs(sftp, path.parent)
    with sftp.open(str(path), 'rb') as stream:
        opened = stream.stat()
        require(stat.S_ISREG(opened.st_mode) and 0 < opened.st_size <= protocol.MAX_METADATA_BYTES
                and (not stable or opened.st_size == info.st_size), 'Invalid opened metadata')
        data = stream.read(opened.st_size + 1)
    require(len(data) == opened.st_size, f'Remote metadata changed during read: {path}')
    return data


def upload_ack(sftp, path, data):
    existing = remote_metadata(sftp, path)
    if existing is not None:
        require(existing == data, 'Remote acknowledgement already exists with different bytes')
        return
    temp = path.with_name('.ack-' + secrets.token_hex(16) + '.partial')
    with sftp.open(str(temp), 'wx') as stream:
        stream.write(data)
        stream.flush()
    sftp.posix_rename(str(temp), str(path))
    require(remote_metadata(sftp, path) == data, 'Remote acknowledgement did not round-trip')


def replace_metadata(path, data):
    temporary = path.with_name(path.name + '.partial')
    if temporary.exists():
        protocol.regular(temporary)
        temporary.unlink()
    protocol.write_atomic(path, data)


def keep_metadata(path, data):
    if path.exists():
        require(protocol.local_metadata(path) == data, f'Existing backup metadata differs: {path}')
    else:
        replace_metadata(path, data)


class BackupStore:
    def __init__(self, root, allowed_root, maximum=MAX_BACKUP_BYTES, reserve=MIN_FREE_BYTES):
        self.root = protocol.checked_local(Path(root).absolute(), Path(allowed_root).resolve(strict=True))
        self.objects = self.root / 'objects'
        self.epochs = self.root / 'epochs'
        self.maximum = maximum
        self.reserve = reserve
        self.verified = {}

    def setup(self):
        for path in (self.root, self.objects, self.epochs):
            protocol.checked_local(path, self.root)
            path.mkdir(exist_ok=True)
            require(path.is_dir() and not protocol.is_link(path), f'Invalid backup directory: {path}')
        entries = list(self.objects.iterdir())
        for path in entries:
            require(protocol.HEX_SHA.fullmatch(path.stem) and path.suffix in ('.blob', '.part'),
                    f'Unexpected object path: {path}')
        self.used = sum(protocol.regular(path).st_size for path in entries)
        require(self.used <= self.maximum, 'Backup already exceeds its storage limit')

    def stage(self, epoch):
        name = protocol.phase_name(epoch)
        path = self.root / name if epoch == -1 else self.epochs / name
        return protocol.checked_local(path, self.root)

    def emit(self, status, **details):
        event = {'status': status, 'observed_utc': protocol.utc_now(), **details}
        data = protocol.encode(event)
        replace_metadata(self.root / 'monitor_status.json', data)
        event_log = self.root / 'monitor_events.jsonl'
        if event_log.exists():
            protocol.regular(event_log)
        with event_log.open('ab') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        print(data.decode('utf-8').rstrip(), flush=True)

    def verified_object(self, item):
        sha = item['sha256']
        path = self.objects / f'{sha}.blob'
        info = protocol.regular(path)
        require(info.st_size == item['bytes'], 'Local object size mismatch')
        signature = (info.st_size, info.st_mtime_ns, info.st_ino)
        if self.verified.get(sha) != signature:
            require(protocol.file_sha(path) == sha, f'Local object hash mismatch: {sha}')
            self.verified[sha] = signature
        return path

    def fetch(self, sftp, here, item):
        sha = item['sha256']
        final = self.objects / f'{sha}.blob'
        if final.exists():
            return self.verified_object(item)
        temporary = self.objects / f'{sha}.part'
        offset = protocol.regular(temporary).st_size if temporary.exists() else 0
        size = item['bytes']
        remaining = size - offset
        # Refresh after a broken SFTP transfer left a durable partial file.
        self.used = sum(protocol.regular(path).st_size for path in self.objects.iterdir())
        require(0 <= remaining and self.used + remaining <= self.maximum, 'Backup exceeds the storage limit')
        require(shutil.disk_usage(self.objects).free >= self.reserve + remaining, 'Backup would exhaust local disk space')
        remote = here / item['source']
        check_remote_dirs(sftp, remote.parent)
        info = remote_info(sftp, remote)
        require(info is not None and stat.S_ISREG(info.st_mode) and info.st_size == size,
                f'Remote backup file size or type mismatch: {remote}')
        digest = hashlib.sha256()
        if offset:
            with temporary.open('rb') as prefix:
                while chunk := prefix.read(4 * WINDOW_BYTES):
                    digest.update(chunk)
        mode = 'ab' if temporary.exists() else 'xb'
        with sftp.open(str(remote), 'rb') as src, temporary.open(mode) as dst:
            opened = src.stat()
            require(stat.S_ISREG(opened.st_mode) and opened.st_size == size, 'Remote backup file changed during open')
            src.seek(offset)
            try:
                while offset < size:
                    end = min(size, offset + WINDOW_BYTES)
                    # One window at a time limits outstanding requests on Paramiko 2.x.
                    src.prefetch(file_size=end)
                    data = src.read(end - offset)
                    if len(data) != end - offset:
                        raise EOFError('Short SFTP read')
                    dst.write(data)
                    digest.update(data)
                    offset = end
            finally:
                dst.flush()
                os.fsync(dst.fileno())
            require(src.stat().st_size == size, 'Remote backup file changed during transfer')
        require(digest.hexdigest() == sha, f'SFTP backup hash mismatch: {item["name"]}')
        protocol.durable_replace(temporary, final)
        return self.verified_object(item)

    def verify_stage(self, stage, data, manifest):
        require(protocol.local_metadata(stage / 'manifest.json') == data, 'Local stage manifest changed')
        expected_ack = protocol.ack_bytes(data, manifest)
        require(protocol.local_metadata(stage / 'ack.json') == expected_ack, 'Local stage acknowledgement changed')
        for item in manifest['files']:
            path = protocol.checked_local(stage / item['name'], self.root)
            require(protocol.regular(path).st_size == item['bytes']
                    and os.path.samefile(path, self.verified_object(item)), f'Local stage file differs: {path}')
        protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(stage / name))
        return expected_ack

    def commit_stage(self, sftp, here, data, manifest):
        stage = self.stage(manifest['epoch'])
        if stage.exists():
            return self.verify_stage(stage, data, manifest)
        temporary = protocol.checked_local(stage.with_name('.' + stage.name + '.partial'), self.root)
        temporary.mkdir(exist_ok=True)
        keep_metadata(temporary / 'manifest.json', data)
        for item in manifest['files']:
            path = protocol.checked_local(temporary / item['name'], self.root)
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = self.fetch(sftp, here, item)
            if not path.exists():
                os.link(blob, path)
                protocol.sync_dir(path.parent)
            require(protocol.regular(path).st_size == item['bytes'] and os.path.samefile(path, blob),
                    f'Existing staged backup file differs: {path}')
        protocol.check_metric_files(manifest, lambda name: protocol.local_metadata(temporary / name))
        ack = protocol.ack_bytes(data, manifest)
        keep_metadata(temporary / 'ack.json', ack)
        self.verify_stage(temporary, data, manifest)
        protocol.durable_replace(temporary, stage)
        return ack

    def verify_progress(self, run_id):
        for epoch in range(-1, protocol.FINAL_EPOCH + 1):
            stage = self.stage(epoch)
            if not (stage / 'accepted.json').exists():
                return epoch
            data = protocol.local_metadata(stage / 'manifest.json')
            manifest = protocol.parse_manifest(data, run_id, epoch)
            self.verify_stage(stage, data, manifest)
            require(protocol.local_metadata(stage / 'accepted.json') == protocol.accepted_bytes(data, manifest),
                    'Prior remote acceptance does not match local verified data')
        return protocol.FINAL_EPOCH + 1


def protect_local_lock(root):
    root.mkdir(exist_ok=True)
    lock = protocol.checked_local(root / '.monitor.lock', root)
    descriptor = lock.open('a+b')
    try:
        if descriptor.tell() == 0:
            descriptor.write(b'0')
            descriptor.flush()
        descriptor.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(descriptor.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        descriptor.close()
        raise
    return descriptor


def backup_once(sftp, here, store, epoch):
    remote = here / 'backup_gate' / protocol.phase_name(epoch)
    data = remote_metadata(sftp, remote / 'manifest.json')
    if data is None:
        return False
    manifest = protocol.parse_manifest(data, here.name, epoch)
    ack = store.commit_stage(sftp, here, data, manifest)
    require(remote_metadata(sftp, remote / 'manifest.json') == data, 'Remote manifest changed before acknowledgement')
    upload_ack(sftp, remote / 'ack.json', ack)
    accepted = remote_metadata(sftp, remote / 'accepted.json')
    if accepted is None:
        return False
    require(accepted == protocol.accepted_bytes(data, manifest), 'Remote acceptance differs from verified backup')
    keep_metadata(store.stage(epoch) / 'accepted.json', accepted)
    store.emit('PROBE_VERIFIED' if epoch == -1 else 'EPOCH_BACKUP_VERIFIED', epoch=epoch,
               manifest_sha256=protocol.digest(data), ack_sha256=protocol.digest(ack))
    return True


def check_live_receipt(sftp, here, store, final=False):
    data = remote_metadata(sftp, here / 'run' / 'receipt.json', stable=False)
    if data is None:
        return False
    receipt = protocol.decode(data)
    require(receipt['arm'] == protocol.ARM and receipt['protocol_sha256'] == protocol.PROTOCOL,
            'Unrecognized remote run receipt')
    if receipt['status'] == 'FAILED':
        keep_metadata(store.root / 'failed_run_receipt.json', data)
        raise protocol.BackupError('The remote training process recorded a failure')
    if not final or receipt['status'] != 'COMPLETED':
        return False
    end = store.stage(protocol.FINAL_EPOCH)
    manifest_data = protocol.local_metadata(end / 'manifest.json')
    manifest = protocol.parse_manifest(manifest_data, here.name, protocol.FINAL_EPOCH)
    expected_ack = protocol.ack_bytes(manifest_data, manifest)
    require(receipt['optimizer_step_attempts'] == 517440 and receipt['completed_epochs'] == 320
            and receipt['rolling_checkpoint_sha256'] == manifest['rolling_sha256']
            and receipt['local_backup_verified'] == {'epoch': protocol.FINAL_EPOCH,
                'manifest_sha256': protocol.digest(manifest_data), 'ack_sha256': protocol.digest(expected_ack)},
            'Final run receipt differs from the verified terminal backup')
    keep_metadata(store.root / 'completed_run_receipt.json', data)
    supervisor_data = remote_metadata(sftp, here / 'supervisor.json', stable=False)
    if supervisor_data is None:
        return False
    supervisor = protocol.decode(supervisor_data)
    if supervisor['status'] in ('STARTING', 'RUNNING'):
        return False
    if (supervisor['status'] != 'COMPLETED' or supervisor['exit_code'] != 0
            or supervisor['final_receipt'] != receipt or supervisor['pid'] != receipt['pid']
            or supervisor['runner_sha256'] != receipt['runner_sha256']):
        keep_metadata(store.root / 'rejected_supervisor.json', supervisor_data)
        raise protocol.BackupError('The final supervisor did not confirm successful exit of this training process')
    keep_metadata(store.root / 'completed_supervisor.json', supervisor_data)
    store.emit('COMPLETED_AND_BACKED_UP', epoch=protocol.FINAL_EPOCH,
               rolling_checkpoint_sha256=manifest['rolling_sha256'],
               supervisor_sha256=protocol.digest(supervisor_data))
    return True


def pinned_client(paramiko, host, port, known_hosts):
    client = paramiko.SSHClient()
    client.load_host_keys(str(known_hosts))
    label = host if port == 22 else f'[{host}]:{port}'
    require(client.get_host_keys().lookup(label) is not None, 'No pinned key for this host and port')
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    return client


def connect(paramiko, args, password, here, store):
    client = pinned_client(paramiko, args.host, args.port, args.known_hosts)
    try:
        client.connect(args.host, port=args.port, username=args.user, password=password,
                       look_for_keys=False, allow_agent=False, timeout=20, banner_timeout=20, auth_timeout=20)
        client.get_transport().set_keepalive(15)
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(30)
        saved = store.stage(-1) / 'manifest.json'
        if saved.exists():
            remote = remote_metadata(sftp, here / 'backup_gate' / 'probe' / 'manifest.json')
            require(remote == protocol.local_metadata(saved), 'Remote probe differs from this local backup session')
        return client, sftp
    except BaseException:
        client.close()
        raise


def reconnectable(paramiko, error):
    if isinstance(error, (paramiko.BadHostKeyException, paramiko.AuthenticationException)):
        return False
    if isinstance(error, (paramiko.SSHException, paramiko.ssh_exception.NoValidConnectionsError,
                          socket.timeout, ConnectionError, EOFError)):
        return True
    return isinstance(error, OSError) and (error.errno in (errno.EPIPE, errno.ECONNRESET, errno.ETIMEDOUT)
                                          or str(error) == 'Socket is closed')


def watch(paramiko, args, here, store, password):
    cursor = store.verify_progress(here.name)
    store.emit('MONITOR_READY', next_backup_epoch=cursor, storage_limit_bytes=store.maximum,
               minimum_free_bytes=store.reserve)
    client = sftp = offline_since = None
    try:
        while True:
            try:
                if client is None:
                    client, sftp = connect(paramiko, args, password, here, store)
                    store.emit('PINNED_SFTP_CONNECTED', next_backup_epoch=cursor)
                if cursor <= protocol.FINAL_EPOCH:
                    if backup_once(sftp, here, store, cursor):
                        cursor += 1
                        offline_since = None
                        continue
                    check_live_receipt(sftp, here, store)
                elif check_live_receipt(sftp, here, store, final=True):
                    return
                offline_since = None
            except Exception as error:
                if not reconnectable(paramiko, error):
                    raise
                if client is not None:
                    client.close()
                client = sftp = None
                offline_since = time.monotonic() if offline_since is None else offline_since
                require(time.monotonic() - offline_since < MAX_OFFLINE_SECONDS, 'SFTP reconnect window expired')
                store.emit('RECONNECTING', next_backup_epoch=cursor, error_type=type(error).__name__)
            time.sleep(POLL_SECONDS)
    finally:
        if client is not None:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=30219, choices=[30219, 30848])
    parser.add_argument('--user', required=True)
    parser.add_argument('--known-hosts', required=True, type=Path)
    parser.add_argument('--remote-output', required=True, type=PurePosixPath)
    parser.add_argument('--backup-root', type=Path, default=LOCAL_ROOT / 'backups')
    args = parser.parse_args()
    out = args.remote_output
    require(out.is_absolute() and out.name == 'run' and out.parent != PurePosixPath('/')
            and all(protocol.SAFE_NAME.fullmatch(part) for part in out.parts[1:]), 'Invalid remote output path')
    args.known_hosts = args.known_hosts.resolve(strict=True)
    store = BackupStore(args.backup_root, LOCAL_ROOT)
    require(store.root != LOCAL_ROOT, 'Choose a dedicated local backup subdirectory')
    import paramiko

    with pinned_client(paramiko, args.host, args.port, args.known_hosts):
        pass
    require(sys.stdin.isatty(), 'Start the backup monitor in an interactive terminal')
    with warnings.catch_warnings():
        warnings.simplefilter('error', getpass.GetPassWarning)
        password = getpass.getpass('SSH password for the pinned FULL host: ')
    with protect_local_lock(store.root):
        store.setup()
        try:
            watch(paramiko, args, out.parent, store, password)
        except KeyboardInterrupt:
            store.emit('MONITOR_STOPPED_BY_OPERATOR')
            raise
        except Exception as error:
            store.emit('MONITOR_FAILED_NO_FURTHER_ACKS', error_type=type(error).__name__)
            raise
        finally:
            del password


if __name__ == '__main__':
    main()
