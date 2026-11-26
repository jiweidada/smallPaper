"""One-stage backup transfer using an injected SFTP-like connection.

No connection is created here and this module cannot launch a training job.
The caller must authenticate the host, establish the transport and supply the
frozen run/protocol identity after explicit authorization.
"""
from pathlib import PurePosixPath
import os
import stat

import epoch_backup_guard as guard

CHUNK_BYTES = 1024 * 1024


def require(condition, message):
    if not condition:
        raise guard.BackupError(message)


def remote_stage_path(remote_root, run_id, arm, epoch):
    require(isinstance(remote_root, str) and remote_root.startswith('/'), 'Remote root must be absolute')
    parts = remote_root.split('/')
    require(all(part not in ('.', '..') for part in parts), 'Remote root must not contain traversal')
    root = PurePosixPath(remote_root)
    require(str(root) == remote_root.rstrip('/') and str(root) != '/', 'Remote root must be normalized')
    require(isinstance(run_id, str) and guard.RUN_ID.fullmatch(run_id), 'Invalid run id')
    require(arm in ('BASE', 'FULL'), 'Invalid arm')
    require(type(epoch) is int and 0 <= epoch < guard.EPOCHS, 'Invalid epoch')
    return root / 'runs' / arm / 'epochs' / ('epoch_%04d' % epoch)


def remote_regular(sftp, path, expected_size=None):
    info = sftp.lstat(str(path))
    require(stat.S_ISREG(info.st_mode), 'Remote entry is not a regular file: ' + str(path))
    require(0 < info.st_size <= guard.MAX_FILE_BYTES, 'Invalid remote file size')
    if expected_size is not None:
        require(info.st_size == expected_size, 'Remote file size mismatch: ' + str(path))
    return info


def read_manifest(sftp, stage):
    path = stage / 'manifest.json'
    remote_regular(sftp, path)
    with sftp.open(str(path), 'rb') as stream:
        info = stream.stat()
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= guard.MAX_MANIFEST_BYTES,
                'Invalid opened manifest')
        data = stream.read(guard.MAX_MANIFEST_BYTES + 1)
        require(len(data) == info.st_size and len(data) <= guard.MAX_MANIFEST_BYTES, 'Manifest changed during read')
    return data


def copy_one(sftp, remote, local, item):
    remote_regular(sftp, remote, item['bytes'])
    with sftp.open(str(remote), 'rb') as src, local.open('xb') as dst:
        info = src.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_size == item['bytes'], 'Opened remote file mismatch')
        remaining = item['bytes']
        while remaining:
            data = src.read(min(CHUNK_BYTES, remaining))
            require(isinstance(data, bytes) and 0 < len(data) <= remaining, 'Incomplete remote file')
            dst.write(data)
            remaining -= len(data)
        require(src.read(1) == b'', 'Remote file grew during copy')
        final = src.stat()
        require(final.st_size == info.st_size, 'Remote file changed during copy')
        dst.flush()
        os.fsync(dst.fileno())
    require(local.stat().st_size == item['bytes'], 'Incomplete local copy')


def exclusive_upload(sftp, path, data):
    # The marker is uploaded only after the full acknowledgement closes.
    with sftp.open(str(path), 'wx') as stream:
        if data:
            stream.write(data)
            stream.flush()


def backup_stage(sftp, remote_root, local_root, allowed_root, run_id, arm, epoch, protocol_sha256,
                 reserve_bytes=512 * 1024**2):
    """Copy one immutable stage and publish an ack only after local validation.

    Failures leave any partial stage in place and never retry automatically.
    The training-side process must separately wait for and validate the ack.
    """
    guard.validate_identity(run_id, arm, protocol_sha256)
    stage = remote_stage_path(remote_root, run_id, arm, epoch)
    root = PurePosixPath(remote_root)
    for directory in (*reversed(root.parents), root, root / 'runs', root / 'runs' / arm, stage.parent, stage):
        info = sftp.lstat(str(directory))
        require(stat.S_ISDIR(info.st_mode), 'Remote stage path is not a real directory: ' + str(directory))
    manifest_data = read_manifest(sftp, stage)
    manifest = guard.parse_manifest(manifest_data, run_id, arm, protocol_sha256)
    require(manifest['epoch_index'] == epoch, 'Remote manifest epoch mismatch')
    local_stage = guard.prepare(manifest_data, local_root, allowed_root, run_id, arm,
                                protocol_sha256, reserve_bytes=reserve_bytes)
    for item in manifest['files']:
        copy_one(sftp, stage / item['name'], local_stage / item['name'], item)
    ack = guard.verify(manifest_data, local_root, allowed_root, run_id, arm, protocol_sha256)
    guard.validate_ack(ack, manifest_data, manifest)
    exclusive_upload(sftp, stage / 'ack.uploading', ack)
    exclusive_upload(sftp, stage / 'ack.ready', b'')
    return {'status': 'LOCAL_BACKUP_VERIFIED_AND_ACKNOWLEDGED', 'epoch_index': epoch,
            'manifest_sha256': guard.digest_bytes(manifest_data), 'ack_sha256': guard.digest_bytes(ack),
            'local_stage': str(local_stage)}
