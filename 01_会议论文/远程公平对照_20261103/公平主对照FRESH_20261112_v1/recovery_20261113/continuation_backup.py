"""Use the frozen backup client for a staged continuation on the same archive."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1')
CHECKPOINT_PYTHON = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
TORCH_SITE = PROJECT.parents[1] / '本地CPU核验_20261103/.runtime/site-packages'
sys.path.insert(0, str(PROJECT))
from ssh_control import HOST, credential_path, protect
from continuation_guard import read, require, sha


def verify_remote_acceptance(backup, paramiko, options, password, here, store, cursor, plan_path):
    stage = store.stage(cursor - 1)
    data = backup.protocol.local_metadata(stage / 'manifest.json')
    manifest = backup.protocol.parse_manifest(data, here.name, cursor - 1)
    client, sftp = backup.connect(paramiko, options, password, here, store)
    try:
        status = backup.remote_metadata(sftp, here / 'stage_receipt.json')
        require(backup.protocol.decode(status)['plan_sha256'] == sha(plan_path),
                'Remote staging receipt differs')
        accepted = backup.remote_metadata(
            sftp, here / 'backup_gate' / backup.protocol.phase_name(cursor - 1) / 'accepted.json'
        )
        require(accepted == backup.protocol.accepted_bytes(data, manifest) ==
                backup.protocol.local_metadata(stage / 'accepted.json'), 'Remote anchor acceptance differs')
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=('BASE_FRESH', 'FULL_FRESH'))
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--check', action='store_true', help='Check the staged anchor and pinned SFTP.')
    checks.add_argument('--check-live-prefix', action='store_true',
                        help='Verify the accepted prefix and pinned SFTP without taking the client lock or writing an ACK.')
    parser.add_argument('--full-limit-gib', type=int, choices=(80, 85),
                        help='Set the local backup limit for FULL_FRESH in GiB.')
    args = parser.parse_args()
    if args.full_limit_gib is not None and args.arm != 'FULL_FRESH':
        parser.error('--full-limit-gib applies only to FULL_FRESH')
    plan_path = HERE / args.arm / 'continuation_plan.json'
    plan = read(plan_path)
    require(plan['arm'] == args.arm, 'Unexpected backup arm')
    control = (BACKUPS / args.arm).resolve(strict=True)
    archive = (control / 'backups').resolve(strict=True)
    require(archive == Path(plan['local_backup_root']).resolve(strict=True), 'Unexpected local archive')
    require(sha(control / 'execution_protocol.json') == plan['original_protocol_sha256'], 'Local protocol differs')
    for name in ('backup_monitor.py', 'backup_protocol.py', 'checkpoint_validator.py'):
        require(sha(control / name) == sha(PROJECT / 'driver' / name), 'Backup module differs: ' + name)
    sys.path.insert(0, str(control))
    import backup_monitor as backup
    import paramiko
    backup.CHECKPOINT_PYTHON = CHECKPOINT_PYTHON.resolve(strict=True)
    backup.CHECKPOINT_TORCH_SITE = TORCH_SITE.resolve(strict=True)
    port = plan['port']
    options = SimpleNamespace(host=HOST, port=port, user='root', known_hosts=PROJECT / 'known_hosts')
    here = PurePosixPath(plan['continuation_root'])
    require(here.is_absolute() and here.name == args.arm
            and all(backup.protocol.SAFE_NAME.fullmatch(part) for part in here.parts[1:]),
            'Unexpected remote archive path')
    password = None
    for slot in ('backup', 'control'):
        try:
            password = protect(credential_path(port, slot).read_bytes(), decode=True).decode('utf-8')
            break
        except OSError:
            continue
    require(bool(password), 'No decryptable SSH credential for the backup client')
    maximum = backup.MAX_BACKUP_BYTES if args.full_limit_gib is None else args.full_limit_gib * 1024**3
    allocation_path = HERE / 'full_backup_bindings_80gib.json'
    if args.arm == 'BASE_FRESH' and allocation_path.is_file():
        allocation = read(allocation_path)['local_backup_limit_GiB']
        require(allocation == {'BASE_FRESH': 40, 'FULL_FRESH': 80}, 'Unexpected local backup allocation')
        maximum = allocation[args.arm] * 1024**3
    store = backup.BackupStore(archive, control, maximum=maximum)
    if args.check_live_prefix:
        for path in (store.root, store.objects, store.epochs):
            require(path.is_dir() and not backup.protocol.is_link(path), 'Existing backup directory is missing')
        store.setup()
        free = shutil.disk_usage(store.root).free
        require(free >= store.reserve + store.maximum - store.used, 'Insufficient local free space for the limit')
        cursor = store.verify_progress(here.name)
        require(cursor >= plan['accepted_rounds'], 'Accepted prefix is shorter than the staged anchor')
        verify_remote_acceptance(backup, paramiko, options, password, here, store, cursor, plan_path)
        print(f'{args.arm}: {cursor} accepted rounds, {len(store.verified)} verified objects, '
              f'{store.used}/{store.maximum} bytes; pinned SFTP acceptance matches', flush=True)
        return
    with backup.protect_local_lock(store.root):
        store.setup()
        if args.check:
            cursor = plan['accepted_rounds']
            stage = store.stage(cursor - 1)
            data = backup.protocol.local_metadata(stage / 'manifest.json')
            manifest = backup.protocol.parse_manifest(data, args.arm, cursor - 1)
            store.verify_stage(stage, data, manifest)
            require(sha(stage / 'manifest.json') == plan['checkpoint_manifest_sha256']
                    and not store.stage(cursor).exists(), 'Local continuation cursor differs')
            verify_remote_acceptance(backup, paramiko, options, password, here, store, cursor, plan_path)
            print(f'{args.arm}: pinned SFTP and accepted backup verified through round {cursor}', flush=True)
            return
        previous_failure = store.root / 'failed_run_receipt.json'
        if previous_failure.exists():
            preserved_failure = store.root / 'failed_run_receipt.original_segment.json'
            require(not preserved_failure.exists(), 'Original failure receipt has already been preserved')
            backup.protocol.regular(previous_failure)
            backup.protocol.durable_replace(previous_failure, preserved_failure)
        try:
            backup.watch(paramiko, options, here, store, password)
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
