"""Test the backup handoff with local fake SFTP and inert synthetic files."""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import TestCase, main, mock
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import stat
import tempfile

import epoch_backup_barrier as barrier
import epoch_backup_guard as guard
import epoch_backup_transport as transport

RUN = 'pair_trial_test'
PROTOCOL = 'a' * 64
ROOT = '/isolated_run'
NONCE = 'b' * 32


class LocalHandle:
    def __init__(self, path, mode):
        self.file = path.open('xb' if mode == 'wx' else mode)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def read(self, size=-1):
        return self.file.read(size)

    def write(self, data):
        return self.file.write(data)

    def flush(self):
        return self.file.flush()

    def stat(self):
        return os.fstat(self.file.fileno())


class LocalFakeSFTP:
    def __init__(self, root, fail_marker=False, fake_link=None):
        self.root = root
        self.fail_marker = fail_marker
        self.fake_link = fake_link

    def path(self, remote):
        remote = PurePosixPath(remote)
        if not remote.is_absolute() or '..' in remote.parts:
            raise ValueError('Invalid fake remote path')
        return self.root.joinpath(*remote.parts[1:])

    def lstat(self, remote):
        if remote == self.fake_link:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_size=0)
        return self.path(remote).lstat()

    def open(self, remote, mode):
        if self.fail_marker and remote.endswith('/ack.ready'):
            raise OSError('Simulated interruption during marker upload')
        return LocalHandle(self.path(remote), mode)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class TransportTests(TestCase):
    def setUp(self):
        self.tmp_parent = Path(tempfile.gettempdir()).resolve(strict=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='mpsma_transport_test_', dir=str(self.tmp_parent))
        self.work = Path(self.tmp.name).resolve(strict=True)
        assert self.work.is_relative_to(self.tmp_parent) and self.work != self.tmp_parent
        self.remote = self.work / 'remote'
        self.remote.mkdir()
        self.backup = self.work / 'backup'
        self.sftp = LocalFakeSFTP(self.remote)

    def tearDown(self):
        target = Path(self.tmp.name).resolve(strict=True)
        assert target == self.work and target.is_relative_to(self.tmp_parent) and target != self.tmp_parent
        self.tmp.cleanup()

    def make_stage(self, epoch, arm='BASE', restored=False):
        path = self.remote / ROOT.lstrip('/') / 'runs' / arm / 'epochs' / ('epoch_%04d' % epoch)
        path.mkdir(parents=True)
        for name in sorted(guard.expected_files(restored) - {'phase_record.json'}):
            (path / name).write_bytes((arm + ':' + str(epoch) + ':' + name).encode())
        phase = {'run_id': RUN, 'arm': arm, 'protocol_sha256': PROTOCOL, 'epoch_index': epoch,
                 'optimizer_updates': (epoch + 1) * guard.BATCHES, 'validation_passes': epoch + 1,
                 'restore_after_evaluation': restored,
                 'evaluated_checkpoint_sha256': guard.digest_file(path / 'evaluated.pt'),
                 'prediction_sha256': guard.digest_file(path / 'predictions.json'),
                 'evaluation_arrays_sha256': guard.digest_file(path / 'evaluation_arrays.npz'),
                 'continuation_checkpoint_sha256': guard.digest_file(path / 'continuation.pt') if restored else None,
                 'evaluated_tensor_sha256': 'c' * 64}
        (path / 'phase_record.json').write_bytes(guard.encode_json(phase))
        data = barrier.publish_manifest(path, RUN, arm, PROTOCOL, epoch, restored, nonce=NONCE)
        return path, data

    def copy(self, arm, epoch, sftp=None):
        return transport.backup_stage(sftp or self.sftp, ROOT, self.backup, self.work,
                                      RUN, arm, epoch, PROTOCOL, reserve_bytes=0)

    def await_stage(self, stage, epoch, arm='BASE'):
        clock = FakeClock()
        return barrier.await_ack(stage, RUN, arm, PROTOCOL, epoch, timeout_seconds=3, poll_seconds=1,
                                 monotonic=clock.monotonic, sleep=clock.sleep)

    def prime_previous_fixture(self, epoch):
        remote, data = self.make_stage(epoch)
        manifest = guard.parse_manifest(data, RUN, 'BASE', PROTOCOL)
        local = self.backup / RUN / 'BASE' / ('epoch_%04d' % epoch)
        local.mkdir(parents=True)
        for path in remote.iterdir():
            shutil.copyfile(path, local / path.name)
        ack = {'schema': 1, 'status': 'LOCAL_BACKUP_VERIFIED',
               **{k: manifest[k] for k in ('run_id', 'arm', 'protocol_sha256', 'epoch_index',
                                           'optimizer_updates', 'validation_passes', 'nonce')},
               'manifest_sha256': guard.digest_bytes(data), 'files': manifest['files']}
        (local / 'ack.json').write_bytes(guard.encode_json(ack))
        guard.validate_ack((local / 'ack.json').read_bytes(), data, manifest)
        guard.check_files(local, manifest, include_ack=True)

    def test_two_arms_and_successive_epoch_handoff(self):
        for arm in ('BASE', 'FULL'):
            stage, data = self.make_stage(0, arm)
            result = self.copy(arm, 0)
            self.assertEqual(result['manifest_sha256'], guard.digest_bytes(data))
            self.assertEqual(self.await_stage(stage, 0, arm)['status'], 'ACKNOWLEDGEMENT_VERIFIED')
        stage, _ = self.make_stage(1)
        self.copy('BASE', 1)
        self.assertEqual(self.await_stage(stage, 1)['epoch_index'], 1)

    def test_two_arms_can_backup_concurrently(self):
        base, _ = self.make_stage(0, 'BASE')
        full, _ = self.make_stage(0, 'FULL')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(self.copy, arm, 0) for arm in ('BASE', 'FULL')]
            results = [job.result() for job in jobs]
        self.assertEqual([r['status'] for r in results],
                         ['LOCAL_BACKUP_VERIFIED_AND_ACKNOWLEDGED'] * 2)
        self.assertEqual(self.await_stage(base, 0, 'BASE')['status'], 'ACKNOWLEDGEMENT_VERIFIED')
        self.assertEqual(self.await_stage(full, 0, 'FULL')['status'], 'ACKNOWLEDGEMENT_VERIFIED')

    def test_corrupt_remote_copy_never_releases_barrier(self):
        stage, _ = self.make_stage(0)
        path = stage / 'predictions.json'
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        with self.assertRaises(guard.BackupError):
            self.copy('BASE', 0)
        self.assertFalse((stage / 'ack.ready').exists())
        with self.assertRaises(TimeoutError):
            self.await_stage(stage, 0)

    def test_no_marker_never_releases_barrier(self):
        stage, _ = self.make_stage(0)
        (stage / 'ack.uploading').write_bytes(b'{"incomplete":true}')
        with self.assertRaises(TimeoutError):
            self.await_stage(stage, 0)

    def test_delayed_local_copy_releases_only_after_ack(self):
        stage, _ = self.make_stage(0)
        clock = FakeClock()
        def progress(seconds):
            clock.sleep(seconds)
            if clock.value == 2:
                self.copy('BASE', 0)
        result = barrier.await_ack(stage, RUN, 'BASE', PROTOCOL, 0, timeout_seconds=5, poll_seconds=1,
                                   monotonic=clock.monotonic, sleep=progress)
        self.assertEqual(result['status'], 'ACKNOWLEDGEMENT_VERIFIED')
        self.assertEqual(result['wait_seconds'], 2)

    def test_ack_arriving_after_deadline_never_releases_barrier(self):
        stage, _ = self.make_stage(0)
        clock = FakeClock()
        def late_progress(seconds):
            clock.sleep(seconds + 1)
            self.copy('BASE', 0)
        with self.assertRaises(TimeoutError):
            barrier.await_ack(stage, RUN, 'BASE', PROTOCOL, 0, timeout_seconds=1, poll_seconds=1,
                              monotonic=clock.monotonic, sleep=late_progress)
        self.assertTrue((stage / 'ack.ready').is_file())

    def test_verification_itself_must_finish_before_deadline(self):
        stage, _ = self.make_stage(0)
        self.copy('BASE', 0)
        clock = FakeClock()
        original_digest = guard.digest_file
        def slow_hash(path):
            clock.sleep(1)
            return original_digest(path)
        with mock.patch.object(guard, 'digest_file', side_effect=slow_hash):
            with self.assertRaises(TimeoutError):
                barrier.await_ack(stage, RUN, 'BASE', PROTOCOL, 0, timeout_seconds=2, poll_seconds=1,
                                  monotonic=clock.monotonic, sleep=clock.sleep)

    def test_marker_upload_failure_does_not_retry_or_release(self):
        stage, _ = self.make_stage(0)
        interrupted = LocalFakeSFTP(self.remote, fail_marker=True)
        with self.assertRaises(OSError):
            self.copy('BASE', 0, interrupted)
        self.assertTrue((stage / 'ack.uploading').is_file())
        self.assertFalse((stage / 'ack.ready').exists())
        with self.assertRaises(TimeoutError):
            self.await_stage(stage, 0)
        with self.assertRaises(FileExistsError):
            self.copy('BASE', 0)

    def test_tampered_ack_and_changed_source_are_rejected(self):
        stage, _ = self.make_stage(0)
        self.copy('BASE', 0)
        ack_path = stage / 'ack.uploading'
        original = ack_path.read_bytes()
        ack = json.loads(original)
        ack['nonce'] = 'd' * 32
        ack_path.write_bytes(guard.encode_json(ack))
        with self.assertRaises(guard.BackupError):
            self.await_stage(stage, 0)
        ack_path.write_bytes(original)
        evaluated = stage / 'evaluated.pt'
        data = bytearray(evaluated.read_bytes())
        data[-1] ^= 1
        evaluated.write_bytes(data)
        with self.assertRaises(guard.BackupError):
            self.await_stage(stage, 0)

    def test_missing_predecessor_and_unsafe_paths_fail(self):
        self.make_stage(1)
        with self.assertRaises(guard.BackupError):
            self.copy('BASE', 1)
        for root in ('/', '/a/../b', 'relative', '/a//b'):
            with self.assertRaises(guard.BackupError):
                transport.remote_stage_path(root, RUN, 'BASE', 0)
        linked = LocalFakeSFTP(self.remote, fake_link=ROOT + '/runs/BASE')
        self.make_stage(0)
        with self.assertRaises(guard.BackupError):
            self.copy('BASE', 0, linked)

    def test_restored_stage_requires_bound_continuation(self):
        # A synthetic epoch-295 predecessor is sufficient for this isolated
        # boundary test; it does not assert that epochs 0–294 exist.
        self.prime_previous_fixture(295)
        stage, data = self.make_stage(296, restored=True)
        manifest = guard.parse_manifest(data, RUN, 'BASE', PROTOCOL)
        self.assertIn('continuation.pt', {item['name'] for item in manifest['files']})
        self.copy('BASE', 296)
        self.assertEqual(self.await_stage(stage, 296)['status'], 'ACKNOWLEDGEMENT_VERIFIED')
        self.assertTrue((self.backup / RUN / 'BASE' / 'epoch_0296' / 'continuation.pt').is_file())
        (stage / 'continuation.pt').unlink()
        with self.assertRaises(guard.BackupError):
            self.await_stage(stage, 296)


if __name__ == '__main__':
    main(verbosity=2)
