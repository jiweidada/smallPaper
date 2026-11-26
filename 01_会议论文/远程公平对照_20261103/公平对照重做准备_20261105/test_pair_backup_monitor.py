"""Exercise pair coordination with temporary synthetic files and fake SFTP."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest import TestCase, main
import tempfile

import epoch_backup_barrier as barrier
import epoch_backup_guard as guard
from pair_backup_monitor import MonitorHalted, PairBackupMonitor
from test_epoch_backup_transport import LocalFakeSFTP

RUN, PROTOCOL, ROOT = 'pair_monitor_test', 'a' * 64, '/isolated_run'


class CountingSFTP(LocalFakeSFTP):
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        self.calls = 0

    def lstat(self, remote):
        self.calls += 1
        return super().lstat(remote)

    def open(self, remote, mode):
        self.calls += 1
        return super().open(remote, mode)


class BrokenSFTP:
    def lstat(self, path):
        raise ConnectionResetError('Synthetic connection failure')

    def open(self, path, mode):
        raise AssertionError('Must not open after failed lstat')


class BlockingReader:
    def __init__(self, wrapped, started, release):
        self.wrapped, self.started, self.release = wrapped, started, release

    def __enter__(self):
        self.wrapped.__enter__()
        return self

    def __exit__(self, *args):
        return self.wrapped.__exit__(*args)

    def stat(self):
        return self.wrapped.stat()

    def read(self, size=-1):
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError('Synthetic copy was not released')
        return self.wrapped.read(size)


class BlockingSFTP(LocalFakeSFTP):
    def __init__(self, root, started, release):
        super().__init__(root)
        self.started, self.release = started, release

    def open(self, remote, mode):
        handle = super().open(remote, mode)
        if mode == 'rb' and remote.endswith('/predictions.json'):
            return BlockingReader(handle, self.started, self.release)
        return handle


class PairMonitorTests(TestCase):
    def setUp(self):
        self.tmp_parent = Path(tempfile.gettempdir()).resolve(strict=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='mpsma_pair_monitor_', dir=str(self.tmp_parent))
        self.work = Path(self.tmp.name).resolve(strict=True)
        assert self.work.is_relative_to(self.tmp_parent) and self.work != self.tmp_parent
        self.remote = self.work / 'remote'
        self.remote.mkdir()
        self.backup = self.work / 'backup'
        self.connections = {arm: CountingSFTP(self.remote) for arm in ('BASE', 'FULL')}
        self.roots = {arm: ROOT for arm in ('BASE', 'FULL')}

    def tearDown(self):
        target = Path(self.tmp.name).resolve(strict=True)
        assert target == self.work and target.is_relative_to(self.tmp_parent) and target != self.tmp_parent
        self.tmp.cleanup()

    def monitor(self, connections=None):
        return PairBackupMonitor(connections or self.connections, self.roots, self.backup, self.work,
                                 RUN, PROTOCOL, reserve_bytes=0)

    def make_stage(self, arm, epoch, run_id=RUN):
        path = self.remote / ROOT.lstrip('/') / 'runs' / arm / 'epochs' / ('epoch_%04d' % epoch)
        path.mkdir(parents=True)
        for name in sorted(guard.BASE_FILES - {'phase_record.json'}):
            (path / name).write_bytes((arm + ':' + str(epoch) + ':' + name).encode())
        phase = {'run_id': run_id, 'arm': arm, 'protocol_sha256': PROTOCOL, 'epoch_index': epoch,
                 'optimizer_updates': (epoch + 1) * guard.BATCHES, 'validation_passes': epoch + 1,
                 'restore_after_evaluation': False,
                 'evaluated_checkpoint_sha256': guard.digest_file(path / 'evaluated.pt'),
                 'prediction_sha256': guard.digest_file(path / 'predictions.json'),
                 'evaluation_arrays_sha256': guard.digest_file(path / 'evaluation_arrays.npz'),
                 'continuation_checkpoint_sha256': None, 'evaluated_tensor_sha256': 'c' * 64}
        (path / 'phase_record.json').write_bytes(guard.encode_json(phase))
        barrier.publish_manifest(path, run_id, arm, PROTOCOL, epoch, False, nonce='b' * 32)
        return path

    def test_waiting_is_not_a_process_or_completion_claim(self):
        monitor = self.monitor()
        result = monitor.poll_once('BASE')
        self.assertEqual(result['status'], 'AWAITING_NEXT_MANIFEST_PROCESS_LIVENESS_UNVERIFIED')
        self.assertEqual(monitor.snapshot()['status'], 'ACTIVE_PROCESS_LIVENESS_UNVERIFIED')
        self.assertFalse(self.backup.exists())
        stage = self.make_stage('BASE', 0)
        self.assertEqual(monitor.poll_once('BASE')['status'], 'LOCAL_BACKUP_VERIFIED_AND_ACKNOWLEDGED')
        self.assertTrue((stage / 'ack.ready').is_file())
        self.assertEqual(monitor.poll_once('BASE')['epoch'], 1)
        self.assertEqual(monitor.snapshot()['next_epoch'], {'BASE': 1, 'FULL': 0})

    def test_both_arms_are_coordinated_when_polled_concurrently(self):
        base, full = self.make_stage('BASE', 0), self.make_stage('FULL', 0)
        monitor = self.monitor()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(monitor.poll_once, arm) for arm in ('BASE', 'FULL')]
            self.assertTrue(all(job.result()['status'] == 'LOCAL_BACKUP_VERIFIED_AND_ACKNOWLEDGED' for job in jobs))
        self.assertTrue((base / 'ack.ready').is_file() and (full / 'ack.ready').is_file())
        self.assertEqual(monitor.snapshot()['next_epoch'], {'BASE': 1, 'FULL': 1})

    def test_corrupt_one_arm_halts_the_pair_without_retry(self):
        base, full = self.make_stage('BASE', 0), self.make_stage('FULL', 0)
        path = full / 'predictions.json'
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        monitor = self.monitor()
        with self.assertRaises(MonitorHalted):
            monitor.poll_once('FULL')
        full_calls = self.connections['FULL'].calls
        base_calls = self.connections['BASE'].calls
        self.assertEqual(monitor.poll_once('BASE')['status'], 'HALTED_NO_AUTOMATIC_RETRY')
        self.assertEqual(monitor.poll_once('FULL')['status'], 'HALTED_NO_AUTOMATIC_RETRY')
        self.assertEqual((base_calls, full_calls), (self.connections['BASE'].calls, self.connections['FULL'].calls))
        self.assertFalse((base / 'ack.ready').exists() or (full / 'ack.ready').exists())
        with self.assertRaises(guard.BackupError):
            self.monitor()

    def test_marker_failure_halts_without_releasing_other_arm(self):
        base, full = self.make_stage('BASE', 0), self.make_stage('FULL', 0)
        connections = dict(self.connections)
        connections['FULL'] = CountingSFTP(self.remote, fail_marker=True)
        monitor = self.monitor(connections)
        with self.assertRaises(MonitorHalted):
            monitor.poll_once('FULL')
        self.assertTrue((full / 'ack.uploading').is_file())
        self.assertFalse((full / 'ack.ready').exists())
        self.assertEqual(monitor.poll_once('BASE')['status'], 'HALTED_NO_AUTOMATIC_RETRY')
        self.assertFalse((base / 'ack.ready').exists())

    def test_wrong_run_identity_halts_before_acknowledgement(self):
        stage = self.make_stage('BASE', 0, run_id='different_run')
        monitor = self.monitor()
        with self.assertRaises(MonitorHalted):
            monitor.poll_once('BASE')
        self.assertFalse((stage / 'ack.ready').exists())
        self.assertEqual(monitor.snapshot()['status'], 'HALTED_NO_AUTOMATIC_RETRY')

    def test_connection_failure_is_not_treated_as_missing_manifest(self):
        connections = dict(self.connections)
        connections['BASE'] = BrokenSFTP()
        monitor = self.monitor(connections)
        with self.assertRaises(MonitorHalted):
            monitor.poll_once('BASE')
        self.assertEqual(monitor.snapshot()['failure']['reason'], 'ConnectionResetError')
        self.assertEqual(monitor.poll_once('FULL')['status'], 'HALTED_NO_AUTOMATIC_RETRY')

    def test_operator_halt_prevents_future_network_operations(self):
        self.make_stage('BASE', 0)
        monitor = self.monitor()
        monitor.halt('OPERATOR_STOP')
        self.assertEqual(monitor.poll_once('BASE')['status'], 'HALTED_NO_AUTOMATIC_RETRY')
        self.assertEqual(self.connections['BASE'].calls, 0)

    def test_operator_halt_during_copy_prevents_new_marker(self):
        stage = self.make_stage('BASE', 0)
        started, release = Event(), Event()
        connections = dict(self.connections)
        connections['BASE'] = BlockingSFTP(self.remote, started, release)
        monitor = self.monitor(connections)
        with ThreadPoolExecutor(max_workers=2) as pool:
            copying = pool.submit(monitor.poll_once, 'BASE')
            self.assertTrue(started.wait(2))
            stopping = pool.submit(monitor.halt, 'OPERATOR_STOP')
            self.assertTrue(monitor._stop_event.wait(2))
            release.set()
            with self.assertRaises(MonitorHalted):
                copying.result(timeout=5)
            stopping.result(timeout=5)
        self.assertFalse((stage / 'ack.ready').exists())
        self.assertEqual(monitor.snapshot()['failure']['reason'], 'OPERATOR_STOP')

    def test_corrupt_previous_backup_halts_at_the_next_boundary(self):
        self.make_stage('BASE', 0)
        monitor = self.monitor()
        monitor.poll_once('BASE')
        old = self.backup / RUN / 'BASE' / 'epoch_0000' / 'evaluated.pt'
        data = bytearray(old.read_bytes())
        data[0] ^= 1
        old.write_bytes(data)
        next_stage = self.make_stage('BASE', 1)
        with self.assertRaises(MonitorHalted):
            monitor.poll_once('BASE')
        self.assertFalse((next_stage / 'ack.ready').exists())
        self.assertEqual(monitor.snapshot()['next_epoch']['BASE'], 1)

    def test_all_backup_markers_are_not_training_completion(self):
        monitor = self.monitor()
        # Synthetic counters only exercise the status label, not a 320-epoch run.
        monitor._next = {'BASE': guard.EPOCHS, 'FULL': guard.EPOCHS}
        self.assertEqual(monitor.snapshot()['status'], 'ALL_BACKUPS_ACKNOWLEDGED_TRAINING_COMPLETION_UNVERIFIED')
        self.assertEqual(monitor.poll_once('BASE')['status'], 'ARM_BACKUPS_ACKNOWLEDGED_TRAINING_COMPLETION_UNVERIFIED')


if __name__ == '__main__':
    main(verbosity=2)
