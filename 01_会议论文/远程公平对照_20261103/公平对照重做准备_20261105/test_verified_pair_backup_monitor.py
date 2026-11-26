"""Test identity-bound acknowledgements with synthetic files and fake proc."""
from io import BytesIO
from pathlib import Path
from unittest import TestCase, main
import copy
import tempfile

import epoch_backup_barrier as barrier
import epoch_backup_guard as guard
from pair_backup_monitor import MonitorHalted
from test_epoch_backup_transport import LocalFakeSFTP
from test_remote_process_identity import BOOT, NAMESPACE, stat_record
from verified_pair_backup_monitor import VerifiedPairBackupMonitor

RUN, PROTOCOL, ROOT = 'verified_pair_test', 'a' * 64, '/isolated_run'


def identity_for(arm, pid):
    return {'boot_id': BOOT, 'pid_namespace': NAMESPACE, 'pid': pid, 'start_ticks': 123456 + pid,
            'argv': ['/opt/python', '-B', '-u', ROOT + '/guarded_fair_runner.py', 'train', '--arm', arm]}


class FakeIdentitySFTP(LocalFakeSFTP):
    def __init__(self, root, expected, fail_at_verification=None):
        super().__init__(root)
        self.expected = copy.deepcopy(expected)
        self.boot_reads = 0
        self.fail_at_verification = fail_at_verification

    def open(self, remote, mode):
        pid = self.expected['pid']
        if remote == '/proc/sys/kernel/random/boot_id':
            self.boot_reads += 1
            if self.fail_at_verification and (self.boot_reads + 1) // 2 >= self.fail_at_verification:
                raise FileNotFoundError('Synthetic process environment disappeared')
            return BytesIO((self.expected['boot_id'] + '\n').encode())
        if remote == f'/proc/{pid}/stat':
            return BytesIO(stat_record(pid=pid, ticks=self.expected['start_ticks']))
        if remote == f'/proc/{pid}/cmdline':
            return BytesIO(b'\x00'.join(x.encode() for x in self.expected['argv']) + b'\x00')
        return super().open(remote, mode)

    def readlink(self, path):
        assert path == '/proc/self/ns/pid'
        return self.expected['pid_namespace']


class VerifiedPairTests(TestCase):
    def setUp(self):
        self.tmp_parent = Path(tempfile.gettempdir()).resolve(strict=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='mpsma_verified_pair_', dir=str(self.tmp_parent))
        self.work = Path(self.tmp.name).resolve(strict=True)
        assert self.work.is_relative_to(self.tmp_parent) and self.work != self.tmp_parent
        self.remote = self.work / 'remote'
        self.remote.mkdir()
        self.backup = self.work / 'backup'
        self.expected = {'BASE': identity_for('BASE', 4242), 'FULL': identity_for('FULL', 4243)}
        self.roots = {'BASE': ROOT, 'FULL': ROOT}
        self.connections = {arm: FakeIdentitySFTP(self.remote, self.expected[arm]) for arm in ('BASE', 'FULL')}

    def tearDown(self):
        target = Path(self.tmp.name).resolve(strict=True)
        assert target == self.work and target.is_relative_to(self.tmp_parent) and target != self.tmp_parent
        self.tmp.cleanup()

    def monitor(self, connections=None, identities=None):
        return VerifiedPairBackupMonitor(connections or self.connections, self.roots, identities or self.expected,
                                        self.backup, self.work, RUN, PROTOCOL, reserve_bytes=0)

    def stage(self, arm, epoch=0):
        path = self.remote / ROOT.lstrip('/') / 'runs' / arm / 'epochs' / ('epoch_%04d' % epoch)
        path.mkdir(parents=True)
        for name in sorted(guard.BASE_FILES - {'phase_record.json'}):
            (path / name).write_bytes((arm + ':' + str(epoch) + ':' + name).encode())
        phase = {'run_id': RUN, 'arm': arm, 'protocol_sha256': PROTOCOL, 'epoch_index': epoch,
                 'optimizer_updates': (epoch + 1) * guard.BATCHES, 'validation_passes': epoch + 1,
                 'restore_after_evaluation': False,
                 'evaluated_checkpoint_sha256': guard.digest_file(path / 'evaluated.pt'),
                 'prediction_sha256': guard.digest_file(path / 'predictions.json'),
                 'evaluation_arrays_sha256': guard.digest_file(path / 'evaluation_arrays.npz'),
                 'continuation_checkpoint_sha256': None, 'evaluated_tensor_sha256': 'c' * 64}
        (path / 'phase_record.json').write_bytes(guard.encode_json(phase))
        barrier.publish_manifest(path, RUN, arm, PROTOCOL, epoch, False, nonce='b' * 32)
        return path

    def test_matching_identities_allow_both_acknowledgements(self):
        base, full = self.stage('BASE'), self.stage('FULL')
        monitor = self.monitor()
        for arm in ('BASE', 'FULL'):
            self.assertEqual(monitor.poll_once(arm)['status'], 'LOCAL_BACKUP_VERIFIED_AND_ACKNOWLEDGED')
            self.assertGreaterEqual(self.connections[arm].boot_reads, 6)
        self.assertTrue((base / 'ack.ready').is_file() and (full / 'ack.ready').is_file())
        self.assertEqual(monitor.snapshot()['status'], 'ACTIVE_PROCESS_LIVENESS_UNVERIFIED')

    def test_missing_manifest_does_not_assert_process_liveness(self):
        monitor = self.monitor()
        self.assertEqual(monitor.poll_once('BASE')['status'], 'AWAITING_NEXT_MANIFEST_PROCESS_LIVENESS_UNVERIFIED')
        self.assertEqual(self.connections['BASE'].boot_reads, 0)

    def test_identity_disappearing_before_ack_halts_both_arms(self):
        base, full = self.stage('BASE'), self.stage('FULL')
        connections = dict(self.connections)
        connections['FULL'] = FakeIdentitySFTP(self.remote, self.expected['FULL'], fail_at_verification=3)
        monitor = self.monitor(connections=connections)
        with self.assertRaises(MonitorHalted):
            monitor.poll_once('FULL')
        self.assertFalse((full / 'ack.uploading').exists() or (full / 'ack.ready').exists())
        self.assertEqual(monitor.poll_once('BASE')['status'], 'HALTED_NO_AUTOMATIC_RETRY')
        self.assertFalse((base / 'ack.ready').exists())

    def test_process_replacement_after_ack_upload_prevents_marker(self):
        stage = self.stage('BASE')
        connections = dict(self.connections)
        connections['BASE'] = FakeIdentitySFTP(self.remote, self.expected['BASE'], fail_at_verification=4)
        monitor = self.monitor(connections=connections)
        with self.assertRaises(MonitorHalted):
            monitor.poll_once('BASE')
        self.assertTrue((stage / 'ack.uploading').is_file())
        self.assertFalse((stage / 'ack.ready').exists())

    def test_wrong_arm_command_or_same_process_is_rejected(self):
        invalid = copy.deepcopy(self.expected)
        invalid['BASE']['argv'][-1] = 'FULL'
        with self.assertRaises(guard.BackupError):
            self.monitor(identities=invalid)
        duplicate = copy.deepcopy(self.expected)
        duplicate['FULL']['pid'] = duplicate['BASE']['pid']
        duplicate['FULL']['start_ticks'] = duplicate['BASE']['start_ticks']
        with self.assertRaises(guard.BackupError):
            self.monitor(identities=duplicate)

    def test_expected_identity_is_copied_not_mutable_from_caller(self):
        stage = self.stage('BASE')
        monitor = self.monitor()
        self.expected['BASE']['argv'][0] = '/changed'
        self.assertEqual(monitor.poll_once('BASE')['status'], 'LOCAL_BACKUP_VERIFIED_AND_ACKNOWLEDGED')
        self.assertTrue((stage / 'ack.ready').exists())


if __name__ == '__main__':
    main(verbosity=2)
