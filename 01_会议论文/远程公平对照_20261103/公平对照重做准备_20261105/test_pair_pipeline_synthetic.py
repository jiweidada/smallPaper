"""Exercise the complete evidence handoff with fake bytes and fake SFTP only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import TestCase, main
import json
import shutil
import tempfile

import epoch_backup_barrier as barrier
import epoch_backup_guard as guard
import epoch_evidence_stager as stager
from pair_backup_monitor import MonitorHalted
from test_epoch_backup_transport import FakeClock
from test_epoch_evidence_stager import rates, rng_fixture, writer
from test_verified_pair_backup_monitor import FakeIdentitySFTP, identity_for
from verified_pair_backup_monitor import VerifiedPairBackupMonitor

RUN, PROTOCOL, ROOT = 'pipeline_fixture', 'a' * 64, '/isolated_run'


class PairPipelineTests(TestCase):
    def setUp(self):
        self.tmp_parent = Path(tempfile.gettempdir()).resolve(strict=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='mpsma_pipeline_', dir=str(self.tmp_parent))
        self.work = Path(self.tmp.name).resolve(strict=True)
        assert self.work.is_relative_to(self.tmp_parent) and self.work != self.tmp_parent
        self.remote = self.work / 'remote'
        self.run_root = self.remote / ROOT.lstrip('/')
        for arm in ('BASE', 'FULL'):
            (self.run_root / 'runs' / arm).mkdir(parents=True)
        self.backup = self.work / 'backup'
        self.rng = rng_fixture()
        self.identities = {key: 'd' * 64 for key in stager.IDENTITY_FIELDS}
        self.processes = {'BASE': identity_for('BASE', 4242), 'FULL': identity_for('FULL', 4243)}
        self.connections = {arm: FakeIdentitySFTP(self.remote, self.processes[arm]) for arm in ('BASE', 'FULL')}

    def tearDown(self):
        target = Path(self.tmp.name).resolve(strict=True)
        assert target == self.work and target.is_relative_to(self.tmp_parent) and target != self.tmp_parent
        self.tmp.cleanup()

    def monitor(self, connections=None):
        return VerifiedPairBackupMonitor(connections or self.connections, {'BASE': ROOT, 'FULL': ROOT},
                                         self.processes, self.backup, self.work, RUN, PROTOCOL, reserve_bytes=0)

    def stage(self, arm, epoch, ap, before, after, restored=False):
        out = self.run_root / 'runs' / arm
        predictions = out / 'latest.predictions.json'
        arrays = out / 'latest.evaluation_arrays.npz'
        predictions.write_bytes((arm + ':' + str(epoch) + ':predictions').encode())
        arrays.write_bytes((arm + ':' + str(epoch) + ':arrays').encode())
        metrics = {key: 0.2 for key in stager.METRICS}
        metrics['AP'] = ap
        evaluated = ('evaluated:' + arm + ':' + str(epoch)).encode()
        handle = stager.begin(self.run_root, RUN, arm, PROTOCOL, epoch, self.identities, metrics,
                              guard.digest_file(predictions), guard.digest_file(arrays), 'e' * 64,
                              self.rng, before, rates(epoch), writer(evaluated))
        continuation = ('continuation:' + arm + ':' + str(epoch)).encode() if restored else None
        data = stager.finish(handle, restored, ('f' if restored else 'e') * 64, self.rng, after,
                             lambda: self.rng, continuation_writer=writer(continuation) if restored else None,
                             nonce='b' * 32)
        return handle['stage'], data, evaluated, continuation

    @staticmethod
    def control(epoch, best_epoch, best_ap, last_epoch=None, updates=None):
        return {'last_epoch': epoch if last_epoch is None else last_epoch,
                'ema_updates': (epoch + 1) * guard.BATCHES if updates is None else updates,
                'ema_decay': 0.9999, 'best_epoch': best_epoch, 'best_ap': best_ap}

    @staticmethod
    def wait(stage, arm, epoch):
        clock = FakeClock()
        return barrier.await_ack(stage, RUN, arm, PROTOCOL, epoch, timeout_seconds=3, poll_seconds=1,
                                 monotonic=clock.monotonic, sleep=clock.sleep)

    def prime_previous_fixture(self, arm, epoch, stage, manifest_data):
        """Construct just one prior local stage; no earlier history is claimed."""
        manifest = guard.parse_manifest(manifest_data, RUN, arm, PROTOCOL)
        local = self.backup / RUN / arm / ('epoch_%04d' % epoch)
        local.mkdir(parents=True)
        for path in stage.iterdir():
            shutil.copyfile(path, local / path.name)
        ack = {'schema': 1, 'status': 'LOCAL_BACKUP_VERIFIED',
               **{key: manifest[key] for key in ('run_id', 'arm', 'protocol_sha256', 'epoch_index',
                                                  'optimizer_updates', 'validation_passes', 'nonce')},
               'manifest_sha256': guard.digest_bytes(manifest_data), 'files': manifest['files']}
        (local / 'ack.json').write_bytes(guard.encode_json(ack))
        guard.validate_ack((local / 'ack.json').read_bytes(), manifest_data, manifest)
        guard.check_files(local, manifest, include_ack=True)

    def test_two_arms_two_epochs_preserve_bound_evidence(self):
        monitor = self.monitor()
        for epoch in (0, 1):
            stages = {}
            for arm in ('BASE', 'FULL'):
                ap = 0.2 if epoch == 0 else (0.25 if arm == 'BASE' else 0.19)
                before = self.control(epoch, -1, None) if epoch == 0 else self.control(epoch, 0, 0.2)
                if epoch == 0:
                    after = self.control(epoch, 0, 0.2)
                elif arm == 'BASE':
                    after = self.control(epoch, 1, 0.25)
                else:
                    after = self.control(epoch, 0, 0.2)
                stages[arm] = self.stage(arm, epoch, ap, before, after)
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(monitor.poll_once, arm) for arm in ('BASE', 'FULL')]
                self.assertTrue(all(job.result()['status'] == 'LOCAL_BACKUP_VERIFIED_AND_ACKNOWLEDGED' for job in jobs))
            for arm in ('BASE', 'FULL'):
                stage, data, evaluated, _ = stages[arm]
                self.assertEqual(self.wait(stage, arm, epoch)['status'], 'ACKNOWLEDGEMENT_VERIFIED')
                saved = self.backup / RUN / arm / ('epoch_%04d' % epoch)
                self.assertEqual((saved / 'evaluated.pt').read_bytes(), evaluated)
                self.assertEqual((saved / 'manifest.json').read_bytes(), data)
                phase = json.loads((saved / 'phase_record.json').read_bytes())
                self.assertEqual(phase['metrics']['AP'], 0.2 if epoch == 0 else (0.25 if arm == 'BASE' else 0.19))
        self.assertEqual(monitor.snapshot()['next_epoch'], {'BASE': 2, 'FULL': 2})

    def test_late_rollback_retains_distinct_states_with_synthetic_predecessor(self):
        monitor = self.monitor()
        previous = self.stage('BASE', 295, 0.3, self.control(295, 290, 0.3), self.control(295, 290, 0.3))
        self.prime_previous_fixture('BASE', 295, previous[0], previous[1])
        monitor.monitor._next['BASE'] = 296  # Isolated fixture; epochs 0–294 are not asserted.
        before = self.control(296, 290, 0.3)
        after = self.control(296, 290, 0.3, last_epoch=290, updates=470000)
        stage, _, evaluated, continuation = self.stage('BASE', 296, 0.25, before, after, restored=True)
        monitor.poll_once('BASE')
        self.wait(stage, 'BASE', 296)
        saved = self.backup / RUN / 'BASE' / 'epoch_0296'
        self.assertEqual((saved / 'evaluated.pt').read_bytes(), evaluated)
        self.assertEqual((saved / 'continuation.pt').read_bytes(), continuation)
        self.assertNotEqual(evaluated, continuation)

    def test_corrupt_checkpoint_halts_both_before_marker(self):
        before, after = self.control(0, -1, None), self.control(0, 0, 0.2)
        base = self.stage('BASE', 0, 0.2, before, after)[0]
        full = self.stage('FULL', 0, 0.2, before, after)[0]
        path = full / 'evaluated.pt'
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        monitor = self.monitor()
        with self.assertRaises(MonitorHalted):
            monitor.poll_once('FULL')
        self.assertEqual(monitor.poll_once('BASE')['status'], 'HALTED_NO_AUTOMATIC_RETRY')
        self.assertFalse((full / 'ack.ready').exists() or (base / 'ack.ready').exists())
        with self.assertRaises(TimeoutError):
            self.wait(full, 'FULL', 0)

    def test_identity_loss_after_upload_never_releases_barrier(self):
        before, after = self.control(0, -1, None), self.control(0, 0, 0.2)
        stage = self.stage('BASE', 0, 0.2, before, after)[0]
        connections = dict(self.connections)
        connections['BASE'] = FakeIdentitySFTP(self.remote, self.processes['BASE'], fail_at_verification=4)
        monitor = self.monitor(connections)
        with self.assertRaises(MonitorHalted):
            monitor.poll_once('BASE')
        self.assertTrue((stage / 'ack.uploading').is_file())
        self.assertFalse((stage / 'ack.ready').exists())
        with self.assertRaises(TimeoutError):
            self.wait(stage, 'BASE', 0)


if __name__ == '__main__':
    main(verbosity=2)
