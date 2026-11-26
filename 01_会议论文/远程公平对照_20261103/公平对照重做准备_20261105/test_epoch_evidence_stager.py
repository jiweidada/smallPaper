"""Exercise epoch staging using inert checkpoint bytes and local fake SFTP."""
from pathlib import Path
from unittest import TestCase, main
import copy
import gzip
import json
import random
import tempfile

import epoch_backup_barrier as barrier
import epoch_backup_guard as guard
import epoch_backup_transport as transport
import epoch_evidence_stager as stager
from test_epoch_backup_transport import LocalFakeSFTP

RUN, PROTOCOL, ROOT = 'pair_trial_test', 'a' * 64, '/isolated_run'
NONCE = 'b' * 32


def rng_fixture():
    return {'python': random.Random(0).getstate(),
            'numpy': {'name': 'MT19937', 'values': list(range(624)), 'pos': 624,
                      'has_gauss': 0, 'cached_gaussian': 0.0},
            'torch_cpu': [1, 2, 3], 'torch_cuda_all': [[4, 5]],
            'train_loader_generator': [6, 7], 'train_sampler_generator': [8, 9],
            'validation_loader_generator': [10, 11]}


def control(epoch, after=False):
    best_epoch = (0 if after else -1) if epoch == 0 else max(0, epoch - 6)
    best_ap = (0.25 if after else None) if epoch == 0 else 0.3
    return {'last_epoch': epoch, 'ema_updates': (epoch + 1) * guard.BATCHES,
            'ema_decay': 0.9999, 'best_epoch': best_epoch, 'best_ap': best_ap}


def rates(epoch):
    return [(epoch * guard.BATCHES + i, epoch, i, [0.0004, 0.0008]) for i in range(guard.BATCHES)]


def writer(data):
    def save(stream):
        stream.write(data)
    return save


class EvidenceStageTests(TestCase):
    def setUp(self):
        self.tmp_parent = Path(tempfile.gettempdir()).resolve(strict=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='mpsma_stage_test_', dir=str(self.tmp_parent))
        self.work = Path(self.tmp.name).resolve(strict=True)
        assert self.work.is_relative_to(self.tmp_parent) and self.work != self.tmp_parent
        self.remote = self.work / 'remote'
        self.run_root = self.remote / ROOT.lstrip('/')
        self.out = self.run_root / 'runs' / 'BASE'
        self.out.mkdir(parents=True)
        (self.out / 'latest.predictions.json').write_bytes(b'{"synthetic_predictions":true}')
        (self.out / 'latest.evaluation_arrays.npz').write_bytes(b'synthetic-array-bytes')
        self.pred_sha = guard.digest_file(self.out / 'latest.predictions.json')
        self.arr_sha = guard.digest_file(self.out / 'latest.evaluation_arrays.npz')
        self.identity = {key: 'd' * 64 for key in stager.IDENTITY_FIELDS}
        self.metrics = {key: 0.25 for key in stager.METRICS}
        self.rng = rng_fixture()
        self.backup = self.work / 'backup'

    def tearDown(self):
        target = Path(self.tmp.name).resolve(strict=True)
        assert target == self.work and target.is_relative_to(self.tmp_parent) and target != self.tmp_parent
        self.tmp.cleanup()

    def begin(self, epoch=0, **overrides):
        args = {'run_root': self.run_root, 'run_id': RUN, 'arm': 'BASE', 'protocol_sha256': PROTOCOL,
                'epoch': epoch, 'identities': self.identity, 'metrics': self.metrics,
                'prediction_sha': self.pred_sha, 'arrays_sha': self.arr_sha, 'evaluated_tensor_sha': 'e' * 64,
                'rng_before': self.rng, 'control_before': control(epoch), 'learning_rates': rates(epoch),
                'checkpoint_writer': writer(b'evaluated-weights')}
        args.update(overrides)
        return stager.begin(**args)

    def finish(self, handle, restored=False, **overrides):
        epoch = handle['epoch_index']
        args = {'handle': handle, 'restored': restored,
                'continuation_tensor_sha': ('f' if restored else 'e') * 64,
                'rng_after': self.rng, 'control_after': control(epoch, after=True),
                'rng_probe': lambda: self.rng,
                'continuation_writer': writer(b'continuation-weights') if restored else None,
                'nonce': NONCE}
        args.update(overrides)
        return stager.finish(**args)

    def test_no_restore_stage_to_synthetic_backup_and_ack(self):
        handle = self.begin()
        manifest_data = self.finish(handle)
        stage = handle['stage']
        manifest = guard.parse_manifest(manifest_data, RUN, 'BASE', PROTOCOL)
        self.assertFalse(manifest['restore_after_evaluation'])
        self.assertNotIn('continuation.pt', {item['name'] for item in manifest['files']})
        phase = json.loads((stage / 'phase_record.json').read_bytes())
        self.assertEqual(phase['metrics'], self.metrics)
        self.assertEqual(phase['rng_before'], phase['rng_after'])
        self.assertEqual(phase['control_after']['best_epoch'], 0)
        with gzip.open(stage / 'learning_rates.csv.gz', 'rt', encoding='utf-8') as stream:
            lines = stream.readlines()
        self.assertEqual(len(lines), guard.BATCHES + 1)
        self.assertEqual(lines[0].strip(), 'attempt,actual_epoch,batch,group0,group1')
        self.assertTrue(lines[-1].startswith('1616,0,1616,'))
        result = transport.backup_stage(LocalFakeSFTP(self.remote), ROOT, self.backup, self.work,
                                        RUN, 'BASE', 0, PROTOCOL, reserve_bytes=0)
        self.assertEqual(result['status'], 'LOCAL_BACKUP_VERIFIED_AND_ACKNOWLEDGED')
        self.assertEqual(barrier.await_ack(stage, RUN, 'BASE', PROTOCOL, 0, timeout_seconds=3, poll_seconds=0.01)['status'],
                         'ACKNOWLEDGEMENT_VERIFIED')

    def test_restore_creates_distinct_evaluated_and_continuation_states(self):
        handle = self.begin(epoch=296)
        after = {'last_epoch': 290, 'ema_updates': 470000, 'ema_decay': 0.9998,
                 'best_epoch': 290, 'best_ap': 0.3}
        manifest_data = self.finish(handle, restored=True, control_after=after)
        manifest = guard.parse_manifest(manifest_data, RUN, 'BASE', PROTOCOL)
        stage = handle['stage']
        self.assertEqual((stage / 'evaluated.pt').read_bytes(), b'evaluated-weights')
        self.assertEqual((stage / 'continuation.pt').read_bytes(), b'continuation-weights')
        self.assertTrue(manifest['restore_after_evaluation'])
        phase = json.loads((stage / 'phase_record.json').read_bytes())
        self.assertEqual(phase['control_before']['last_epoch'], 296)
        self.assertEqual(phase['control_after']['last_epoch'], 290)
        self.assertNotEqual(phase['evaluated_checkpoint_sha256'], phase['continuation_checkpoint_sha256'])
        guard.check_files(stage, manifest, include_ack=False)

    def test_no_restore_cannot_change_ema_or_control_state(self):
        handle = self.begin()
        with self.assertRaises(guard.BackupError):
            self.finish(handle, continuation_tensor_sha='f' * 64)
        self.assertFalse((handle['stage'] / 'manifest.json').exists())

    def test_changed_rng_or_rng_consumed_by_checkpoint_writer_stops(self):
        handle = self.begin(epoch=296)
        changed = copy.deepcopy(self.rng)
        changed['torch_cpu'][0] ^= 1
        with self.assertRaises(guard.BackupError):
            self.finish(handle, restored=True, rng_after=changed)
        def consuming_writer(stream):
            stream.write(b'continuation')
            self.rng['torch_cpu'][0] ^= 1
        with self.assertRaises(guard.BackupError):
            self.finish(handle, restored=True, continuation_writer=consuming_writer)
        self.assertFalse((handle['stage'] / 'manifest.json').exists())

    def test_invalid_learning_rate_and_rng_fail_before_staging(self):
        invalid = rates(0)
        invalid[4] = (99, 0, 4, [0.0004, 0.0008])
        with self.assertRaises(guard.BackupError):
            self.begin(learning_rates=invalid)
        invalid_rng = copy.deepcopy(self.rng)
        invalid_rng.pop('train_sampler_generator')
        with self.assertRaises(guard.BackupError):
            self.begin(rng_before=invalid_rng)
        self.assertFalse((self.out / 'epochs').exists())

    def test_invalid_metrics_or_identity_fail_before_staging(self):
        with self.assertRaises(guard.BackupError):
            self.begin(metrics={**self.metrics, 'AP': float('nan')})
        with self.assertRaises(guard.BackupError):
            self.begin(identities={})
        self.assertFalse((self.out / 'epochs').exists())

    def test_failed_checkpoint_or_mismatched_prediction_never_publishes(self):
        def failed_writer(stream):
            stream.write(b'partial')
            raise OSError('Simulated checkpoint failure')
        with self.assertRaises(OSError):
            self.begin(checkpoint_writer=failed_writer)
        self.assertFalse((self.out / 'epochs' / 'epoch_0000' / 'manifest.json').exists())
        with self.assertRaises(FileExistsError):
            self.begin()

    def test_wrong_prediction_hash_stops_without_manifest(self):
        with self.assertRaises(guard.BackupError):
            self.begin(prediction_sha='0' * 64)
        self.assertFalse((self.out / 'epochs' / 'epoch_0000' / 'manifest.json').exists())

    def test_no_restore_rejects_unexpected_state_checkpoint(self):
        handle = self.begin()
        with self.assertRaises(guard.BackupError):
            self.finish(handle, continuation_writer=writer(b'unexpected'))
        self.assertFalse((handle['stage'] / 'continuation.pt').exists())

    def test_non_improving_late_epoch_must_mark_restore(self):
        handle = self.begin(epoch=296)
        with self.assertRaises(guard.BackupError):
            self.finish(handle, restored=False)
        self.assertFalse((handle['stage'] / 'manifest.json').exists())

    def test_improving_late_epoch_cannot_mark_restore(self):
        before = {**control(296), 'best_epoch': 290, 'best_ap': 0.2}
        handle = self.begin(epoch=296, control_before=before)
        after = {**control(296), 'best_epoch': 296, 'best_ap': 0.25}
        with self.assertRaises(guard.BackupError):
            self.finish(handle, restored=True, control_after=after)
        self.assertFalse((handle['stage'] / 'manifest.json').exists())

    def test_best_update_must_match_evaluated_ap(self):
        handle = self.begin()
        after = {**control(0, after=True), 'best_ap': 0.24}
        with self.assertRaises(guard.BackupError):
            self.finish(handle, control_after=after)
        self.assertFalse((handle['stage'] / 'manifest.json').exists())

    def test_early_non_improving_epoch_keeps_best_without_restore(self):
        handle = self.begin(epoch=1)
        manifest_data = self.finish(handle)
        manifest = guard.parse_manifest(manifest_data, RUN, 'BASE', PROTOCOL)
        self.assertFalse(manifest['restore_after_evaluation'])
        phase = json.loads((handle['stage'] / 'phase_record.json').read_bytes())
        self.assertEqual(phase['control_before']['best_epoch'], 0)
        self.assertEqual(phase['control_after']['best_epoch'], 0)

    def test_best_before_evaluation_cannot_be_current_epoch(self):
        invalid = {**control(0), 'best_epoch': 0, 'best_ap': 0.3}
        with self.assertRaises(guard.BackupError):
            self.begin(control_before=invalid)
        self.assertFalse((self.out / 'epochs').exists())


if __name__ == '__main__':
    main(verbosity=2)
