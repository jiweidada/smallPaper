"""Exercise integrity failures with synthetic bytes; no model or network use."""
from pathlib import Path
from unittest import TestCase, main, mock
import json
import os
import shutil
import tempfile

import epoch_backup_guard as g

RUN = 'pair_trial_test'
PROTOCOL = 'a' * 64
NONCE = 'b' * 32


class GuardTests(TestCase):
    def setUp(self):
        self.tmp_parent = Path(tempfile.gettempdir()).resolve(strict=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='mpsma_guard_test_', dir=str(self.tmp_parent))
        self.work = Path(self.tmp.name).resolve(strict=True)
        assert self.work.is_relative_to(self.tmp_parent) and self.work != self.tmp_parent
        self.root = self.work / 'backup'

    def tearDown(self):
        target = Path(self.tmp.name).resolve(strict=True)
        assert target == self.work and target.is_relative_to(self.tmp_parent) and target != self.tmp_parent
        self.tmp.cleanup()

    def make_source(self, epoch, arm='BASE', restored=False):
        folder = self.work / ('src_%s_%04d_%d' % (arm, epoch, int(restored)))
        folder.mkdir()
        for name in sorted(g.expected_files(restored) - {'phase_record.json'}):
            (folder / name).write_bytes((name + ':' + str(epoch) + ':' + arm).encode())
        phase = {'run_id': RUN, 'arm': arm, 'protocol_sha256': PROTOCOL, 'epoch_index': epoch,
                 'optimizer_updates': (epoch + 1) * g.BATCHES, 'validation_passes': epoch + 1,
                 'restore_after_evaluation': restored,
                 'evaluated_checkpoint_sha256': g.digest_file(folder / 'evaluated.pt'),
                 'prediction_sha256': g.digest_file(folder / 'predictions.json'),
                 'evaluation_arrays_sha256': g.digest_file(folder / 'evaluation_arrays.npz'),
                 'continuation_checkpoint_sha256': g.digest_file(folder / 'continuation.pt') if restored else None,
                 'evaluated_tensor_sha256': 'c' * 64}
        (folder / 'phase_record.json').write_bytes(g.encode_json(phase))
        manifest = g.build_manifest(folder, RUN, arm, PROTOCOL, epoch, restored, NONCE)
        return folder, manifest

    def prepare_copy(self, folder, manifest, arm='BASE'):
        target = g.prepare(manifest, self.root, self.work, RUN, arm, PROTOCOL, reserve_bytes=0)
        for path in folder.iterdir():
            shutil.copyfile(path, target / path.name)
        return target

    def complete(self, folder, manifest, arm='BASE'):
        target = self.prepare_copy(folder, manifest, arm)
        ack = g.verify(manifest, self.root, self.work, RUN, arm, PROTOCOL)
        parsed = g.parse_manifest(manifest, RUN, arm, PROTOCOL)
        g.validate_ack(ack, manifest, parsed)
        return target, ack

    def test_sequential_backups_and_separate_arms(self):
        first, first_manifest = self.make_source(0)
        later, later_manifest = self.make_source(1)
        with self.assertRaises(g.BackupError):
            g.prepare(later_manifest, self.root, self.work, RUN, 'BASE', PROTOCOL, reserve_bytes=0)
        target, ack = self.complete(first, first_manifest)
        self.assertTrue((target / 'ack.json').is_file())
        self.assertEqual(json.loads(ack)['optimizer_updates'], 1617)
        second, _ = self.complete(later, later_manifest)
        self.assertTrue((second / 'ack.json').is_file())
        full, full_manifest = self.make_source(0, arm='FULL')
        full_target, _ = self.complete(full, full_manifest, arm='FULL')
        self.assertNotEqual(target, full_target)

    def test_corrupt_download_never_produces_ack(self):
        folder, manifest = self.make_source(0)
        stage = self.prepare_copy(folder, manifest)
        path = stage / 'predictions.json'
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        with self.assertRaises(g.BackupError):
            g.verify(manifest, self.root, self.work, RUN, 'BASE', PROTOCOL)
        self.assertFalse((stage / 'ack.json').exists())

    def test_changed_previous_backup_blocks_next_epoch(self):
        first, manifest = self.make_source(0)
        previous, _ = self.complete(first, manifest)
        path = previous / 'evaluated.pt'
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        later, later_manifest = self.make_source(1)
        with self.assertRaises(g.BackupError):
            g.prepare(later_manifest, self.root, self.work, RUN, 'BASE', PROTOCOL, reserve_bytes=0)
        self.assertFalse((self.root / RUN / 'BASE' / 'epoch_0001').exists())

    def test_post_validation_restore_requires_separate_bound_state(self):
        folder, manifest = self.make_source(296, restored=True)
        parsed = g.parse_manifest(manifest, RUN, 'BASE', PROTOCOL)
        self.assertIn('continuation.pt', {x['name'] for x in parsed['files']})
        phase_path = folder / 'phase_record.json'
        phase = json.loads(phase_path.read_bytes())
        phase['continuation_checkpoint_sha256'] = 'd' * 64
        phase_path.write_bytes(g.encode_json(phase))
        with self.assertRaises(g.BackupError):
            g.build_manifest(folder, RUN, 'BASE', PROTOCOL, 296, True, NONCE)
        with self.assertRaises(g.BackupError):
            g.parse_manifest(g.encode_json({**parsed, 'epoch_index': 10,
                                            'optimizer_updates': 11 * g.BATCHES,
                                            'validation_passes': 11}), RUN, 'BASE', PROTOCOL)

    def test_identity_counts_paths_and_duplicate_keys_rejected(self):
        _, data = self.make_source(0)
        m = g.parse_manifest(data, RUN, 'BASE', PROTOCOL)
        for bad in [{**m, 'arm': 'FULL'}, {**m, 'optimizer_updates': 1616},
                    {**m, 'validation_passes': True}, {**m, 'nonce': '../escape'},
                    {**m, 'files': [{**m['files'][0], 'name': '../escape'}] + m['files'][1:]}]:
            with self.assertRaises(g.BackupError):
                g.parse_manifest(g.encode_json(bad), RUN, 'BASE', PROTOCOL)
        with self.assertRaises(g.BackupError):
            g.parse_json(b'{"schema":1,"schema":1}')
        with self.assertRaises(g.BackupError):
            g.checked_root(self.work.parent / 'outside', self.work)

    def test_ack_is_bound_and_cannot_be_replayed(self):
        folder, manifest = self.make_source(0)
        _, ack = self.complete(folder, manifest)
        parsed = g.parse_manifest(manifest, RUN, 'BASE', PROTOCOL)
        tampered = json.loads(ack)
        for field, value in [('nonce', 'f' * 32), ('arm', 'FULL'), ('manifest_sha256', '0' * 64),
                             ('optimizer_updates', 0)]:
            bad = {**tampered, field: value}
            with self.assertRaises(g.BackupError):
                g.validate_ack(g.encode_json(bad), manifest, parsed)
        with self.assertRaises(g.BackupError):
            g.verify(manifest, self.root, self.work, RUN, 'BASE', PROTOCOL)

    def test_space_exhaustion_stops_before_stage_creation(self):
        _, manifest = self.make_source(0)
        with mock.patch.object(g.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(100, 99, 1)):
            with self.assertRaises(g.BackupError):
                g.prepare(manifest, self.root, self.work, RUN, 'BASE', PROTOCOL, reserve_bytes=0)
        self.assertFalse(self.root.exists())

    def test_linked_path_is_rejected_when_supported(self):
        _, manifest = self.make_source(0)
        self.root.mkdir()
        elsewhere = self.work / 'elsewhere'
        elsewhere.mkdir()
        try:
            os.symlink(elsewhere, self.root / RUN, target_is_directory=True)
        except OSError:
            self.skipTest('Windows account cannot create symbolic links')
        with self.assertRaises(g.BackupError):
            g.prepare(manifest, self.root, self.work, RUN, 'BASE', PROTOCOL, reserve_bytes=0)

    def test_junction_branch_is_rejected_with_simulated_path(self):
        _, manifest = self.make_source(0)
        forbidden = self.root / RUN
        with mock.patch.object(g.Path, 'is_junction', autospec=True, side_effect=lambda path: path == forbidden):
            with self.assertRaises(g.BackupError):
                g.prepare(manifest, self.root, self.work, RUN, 'BASE', PROTOCOL, reserve_bytes=0)


if __name__ == '__main__':
    main(verbosity=2)
