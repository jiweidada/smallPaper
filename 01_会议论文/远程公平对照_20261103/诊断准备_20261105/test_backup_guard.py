"""Exercise backup integrity barriers with tiny synthetic files; no model or network."""
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('backup_guard', HERE / 'backup_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

def content(name):
    return ('fixture:' + name).encode()

def manifest(phase, epoch, updates, passes):
    names = guard.EPOCH_FILES if phase.startswith('epoch_') else guard.EVAL_FILES
    return {'schema': 1, 'status': 'READY_FOR_BACKUP', 'run_id': guard.RUN_ID,
            'plan_sha256': guard.PLAN_SHA, 'storage_plan_sha256': guard.STORAGE_PLAN_SHA,
            'input_checkpoint_sha256': guard.CHECKPOINT_SHA, 'nonce': '0123456789abcdef0123456789abcdef',
            'phase': phase, 'epoch_index': epoch, 'optimizer_updates': updates,
            'validation_passes': passes,
            'files': [{'name': n, 'bytes': len(content(n)), 'sha256': sha256(content(n)).hexdigest()}
                      for n in sorted(names)]}

class BackupGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix='.backup_guard_test_', dir=HERE)).resolve()
        self.base = self.temp / 'copies'
        self.counter = 0

    def tearDown(self):
        # Verify the exact newly created directory before recursive cleanup.
        target = self.temp.resolve()
        assert target.parent == HERE.resolve() and target.name.startswith('.backup_guard_test_')
        assert target.is_relative_to(guard.WORKSPACE.resolve()) and target != HERE.resolve()
        shutil.rmtree(target)

    def write_manifest(self, obj):
        self.counter += 1
        path = self.temp / f'manifest_{self.counter}.json'
        path.write_text(json.dumps(obj), encoding='utf-8')
        return path

    def populate(self, stage):
        for item in stage['downloads']:
            Path(item['local']).write_bytes(content(Path(item['local']).name))

    def good_initial(self):
        path = self.write_manifest(manifest('initial_ema', 213, 0, 1))
        stage = guard.prepare(path, self.base)
        self.populate(stage)
        ack = guard.verify(path, self.base)
        return path, stage, ack

    def test_verified_copy_and_predecessor(self):
        path, stage, ack = self.good_initial()
        self.assertEqual(ack['status'], 'LOCAL_BACKUP_VERIFIED')
        data = json.loads(Path(ack['local_ack']).read_text(encoding='utf-8'))
        self.assertEqual(data['manifest_sha256'], sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(data['files']), 4)
        with self.assertRaises(FileExistsError):
            guard.prepare(path, self.base)
        raw = self.write_manifest(manifest('initial_raw', 213, 0, 2))
        raw_stage = guard.prepare(raw, self.base)
        self.populate(raw_stage)
        self.assertEqual(guard.verify(raw, self.base)['status'], 'LOCAL_BACKUP_VERIFIED')
        ep = self.write_manifest(manifest('epoch_0214', 214, 1617, 3))
        self.assertEqual(guard.prepare(ep, self.base)['phase'], 'epoch_0214')

    def test_no_predecessor_or_corrupt_download(self):
        raw = self.write_manifest(manifest('initial_raw', 213, 0, 2))
        with self.assertRaises(guard.BackupError):
            guard.prepare(raw, self.base)
        self.good_initial()
        stage = guard.prepare(raw, self.base)
        self.populate(stage)
        p = Path(stage['local_stage']) / 'metrics.json'
        p.write_bytes(b'X' * len(content('metrics.json')))
        with self.assertRaises(guard.BackupError):
            guard.verify(raw, self.base)
        self.assertFalse((Path(stage['local_stage']) / 'ack.json').exists())

    def test_previous_copy_tampering_blocks_next_phase(self):
        _, stage, _ = self.good_initial()
        (Path(stage['local_stage']) / 'metrics.json').write_bytes(b'X' * len(content('metrics.json')))
        raw = self.write_manifest(manifest('initial_raw', 213, 0, 2))
        with self.assertRaises(guard.BackupError):
            guard.prepare(raw, self.base)

    def test_manifest_attacks_and_budget(self):
        good = manifest('initial_ema', 213, 0, 1)
        unsafe = json.loads(json.dumps(good))
        unsafe['files'][0]['name'] = '../metrics.json'
        with self.assertRaises(guard.BackupError):
            guard.parse_manifest(json.dumps(unsafe).encode())
        duplicate = b'{"schema":1,' + json.dumps(good).encode()[1:]
        with self.assertRaises(guard.BackupError):
            guard.parse_manifest(duplicate)
        wrong = manifest('epoch_0225', 225, 12 * 1617, 14)
        with self.assertRaises(guard.BackupError):
            guard.parse_manifest(json.dumps(wrong).encode())
        outside = guard.WORKSPACE.parent / 'outside_backup_guard_test'
        with self.assertRaises(guard.BackupError):
            guard.checked_backup_root(outside)
        with self.assertRaises(guard.BackupError):
            guard.checked_backup_root(guard.WORKSPACE / 'a' / '..' / 'b')

if __name__ == '__main__':
    unittest.main(verbosity=2)
