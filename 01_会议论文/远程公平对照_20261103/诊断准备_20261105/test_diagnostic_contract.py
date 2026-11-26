"""Synthetic budget and backup-handshake checks; no model, dataset or network."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import shutil
import tempfile
import unittest

import diagnostic_contract as c
import backup_guard as g
from test_backup_guard import manifest, content

HERE = Path(__file__).resolve().parent

class ContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix='.diagnostic_test_', dir=HERE)).resolve()

    def tearDown(self):
        target = self.temp.resolve()
        assert target.parent == HERE.resolve() and target.name.startswith('.diagnostic_test_')
        assert target.is_relative_to(g.WORKSPACE.resolve()) and target != HERE.resolve()
        shutil.rmtree(target)

    def test_budgets_match_local_guard(self):
        stages = [('initial_ema', 213), ('initial_raw', 213)]
        stages += [('epoch_%04d' % n, n) for n in range(214, 225)]
        stages += [('terminal_raw', n) for n in range(214, 225)]
        for phase, epoch in stages:
            updates, passes = c.expected_counters(phase, epoch)
            parsed = g.parse_manifest(json.dumps(manifest(phase, epoch, updates, passes)).encode())
            self.assertLessEqual(parsed['optimizer_updates'], 17787)
            self.assertLessEqual(parsed['validation_passes'], 14)
        for phase, epoch in [('initial_ema', 214), ('epoch_0225', 225), ('terminal_raw', 213)]:
            with self.assertRaises(RuntimeError):
                c.expected_counters(phase, epoch)

    def test_acknowledgement_waits_for_marker_and_checks_bytes(self):
        stage = self.temp / 'phases' / 'initial_ema'
        stage.mkdir(parents=True)
        obj = manifest('initial_ema', 213, 0, 1)
        data = json.dumps(obj).encode()
        (stage / 'manifest.json').write_bytes(data)
        for item in obj['files']:
            (stage / item['name']).write_bytes(content(item['name']))
        ack = {'status': 'LOCAL_BACKUP_VERIFIED', 'run_id': c.RUN_ID, 'phase': 'initial_ema',
               'epoch_index': 213, 'optimizer_updates': 0, 'validation_passes': 1,
               'nonce': obj['nonce'], 'manifest_sha256': hashlib.sha256(data).hexdigest(), 'files': obj['files']}
        (stage / 'ack.uploading').write_text(json.dumps(ack), encoding='utf-8')
        with patch.object(c, 'ROOT', self.temp), patch.object(c, 'BACKUP_TIMEOUT', 0):
            with self.assertRaises(TimeoutError):
                c.await_backup('initial_ema', obj)
            (stage / 'ack.ready').write_bytes(b'')
            result = c.await_backup('initial_ema', obj)
            self.assertEqual(result['manifest_sha256'], ack['manifest_sha256'])
            (stage / 'metrics.json').write_bytes(b'tampered')
            with self.assertRaises(RuntimeError):
                c.await_backup('initial_ema', obj)

if __name__ == '__main__':
    unittest.main()
