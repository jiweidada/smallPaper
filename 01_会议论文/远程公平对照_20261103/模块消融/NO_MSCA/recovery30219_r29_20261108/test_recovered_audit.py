"""Focused rejection checks for the two-comparison recovery audit adapter."""

import ast
from pathlib import Path
import tempfile
import unittest

import audit_recovered_ablation_lineage as audit


class RecoveryAuditChecks(unittest.TestCase):
    def test_scientific_body_discovery_in_nested_try(self):
        result = audit.verify_runner_ast(audit.ARM_ROOT)
        self.assertTrue(result['entire_epoch_loop_identical'])
        self.assertEqual(result['nested_functions_identical'], ['state', 'load_state', 'restore', 'commit'])

    def test_only_two_exact_old_final_checks_change(self):
        tree, changes = audit.patched_auditor_tree()
        self.assertEqual(len(changes), 2)
        self.assertEqual({entry['before'] for entry in changes}, {
            "receipt['first_epoch_in_this_segment'] == 0",
            "receipt['runner_sha256'] == frozen['train_runner.py']",
        })
        compile(tree, 'exact_two_comparison_adapter', 'exec')
        self.assertEqual(audit.sha(audit.MODULE_ROOT / 'audit_ablation_lineage.py'), audit.OLD_AUDITOR_SHA)

    def test_modified_addendum_copy_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='audit_tamper_', dir=audit.HERE) as temporary:
            root = Path(temporary).resolve(strict=True)
            self.assertEqual(root.parent, audit.HERE)
            self.assertTrue(root.name.startswith('audit_tamper_'))
            copied = root / 'recovery_addendum.json'
            original = (audit.HERE / 'recovery_addendum.json').read_bytes()
            copied.write_bytes(original)
            self.assertEqual(audit.checked_addendum(copied)['checkpoint_sha256'], audit.ANCHOR_SHA)
            copied.write_bytes(original + b'\n')
            with self.assertRaisesRegex(RuntimeError, 'addendum'):
                audit.checked_addendum(copied)
            self.assertEqual(audit.sha(audit.HERE / 'recovery_addendum.json'), audit.ADDENDUM_SHA)


if __name__ == '__main__':
    unittest.main(verbosity=2)
