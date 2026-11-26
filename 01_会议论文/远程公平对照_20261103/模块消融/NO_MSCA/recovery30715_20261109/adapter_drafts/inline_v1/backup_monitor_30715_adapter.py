"""Allow the newly authorized SSH port; preserve every native backup operation."""
import ast
import copy
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backup_monitor.py')
EXPECTED = '4e5c38bb7dcf4dbd8131157b34f2637de7e45379796b75c9e211a63a9f25b37b'
payload = SOURCE.read_bytes()
assert sha256(payload).hexdigest() == EXPECTED
original = ast.parse(payload.decode('utf-8'), filename=str(SOURCE))
adapted = copy.deepcopy(original)
matches = []
for node in ast.walk(adapted):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument' and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--port'):
        choices = next(item for item in node.keywords if item.arg == 'choices')
        assert ast.literal_eval(choices.value) == [30219, 30848]
        choices.value.elts.append(ast.Constant(value=30715))
        matches.append(choices)
assert len(matches) == 1
roundtrip = copy.deepcopy(adapted)
for node in ast.walk(roundtrip):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument' and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--port'):
        choices = next(item for item in node.keywords if item.arg == 'choices')
        assert ast.literal_eval(choices.value) == [30219, 30848, 30715]
        choices.value.elts.pop()
assert ast.dump(roundtrip, include_attributes=False) == ast.dump(original, include_attributes=False)
ast.fix_missing_locations(adapted)
if '--verify-adapter-only' in sys.argv:
    assert len(sys.argv) == 2
    report = {'status': 'NATIVE_BACKUP_AST_IDENTICAL_EXCEPT_AUTHORIZED_SSH_PORT_ALLOWLIST',
        'original_source': str(SOURCE), 'original_source_sha256': EXPECTED,
        'adapter_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'only_change': {'argument': '--port', 'before': [30219, 30848], 'after': [30219, 30848, 30715]},
        'entire_AST_equal_after_reversing_port_append': True,
        'native_backup_fetch_commit_verify_ACK_functions_changed': False,
        'original_file_changed': False, 'native___file___preserved': str(SOURCE),
        'training_actions': 0, 'remote_connections': 0}
    with (HERE / 'backup_port_adapter_verified.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report))
else:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE.parent))
    namespace = {'__name__': '__main__', '__file__': str(SOURCE), '__package__': None}
    exec(compile(adapted, str(SOURCE), 'exec'), namespace)
