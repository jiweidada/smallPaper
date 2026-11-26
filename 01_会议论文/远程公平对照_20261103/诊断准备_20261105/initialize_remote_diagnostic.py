"""Create only the unique isolated diagnostic directory, without model execution."""
from pathlib import Path
import hashlib
import json
import os

original = Path('/25085400120/mpsma_fair_20261103_v1')
parent = original / 'diagnostics'
root = parent / 'mpsma_diag_base_apdrop_20261105_v1'
expected = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
protocol = original / 'execution_protocol.json'
assert original.is_dir() and not original.is_symlink() and protocol.is_file()
assert hashlib.sha256(protocol.read_bytes()).hexdigest() == expected
if not parent.exists():
    parent.mkdir(exist_ok=False)
assert parent.is_dir() and not parent.is_symlink() and parent.resolve().parent == original.resolve()
assert not root.exists() and not root.is_symlink()
root.mkdir(exist_ok=False)
fd = os.open(str(parent), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
try:
    os.fsync(fd)
finally:
    os.close(fd)
print(json.dumps({'status': 'ISOLATED_DIRECTORY_CREATED', 'root': str(root), 'model_executions': 0}))
