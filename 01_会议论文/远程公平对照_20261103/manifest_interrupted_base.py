"""Hash the stopped BASE run and small provenance files, read-only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path('/25085400120/mpsma_fair_20261103_v1')
EXPECTED = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def live_same(pid, ticks):
    try:
        p = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return p[19] == ticks and p[0] not in ['Z', 'X']
    except (FileNotFoundError, ProcessLookupError):
        return False

assert sha(ROOT / 'execution_protocol.json') == EXPECTED
assert not live_same(1193, '39931198') and not live_same(1194, '39931668')
paths = []
run = ROOT / 'runs/BASE'
assert run.is_dir()
for base, dirs, files in os.walk(run):
    dirs.sort()
    for name in sorted(files):
        p = Path(base) / name
        if p.is_file() and not p.is_symlink():
            paths.append(p)
for name in ['execution_protocol.json', 'queue.json', 'BASE.train.log', 'FULL.train.log',
             'training_supervisor.log', 'one_shot.lock', 'primary_full_reservation.json',
             'parallel_execution_addendum.json', 'environment.current.json',
             'execution_configs/BASE.yml', 'execution_configs/FULL.yml']:
    p = ROOT / name
    if p.is_file() and not p.is_symlink():
        paths.append(p)
for p in sorted((ROOT / 'preflight').glob('BASE*')):
    if p.is_file() and not p.is_symlink():
        paths.append(p)
assert len(paths) == len(set(paths))
files = []
for p in paths:
    before = p.stat()
    digest = sha(p)
    after = p.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), str(p)
    files.append({'relative_path': str(p.relative_to(ROOT)), 'bytes': after.st_size,
                  'mtime_ns': after.st_mtime_ns, 'sha256': digest})
assert not live_same(1193, '39931198') and not live_same(1194, '39931668')
print(json.dumps({'status': 'INTERRUPTED_BASE_FILES_HASHED_READ_ONLY',
                  'observed_remote_utc': datetime.now(timezone.utc).isoformat(),
                  'protocol_sha256': EXPECTED, 'files': files,
                  'total_bytes': sum(f['bytes'] for f in files),
                  'old_supervisor_live': False, 'old_train_process_live': False,
                  'model_executions': 0, 'evaluator_executions': 0}))
