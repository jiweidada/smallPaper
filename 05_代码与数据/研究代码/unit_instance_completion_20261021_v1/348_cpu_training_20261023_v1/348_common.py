"""348 execution ledger; all training remains in sealed train347."""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
S347 = WORK / '347_codex_cpu_entry_20261023_v1'
S345 = WORK / '345_codex_contracts_20261022_v1'
S5 = WORK / '342_codex_stage5_review_20261022_v1'
PLAN = HERE / '348_执行绑定.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with open(path, 'x', encoding='utf-8', newline='\n') as f:
        f.write(data)


def text(path, value):
    with open(path, 'x', encoding='utf-8', newline='\n') as f:
        f.write(value)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_record(path, expected):
    check(sha(path) == expected, 'SHA mismatch: ' + str(path))


def plan():
    anchor = read(HERE / '348_执行绑定SHA.json')
    verify_record(PLAN, anchor['sha256'])
    document = read(PLAN)
    for path, expected in document['sources'].items():
        verify_record(path, expected)
    for record in document['frozen_records']:
        verify_record(record['path'], record['sha256'])
    return document


def verify_seal(path):
    document = read(path)
    for name, record in document['artifacts'].items():
        check(Path(name).stat().st_size == record['bytes'], 'sealed length: ' + name)
        verify_record(name, record['sha256'])
    return {'path': str(path), 'sha256': sha(path), 'verified_files': len(document['artifacts'])}


def seal_tree(root, output_name):
    root = Path(root)
    output = root / output_name
    check(not output.exists(), 'seal already exists: ' + str(output))
    records = {}
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix != '.lock':
            records[str(p)] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    write(output, {'status': '已封存', 'created_utc': now(),
                   'excluded': 'Process-lifetime .lock files are synchronization primitives, not results.',
                   'artifacts': records})
    return {'path': str(output), 'sha256': sha(output), 'verified_files': len(records)}


@contextlib.contextmanager
def lock(path):
    """Windows process-lifetime lock; existence of this file is not a lock."""
    import msvcrt
    path = Path(path)
    if not path.exists():
        with open(path, 'xb') as f:
            f.write(b'0')
    with open(path, 'r+b') as f:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def cpu_env():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MKL_THREADING_LAYER='SEQUENTIAL', PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    return env
