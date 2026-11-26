"""Inventory saved experiment artifacts and likely mount roots, without writing remotely."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path('/25085400120/mpsma_fair_20261103_v1')
out = {'observed_remote_utc': datetime.now(timezone.utc).isoformat(), 'root_exists': ROOT.exists(),
       'targets': {}, 'likely_mount_roots': {}, 'candidate_experiment_paths': [],
       'scope': 'Read-only inventory. No model, evaluator, training, deletion, or server configuration changes.'}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def tail(path, count=8000):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - count))
            return f.read().decode('utf-8', errors='replace')
    except OSError as error:
        return repr(error)

if ROOT.exists():
    for arm in ['BASE', 'FULL']:
        folder = ROOT / 'runs' / arm
        files = []
        if folder.exists():
            for base, dirs, names in os.walk(folder):
                dirs.sort()
                for name in sorted(names):
                    path = Path(base) / name
                    if not path.is_file() or path.is_symlink():
                        continue
                    stat = path.stat()
                    item = {'path': str(path.relative_to(ROOT)), 'bytes': stat.st_size,
                            'mtime_ns': stat.st_mtime_ns}
                    if stat.st_size <= 32 * 1024 * 1024:
                        item['sha256'] = digest(path)
                    files.append(item)
        obj = {'exists': folder.exists(), 'files': files, 'total_bytes': sum(x['bytes'] for x in files)}
        for name in ['receipt.json', 'epochs.jsonl']:
            p = folder / name
            if p.exists():
                obj[name + '_tail'] = tail(p, 12000 if name == 'epochs.jsonl' else 4000)
        trainlog = ROOT / (arm + '.train.log')
        if trainlog.exists():
            obj['train_log'] = {'bytes': trainlog.stat().st_size, 'tail': tail(trainlog)}
        out['targets'][arm] = obj
    for name in ['queue.json', 'secondary_queue.json', 'training_supervisor.log', 'secondary_supervisor.log']:
        p = ROOT / name
        if p.exists():
            out['targets'][name] = {'bytes': p.stat().st_size, 'sha256': digest(p), 'tail': tail(p)}

for p in [Path('/'), Path('/workspace'), Path('/mnt'), Path('/data'), Path('/root'), Path('/tmp')]:
    try:
        if p.is_dir():
            entries = sorted(x.name for x in p.iterdir())
            out['likely_mount_roots'][str(p)] = entries[:120]
            for name in entries:
                child = p / name
                if 'mpsma' in name.lower() or name == '25085400120':
                    out['candidate_experiment_paths'].append(str(child))
                if name == '25085400120' and (child / 'mpsma_fair_20261103_v1').exists():
                    out['candidate_experiment_paths'].append(str(child / 'mpsma_fair_20261103_v1'))
    except OSError as error:
        out['likely_mount_roots'][str(p)] = {'error': repr(error)}
print(json.dumps(out, ensure_ascii=False))
