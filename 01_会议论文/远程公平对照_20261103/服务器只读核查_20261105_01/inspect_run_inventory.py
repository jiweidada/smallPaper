"""Inspect only the two known run directories; never execute a model or write remotely."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path('/25085400120/mpsma_fair_20261103_v1')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def inspect_arm(arm):
    folder = ROOT / 'runs' / arm
    result = {'exists': folder.is_dir(), 'files': [], 'epochs': None}
    if not folder.is_dir() or folder.is_symlink():
        return result
    for base, dirs, names in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
        depth = len(Path(base).relative_to(folder).parts)
        if depth >= 2:
            dirs[:] = []
        for name in sorted(names):
            path = Path(base) / name
            if path.is_symlink() or not path.is_file():
                continue
            before = path.stat()
            item = {'path': str(path.relative_to(ROOT)), 'bytes': before.st_size}
            if before.st_size <= 256 * 1024 * 1024:
                item['sha256'] = sha256(path)
                after = path.stat()
                item['unchanged_during_read'] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            result['files'].append(item)
            if len(result['files']) >= 200:
                result['truncated'] = True
                return result
    log = folder / 'epochs.jsonl'
    if log.is_file() and not log.is_symlink():
        count, invalid, last = 0, 0, None
        with log.open('r', encoding='utf-8') as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                count += 1
                last = {'epoch': value.get('epoch'), 'AP': value.get('metrics', {}).get('AP')}
        result['epochs'] = {'valid_lines': count, 'invalid_lines': invalid, 'last': last}
    return result

print(json.dumps({
    'observed_remote_utc': datetime.now(timezone.utc).isoformat(),
    'root_exists': ROOT.is_dir(),
    'runs': {arm: inspect_arm(arm) for arm in ['BASE', 'FULL']},
    'scope': 'Read-only inventory and hashes of known run directories, no model or evaluator execution.'
}, ensure_ascii=False))
