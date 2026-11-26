"""Compare server run-file hashes with the existing local BASE backup."""
from pathlib import Path
import hashlib
import json

here = Path(__file__).resolve().parent
root = here.parent
wrapper = json.loads((here / 'primary_run_inventory.json').read_text(encoding='utf-8'))
if wrapper['exit_code'] != 0:
    raise RuntimeError('Server inventory failed')
inventory = json.loads(wrapper['stdout'])
backup = root / 'remote_evidence' / 'interrupted_BASE_20261105'
results = []
for item in inventory['runs']['BASE']['files']:
    local = backup / Path(item['path'])
    entry = {'path': item['path'], 'local_exists': local.is_file()}
    if local.is_file() and item.get('sha256'):
        digest = hashlib.sha256()
        with local.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        entry.update({'local_bytes': local.stat().st_size,
                      'matches_server': local.stat().st_size == item['bytes'] and digest.hexdigest() == item['sha256'],
                      'server_unchanged_during_read': item.get('unchanged_during_read')})
    results.append(entry)
report = {'server_files': len(results),
          'matching_local_files': sum(x.get('matches_server') is True for x in results),
          'files': results,
          'scope': 'Byte size and SHA-256 comparison; no remote writes.'}
(here / 'base_backup_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
