"""Observe the single preparation attempt without re-running it."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1')
prep, target = root / 'recovery_no_msca_r29_20261108', root / 'no_msca_20261107_v1'
out = {'observed_server_utc': datetime.now(timezone.utc).isoformat(), 'target_exists': target.exists(), 'files': {}}
for name in ['execution_protocol.json', 'train_runner.py', 'resume_train_runner.py',
             'recovery_addendum.json', 'run/next_epoch.pt', 'recovery_anchor/next_epoch.pt']:
    path = target / name
    out['files'][name] = {'exists': path.exists(), 'bytes': path.stat().st_size if path.is_file() else None}
report = prep / 'recovery_installation_verified.json'
out['installation_report'] = json.loads(report.read_text()) if report.is_file() else None
out['python_processes'] = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) == os.getpid(): continue
    try:
        argv = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        if argv.startswith('/25085400120/envs/deim/bin/python'):
            out['python_processes'].append({'pid': int(p.name), 'command': argv[:400]})
    except (FileNotFoundError, PermissionError, ProcessLookupError): pass
print(json.dumps(out))
