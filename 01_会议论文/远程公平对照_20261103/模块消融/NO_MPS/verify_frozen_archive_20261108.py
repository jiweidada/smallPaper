"""Validate and fill missing frozen artifacts, preserving existing local files."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
ROOT = Path(__file__).resolve().parent
protocol = ROOT / 'execution_protocol.json'
protocol_sha = sha256(protocol.read_bytes()).hexdigest()
wrapper = json.loads((ROOT / 'frozen_archive_remote_20261108.json').read_text(encoding='utf-8'))
assert wrapper['exit_code'] == 0 and wrapper['stderr'] == ''
remote = json.loads(wrapper['stdout'])
assert remote['status'] == 'ALL_FROZEN_INPUTS_ARCHIVED_AND_VERIFIED'
assert remote['protocol_sha256'] == protocol_sha
path = ROOT / 'frozen_inputs_20261108.zip'
assert path.stat().st_size == remote['bytes']
assert sha256(path.read_bytes()).hexdigest() == remote['archive_sha256']
plan = json.loads(protocol.read_text(encoding='utf-8'))
expected = {item['path']: item['sha256'] for item in plan['frozen_files']}
expected['execution_protocol.json'] = protocol_sha
new_files, reused = [], []
with ZipFile(path) as archive:
    assert len(archive.namelist()) == len(set(archive.namelist())) == remote['files'] == len(expected)
    assert set(archive.namelist()) == set(expected) and archive.testzip() is None
    for name, value in sorted(expected.items()):
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        target = ROOT / name
        assert target.resolve().is_relative_to(ROOT.resolve()) and not target.is_symlink()
        assert sha256(archive.read(name)).hexdigest() == value
        if target.exists():
            assert target.is_file() and sha256(target.read_bytes()).hexdigest() == value, name
            reused.append(name)
        else:
            new_files.append(name)
    for name in new_files:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(archive.read(name))
for name, value in expected.items():
    assert sha256((ROOT / name).read_bytes()).hexdigest() == value
report = {'status': 'ALL_FROZEN_INPUTS_LOCALLY_ARCHIVED_AND_VERIFIED',
          'observed_local_utc': datetime.now(timezone.utc).isoformat(),
          'protocol_sha256': protocol_sha, 'archive_sha256': remote['archive_sha256'],
          'archive_bytes': path.stat().st_size, 'files': len(expected),
          'added_missing_files': new_files, 'already_present_and_identical': reused,
          'existing_files_overwritten': False, 'model_execution': False,
          'remote_receipt_sha256': sha256((ROOT / 'frozen_archive_remote_20261108.json').read_bytes()).hexdigest()}
with (ROOT / 'frozen_inputs_verified_20261108.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'files': len(expected),
                  'missing_files_added': len(new_files), 'identical_existing_files': len(reused)}))
