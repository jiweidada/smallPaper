"""Archive immutable input artifacts on the known NO_MPS host, without model use."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_mps_20261107_v1')
EXPECTED_PROTOCOL = '62c8a9012df8695b400c55b21fac6210d320492f911d551a34eb9015bd4571ad'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
processes = {}
for pid, ticks in [(621, 311184), (622, 311393)]:
    actual = int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19])
    assert actual == ticks
    processes[str(pid)] = actual
protocol_path = ROOT / 'execution_protocol.json'
protocol_bytes = protocol_path.read_bytes()
assert sha256(protocol_bytes).hexdigest() == EXPECTED_PROTOCOL
plan = json.loads(protocol_bytes)
assert plan['arm'] == 'NO_MPS' and plan['authorized'] is True
assert shutil.disk_usage(ROOT).free > 2 * 1024**3
target = ROOT / 'no_mps_frozen_inputs_20261108.zip'
partial = target.with_suffix('.zip.partial')
assert not target.exists() and not partial.exists()
expected_files = {item['path']: item['sha256'] for item in plan['frozen_files']}
expected_files['execution_protocol.json'] = EXPECTED_PROTOCOL
assert len(expected_files) == len(plan['frozen_files']) + 1
with ZipFile(partial, 'x', compression=ZIP_DEFLATED, compresslevel=1) as archive:
    for name, expected in sorted(expected_files.items()):
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path = ROOT / name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve())
        assert sha256(path.read_bytes()).hexdigest() == expected, name
        archive.write(path, name)
with ZipFile(partial) as archive:
    assert archive.testzip() is None and set(archive.namelist()) == set(expected_files)
    for name, expected in expected_files.items():
        assert sha256(archive.read(name)).hexdigest() == expected, name
os.replace(partial, target)
print(json.dumps({'status': 'ALL_FROZEN_INPUTS_ARCHIVED_AND_VERIFIED',
                  'observed_server_utc': datetime.now(timezone.utc).isoformat(),
                  'protocol_sha256': EXPECTED_PROTOCOL, 'archive': str(target),
                  'archive_sha256': sha256(target.read_bytes()).hexdigest(),
                  'bytes': target.stat().st_size, 'files': len(expected_files),
                  'process_start_ticks': processes, 'model_execution': False}))
