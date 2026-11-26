"""Archive old segment bytes before restoring164; never start or stop training."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1')
PREP = ROOT / 'recovery30715_preparation'
RELATIVE_ARCHIVE = Path('segment_history/segment30219_until164')
ARCHIVE = ROOT / RELATIVE_ARCHIVE
STANDARD = ['launch.lock', 'supervisor.json', 'supervisor.log', 'training.log']


def digest(path):
    result = sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024 ** 2):
            result.update(block)
    return result.hexdigest()


def workers():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / 'cmdline').read_bytes()
            if str(ROOT).encode() in command and (b'train_runner.py' in command or b'launch_arm.py' in command):
                fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                found.append({'pid': int(proc.name), 'start_ticks': int(fields[19]),
                    'command': command.replace(b'\0', b' ').decode(errors='replace')})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


assert digest(ROOT / 'execution_protocol.json') == '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
assert digest(ROOT / 'run/next_epoch.pt') == '512dfe64c5b99479ffc2235054ee1bc1b2b5378ac776e7bead4d57395b3e545d'
preflight_path = PREP / 'runtime_and_state_preflight.json'
assert preflight_path.is_file(), 'The original preflight must finish before any archival mutation'
assert json.loads(preflight_path.read_text())['status'] == 'FROZEN_RUNTIME_AND_PREFIX_VERIFIED_SOURCE_RESTORE_PENDING'
preflight_sha256 = digest(preflight_path)
assert not ARCHIVE.exists()
before = workers()
assert not before
gpu_before = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
assert not gpu_before
manifest = json.loads((ROOT / 'backup_gate/epoch0163/manifest.json').read_text())
sources = {item['source'] for item in manifest['files']} | set(STANDARD)
extra_dirs = [path for path in (ROOT / 'run/selected').iterdir()
    if path.is_dir() and path.name.startswith('epoch') and int(path.name[5:]) > 163]
assert all(any(p.is_file() for p in directory.rglob('*')) for directory in extra_dirs), 'Empty unaccepted package needs explicit preservation review'
stale_temporaries = [path for folder in [ROOT, ROOT / 'run'] for path in folder.glob('*.partial') if path.is_file()]
for directory in extra_dirs:
    sources.update(str(path.relative_to(ROOT)) for path in directory.rglob('*') if path.is_file())
sources.update(str(path.relative_to(ROOT)) for path in stale_temporaries)
records = []
for relative in sorted(sources):
    source = (ROOT / relative).resolve(strict=True)
    assert source.is_relative_to(ROOT) and source.is_file() and not (ROOT / relative).is_symlink()
    moved = relative in STANDARD or source in stale_temporaries or any(source.is_relative_to(path) for path in extra_dirs)
    records.append({'original_path': relative,
        'archive_path': str(RELATIVE_ARCHIVE / relative), 'archive_member': relative,
        'sha256': digest(source), 'bytes': source.stat().st_size, 'moved': moved})
ARCHIVE.mkdir(parents=True, exist_ok=False)
for directory in extra_dirs:
    target = ARCHIVE / directory.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.rename(directory, target)
for item in records:
    original, target = ROOT / item['original_path'], ROOT / item['archive_path']
    if target.exists():
        assert any(target.is_relative_to(ARCHIVE / p.relative_to(ROOT)) for p in extra_dirs)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if item['moved']:
            os.rename(original, target)
        else:
            shutil.copy2(original, target)
    assert target.stat().st_size == item['bytes'] and digest(target) == item['sha256']
after = workers()
assert not after and all(not (ROOT / name).exists() for name in STANDARD)
archive_zip = PREP / 'prior_segment_archive.zip'
with zipfile.ZipFile(archive_zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
    for item in records:
        bundle.write(ROOT / item['archive_path'], arcname=item['archive_member'])
result = {'status': 'PRIOR_SEGMENT_ARCHIVED_BEFORE_ANY_CURSOR_RESTORATION',
    'observed_utc': datetime.now(timezone.utc).isoformat(),
    'archive_root': str(ARCHIVE), 'archive_root_relative': str(RELATIVE_ARCHIVE),
    'files': records, 'archive_zip_sha256': digest(archive_zip),
    'archive_zip_bytes': archive_zip.stat().st_size,
    'project_processes_before': before, 'project_processes_after': after,
    'gpu_compute_before': gpu_before, 'old_standard_names_moved': STANDARD,
    'preflight_sha256': preflight_sha256,
    'archive_source_sha256': digest(Path(__file__)),
    'old_receipt_scope': 'Stale round165 partial trajectory after164 accepted; not an accepted165 round',
    'training_actions': 0}
with (PREP / 'prior_segment_archive_manifest.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(result))
