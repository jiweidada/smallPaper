"""Install one hash-bound recovery tree in an absent experiment directory; no training."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

parent = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1').resolve(strict=True)
prep = parent / 'recovery_no_msca_r29_20261108'
target = parent / 'no_msca_20261107_v1'
archive = prep / 'no_msca_round29_recovery_package.zip'
def sha(path):
    h = sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024**2): h.update(block)
    return h.hexdigest()
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
assert target.is_dir() and not any(target.iterdir()), 'Partial target is not the expected empty directory'
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
assert sha(archive) == '7dd93c476068cd8a85a865bac41d01223b51917c9850d9188f8a4fde62504f1b'
with (prep / 'installation_python310.lock').open('x') as handle:
    handle.write(datetime.now(timezone.utc).isoformat() + '\n')
assert (prep / 'installation.lock').is_file(), 'Missing original failed-preparation marker'
with zipfile.ZipFile(archive) as bundle:
    names = set()
    for item in bundle.infolist():
        relative = PurePosixPath(item.filename)
        assert item.filename not in names and not relative.is_absolute() and '..' not in relative.parts
        assert not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16)
        names.add(item.filename)
        path = target.joinpath(*relative.parts)
        assert path.resolve().is_relative_to(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as output, bundle.open(item) as input_stream:
            while block := input_stream.read(4 * 1024**2): output.write(block)
            output.flush()
            os.fsync(output.fileno())
protocol_path = target / 'execution_protocol.json'
assert sha(protocol_path) == '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
protocol = json.loads(protocol_path.read_text())
for item in protocol['frozen_files']:
    assert sha(target / item['path']) == item['sha256'], item['path']
addendum_path = target / 'recovery_addendum.json'
assert sha(addendum_path) == 'e2db4f390756e33e742684e8f1eff526047403ba2db6bad17d5c9e21132e9d54'
recovery = json.loads(addendum_path.read_text())
for item in recovery['recovery_files']:
    assert sha(target / item['path']) == item['sha256'], item['path']
assert sha(target / 'resume_train_runner.py') == recovery['resume_runner_sha256']
anchor = target / 'recovery_anchor/next_epoch.pt'
assert sha(anchor) == recovery['checkpoint_sha256']
os.link(anchor, target / 'run/next_epoch.pt')
assert sha(target / 'run/next_epoch.pt') == recovery['checkpoint_sha256']
manifest = json.loads((target / 'recovery_anchor/original_manifest.json').read_text())
for item in manifest['files']:
    assert sha(target / item['source']) == item['sha256'], item['source']
value = {'status': 'RECOVERY_INSTALLED_NO_TRAINING', 'installed_utc': datetime.now(timezone.utc).isoformat(),
    'target': str(target), 'package_sha256': sha(archive), 'files': len(names),
    'frozen_files_verified': len(protocol['frozen_files']) + 1,
    'rolling_checkpoint_sha256': sha(anchor), 'first_epoch_index': 29,
    'original_gate_metadata_restored': True, 'training_started': False}
with (prep / 'recovery_installation_verified.json').open('x') as handle:
    json.dump(value, handle, indent=2)
    handle.write('\n')
print(json.dumps(value))
