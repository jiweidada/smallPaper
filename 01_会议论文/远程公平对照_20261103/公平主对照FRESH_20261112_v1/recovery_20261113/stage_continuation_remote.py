"""Build a separate continuation directory without starting a worker."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
original = Path(plan['original_root'])
target = Path(plan['continuation_root'])
pending = target.parent / (target.name + '.staging')
assert original.is_dir() and target.parent.parent == original.parent
assert target.name == original.name == plan['arm'] and not target.exists()
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == plan['boot_id']
sys.path.insert(0, str(original))
from backup_protocol import EpochGate, accepted_bytes, ack_bytes, file_sha, parse_manifest, sync_dir

assert not (pending / 'stage_receipt.json').exists()
for name, expected in plan['original_failure_records'].items():
    assert file_sha(original / name) == expected, name
supervisor = json.loads((original / 'supervisor.json').read_text())
assert not any(Path('/proc', str(supervisor[k])).exists() for k in ('supervisor_pid', 'pid'))
protocol = original / 'execution_protocol.json'
assert file_sha(protocol) == plan['original_protocol_sha256']
protocol_data = json.loads(protocol.read_text())
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                               '--format=csv,noheader,nounits'], text=True).strip().split(', ')
assert len(gpu) == 2 and gpu[0] == protocol_data['gpu_uuid'] and int(gpu[1]) >= 20000
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).strip()
gate = EpochGate(original / 'run')
gate.require_probe()
cursor = plan['accepted_rounds']
assert not gate.stage(cursor).exists()
paths = [gate.stage(-1)] + [gate.stage(epoch) for epoch in range(cursor)]
anchor = paths[-1]
data = (anchor / 'manifest.json').read_bytes()
manifest = parse_manifest(data, original.name, cursor - 1)
assert file_sha(anchor / 'manifest.json') == plan['checkpoint_manifest_sha256']
assert manifest['rolling_sha256'] == plan['checkpoint_sha256'] and manifest['evaluated_state_is_next_state']


def copy_checked(source, destination, expected):
    assert source.is_file() and not source.is_symlink(), source
    destination.parent.mkdir(exist_ok=True, parents=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    assert destination.is_file() and not destination.is_symlink() and file_sha(destination) == expected, destination


for name, expected in plan['continuation_code'].items():
    source = pending / name
    assert file_sha(source) == expected
    compile(source.read_bytes(), str(source), 'exec')
for item in protocol_data['frozen_files']:
    copy_checked(original / item['path'], pending / item['path'], item['sha256'])
copy_checked(protocol, pending / protocol.name, plan['original_protocol_sha256'])
for epoch, folder in enumerate(paths, start=-1):
    payload = (folder / 'manifest.json').read_bytes()
    stage_manifest = parse_manifest(payload, original.name, epoch)
    expected_ack = ack_bytes(payload, stage_manifest)
    expected_acceptance = accepted_bytes(payload, stage_manifest)
    assert (folder / 'ack.json').read_bytes() == expected_ack
    assert (folder / 'accepted.json').read_bytes() == expected_acceptance
    for name in ('manifest.json', 'ack.json', 'accepted.json'):
        copy_checked(folder / name, pending / 'backup_gate' / folder.name / name, file_sha(folder / name))
    if epoch >= 0:
        metric = next(i for i in stage_manifest['files'] if i['name'] == 'epoch.metrics.json')
        copy_checked(folder / metric['name'], pending / metric['source'], metric['sha256'])
    else:
        for item in stage_manifest['files']:
            copy_checked(original / item['source'], pending / item['source'], item['sha256'])

run_sources = set()
for item in manifest['files']:
    source = item['source']
    if not source.startswith('run/') or source == 'run/receipt.json' or source in run_sources:
        continue
    copy_checked(original / source, pending / source, item['sha256'])
    run_sources.add(source)
assert not (pending / 'run/receipt.json').exists()
assert not (pending / 'launch.lock').exists()
report = {'status': 'CONTINUATION_STAGED_NO_WORKER', 'arm': plan['arm'], 'root': str(target),
          'recorded_remote_utc': datetime.now(timezone.utc).isoformat(),
          'plan_sha256': file_sha(pending / 'continuation_plan.json'),
          'frozen_file_count': len(protocol_data['frozen_files']), 'original_source_preserved': True,
          'accepted_backup_gate_prefix': cursor, 'restored_run_files': sorted(run_sources),
          'checkpoint_sha256': manifest['rolling_sha256'], 'model_forwards': 0,
          'evaluator_runs': 0, 'optimizer_updates': 0}
with (pending / 'stage_receipt.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
pending.rename(target)
sync_dir(target.parent)
print(json.dumps(report), flush=True)
