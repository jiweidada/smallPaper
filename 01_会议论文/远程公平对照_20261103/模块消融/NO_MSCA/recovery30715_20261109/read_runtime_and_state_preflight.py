"""Inspect the retained scientific inputs and CPU-loaded164 state; never run a model."""
import os
os.environ['ATEN_CPU_CAPABILITY'] = 'avx2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1')
PREP = ROOT / 'recovery30715_preparation'
PROTOCOL = '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
ANCHOR = '512dfe64c5b99479ffc2235054ee1bc1b2b5378ac776e7bead4d57395b3e545d'


def digest(path):
    result = sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024 ** 2):
            result.update(block)
    return result.hexdigest()


def record(path, relative):
    target = path.resolve(strict=True)
    assert target.is_relative_to(ROOT)
    return {'path': relative, 'sha256': digest(target), 'bytes': target.stat().st_size}


assert sys.executable == '/25085400120/envs/deim/bin/python' and sys.version_info[:2] == (3, 10)
assert digest(ROOT / 'execution_protocol.json') == PROTOCOL
plan = json.loads((ROOT / 'execution_protocol.json').read_text())
reference_path = PREP / 'prefix164_metadata_reference.json'
assert digest(reference_path) == 'b36b5591326e6e9dbfbf24a0262f75dc6ac6560d863fa6375e6a86460cfd4fee'
reference = json.loads(reference_path.read_text())
restoration = json.loads((PREP / 'data_restoration.json').read_text())
assert restoration['status'] == 'ORIGINAL_CACHE_PATH_7021_FROZEN_FILES_RESTORED_AND_VERIFIED'
assert restoration['files'] == 7021 and not restoration['source_config_changed']
assert restoration['cache_root'] == '/tmp/mpsma_module_ablation_20261107_data'
import torch
actual_runtime = {'python': sys.version, 'executable': sys.executable,
    'torch': str(torch.__version__), 'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(), 'torch_build': torch.__config__.show(),
    'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
saved_runtime = json.loads((ROOT / 'environment.current.json').read_text())
runtime_mismatches = {key: {'actual': value, 'expected': saved_runtime[key]}
    for key, value in actual_runtime.items() if value != saved_runtime[key]}
frozen = [record(ROOT / item['path'], item['path']) for item in plan['frozen_files']]
frozen_bad = [item['path'] for item, expected in zip(frozen, plan['frozen_files']) if item['sha256'] != expected['sha256']]
prefix = [record(ROOT / item['path'], item['path']) for item in reference['entries']]
prefix_bad = [item['path'] for item, expected in zip(prefix, reference['entries'])
    if item['sha256'] != expected['sha256'] or item['bytes'] != expected['bytes']]
manifest = json.loads((ROOT / 'backup_gate/epoch0163/manifest.json').read_text())
assert digest(ROOT / 'backup_gate/epoch0163/manifest.json') == '8bf322a46106ffc2113e65ddc4710ef5e280f43b58466be18866ebf2dae9e8a0'
sources = []
for item in manifest['files']:
    entry = record(ROOT / item['source'], item['source'])
    entry.update(name=item['name'], source=item['source'], expected_sha256=item['sha256'], expected_bytes=item['bytes'])
    sources.append(entry)
checkpoint_path = ROOT / 'run/next_epoch.pt'
assert digest(checkpoint_path) == ANCHOR
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
metadata = {name: checkpoint[name] for name in ['resume_format', 'next_epoch', 'actual_loop_epoch',
    'last_epoch', 'optimizer_step_attempts', 'ema_decay_attribute']}
metadata['rng_keys'] = sorted(checkpoint['rng_state'])
metadata['cuda_rng_state_count'] = len(checkpoint['rng_state']['torch_cuda_all'])
metadata['state_sections'] = sorted(k for k in checkpoint if k in ['model', 'ema', 'optimizer',
    'criterion', 'postprocessor', 'lr_warmup_scheduler', 'rng_state'])
assert metadata['next_epoch'] == 164 and metadata['actual_loop_epoch'] == metadata['last_epoch'] == 163
assert metadata['optimizer_step_attempts'] == 265188
assert set(metadata['rng_keys']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda_all',
    'train_loader', 'train_sampler', 'validation_loader'}
processes = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        command = (proc / 'cmdline').read_bytes()
        if str(ROOT).encode() in command and (b'train_runner.py' in command or b'launch_arm.py' in command):
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            processes.append({'pid': int(proc.name), 'state': fields[0], 'start_ticks': int(fields[19]),
                'command': command.replace(b'\0', b' ').decode(errors='replace')})
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
gpu_raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.free,memory.total,driver_version',
    '--format=csv,noheader,nounits'], text=True).strip()
compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
    '--format=csv,noheader,nounits'], text=True).strip()
assert gpu_raw.split(', ')[0] == 'GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a'
assert int(gpu_raw.split(', ')[2]) >= 20000 and not compute
assert not runtime_mismatches and not frozen_bad and not prefix_bad and not processes
result = {'status': 'FROZEN_RUNTIME_AND_PREFIX_VERIFIED_SOURCE_RESTORE_PENDING',
    'observed_server_utc': datetime.now(timezone.utc).isoformat(),
    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'runtime': actual_runtime, 'runtime_mismatches': runtime_mismatches,
    'gpu_raw': gpu_raw, 'compute_processes': compute, 'project_processes': processes,
    'frozen_files_actual': frozen, 'prefix_metadata_actual': prefix,
    'manifest_source_actual': sources, 'rolling_checkpoint_sha256': ANCHOR,
    'rolling_metadata': metadata, 'data_restoration': restoration,
    'source_restore_required': [v['source'] for v in sources
        if v['sha256'] != v['expected_sha256'] or v['bytes'] != v['expected_bytes']],
    'extra_selected_packages': [p.name for p in (ROOT / 'run/selected').iterdir()
        if p.is_dir() and p.name.startswith('epoch') and int(p.name[5:]) > 163],
    'preflight_source_sha256': digest(Path(__file__)),
    'model_forwards': 0, 'optimizer_updates': 0, 'evaluator_runs': 0}
assert result['boot_id'] == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
with (PREP / 'runtime_and_state_preflight.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(result))
