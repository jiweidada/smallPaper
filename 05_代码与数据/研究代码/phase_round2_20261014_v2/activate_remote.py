"""GPU preflight then backed-up activation. Refuses to alter a busy code tree."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

storage = pathlib.Path('/25085400120' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
stage = phase/'artifacts/round2_20261014_v1'
code = phase/'code'
receipt = json.loads((stage/'staging_receipt.json').read_text())
assert receipt['status'] == 'cpu_verified_pending_gpu_and_activation'
target_receipt = stage/'activation_receipt.json'
assert not target_receipt.exists(), 'Activation has already been attempted; inspect its receipt'
def busy():
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, capture_output=True)
    if query.returncode:
        raise RuntimeError('Could not inspect GPU compute processes')
    pids = [line.strip() for line in query.stdout.splitlines() if line.strip().isdigit()]
    trainers = []
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc/'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
            if 'train.py ' in cmd and proc.joinpath('cwd').resolve() == code.resolve():
                trainers.append(int(proc.name))
        except (OSError, PermissionError):
            continue
    return {'gpu_compute_pids': pids, 'trainers_using_target_code': trainers}

status = busy()
if any(status.values()):
    print(json.dumps({'status': 'deferred_existing_training_active', 'phase_root': str(phase), **status}))
    raise SystemExit(3)
activation = {'started': time.time(), 'phase_root': str(phase), 'status': 'gpu_preflight', 'training_launched': False}
def save():
    target_receipt.write_text(json.dumps(activation, indent=2))

try:
    save()
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    with (stage/'gpu_preflight.log').open('xb') as log:
        completed = subprocess.run([sys.executable, '-B', str(stage/'preflight_gpu.py')],
                                   cwd=stage/'code', env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    activation['gpu_preflight_returncode'] = completed.returncode
    if completed.returncode != 0:
        raise RuntimeError('Real model GPU preflight failed; original code has not been replaced')
    assert not any(busy().values()), 'GPU or target code became busy before activation'
    original_hashes = receipt['original_sha256']
    for relative, expected in original_hashes.items():
        assert hashlib.sha256((code/relative).read_bytes()).hexdigest() == expected, 'Original changed: ' + relative
    replacements = ['engine/backbone/hgnetv2_light_phase.py', 'tests/engine/test_p2_phase_controls.py']
    additions = ['tests/engine/test_p2_round2_acceptance.py', 'tests/engine/lowrank_branch_ref_20261011_v3.py']
    additions += ['configs/experiments/phase/' + name for name in receipt['new_configs']]
    for relative in replacements:
        assert not (code/(relative+'.bak_round2_20261014_v1')).exists()
    for relative in additions:
        assert not (code/relative).exists(), 'Refusing to overwrite new file: ' + relative
    queue_target = phase/'queue_round2_20261014_v1.sh'
    assert not queue_target.exists()
    subprocess.run(['bash', '-n', str(stage/'queue_round2.sh')], check=True)
    activation['backups'] = {}
    for relative in replacements:
        target = code/relative
        backup = code/(relative+'.bak_round2_20261014_v1')
        with backup.open('xb') as handle:
            handle.write(target.read_bytes())
        activation['backups'][relative] = {'path': str(backup), 'sha256': hashlib.sha256(backup.read_bytes()).hexdigest()}
    for relative in additions:
        with (code/relative).open('xb') as handle:
            handle.write((stage/'code'/relative).read_bytes())
    with queue_target.open('xb') as handle:
        handle.write((stage/'queue_round2.sh').read_bytes())
    for relative in replacements:
        target = code/relative
        temporary = code/(relative+'.round2_new')
        with temporary.open('xb') as handle:
            handle.write((stage/'code'/relative).read_bytes())
        os.replace(temporary, target)
    activation['deployed_sha256'] = {relative: hashlib.sha256((code/relative).read_bytes()).hexdigest()
                                    for relative in replacements + additions}
    for relative in ['engine/backbone/hgnetv2_light.py', 'engine/backbone/hgnetv2_light_control.py',
                     'engine/solver/det_solver.py', 'engine/solver/_solver.py']:
        assert hashlib.sha256((code/relative).read_bytes()).hexdigest() == original_hashes[relative]
    activation.update(status='activated_gpu_verified', ended=time.time(), protected_sources_unchanged=True)
    save()
    print(json.dumps(activation, indent=2))
except BaseException as exc:
    activation.update(status='failed', error=type(exc).__name__ + ': ' + str(exc), ended=time.time())
    save()
    print(json.dumps(activation, indent=2))
    raise
