"""Stage an isolated copy, verify it, and retain a receipt. Never launches training."""
import ast
import base64
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

PAYLOAD = {}  # BUNDLE_INSERTION
storage = pathlib.Path('/25085400120' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
code = phase/'code'
stage = phase/'artifacts/round2_20261014_v1'
stage.mkdir(exist_ok=False)
receipt = {'phase_root': str(phase), 'stage': str(stage), 'started': time.time(), 'status': 'staging', 'training_launched': False}
def save():
    (stage/'staging_receipt.json').write_text(json.dumps(receipt, indent=2))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

try:
    originals = [code/'engine/backbone/hgnetv2_light.py', code/'engine/backbone/hgnetv2_light_control.py',
                 code/'engine/backbone/hgnetv2_light_phase.py', code/'tests/engine/test_p2_phase_controls.py',
                 code/'engine/solver/det_solver.py', code/'engine/solver/_solver.py']
    receipt['original_sha256'] = {str(p.relative_to(code)): digest(p) for p in originals}
    # Preserve existing diagnostic classes byte-for-byte at the AST level.
    old_tree = ast.parse((code/'engine/backbone/hgnetv2_light_phase.py').read_text())
    new_tree = ast.parse(base64.b64decode(PAYLOAD['hgnetv2_light_phase.py']['base64']))
    for name in ['P2ToP3PhaseAblation', 'P2ToP3SinglePhaseResidual', '_pad_even']:
        old = next(n for n in old_tree.body if getattr(n, 'name', None) == name)
        new = next(n for n in new_tree.body if getattr(n, 'name', None) == name)
        assert ast.dump(old) == ast.dump(new), 'Protected diagnostic changed: ' + name
    staged_code = stage/'code'
    shutil.copytree(code, staged_code, symlinks=True,
                    ignore=shutil.ignore_patterns('outputs', 'logs', 'jobs', '__pycache__', '.git'))
    for name, row in PAYLOAD.items():
        data = base64.b64decode(row['base64'])
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        if name == 'hgnetv2_light_phase.py':
            target = staged_code/'engine/backbone'/name
        elif name.startswith('test_') or name.startswith('lowrank_branch_ref'):
            target = staged_code/'tests/engine'/name
        else:
            target = stage/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    configs = staged_code/'configs/experiments/phase'
    template = (configs/'singlephase_full96.yml').read_text()
    receipt['new_configs'] = {}
    for mode in ['c_mean', 'rank16', 'rank16_mean']:
        filename = mode + '_full96.yml'
        if (code/'configs/experiments/phase'/filename).exists():
            raise RuntimeError('Existing new-arm configuration: ' + filename)
        text = template.replace('singlephase_full96', mode+'_full96').replace('p2_skip_mode: single_phase', 'p2_skip_mode: '+mode)
        text = '\n'.join(line for line in text.splitlines() if 'p2_expanded:' not in line) + '\n'
        with (configs/filename).open('x') as f:
            f.write(text)
        receipt['new_configs'][filename] = digest(configs/filename)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    receipt['cpu_tests'] = []
    save()
    for name in ['test_p2_phase_controls.py', 'test_p2_round2_acceptance.py']:
        with (stage/(name+'.log')).open('xb') as log:
            completed = subprocess.run([sys.executable, '-B', str(staged_code/'tests/engine'/name)],
                                       cwd=staged_code, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        receipt['cpu_tests'].append({'file': name, 'returncode': completed.returncode})
        save()
        assert completed.returncode == 0, 'CPU tests failed: ' + name
    receipt['status'] = 'cpu_verified_pending_gpu_and_activation'
    assert all(digest(p) == receipt['original_sha256'][str(p.relative_to(code))] for p in originals)
    receipt['original_files_unchanged'] = True
    receipt['ended'] = time.time()
    save()
    print(json.dumps(receipt, indent=2))
except BaseException as exc:
    receipt.update(status='failed', error=type(exc).__name__+': '+str(exc), ended=time.time())
    save()
    print(json.dumps(receipt, indent=2))
    raise
