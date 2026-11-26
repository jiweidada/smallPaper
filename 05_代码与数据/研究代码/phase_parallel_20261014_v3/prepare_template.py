"""Append a fully verified copy; never touch the code used by a live trainer."""
import base64
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ASSETS = {}  # INSERT_ASSETS
account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A' else '/24085403014/experiments/phase_mechanism_20261012')
acceptance = phase/'artifacts/round2_20261014_v1'
stage = acceptance/'execution_v3'
stage.mkdir(exist_ok=False)
env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
record = {'account': account, 'started': time.time(), 'status': 'preparing_isolated_code', 'new_training_started': False}
receipt = stage/'preparation_receipt.json'

def save():
    receipt.write_text(json.dumps(record, indent=2))

def run(command, log, cwd):
    with (stage/log).open('xb') as handle:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(log+' failed, returncode='+str(result.returncode))

try:
    save()
    plan = json.loads((acceptance/'addons_v2/queue_plan.json').read_text())
    with (stage/'queue_plan.json').open('x') as handle:
        json.dump(plan, handle, indent=2)
    original = phase/'code'
    record['protected_live_sources'] = {name: hashlib.sha256((original/name).read_bytes()).hexdigest() for name in [
        'engine/backbone/hgnetv2_light.py', 'engine/backbone/hgnetv2_light_phase.py',
        'engine/backbone/hgnetv2_light_control.py', 'engine/solver/det_solver.py', 'engine/solver/_solver.py']}
    code = stage/'code'
    shutil.copytree(original, code, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '.git', 'experiment_records'))
    tools = stage/'tools'
    shutil.copytree(acceptance/'addons_v2', tools, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', 'prelaunch_four_images'))
    # Changes only in the new copy. Canonical assets are the already accepted round-two sources.
    for relative, item in ASSETS.items():
        data = base64.b64decode(item['base64'])
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        target = stage/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            (target.with_name(target.name+'.copied_original')).write_bytes(target.read_bytes())
        target.write_bytes(data)
    for name in ['c_mean_full96.yml', 'rank16_full96.yml', 'rank16_mean_full96.yml']:
        source = acceptance/'code/configs/experiments/phase'/name
        target = code/'configs/experiments/phase'/name
        if target.exists():
            assert target.read_bytes() == source.read_bytes(), 'Config changed: '+name
        else:
            with target.open('xb') as handle:
                handle.write(source.read_bytes())
    # Cost measurement must reference the unchanged initialization even from this deeper copy.
    benchmark = tools/'benchmark_soep_lite.py'
    source = benchmark.read_text()
    needle = 'phase = pathlib.Path.cwd().parent'
    assert source.count(needle) == 1
    benchmark.write_text(source.replace(needle, 'phase = pathlib.Path('+repr(str(phase))+')'))
    record['source_sha256'] = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in stage.rglob('*.py') if not p.is_symlink()}
    for name, digest in record['protected_live_sources'].items():
        assert hashlib.sha256((original/name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((phase/'artifacts/common_init_seed0.pth').read_bytes()).hexdigest() == plan['common_init_sha256']
    run([sys.executable, '-B', str(stage/'test_scheduler.py')], 'scheduler_tests.log', stage)
    run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests/engine', '-p', 'test_p2_phase_controls.py', '-v'], 'branch_tests.log', code)
    run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests/engine', '-p', 'test_p2_round2_acceptance.py', '-v'], 'branch_acceptance.log', code)
    run([sys.executable, '-B', str(tools/'test_prediction_analysis.py')], 'analysis_tests.log', code)
    run([sys.executable, '-B', str(tools/'test_soep_lite.py')], 'soep_tests.log', code)
    # This finite forward check uses the otherwise free second slot; it is not a timing measurement.
    run([sys.executable, '-B', str(stage/'preflight_gpu.py')], 'gpu_preflight.log', code)
    for name, digest in record['protected_live_sources'].items():
        assert hashlib.sha256((original/name).read_bytes()).hexdigest() == digest
    record.update(status='ready_for_noninterrupting_handoff', finished=time.time(),
                  live_training_sources_unchanged=True, maximum_concurrent_trainings=2)
    save()
    print(json.dumps(record, indent=2))
except BaseException as exc:
    record.update(status='failed_before_handoff', error=type(exc).__name__+': '+str(exc), finished=time.time())
    save()
    print(json.dumps(record, indent=2))
    raise
