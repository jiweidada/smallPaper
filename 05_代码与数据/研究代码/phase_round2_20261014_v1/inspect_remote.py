"""Read-only snapshot of relevant experiment state and configuration paths."""
import hashlib
import json
import pathlib
import subprocess

storage = pathlib.Path('/25085400120' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014')
root = storage / 'DEIM'
experiments = root / 'experiments'
phase = next((p for p in [experiments/'phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir()), experiments/'phase_mechanism_20261012')
result = {'storage': str(storage), 'phase_root': str(phase), 'phase_exists': phase.is_dir(), 'files': {}, 'jobs': {}, 'old_c_args': []}
for rel in ['code/configs/experiments/phase/proto96.yml', 'code/configs/experiments/phase/singlephase_full96.yml',
            'code/engine/backbone/hgnetv2_light_phase.py', 'code/engine/backbone/hgnetv2_light.py']:
    p = phase / rel
    if p.is_file():
        data = p.read_bytes()
        result['files'][rel] = {'sha256': hashlib.sha256(data).hexdigest(),
                              'text': data.decode('utf-8') if '/configs/' in rel else None}
for p in sorted((phase / 'jobs').glob('*.json')):
    result['jobs'][p.name] = json.loads(p.read_text())
init = phase / 'artifacts/common_init_seed0.pth'
if init.exists():
    result['init_sha256'] = hashlib.sha256(init.read_bytes()).hexdigest()
patterns = ['visdrone_light_20260928_090716/structural_candidates/outputs/p2_skip_main_seed0/args.json',
            'c_mainline_seed1_full96_20261003_v1/**/affine_epoch3_resume_v1/args.json']
for base in [experiments, storage/'experiments']:
    for pattern in patterns:
        for p in base.glob(pattern):
            result['old_c_args'].append({'path': str(p), 'args': json.loads(p.read_text())})
for cmd, label in [(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], 'gpu'),
                   (['ps', '-eo', 'pid,ppid,stat,args'], 'processes')]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    lines = completed.stdout.splitlines()
    if label == 'processes':
        lines = [s for s in lines if ('train.py' in s or 'queue.sh' in s) and 'base64' not in s]
    result[label] = {'returncode': completed.returncode, 'lines': lines}
result['python_candidates'] = [str(p) for p in [storage/'envs/deim/bin/python', storage/'deim_migration_runtime/envs/deim/bin/python', pathlib.Path('/opt/conda/bin/python')] if p.exists()]
print(json.dumps(result, ensure_ascii=False, indent=2))
