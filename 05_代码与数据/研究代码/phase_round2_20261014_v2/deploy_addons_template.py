"""Append queue and analysis tools. No existing source or running job is changed."""
import base64
import hashlib
import json
import pathlib

PAYLOAD = {}  # INSERT_FILES
PLANS = {}  # INSERT_PLANS
CONFIGS = {}  # INSERT_CONFIGS
account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
storage = pathlib.Path('/25085400120' if account == 'A' else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
stage = phase/'artifacts/round2_20261014_v1'
addons = stage/'addons'
addons.mkdir(exist_ok=False)
protected = ['engine/backbone/hgnetv2_light.py', 'engine/backbone/hgnetv2_light_control.py', 'engine/backbone/hgnetv2_light_phase.py', 'engine/solver/det_solver.py', 'engine/solver/_solver.py']
before = {name: hashlib.sha256((phase/'code'/name).read_bytes()).hexdigest() for name in protected}
for name, row in PAYLOAD.items():
    data = base64.b64decode(row['base64'])
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    with (addons/name).open('xb') as f:
        f.write(data)
for name, relative in [('soep_lite_round2.py', 'engine/deim/soep_lite_round2.py'), ('train_round2.py', 'train_round2.py')]:
    with (phase/'code'/relative).open('xb') as f:
        f.write((addons/name).read_bytes())
with (phase/'code/configs/experiments/phase/soep_lite_full96.yml').open('x') as f:
    f.write(CONFIGS[account])
with (addons/'queue_plan.json').open('x') as f:
    json.dump(PLANS[account], f, indent=2)
assert before == {name: hashlib.sha256((phase/'code'/name).read_bytes()).hexdigest() for name in protected}
with (addons/'deployment_manifest.json').open('x') as f:
    json.dump({name: row['sha256'] for name, row in PAYLOAD.items()}, f, indent=2)
print(json.dumps({'account': account, 'addons': str(addons), 'protected_sources_unchanged': True, 'jobs': PLANS[account]['jobs'], 'training_launched': False}, indent=2))
