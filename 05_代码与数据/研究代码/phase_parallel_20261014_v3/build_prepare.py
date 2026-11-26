import ast
import base64
import hashlib
import json
import pathlib

here = pathlib.Path(__file__).resolve().parent
accepted = here.with_name('phase_round2_20261014_v2')
assets = {}
for relative, source in {
    'run_parallel_queue.py': here/'run_parallel_queue.py',
    'test_scheduler.py': here/'test_scheduler.py',
    'preflight_gpu.py': accepted/'preflight_gpu.py',
    'code/engine/backbone/hgnetv2_light_phase.py': accepted/'hgnetv2_light_phase.py',
    'code/engine/deim/soep_lite_round2.py': accepted/'soep_lite_round2.py',
    'code/train_round2.py': accepted/'train_round2.py',
    'code/tests/engine/test_p2_phase_controls.py': accepted/'test_p2_phase_controls.py',
    'code/tests/engine/test_p2_round2_acceptance.py': accepted/'test_p2_round2_acceptance.py',
    'code/tests/engine/lowrank_branch_ref_20261011_v3.py': accepted/'lowrank_branch_ref_20261011_v3.py',
}.items():
    raw = source.read_bytes()
    ast.parse(raw)
    assets[relative] = {'sha256': hashlib.sha256(raw).hexdigest(), 'base64': base64.b64encode(raw).decode()}
template = (here/'prepare_template.py').read_text(encoding='utf-8')
request = template.replace('ASSETS = {}  # INSERT_ASSETS', 'ASSETS = '+repr(assets))
ast.parse(request)
(here/'prepare_remote.py').write_text(request, encoding='utf-8', newline='\n')
(here/'asset_manifest.json').write_text(json.dumps({name: value['sha256'] for name, value in assets.items()}, indent=2))
print(json.dumps({'request_bytes': len(request.encode()), 'files': len(assets), 'manifest': str(here/'asset_manifest.json')}))
