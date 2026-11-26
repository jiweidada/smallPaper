"""Generate a self-contained staging script from the reviewed local bundle."""
import base64
import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
names = ['hgnetv2_light_phase.py', 'test_p2_phase_controls.py', 'test_p2_round2_acceptance.py',
         'lowrank_branch_ref_20261011_v3.py', 'preflight_gpu.py', 'queue_round2.sh']
payload = {name: {'sha256': hashlib.sha256((HERE/name).read_bytes()).hexdigest(),
                  'base64': base64.b64encode((HERE/name).read_bytes()).decode('ascii')} for name in names}
source = (HERE/'stage_remote_template.py').read_text(encoding='utf-8')
source = source.replace('PAYLOAD = {}  # BUNDLE_INSERTION', 'PAYLOAD = ' + repr(payload))
with (HERE/'stage_round2_remote.py').open('x', encoding='utf-8', newline='\n') as f:
    f.write(source)
with (HERE/'deployment_manifest.json').open('x', encoding='utf-8') as f:
    json.dump({name: row['sha256'] for name, row in payload.items()}, f, indent=2)
print(json.dumps({'files': len(payload), 'script': str(HERE/'stage_round2_remote.py')}))
