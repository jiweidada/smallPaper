from pathlib import Path
import base64
import datetime
import hashlib
import json
import zlib

code = Path(__file__).resolve().parent
root = code.parents[2]
out = root / 'outputs/assignment_mechanism_diagnostic_20261010_v1'
protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8-sig'))
files = {p.name: p.read_bytes() for p in [code/'capture_assignment_panel.py', code/'controller.py', code/'training_evidence.py']}
hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
protocol.update(execution_revision='v2_pre_execution_evidence_completeness', frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), code_sha256=hashes, changes_before_first_execution='Persist all three actual leaf-box gradient arrays and native scalar losses; retain protected-small denominators. Scientific cohort, hypothesis, counterfactual and decision boundary unchanged. Preserve earlier prepared code.', panel_geometry_precheck={'max_raw_GT':229,'raw_GT_total':2712,'images':32}, output_name='frozen_panel_v1')
(out/'protocol_execution_v2.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files['protocol_execution_v2.json'] = (out/'protocol_execution_v2.json').read_bytes()
payload = {'files':{name:base64.b64encode(value).decode() for name,value in files.items()}, 'hashes':{name:hashlib.sha256(value).hexdigest() for name,value in files.items()}}
packed = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
(out/'deployment_payload_v2.json').write_text(json.dumps({'packed':packed,'hashes':payload['hashes']}),encoding='utf-8')
print(packed)
