"""Save byte-exact positive collection receipts for the completed attention arm."""
from pathlib import Path
import base64,hashlib,json
ROOT=Path.cwd();inputs=json.loads((ROOT/'outputs/user_primary_v2_seed2_attention_receipts12_inputs.json').read_bytes())
assert set(inputs)=={'backbone_attention'}
for arm,v in inputs.items():
    raw=v['payload'].encode();assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256']
    o=json.loads(raw);assert o['seed']==2 and o['arm']==arm and o['waiter_proof']['exit_code']==0 and o['source_v1_failure_preserved'] and o['actual_collection_handle_stopped']
    d=ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed2_{arm}_v1';d.mkdir(parents=True,exist_ok=False)
    for key,name in (('execution','remote_collection_execution_v1.json'),('metadata','remote_collection_metadata_v1.json')):
        b=base64.b64decode(o[key]['base64']);assert len(b)==o[key]['bytes'] and hashlib.sha256(b).hexdigest()==o[key]['sha256']
        with (d/name).open('xb') as f:f.write(b)
    with (d/'remote_collection_transport_v1.json').open('x',encoding='utf8') as f:json.dump({k:o['execution'][k] for k in ('bytes','sha256')},f,indent=2)
    with (d/'remote_waiter_completion_proof_v2.json').open('x',encoding='utf8') as f:json.dump(o['waiter_proof'],f,indent=2)
    print(json.dumps(dict(arm=arm,status='REMOTE_ATTENTION_RECEIPTS_PERSISTED_BYTE_EXACT')))
