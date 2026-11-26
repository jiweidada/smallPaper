from pathlib import Path
import base64,hashlib,json
read=lambda p:json.loads(Path(p).read_bytes())
inputs=read('outputs/stem_seed1_controls_primary_collection12_20261011_v1_inputs.json');obs={}
for label,item in inputs.items():
    b=item['payload'].encode('utf8');t=item['transport']
    assert len(b)==t['bytes'] and hashlib.sha256(b).hexdigest()==t['sha256'];obs[label]=json.loads(b)
assert obs['collection']['status']=='SEED1_BOTH_CONTROL_PRIMARY_COLLECTION_EXECUTION_SUCCESS' and obs['collection']['exit_code']==0
root=Path('outputs/stem_seed1_controls_primary_collection12_20261011_v1');root.mkdir(exist_ok=False)
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
for label,item in inputs.items():(root/f'{label}.json').write_bytes(item['payload'].encode('utf8'))
write(root/'transport.json',{l:i['transport'] for l,i in inputs.items()})
for arm,raw in obs['files']['entries'].items():
    b=base64.b64decode(raw['base64'],validate=True)
    assert len(b)==raw['bytes'] and hashlib.sha256(b).hexdigest()==raw['sha256']
    e=json.loads(b);assert e==obs['collection']['executed'][arm] and e['exit_code']==0
    m=e['metadata'];assert m['seed']==1 and m['arm']==arm and not m['saved_EMA_independent_replay_included']
    folder=Path(f'outputs/stem_component_primary_collection_20261011_v2/seed1_{arm}_v1');folder.mkdir(exist_ok=False)
    (folder/'remote_collection_execution_v1.json').write_bytes(b)
    write(folder/'remote_collection_transport_v1.json',{k:raw[k] for k in ('sha256','bytes')})
    write(folder/'remote_collection_metadata_v1.json',m)
write(root/'execution_state_v1.json',dict(status='SEED1_BOTH_CLOSED_CONTROL_PRIMARY_PACKAGES_COLLECTED_DOWNLOAD_PENDING',independent_saved_EMA_pending=True,goal_complete=False))
print(json.dumps(dict(status='LOCAL_DESTINATIONS_PREPARED_FOR_TWO_SOURCE_PINNED_SEED1_CONTROLS')))
