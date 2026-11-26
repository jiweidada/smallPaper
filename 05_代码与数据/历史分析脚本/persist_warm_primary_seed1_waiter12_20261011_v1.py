from pathlib import Path
import base64,hashlib,json
i=json.loads(Path('outputs/warm_primary_seed1_waiter12_20261011_v1_inputs.json').read_bytes())
def verify(v):
 b=v['payload'].encode();t=v['transport'];assert len(b)==t['bytes'] and hashlib.sha256(b).hexdigest()==t['sha256'];return json.loads(b)
rs=verify(i['records']);d=verify(i['deployment']);o=verify(i['observation'])
assert o['failure'] is None and o['collector_failure'] is None and not o['worker']['same_live']
root=Path('outputs/warm_stem_primary_collection12_20261011_v1');assert root.is_dir()
folder=root/'seed1_warm_stem_v1';folder.mkdir(exist_ok=False)
for n,x in rs['files'].items():
 b=base64.b64decode(x['base64'],validate=True);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
 (folder/n).write_bytes(b)
execution=json.loads((folder/'remote_collection_execution_v1.json').read_bytes());w=json.loads((folder/'remote_waiter_completion_v1.json').read_bytes())
assert w==o['completion'] and w['exit_code']==0 and execution['exit_code']==0
assert w['collection_completion_sha256']==rs['files']['remote_collection_execution_v1.json']['sha256']
meta=json.loads((folder/'remote_collection_metadata_v1.json').read_bytes());assert meta==execution['metadata']==w['metadata']
assert execution['receipt']['pid']==w['receipt']['pid']==d['pid'] and execution['receipt']['start_ticks']==d['start_ticks']
assert execution['receipt']['seed']==1 and not execution['automatic_retry'] and execution['new_model_forwards']==execution['new_training_epochs']==0
(folder/'remote_collection_transport_v1.json').write_text(json.dumps({k:v for k,v in rs['files']['remote_collection_execution_v1.json'].items() if k!='base64'},indent=2),encoding='utf8')
(folder/'remote_deployment.json').write_bytes(i['deployment']['payload'].encode());(folder/'remote_observation.json').write_bytes(i['observation']['payload'].encode())
(folder/'remote_record_transports.json').write_text(json.dumps({k:v['transport'] for k,v in i.items()},indent=2),encoding='utf8')
print(json.dumps({'status':'WARM_SEED1_ACTUAL_RECORDS_VERIFIED_AND_PERSISTED','archive_bytes':meta['bytes'],'archive_sha256':meta['sha256'],'archive':meta['archive']}))
