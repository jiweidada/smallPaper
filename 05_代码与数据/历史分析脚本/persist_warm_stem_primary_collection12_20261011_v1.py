from pathlib import Path
import base64,hashlib,json
i=json.loads(Path('outputs/warm_stem_primary_collection12_20261011_v1_inputs.json').read_bytes())
def verify(v):
 b=v['payload'].encode();t=v['transport'];assert len(b)==t['bytes'] and hashlib.sha256(b).hexdigest()==t['sha256'];return json.loads(b)
rs=verify(i['records']);d=verify(i['deployment']);o=verify(i['observation'])
assert o['failure'] is None and not o['worker']['same_live']
root=Path('outputs/warm_stem_primary_collection12_20261011_v1');root.mkdir(exist_ok=False)
folder=root/'seed2_warm_stem_v1';folder.mkdir()
for n,x in rs['files'].items():
 b=base64.b64decode(x['base64'],validate=True);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
 (folder/n).write_bytes(b)
execution=json.loads((folder/'remote_collection_execution_v1.json').read_bytes())
assert execution==o['completion'] and execution['status']=='STOPPED_WARM_PRIMARY12_PACKAGE_COLLECTED'
meta=json.loads((folder/'remote_collection_metadata_v1.json').read_bytes());assert meta==execution['metadata']
assert execution['receipt']['pid']==d['pid'] and execution['receipt']['start_ticks']==d['start_ticks']
(folder/'remote_collection_transport_v1.json').write_text(json.dumps({k:v for k,v in rs['files']['remote_collection_execution_v1.json'].items() if k!='base64'},indent=2),encoding='utf8')
(root/'deployment.json').write_bytes(i['deployment']['payload'].encode());(root/'observation.json').write_bytes(i['observation']['payload'].encode())
(root/'record_transports.json').write_text(json.dumps({k:v['transport'] for k,v in i.items()},indent=2),encoding='utf8')
print(json.dumps({'status':'ACTUAL_REMOTE_COLLECTION_RECORDS_PERSISTED','archive_bytes':meta['bytes'],'archive_sha256':meta['sha256']}))
