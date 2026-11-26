import datetime,json
from pathlib import Path
data=Path(CONFIG['storage'])/'datasets_verified_20261017/UAVDT_v1'
result={'account':CONFIG['account'],'data_view_exists':data.exists(),'ready':(data/'data_ready.json').is_file(),
        'native_root_exists':(data.parent/'UAVDT_official/native_frames_v2').is_dir()}
for name in ['preparation_started.json','data_ready.json']:
 p=data/name
 if p.is_file():
  value=json.loads(p.read_text())
  result[name]={k:v for k,v in value.items() if k in ['created_utc','pid','state','images','data_metadata_sha256','annotations_sha256']}
print(json.dumps(result),flush=True)
