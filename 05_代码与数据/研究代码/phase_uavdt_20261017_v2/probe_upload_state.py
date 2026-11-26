import json,os,subprocess
from pathlib import Path
root=Path(CONFIG['storage'])/'datasets_verified_20261017/UAVDT_official/raw'
result={'account':CONFIG['account'],'files':{}}
for name in ['UAV-benchmark-M.zip','UAV-benchmark-M.zip.part']:
 p=root/name;result['files'][name]={'exists':p.exists(),'bytes':p.stat().st_size if p.is_file() else None}
result['memory_events']={}
for name in ['memory.events','memory.max','memory.current']:
 p=Path('/sys/fs/cgroup')/name
 if p.is_file():result['memory_events'][name]=p.read_text()
print(json.dumps(result),flush=True)
