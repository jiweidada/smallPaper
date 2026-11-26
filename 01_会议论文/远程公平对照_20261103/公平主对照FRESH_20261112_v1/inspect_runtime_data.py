"""Read-only bounded discovery of existing environments and data locations."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess

def run(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=30)
        return {"exit_code":p.returncode,"stdout":p.stdout[-18000:],"stderr":p.stderr[-2000:]}
    except Exception as e:return {"error":type(e).__name__,"message":str(e)}

root=Path('/25085400120') if Path('/25085400120').is_mount() else Path('/24085403014')
python_paths=[root/'envs/deim/bin/python',root/'deim_migration_runtime/bin/python',root/'deim_migration_runtime/venv/bin/python',root/'DEIM/.venv/bin/python',root/'DEIM/venv/bin/python',Path('/opt/conda/bin/python'),Path('/usr/local/bin/python')]
listings={}
for folder in [root/'deim_migration_runtime',root/'DEIM',root/'datasets_verified_20261017',Path('/tmp/mpsma_fair_20261103_data'),root/'mpsma_fair_20261103_v1']:
    if folder.is_dir():listings[str(folder)]=[{"name":p.name,"directory":p.is_dir(),"bytes":p.stat().st_size if p.is_file() else None} for p in sorted(folder.iterdir()) if not p.name.startswith('.')][:80]
python_info=[]
for p in python_paths:
    if not p.is_file():continue
    code="import sys,importlib.util,json; names=['torch','torchvision','faster_coco_eval','yaml','timm','scipy','cv2','PIL'];print(json.dumps({'python':sys.version,'modules':{n:bool(importlib.util.find_spec(n)) for n in names}}))"
    python_info.append({"path":str(p),"info":run([str(p),'-c',code])})
project=root/'mpsma_fair_20261103_v1'
protocol=None
if (project/'execution_protocol.json').is_file():
    d=json.loads((project/'execution_protocol.json').read_text())
    protocol={"keys":list(d),"paths":{k:v for k,v in d.items() if 'path' in k or k in ['remote_root','data_root','python','data','dataset']}}
print(json.dumps({"observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"root":str(root),"python_info":python_info,"listings":listings,"protocol_shape":protocol,
                  "ram_available_kib":next((x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),None),"tmp_disk_free_bytes":shutil.disk_usage('/tmp').free}))
