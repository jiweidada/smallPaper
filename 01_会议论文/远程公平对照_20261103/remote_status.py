from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
R=Path('/25085400120/mpsma_fair_20261103_v1')
out={'observed_utc':datetime.now(timezone.utc).isoformat(),'receipts':{},'processes':{}}
for name in ['cache_progress.json','preparation_queue.json','preparation_launch.json','preparation_queue_v2.json','preparation_launch_v2.json','queue.json','preflight/BASE.json','preflight/FULL.json']:
 p=R/name
 if p.exists():
  d=json.loads(p.read_text());out['receipts'][name]=d
  entries=[d]+d.get('steps',[])+d.get('runs',[])
  for item in entries:
   pid=item.get('pid',item.get('supervisor_pid'))
   if pid:
    proc=Path(f'/proc/{pid}')
    if proc.exists():
     fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
     out['processes'][str(pid)]={'state':fields[0],'start_ticks':fields[19],
      'identity_matches':str(item.get('start_ticks',item.get('supervisor_start_ticks',fields[19])))==fields[19]}
    else:out['processes'][str(pid)]={'present':False}
for name in ['cache.log','BASE_preflight_v2.log','FULL_preflight_v2.log','BASE.train.log','FULL.train.log']:
 p=R/name
 if p.exists():out[name]=p.read_text(errors='replace')[-1800:]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps(out))
