from pathlib import Path
import json,os,subprocess
r=Path('/25085400120/mpsma_fair_20261103_v1')
out={'files':sorted(p.name for p in r.iterdir()),'receipts':{}}
for n in ['input_verification.json','environment.json']:
    p=r/n
    if p.exists():
        x=json.loads(p.read_text())
        if n=='environment.json': x={k:x[k] for k in ['executable','torch','cuda','gpu']}
        out['receipts'][n]=x
out['python_processes']=subprocess.check_output(['ps','-eo','pid,ppid,etimes,comm'],text=True).splitlines()
out['python_processes']=[x for x in out['python_processes'] if 'python' in x or x.startswith('    PID')]
print(json.dumps(out))
