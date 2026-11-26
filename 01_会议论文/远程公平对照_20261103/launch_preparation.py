from pathlib import Path
import subprocess,sys,json
r=Path('/25085400120/mpsma_fair_20261103_v1')
assert not (r/'preparation_once.lock').exists()
assert not (r/'preparation_launch.json').exists()
with (r/'preparation_supervisor.log').open('x') as log:
 p=subprocess.Popen(['/25085400120/envs/deim/bin/python','-B','-u',str(r/'prepare_supervisor.py')],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'start_ticks':Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
         'scope':'Cache validation and two preflights; no optimizer updates or formal training'}
(r/'preparation_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
