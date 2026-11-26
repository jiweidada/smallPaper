from pathlib import Path
import subprocess,json,hashlib
root=Path('/25085400120/mpsma_fair_20261103_v1')
for name,expected in {'fair_runner.py':'cf054375eb10d9522f6e638fd6e85d985adf80cec8ae359eacf9dd2af3f9bef5',
                      'preflight_supervisor_v2.py':'8597c82bf91a41021052c6c6f017430b903ba93fb8c3d76690dc6e8c4209d10d'}.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
assert (root/'preparation_attempt1/failure_audit.json').exists()
assert not (root/'preparation_once_v2.lock').exists() and not (root/'preparation_launch_v2.json').exists()
assert not (root/'preflight').exists() and not (root/'one_shot.lock').exists()
with (root/'preparation_supervisor_v2.log').open('x') as log:
    proc=subprocess.Popen(['/25085400120/envs/deim/bin/python','-B','-u',str(root/'preflight_supervisor_v2.py')],
                          cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':proc.pid,'start_ticks':Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
         'scope':'Two compatibility preflights after archived Mosaic registration repair; no optimizer updates.'}
(root/'preparation_launch_v2.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
