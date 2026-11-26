from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path('/root/uav_a52_20261101_v1')
assert sys.executable=='/opt/conda/envs/mamba/bin/python'
assert not (ROOT/'capacity_launch_identity.json').exists()
with (ROOT/'capacity_launch_once.lock').open('x') as f:f.write(str(os.getpid()))
with (ROOT/'capacity_stdout.log').open('xb') as out,(ROOT/'capacity_stderr.log').open('xb') as err:
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'capacity_A52.py')],cwd=str(ROOT),stdout=out,stderr=err,start_new_session=True)
ticks=int(Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()[19])
identity={'pid':child.pid,'start_ticks':ticks,'stage':'CAPACITY_ONLY','script':'capacity_A52.py','launch_server_unix':time.time(),'no_formal_queue':True,'no_auto_retry':True}
(ROOT/'capacity_launch_identity.json').write_text(json.dumps(identity,indent=2))
print(json.dumps(identity),flush=True)

