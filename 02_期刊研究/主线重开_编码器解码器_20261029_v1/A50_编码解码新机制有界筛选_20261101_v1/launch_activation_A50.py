from pathlib import Path
import hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parent
with (ROOT/'activation_launch_once.lock').open('x') as f:f.write(str(os.getpid()))
assert not (ROOT/'activation_result.json').exists()
with (ROOT/'activation_stdout.log').open('xb') as out,(ROOT/'activation_stderr.log').open('xb') as err:
    process=subprocess.Popen(['/opt/conda/envs/mamba/bin/python','-u',str(ROOT/'activation_A50.py')],
       cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
stat=Path('/proc')/str(process.pid)/'stat'
info={'pid':process.pid,'start_ticks':stat.read_text().split()[21],
 'protocol_sha256':hashlib.sha256((ROOT/'activation_protocol_FROZEN_v1.json').read_bytes()).hexdigest(),
 'maximum_model_forwards':16,'optimizer_updates':0,'no_auto_retry':True}
(ROOT/'activation_launch_identity.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info))
