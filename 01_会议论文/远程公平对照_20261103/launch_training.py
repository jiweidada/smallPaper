"""Launch exactly the frozen authorized pair; run only after filling its expected hash."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
ROOT=Path('/25085400120/mpsma_fair_20261103_v1')
EXPECTED_PROTOCOL_SHA256='8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while b:=stream.read(4*1024*1024):h.update(b)
    return h.hexdigest()
protocol_path=ROOT/'execution_protocol.json'
assert sha(protocol_path)==EXPECTED_PROTOCOL_SHA256,'Protocol does not match the reviewed local artifact'
protocol=json.loads(protocol_path.read_text())
assert protocol['authorized'] and protocol['epochs']==320 and protocol['arms']==['BASE','FULL']
assert protocol['budget']['optimizer_step_attempts_per_group']==517440
assert protocol['budget']['hard_timeout_seconds_per_arm']==604800
prep=json.loads((ROOT/'preparation_queue_v2.json').read_text())
assert prep['status']=='PREFLIGHTS_COMPLETED_FORMAL_TRAINING_NOT_STARTED'
for job in prep['steps']:
    assert job['status']=='COMPLETED' and job['exit_code']==0
    assert not Path(f"/proc/{job['pid']}").exists(),'Preflight has not fully exited'
for row in protocol['frozen_files']:
    # Full source and all data bytes are checked by each runner before its first update.
    if not row['path'].startswith('source_snapshot/'):
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
assert not (ROOT/'runs').exists() and not (ROOT/'one_shot.lock').exists()
assert not (ROOT/'training_launch.json').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip()
assert gpu.split(',')[0].strip()=='GPU-85eca9af-10f3-4279-d72a-05ff1ecdcd3a'
assert int(gpu.split(',')[1])<512,gpu
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert not apps,'GPU has another compute process: '+apps
with (ROOT/'training_launch_once.lock').open('x') as stream:
    stream.write(json.dumps({'protocol_sha256':EXPECTED_PROTOCOL_SHA256,'remote_utc':datetime.now(timezone.utc).isoformat()})+'\n')
command=['/25085400120/envs/deim/bin/python','-B','-u',str(ROOT/'supervise.py')]
with (ROOT/'training_supervisor.log').open('x') as log:
    proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                          env=dict(os.environ,PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1'))
receipt={'status':'SUPERVISOR_LAUNCHED','pid':proc.pid,
         'start_ticks':Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
         'command':command,'remote_utc':datetime.now(timezone.utc).isoformat(),
         'protocol_sha256':EXPECTED_PROTOCOL_SHA256,'gpu_before_launch':gpu,'no_extra_gpu_processes':True,
         'scope':'BASE then FULL, seed 0, batch 4, FP32, 320 epochs each; no retry, tuning or extra arm.'}
(ROOT/'training_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
