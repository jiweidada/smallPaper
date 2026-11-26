from pathlib import Path, PurePosixPath
import argparse
import datetime as dt
import hashlib
import json
import shlex
from ssh_control import connect
from stage_and_preflight import SPECS

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--port',type=int,choices=list(SPECS),required=True)
p.add_argument('--phase',choices=['finalize','publish-probe','verify-probe','launch'],required=True);a=p.parse_args()
volume,arm,python,gpu=SPECS[a.port]
root=PurePosixPath('/'+volume)/'mpsma_fresh_pair_20261112_v1';armdir=root/arm
result_path=HERE/f'{a.phase}_{a.port}.json';assert not result_path.exists(),'Refusing duplicate operation'
before=json.loads((HERE/f'preflight_{a.port}_attempt2.json').read_text(encoding='utf-8'))
assert before['exit_code']==0
pre=json.loads(before['stdout'].splitlines()[-1])
assert pre['model_forwards']==1 and pre['optimizer_updates']==pre['full_validation_passes']==0
assert pre['status']=='PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE' and pre['loss_and_gradients_finite']
assert pre['first100_batches_sha256']=='f6f18554f6e4b06feb666acf772ae99a5180bbe9ccd3427a63ef1fee9a4b40b9'
assert pre['trainable_parameters']==(3737611 if arm=='BASE_FRESH' else 7540942)
if a.phase=='finalize':command=[python,str(root/'driver/stage_arm.py'),'finalize','--arm-dir',str(armdir)]
elif a.phase in ['publish-probe','verify-probe']:
    command=[python,str(armdir/'backup_protocol.py'),a.phase,'--output',str(armdir/'run')]
    if a.phase=='publish-probe':command+=['--anchor',str(armdir/'inputs')]
else:
    proof=json.loads((HERE/f'verify-probe_{a.port}.json').read_text(encoding='utf-8'))
    assert proof['exit_code']==0 and json.loads(proof['stdout'])['status']=='TRANSPORT_PROBE_VERIFIED'
    inventory=json.loads(json.loads((HERE/f'prelaunch_inventory_{a.port}.json').read_text(encoding='utf-8'))['stdout'])
    assert inventory['compute_processes']['exit_code']==0 and not inventory['compute_processes']['stdout'].strip()
    gpu_fields=[v.strip() for v in inventory['gpu']['stdout'].split(',')]
    assert gpu_fields[2]==gpu and int(gpu_fields[3])>=24000 and int(gpu_fields[4])<1024 and int(gpu_fields[5])<=5
    protocol=(HERE/f'execution_protocol_{a.port}.json').read_bytes()
    command=[python,str(armdir/'launch_arm.py'),'launch',hashlib.sha256(protocol).hexdigest()]
environment=['ATEN_CPU_CAPABILITY=avx2','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
             'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PYTHONHASHSEED=0','CUBLAS_WORKSPACE_CONFIG=:4096:8']
result={'port':a.port,'arm':arm,'phase':a.phase,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'command':['env',*environment,*command]}
client=connect(a.port)
try:
    if a.phase=='launch':
        live_command='nvidia-smi --query-gpu=uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'
        _,live_out,live_err=client.exec_command(live_command,timeout=30)
        live_text=live_out.read().decode().strip();live_error=live_err.read().decode()
        assert live_out.channel.recv_exit_status()==0,live_error
        fields=[x.strip() for x in live_text.split(',')]
        assert len(fields)==4 and fields[0]==gpu and int(fields[1])>=24000 and int(fields[2])<1024 and int(fields[3])<=5,live_text
        _,apps_out,apps_err=client.exec_command('nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits',timeout=30)
        apps=apps_out.read().decode().strip();apps_error=apps_err.read().decode()
        assert apps_out.channel.recv_exit_status()==0 and not apps,apps_error or apps
        result['immediate_gpu_idle_check']={'gpu':live_text,'compute_apps':apps,'observed_local_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    stdin,stdout,stderr=client.exec_command(' '.join(shlex.quote(x) for x in result['command']),timeout=600)
    stdin.channel.shutdown_write()
    result['stdout']=stdout.read().decode('utf-8',errors='replace');result['stderr']=stderr.read().decode('utf-8',errors='replace')
    result['exit_code']=stdout.channel.recv_exit_status()
    if result['exit_code']==0 and a.phase=='finalize':
        with client.open_sftp() as sftp:
            with sftp.open(str(armdir/'execution_protocol.json'),'rb') as f:payload=f.read()
        target=HERE/f'execution_protocol_{a.port}.json';assert not target.exists();target.write_bytes(payload)
        result['execution_protocol_sha256']=hashlib.sha256(payload).hexdigest()
    result['completed_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
finally:client.close()
