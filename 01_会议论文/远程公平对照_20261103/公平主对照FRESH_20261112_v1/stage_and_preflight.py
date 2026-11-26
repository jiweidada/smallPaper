from pathlib import Path, PurePosixPath
import argparse
import datetime as dt
import hashlib
import json
import shlex
import time
from ssh_control import connect

HERE=Path(__file__).resolve().parent
FILES=['stage_arm.py','fair_helpers.py','train_runner.py','backup_protocol.py','launch_arm.py','preflight.py']
SPECS={30715:('25085400120','BASE_FRESH','/25085400120/envs/deim/bin/python','GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a'),
       30048:('24085403014','FULL_FRESH','/24085403014/deim_migration_runtime/envs/deim/bin/python','GPU-85eca9af-10f3-4279-d72a-05ff1ecdcd3a')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,choices=list(SPECS),required=True);p.add_argument('--phase',choices=['prepare','preflight'],required=True);p.add_argument('--attempt',type=int,choices=[1,2],default=1);a=p.parse_args()
    record_path=HERE/(f'{a.phase}_{a.port}.json' if a.attempt==1 else f'{a.phase}_{a.port}_attempt{a.attempt}.json')
    assert not record_path.exists(),'This local execution attempt is already recorded'
    if a.attempt==2:
        previous=json.loads((HERE/f'preflight_{a.port}.json').read_text(encoding='utf-8'))
        assert a.phase=='preflight' and previous['exit_code']==1 and 'SyntaxError: f-string' in previous['stderr']
    volume,arm,python,gpu=SPECS[a.port]
    root=PurePosixPath('/'+volume)/'mpsma_fresh_pair_20261112_v1';driver=root/'driver';armdir=root/arm
    other=30048 if a.port==30715 else 30715
    prepared=json.loads((HERE/'staging_metadata_uploaded.json').read_text(encoding='utf-8'))
    metadata=next(x for x in prepared if x['port']==a.port)
    result={'port':a.port,'arm':arm,'phase':a.phase,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'new_training_runs':0}
    client=connect(a.port)
    try:
        if a.phase=='prepare':
            with client.open_sftp() as sftp:
                try:sftp.stat(str(driver))
                except FileNotFoundError:sftp.mkdir(str(driver))
                result['driver_files']=[]
                for name in FILES:
                    payload=(HERE/'driver'/name).read_bytes()
                    with sftp.open(str(driver/name),'wx') as f:f.write(payload);f.flush()
                    with sftp.open(str(driver/name),'rb') as f:assert f.read()==payload
                    result['driver_files'].append({'name':name,'sha256':hashlib.sha256(payload).hexdigest()})
            command=[python,str(driver/'stage_arm.py'),'prepare','--root',metadata['verified_frozen_root'],'--data-root',metadata['verified_data_root'],
                     '--arm-dir',str(armdir),'--arm',arm,'--gpu-uuid',gpu,'--cuda-device','0',
                     '--authorization-file',str(root/'staging_inputs/user_authorization_20261112.json'),
                     '--environment-file',str(root/'staging_inputs'/f'environment_actual_{a.port}.json'),
                     '--peer-environment-file',str(root/'staging_inputs'/f'environment_actual_{other}.json')]
        else:command=[python,str(armdir/'preflight.py'),'--arm-dir',str(armdir),'--arm',arm]
        environment=['ATEN_CPU_CAPABILITY=avx2','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
                     'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PYTHONHASHSEED=0','CUBLAS_WORKSPACE_CONFIG=:4096:8']
        result['command']=['env',*environment,*command]
        stdin,stdout,stderr=client.exec_command(' '.join(shlex.quote(x) for x in result['command']),timeout=1200)
        stdin.channel.shutdown_write()
        result['stdout']=stdout.read().decode('utf-8',errors='replace');result['stderr']=stderr.read().decode('utf-8',errors='replace')
        result['exit_code']=stdout.channel.recv_exit_status();result['completed_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        path=record_path;path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'phase':a.phase,'port':a.port,'arm':arm,'exit_code':result['exit_code'],'stdout_tail':result['stdout'][-4500:],'stderr_tail':result['stderr'][-3000:],'result':str(path)},ensure_ascii=False))
    finally:client.close()

if __name__=='__main__':main()
