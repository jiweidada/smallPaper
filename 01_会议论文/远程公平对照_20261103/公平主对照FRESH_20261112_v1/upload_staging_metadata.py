from pathlib import Path, PurePosixPath
import hashlib
import json
from ssh_control import connect

HERE=Path(__file__).resolve().parent
envs={}
for port in [30715,30048]:
    wrapper=json.loads((HERE/f'environment_snapshot_{port}.json').read_text(encoding='utf-8'))
    assert wrapper['exit_code']==0
    value=json.loads(wrapper['stdout']);assert value['torch']=='2.5.1+cu124' and value['cuda']=='12.4'
    envs[port]=value
    (HERE/f'environment_actual_{port}.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
for key in ['torch','cuda','cudnn','packages']:assert envs[30715][key]==envs[30048][key]
results=[]
for port,volume in [(30715,'25085400120'),(30048,'24085403014')]:
    prep_wrapper=json.loads((HERE/f'file_preparation_{port}.json').read_text(encoding='utf-8'))
    assert prep_wrapper['exit_code']==0
    prep=json.loads(prep_wrapper['stdout']);assert prep['status']=='ALL_FROZEN_INPUT_BYTES_VERIFIED'
    target=PurePosixPath('/'+volume)/'mpsma_fresh_pair_20261112_v1'/'staging_inputs'
    client=connect(port)
    try:
        with client.open_sftp() as sftp:
            try:sftp.stat(str(target))
            except FileNotFoundError:sftp.mkdir(str(target))
            files=[]
            for name in ['environment_actual_30715.json','environment_actual_30048.json','user_authorization_20261112.json']:
                payload=(HERE/name).read_bytes()
                with sftp.open(str(target/name),'wx') as stream:stream.write(payload);stream.flush()
                with sftp.open(str(target/name),'rb') as stream:assert stream.read()==payload
                files.append({'name':name,'sha256':hashlib.sha256(payload).hexdigest()})
        results.append({'port':port,'remote_staging_inputs':str(target),'verified_frozen_root':prep['frozen_root'],'verified_data_root':prep['data_root'],'files':files})
    finally:client.close()
(HERE/'staging_metadata_uploaded.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
print(json.dumps(results))
