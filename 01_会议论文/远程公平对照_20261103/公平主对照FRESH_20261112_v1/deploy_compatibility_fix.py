from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import json
import shlex
from ssh_control import connect
from stage_and_preflight import SPECS, FILES

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--port',type=int,choices=list(SPECS),required=True);args=p.parse_args()
volume,arm,python,gpu=SPECS[args.port]
root=PurePosixPath('/'+volume)/'mpsma_fresh_pair_20261112_v1'
payload=(HERE/'driver/fair_helpers.py').read_bytes()
newsha=hashlib.sha256(payload).hexdigest();assert newsha=='5cb4325db3a68d728398cd9d04b772239af9c3dd1d9e325a1cf79c76b241a2f5'
prepared=json.loads((HERE/f'prepare_{args.port}.json').read_text(encoding='utf-8'))
oldsha=next(x['sha256'] for x in prepared['driver_files'] if x['name']=='fair_helpers.py')
client=connect(args.port)
result={'port':args.port,'arm':arm,'before_sha256':oldsha,'after_sha256':newsha,'model_forwards':0,'optimizer_updates':0}
try:
    with client.open_sftp() as sftp:
        prep=root/'preparation'
        for label in ['driver',arm]:
            path=root/label/'fair_helpers.py'
            with sftp.open(str(path),'rb') as f:before=f.read()
            assert hashlib.sha256(before).hexdigest()==oldsha
            assert ast.dump(ast.parse(before),include_attributes=False)==ast.dump(ast.parse(payload),include_attributes=False)
            backup=prep/('fair_helpers_'+label+'_before_py310_quote_fix.py')
            with sftp.open(str(backup),'wx') as f:f.write(before);f.flush()
            temp=path.with_name(path.name+'.py310_fix')
            with sftp.open(str(temp),'wx') as f:f.write(payload);f.flush()
            sftp.posix_rename(str(temp),str(path))
            with sftp.open(str(path),'rb') as f:assert f.read()==payload
    code="from pathlib import Path;import sys,json; root=Path("+repr(str(root/arm))+");names="+repr(FILES+['source_loader.py'])+";[(compile((root/name).read_text(),str(root/name),'exec')) for name in names];print(json.dumps({'status':'ALL_CORE_COMPILED_ON_ACTUAL_INTERPRETER','python':sys.version,'files':names,'model_imports':0,'forwards':0}))"
    _,stdout,stderr=client.exec_command(' '.join(shlex.quote(x) for x in [python,'-B','-c',code]),timeout=120)
    result['compile_stdout']=stdout.read().decode();result['compile_stderr']=stderr.read().decode();result['compile_exit_code']=stdout.channel.recv_exit_status()
    path=HERE/f'core_compatibility_{args.port}.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert result['compile_exit_code']==0,result['compile_stderr']
    print(json.dumps(result))
finally:client.close()
