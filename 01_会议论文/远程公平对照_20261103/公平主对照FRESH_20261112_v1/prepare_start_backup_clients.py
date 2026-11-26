from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
TARGET=Path('D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1').resolve(strict=True)
assert TARGET.parent==Path('D:/MPSMA_DEIM_experiment_backups').resolve(strict=True)
assert shutil.disk_usage(TARGET).free>=120*2**30
PYTHON=Path('C:/Users/jiwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
TORCH_SITE=HERE.parents[1]/'本地CPU核验_20261103/.runtime/site-packages'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(HERE/'driver/backup_monitor.py')=='d2889c042416ce45c66db7d15648281662aabb71f337c6b269b5cdd71d6a5662'
assert sha(HERE/'driver/checkpoint_validator.py')=='d7e1962ec61ba871a412387dfd8adf3f627d04e5d6d9063a7c11666843f12c8c'
results=[]
for port,arm,volume in [(30715,'BASE_FRESH','25085400120'),(30048,'FULL_FRESH','24085403014')]:
    control=TARGET/arm
    assert not control.exists(),'Refusing to replace an existing backup control'
    control.mkdir();(control/'backups').mkdir()
    files=[]
    for name in ['backup_protocol.py','backup_monitor.py','checkpoint_validator.py']:
        src=HERE/'driver'/name;dst=control/name;shutil.copyfile(src,dst);assert sha(src)==sha(dst);files.append({'file':name,'sha256':sha(dst)})
    src=HERE/f'execution_protocol_{port}.json';dst=control/'execution_protocol.json';shutil.copyfile(src,dst)
    assert sha(src)==sha(dst);files.append({'file':'execution_protocol.json','sha256':sha(dst)})
    command=[sys.executable,'-W','ignore','-X','utf8',str(HERE/'backup_monitor_entry.py'),
             '--host','172.17.120.160','--port',str(port),'--user','root','--known-hosts',str(HERE/'known_hosts'),
             '--remote-root','/'+volume+'/mpsma_fresh_pair_20261112_v1/'+arm,
             '--local-root',str(control),'--checkpoint-python',str(PYTHON),'--torch-site',str(TORCH_SITE)]
    with (control/'monitor.stdout.log').open('xb') as out,(control/'monitor.stderr.log').open('xb') as err:
        process=subprocess.Popen(command,cwd=str(control),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    result={'arm':arm,'port':port,'local_control':str(control),'pid':process.pid,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'command':command,'files':files,'automatic_retry':False,'training_actions':0}
    (control/'monitor_launch.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    results.append(result)
(HERE/'backup_clients_launched.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'backup_clients':[{'arm':x['arm'],'pid':x['pid'],'local_control':x['local_control']} for x in results],'training_started':False}))
