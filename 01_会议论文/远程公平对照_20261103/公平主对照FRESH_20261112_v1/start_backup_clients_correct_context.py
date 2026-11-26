"""Correct only the pre-training local credential context; preserve failed bootstrap logs."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess
import psutil
from ssh_control import credential_path, protect

HERE=Path(__file__).resolve().parent
old=json.loads((HERE/'backup_clients_launched.json').read_text(encoding='utf-8'))
new=[]
for entry in old:
    assert not psutil.pid_exists(entry['pid']),'Original backup process still exists; do not duplicate it'
    control=Path(entry['local_control']).resolve(strict=True)
    assert control.parent==Path('D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1').resolve(strict=True)
    assert not (control/'backups/monitor_status.json').exists(),'The earlier client performed backup work; inspect before restarting'
    value=protect(credential_path(entry['port'],'backup').read_bytes(),decode=True).decode('utf-8')
    assert isinstance(value,str) and value;value=None
    with (control/'monitor.attempt2.stdout.log').open('xb') as out,(control/'monitor.attempt2.stderr.log').open('xb') as err:
        process=subprocess.Popen(entry['command'],cwd=str(control),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    updated={**entry,'pid':process.pid,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
             'previous_failed_local_bootstrap_pid':entry['pid'],'reason':'Use DPAPI blob created under the actual approved backup process identity; previous client stopped before any backup or training.'}
    (control/'monitor_launch_attempt2.json').write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    new.append(updated)
(HERE/'backup_clients_launched_attempt2.json').write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'backup_clients':[{'arm':x['arm'],'pid':x['pid']} for x in new],'training_actions':0,'no_experiment_retry':True}))
