"""Read the bound preparation processes on both renewed instances."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pair


def observe(server):
    launch = json.loads((HERE / server.arm / 'preparation_launch.json').read_text())
    script = '''
import json
from pathlib import Path
root=Path(ROOT);stat=Path('/proc')/str(PID)/'stat'
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
fields=stat.read_text().rsplit(') ',1)[1].split() if stat.exists() else None
live=bool(fields and fields[0]!='Z' and int(fields[19])==TICKS and boot==BOOT)
result={'arm':ARM,'preparation_live':live,'pid':PID,'files':{},'processes':[]}
if live:
    pending=[PID]
    while pending:
        child=pending.pop(0);proc=Path('/proc')/str(child)
        if not proc.exists():continue
        children=(proc/'task'/str(child)/'children').read_text().split()
        pending.extend(int(value) for value in children)
        io=dict(line.split(': ',1) for line in (proc/'io').read_text().splitlines())
        result['processes'].append({'pid':child,'comm':(proc/'comm').read_text().strip(),
                                    'wchan':(proc/'wchan').read_text().strip(),
                                    'read_bytes':int(io.get('read_bytes',0)),'rchar':int(io.get('rchar',0))})
for name in ['preparation.log.exit','environment.json','data_preparation.json','preparation.log']:
    p=root/name
    result['files'][name]={'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None}
    if p.exists() and p.stat().st_size and name in ['preparation.log.exit','data_preparation.json','preparation.log']:
        result['files'][name]['text']=p.read_text()[-1600:]
print(json.dumps(result))
'''
    prelude = (f'ROOT={launch["remote_directory"]!r}\nPID={launch["pid"]}\nTICKS={launch["start_ticks"]}\n'
               f'BOOT={launch["boot_id"]!r}\nARM={server.arm!r}\n')
    with pair.ssh.connect(server.port) as client:
        result = pair.ssh.run_script(client, prelude + script, timeout=40)
    assert result['exit_code'] == 0, result
    return json.loads(result['stdout'])


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(observe, pair.SERVERS):
            print(json.dumps(record), flush=True)
