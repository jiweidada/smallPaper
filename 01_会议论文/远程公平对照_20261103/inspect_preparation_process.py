from pathlib import Path
from datetime import datetime, timezone
import json, os
root = Path('/25085400120/mpsma_fair_20261103_v1')
qpath = root/'preparation_queue_v2.json'
if not qpath.exists(): qpath=root/'preparation_queue.json'
queue = json.loads(qpath.read_text())
out = {'observed_remote_utc': datetime.now(timezone.utc).isoformat(), 'queue_status': queue['status'], 'processes': []}
for job in queue['steps']:
    if job['status'] != 'RUNNING':
        continue
    p = Path('/proc')/str(job['pid'])
    item = {'name': job['name'], 'pid': job['pid']}
    if p.exists():
        fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
        item.update(state=fields[0], start_ticks=fields[19], identity_matches=fields[19]==job['start_ticks'])
        if item['identity_matches']:
            for name in ['io', 'wchan', 'stack']:
                try: item[name] = (p/name).read_text()
                except OSError as err: item[name] = str(err)
            item['open_files'] = []
            for f in (p/'fd').iterdir():
                try: item['open_files'].append(os.readlink(f))
                except OSError: pass
    else:
        item['present'] = False
    out['processes'].append(item)
print(json.dumps(out))
