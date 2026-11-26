from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
root = Path('/25085400120/mpsma_fair_20261103_v1')
queue = json.loads((root/'preparation_queue.json').read_text())
assert queue['status'] == 'STOPPED_BEFORE_FORMAL_TRAINING'
assert not (root/'one_shot.lock').exists() and not (root/'runs').exists()
for pid in [queue['supervisor_pid']] + [job['pid'] for job in queue['steps']]:
    assert not Path(f'/proc/{pid}').exists(), f'Preparation process still present: {pid}'
report = json.loads((root/'preflight/BASE.json').read_text())
assert report['model_optimizer_steps'] == 0 and report['status'] == 'PREFLIGHT_FAILED_BEFORE_TRAINING'
assert "KeyError: '_pymodule'" in report['traceback']
assert not (root/'preflight/BASE.initial.pt').exists() and not (root/'preflight/FULL.json').exists()
archive = root/'preparation_attempt1'
archive.mkdir(exist_ok=False)
files = ['fair_runner.py','prepare_supervisor.py','preparation_queue.json','preparation_once.lock',
         'preparation_launch.json','preparation_supervisor.log','BASE_preflight.log']
for name in files:
    shutil.copy2(root/name, archive/name)
(root/'preflight').rename(archive/'preflight')
record = {'status':'FAILED_PREPARATION_ARCHIVED_NO_FORMAL_RUN',
          'observed_remote_utc':datetime.now(timezone.utc).isoformat(),
          'model_optimizer_steps':0,'cause':'Training-only Mosaic transform not registered by reused evaluation loader.',
          'repair':'Import the unchanged frozen engine.data.transforms.mosaic module before constructing training transforms.',
          'archived_files':{str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in archive.rglob('*') if p.is_file()}}
(archive/'failure_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
