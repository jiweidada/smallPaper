import base64
import hashlib
import json
import os
import pathlib
import signal
import time

PAYLOAD = {}  # INSERT
account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
storage = pathlib.Path('/25085400120' if account == 'A' else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
stage = phase/'artifacts/round2_20261014_v1'
if account == 'B':
    launch = json.loads((stage/'launch_receipt.json').read_text())
    pid = launch['pid']
    proc = pathlib.Path('/proc', str(pid))
    current_ticks = int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
    cmd = (proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
    state = json.loads((stage/'queue_state.json').read_text())
    assert current_ticks == launch['start_ticks']
    assert str(stage/'addons/run_round2_queue.py') in cmd
    assert state['state'] == 'waiting_existing_experiments' and not state['completed_jobs']
    assert not (phase/'jobs/c_mean_full96_seed1.json').exists()
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not proc.exists() or (proc/'stat').read_text().rsplit(')',1)[1].split()[0] == 'Z':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Waiting controller did not terminate; no new queue launched')
    state.update(state='superseded_before_new_training', superseded_reason='Corrected SOEP-lite parameter-count assertion; no model code changed', ended=time.time())
    (stage/'queue_state.json').write_text(json.dumps(state,indent=2))
addons = stage/'addons_v2'
addons.mkdir(exist_ok=False)
(stage/'execution_v2').mkdir(exist_ok=False)
for name,row in PAYLOAD.items():
    data = base64.b64decode(row['base64'])
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    with (addons/name).open('xb') as f:
        f.write(data)
with (addons/'queue_plan.json').open('xb') as f:
    f.write((stage/'addons/queue_plan.json').read_bytes())
with (addons/'deployment_manifest.json').open('x') as f:
    json.dump({name:row['sha256'] for name,row in PAYLOAD.items()}, f, indent=2)
assert hashlib.sha256((phase/'code/engine/deim/soep_lite_round2.py').read_bytes()).hexdigest() == PAYLOAD['soep_lite_round2.py']['sha256']
print(json.dumps({'account':account,'status':'v2_appended','old_training_untouched':True,'model_source_unchanged':True,'addons':str(addons)},indent=2))
