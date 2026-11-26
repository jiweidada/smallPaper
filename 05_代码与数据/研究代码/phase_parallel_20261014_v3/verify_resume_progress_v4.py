import datetime
import hashlib
import json
import pathlib
import re
import shutil

account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A'
                     else '/24085403014/experiments/phase_mechanism_20261012')
stage = phase/'artifacts/round2_20261014_v1/execution_v4'
plan = json.loads((stage/'resume_plan.json').read_text())
state = json.loads((stage/'queue_state.json').read_text())
result = {'account': account, 'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'queue': state, 'jobs': [], 'disk_free_GiB': shutil.disk_usage(phase).free/1024**3}
for row in plan['resume_jobs']:
    job = row['job']
    log = phase/'logs'/(job+'.log')
    raw = log.read_bytes()
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw.decode(errors='replace'))
    resumed = text.rsplit('RESUME_V4 ', 1)[1]
    steps = []
    for line in resumed.splitlines():
        m = re.search(r'Epoch:\s*(\d+)/96\s+\[\s*(\d+)/1617\]', line)
        if m:
            steps.append({'epoch_zero_based': int(m[1]), 'iteration': int(m[2]), 'line': line})
    restored = {k: ('Load '+k+'.state_dict') in resumed for k in ['model', 'optimizer', 'ema', 'scaler', 'lr_warmup_scheduler']}
    expected = row['last_epoch_zero_based']+1
    correct = bool(steps and steps[0]['epoch_zero_based'] == expected and steps[-1]['iteration'] >= 80)
    item = {'job': job, 'restored': restored, 'expected_epoch_zero_based': expected,
            'new_first_step': steps[0] if steps else None, 'new_latest_step': steps[-1] if steps else None,
            'source_checkpoint_sha256': row['sha256'], 'resume_progress_verified': correct and all(restored.values()),
            'log_sha256': hashlib.sha256(raw).hexdigest(), 'log_bytes': len(raw),
            'tail': resumed.splitlines()[-4:]}
    result['jobs'].append(item)
result['all_four_states_restored_and_progressing_on_this_account'] = all(j['resume_progress_verified'] for j in result['jobs'])
print(json.dumps(result, ensure_ascii=False))
