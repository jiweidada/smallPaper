"""Bind the user-facing status to local backup receipts and read-only SSH proof."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from check_backup_health import observe

HERE = Path(__file__).resolve().parent
state_path = HERE / 'current_execution_state.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
now = datetime.now(timezone.utc).isoformat()
report = {'observed_local_utc': now, 'arms': {}, 'submission_ready': False}
for arm in ['NO_MPS', 'NO_MSCA']:
    evidence_path = HERE / arm / 'live_detail_20261108_2212.json'
    wrapper = json.loads(evidence_path.read_text(encoding='utf-8'))
    assert wrapper['exit_code'] == 0 and wrapper['stderr'] == ''
    remote = json.loads(wrapper['stdout'])
    detail = next(iter(remote['arms'].values()))
    supervisor = detail['supervisor.json']
    assert supervisor['status'] == detail['receipt']['status'] == 'RUNNING'
    assert supervisor['boot_id'] == remote['boot_id']
    for pid_key, ticks_key in [('pid', 'start_ticks'), ('supervisor_pid', 'supervisor_start_ticks')]:
        proof = detail[str(supervisor[pid_key])]
        assert proof['exists'] and proof['start_ticks'] == supervisor[ticks_key]
    protocol = HERE / arm / 'execution_protocol.json'
    assert sha256(protocol.read_bytes()).hexdigest() == supervisor['protocol_sha256']
    root = Path(state[arm]['backup_root'])
    health = observe(arm, state[arm], root)
    assert health['monitor_alive'] and health['status'] == 'EPOCH_BACKUP_VERIFIED'
    events = [json.loads(line) for line in (root / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
    accepted = next(item for item in reversed(events) if item['status'] == 'EPOCH_BACKUP_VERIFIED')
    state[arm].update(status='RUNNING', last_accepted_round=accepted['epoch'] + 1,
                      last_acceptance_utc=accepted['observed_utc'],
                      last_remote_live_observation={
                          'file': str(evidence_path), 'server_observed_utc': remote['observed_utc'],
                          'supervisor_status': supervisor['status'],
                          'trainer_pid': supervisor['pid'], 'boot_id': remote['boot_id'],
                          'receipt': detail['receipt'], 'gpu': remote['gpu']})
    report['arms'][arm] = {'backup': health, 'remote_observation': state[arm]['last_remote_live_observation']}
state['observed_local_utc'] = now
report['completed_main_comparison'] = {'BASE': 320, 'FULL': 320,
                                     'report': '../断点续训_20261105/BASE_FULL_pair_verified_20261108.json'}
report['manuscript_revision'] = state['paper_revision']
temporary = state_path.with_suffix('.status.tmp')
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, state_path)
with (HERE / 'experiment_status_20261108_2213.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'NO_MPS': report['arms']['NO_MPS']['backup'],
                  'NO_MSCA': report['arms']['NO_MSCA']['backup'],
                  'paper_revision': state['paper_revision']}))
