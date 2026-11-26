"""Refresh the local receipt cache; never launch, stop, or retry training."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import uuid
import psutil

HERE = Path(__file__).resolve().parent
state_path = HERE / 'current_execution_state.json'
original = state_path.read_bytes()
state = json.loads(original.decode('utf-8-sig'))
arm = state['NO_MSCA']
root = Path(arm['backup_root'])
process = psutil.Process(arm['monitor_pid'])
assert abs(process.create_time() - arm['monitor_create_time']) < 0.001
events = [json.loads(line) for line in (root / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
accepted = next(e for e in reversed(events) if e['status'] == 'EPOCH_BACKUP_VERIFIED')
status = json.loads((root / 'monitor_status.json').read_text(encoding='utf-8'))
round_number = accepted['epoch'] + 1
assert arm['last_accepted_round'] <= round_number <= 320
assert arm['independent_lineage_audit_round'] <= round_number
now = datetime.now(timezone.utc).isoformat()
arm.update(last_accepted_round=round_number, last_acceptance_utc=accepted['observed_utc'])
arm['last_local_backup_observation'] = {
    'observed_local_utc': now, 'monitor_pid': process.pid,
    'monitor_create_time': process.create_time(), 'monitor_identity_verified': True,
    'transport_verified_round': round_number, 'independent_lineage_verified_round': arm['independent_lineage_audit_round'],
    'monitor_status': status['status'], 'manifest_sha256': accepted['manifest_sha256'],
    'scope': 'Current local client identity and accepted transport receipt only; does not replace saved-state lineage audit or a live remote process observation.'}
state['observed_local_utc'] = now
temporary = state_path.with_name('current_execution_state.' + uuid.uuid4().hex + '.tmp')
assert sha256(state_path.read_bytes()).digest() == sha256(original).digest()
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, state_path)
print(json.dumps({'status': 'LOCAL_BACKUP_OBSERVATION_REFRESHED', 'NO_MSCA_transport_verified_round': round_number,
                  'NO_MSCA_independent_lineage_audit_round': arm['independent_lineage_audit_round'],
                  'monitor_identity_verified': True, 'training_action': None}))
