"""Exercise only the frozen client's local cursor verification; no network or writes to D."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import psutil

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ARM = HERE.parent
STATE = ARM.parent / 'current_execution_state.json'
state = json.loads(STATE.read_text(encoding='utf-8'))
arm = state['NO_MSCA']
root = Path(arm['backup_root'])
client_path = root.parent / 'backup_monitor.py'
protocol_path = root.parent / 'backup_protocol.py'
plan_path = root.parent / 'execution_protocol.json'
digest = lambda p: sha256(p.read_bytes()).hexdigest()
expected = {
    client_path: '4e5c38bb7dcf4dbd8131157b34f2637de7e45379796b75c9e211a63a9f25b37b',
    protocol_path: '2a7bf275be22108cfda5753e660a78d8bbc86bbecc37bd185294f1b7d6a27a8f',
    plan_path: '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'}
for path, expected_sha in expected.items():
    assert digest(path) == expected_sha
try:
    old_process = psutil.Process(arm['monitor_pid'])
    old_alive = abs(old_process.create_time() - arm['monitor_create_time']) < 0.001
except psutil.Error:
    old_alive = False
assert not old_alive
status_path, events_path = root / 'monitor_status.json', root / 'monitor_events.jsonl'
before = {path: digest(path) for path in [status_path, events_path]}
sys.path.insert(0, str(client_path.parent))
spec = importlib.util.spec_from_file_location('frozen_backup_cursor_readonly_check', client_path)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
store = client.BackupStore(root, client_path.parent)
# setup, protect_local_lock, emit, connect and watch deliberately are not called.
# verify_progress verifies existing manifests, acknowledgements, accepted receipts,
# hard-linked objects and their hashes, updating only an in-memory hash cache.
cursor = store.verify_progress('no_msca_20261107_v1')
assert cursor == 29
assert all(digest(path) == expected_sha for path, expected_sha in before.items())
result = {
    'status': 'FROZEN_BACKUP_CLIENT_LOCAL_RESUME_CURSOR_VERIFIED',
    'observed_local_utc': datetime.now(timezone.utc).isoformat(),
    'protocol_sha256': digest(plan_path), 'backup_client_sha256': digest(client_path),
    'backup_protocol_sha256': digest(protocol_path),
    'accepted_training_rounds': 29, 'next_expected_backup_epoch_index': cursor,
    'next_expected_backup_round': cursor + 1, 'checked_object_hash_count': len(store.verified),
    'monitor_metadata_sha256_unchanged': {str(p): h for p, h in before.items()},
    'old_monitor_identity_alive': False,
    'evidence_scope': 'The actual frozen backup-client function can reconstruct its local progress cursor through the probe and all 29 accepted epochs. This proves neither remote process liveness nor the presence of an epoch-30 checkpoint.',
    'exclusive_lock_reacquisition_tested': False, 'remote_endpoint_checked_by_this_script': False,
    'network_connections': 0, 'acknowledgements_written': 0, 'backup_client_started': False,
    'training_started_stopped_or_retried': False, 'model_forwards': 0, 'evaluator_runs': 0,
    'remaining': 'A working authorized SSH endpoint and a fresh check of the original remote process and project are required before client recovery.'}
output = HERE / 'backup_client_resume_preflight_round29.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['status', 'accepted_training_rounds', 'next_expected_backup_round',
                                       'checked_object_hash_count', 'backup_client_started', 'network_connections']}))
