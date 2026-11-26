"""Record an observed epoch boundary and separately label the frozen-policy interpretation."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json
import time

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent / '修订记录_20261103/增强时序核查_20261104/augmentation_schedule_audit.json'
PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

ledger_path = ROOT / 'parallel_execution_ledger.json'
ledger = read(ledger_path)
assert ledger['protocol_sha256'] == PROTOCOL
base = ledger['arms']['BASE']
assert base['status'] == 'RUNNING' and base['pid'] == 1194 and base['start_ticks'] == '39931668'
assert base['completed_epochs'] >= 156 and base['actual_loop_epoch'] >= 156
copies = {}
for arm, filename, expected_pid, expected_ticks in [
    ('BASE', 'primary_parallel_live.json', 1194, '39931668'),
    ('FULL', 'secondary_parallel_live.json', 567, '42791005')
]:
    path = ROOT / filename
    assert time.time() - path.stat().st_mtime < 180, 'Fresh snapshots required'
    assert sha(path) == ledger['snapshots'][arm]['sha256']
    wrapper = read(path)
    assert wrapper['exit_code'] == 0
    snapshot = json.loads(wrapper['stdout'])
    assert snapshot['protocol_sha256'] == PROTOCOL
    proc = next(p for p in snapshot['processes'] if p['pid'] == expected_pid)
    assert proc['live'] and proc['start_ticks'] == expected_ticks
    assert snapshot['critical_frozen_files_match'] and all(snapshot['critical_frozen_files_match'].values())
    snapshot_name = 'BASE_epoch0156_boundary_window_snapshot.json' if arm == 'BASE' else 'FULL_at_BASE_epoch0156_snapshot.json'
    destination = ROOT / 'parallel_evidence' / snapshot_name
    assert not destination.exists()
    with destination.open('xb') as handle:
        handle.write(path.read_bytes())
    copies[arm] = {'path': str(destination), 'sha256': sha(destination),
                   'observed_remote_utc': snapshot['observed_remote_utc']}

assert sha(AUDIT) == '1293e66dac4e6e71429e4edc837b573c1619c991c9bb41c653eddd6f813a9668'
augmentation = read(AUDIT)
assert augmentation['mosaic_selector']['stop_exclusive'] == 156
assert augmentation['batch_mixup']['stop_exclusive'] == 156
for source in augmentation['source_files']:
    assert sha(Path(source['path'])) == source['sha256']
assert augmentation['other_policy_ops']['eligible_stop_exclusive'] == 296
assert base['optimizer_step_attempts'] > 156 * 1617
report = {
    'status': 'BASE_REACHED_SCHEDULED_AUGMENTATION_BOUNDARY',
    'recorded_local_utc': datetime.now(timezone.utc).isoformat(),
    'protocol_sha256': PROTOCOL, 'observed_training_states': ledger['arms'],
    'observed_snapshot_copies': copies,
    'zero_based_epoch_boundary': 156,
    'actual_base_update_attempts_after_boundary': base['optimizer_step_attempts'] - 156 * 1617,
    'augmentation_source_audit': {'path': str(AUDIT), 'sha256': sha(AUDIT)},
    'policy_interpretation': 'In the frozen configuration and inspected implementation, Mosaic selection and batch MixUp are eligible at zero-based epochs 4-155. From epoch156 they are disabled; other specified strong transforms remain eligible until epoch295.',
    'direct_transform_events_observed': False,
    'scope': 'Fresh process identity, actual epoch and update-attempt observation plus separately identified static policy semantics. Not a direct audit of every sampled transform, completed post-boundary epoch, full training or method benefit.',
    'new_model_executions_for_this_observation': 0, 'evaluator_reruns': 0,
    'training_settings_changed': False, 'submission_ready': False
}
target = ROOT / 'BASE_epoch0156_boundary_observation.json'
assert not target.exists()
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'BASE_completed': base['completed_epochs'],
    'BASE_actual_loop': base['actual_loop_epoch'],
    'BASE_post_boundary_update_attempts': report['actual_base_update_attempts_after_boundary'],
    'FULL_completed': ledger['arms']['FULL']['completed_epochs'],
    'direct_transform_events_observed': False}, ensure_ascii=False))
