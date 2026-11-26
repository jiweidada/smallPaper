"""Compare old live-process durations with the current read-only kernel observations."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'kernel_epoch_comparison_20261105.json'
PREVIOUS = {'BASE': ROOT / 'parallel_evidence/BASE_before_connection_loss_20261104.json',
            'FULL': ROOT / 'parallel_evidence/FULL_before_connection_loss_20261104.json'}
CURRENT = {'BASE': ROOT / 'current_kernel_primary_20261105.json',
           'FULL': ROOT / 'current_kernel_secondary_20261105.json'}

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def remote(path):
    wrapper = json.loads(path.read_text(encoding='utf-8-sig'))
    assert wrapper['exit_code'] == 0 and not wrapper['stderr']
    return json.loads(wrapper['stdout'])

assert not REPORT.exists()
clock_audit = json.loads((ROOT / 'clock_observation_20261104.json').read_text(encoding='utf-8'))
assert sha(ROOT / 'supervise.py') == clock_audit['static_timeout_evidence']['BASE_supervisor_sha256']
assert sha(ROOT / 'supervise_secondary_full.py') == clock_audit['static_timeout_evidence']['FULL_supervisor_sha256']
comparison = {}
current = {a: remote(path) for a, path in CURRENT.items()}
assert current['BASE']['boot_id'] == current['FULL']['boot_id']
assert abs(current['BASE']['uptime_seconds'] - current['FULL']['uptime_seconds']) < 2
for arm in ['BASE', 'FULL']:
    old = remote(PREVIOUS[arm])
    q = old['queue']
    run = q['runs'][0]
    assert run['arm'] == arm and run['status'] == 'RUNNING'
    old_identity = next(x for x in old['processes'] if x['pid'] == run['pid'])
    assert old_identity['live'] and old_identity['start_ticks'] == run['start_ticks']
    now_identity = next(x for x in current[arm]['previous_processes'] if x['pid'] == run['pid'])
    assert not now_identity['live'] and not now_identity['same_as_previous']
    elapsed = run['elapsed_seconds']
    uptime = current[arm]['uptime_seconds']
    assert elapsed > 10 * uptime
    comparison[arm] = {'last_verified_snapshot': {'path': str(PREVIOUS[arm]), 'sha256': sha(PREVIOUS[arm])},
                       'current_kernel_snapshot': {'path': str(CURRENT[arm]), 'sha256': sha(CURRENT[arm])},
                       'old_process_pid': run['pid'], 'old_process_start_ticks': run['start_ticks'],
                       'old_supervisor_reported_elapsed_seconds': elapsed,
                       'current_kernel_uptime_seconds': uptime,
                       'old_process_present_in_current_environment': False}
report = {'status': 'CURRENT_KERNEL_EPOCH_DIFFERS_FROM_OLD_LIVE_PROCESS_CONTEXT',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'shared_current_boot_id': current['BASE']['boot_id'],
          'current_hostnames': {a: current[a]['hostname'] for a in current},
          'comparison': comparison,
          'timer_source_evidence': {'primary_supervisor_sha256': sha(ROOT / 'supervise.py'),
                                    'secondary_supervisor_sha256': sha(ROOT / 'supervise_secondary_full.py'),
                                    'scope': 'The recorded elapsed_seconds in both inspected supervisors is computed with time.monotonic().'},
          'interpretation': 'Both currently exposed environments report the same boot ID and about 44 minutes of uptime at observation; old supervisors had reported runs of over 21 hours. This is consistent with a restarted, migrated, or otherwise replaced kernel/time-namespace context, and corroborates the missing original process identities.',
          'limits': ['The original boot IDs and original clock-tick frequencies were not saved.',
                     'Kernel migration, host restart, namespace changes and platform rescheduling are not distinguished by these observations.',
                     'A common currently exposed boot ID does not, by itself, prove physical-host identity or the earlier runs’ fault domains.',
                     'Remote wall-clock timestamps were not compared as exact UTC because the previously observed local/remote offset was nonzero.'],
          'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0,
          'server_configuration_changed': False}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'], 'same_current_boot_id': True,
                  'old_elapsed_seconds': {a: comparison[a]['old_supervisor_reported_elapsed_seconds'] for a in comparison},
                  'current_uptime_seconds': {a: comparison[a]['current_kernel_uptime_seconds'] for a in comparison}}))
