"""Record a bounded, read-only comparison of remote and local wall clocks.

Call `before`, obtain fresh read-only snapshots on both existing SSH sessions,
then call `after`. This does not synchronize clocks or execute a model.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
BEFORE = ROOT / 'clock_observation_before.json'
REPORT = ROOT / 'clock_observation_20261104.json'
SNAPSHOTS = {'BASE': ROOT / 'clock_observation_primary.json',
             'FULL': ROOT / 'clock_observation_secondary.json'}
PROTOCOL = ROOT / 'execution_protocol.json'
PROTOCOL_SHA = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
SECONDARY_SUPERVISOR_SHA = '2df1c11e6b5356b92e892683ce261c2bc46d90628b4f3f64b4abb56f858dc895'

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def exclusive_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def point():
    # Capture both local clocks together to detect a local wall-clock jump.
    monotonic_ns = time.monotonic_ns()
    wall_ns = time.time_ns()
    return {'wall_ns': wall_ns, 'monotonic_ns': monotonic_ns,
            'wall_utc': datetime.fromtimestamp(wall_ns / 1e9, timezone.utc).isoformat()}

assert len(sys.argv) == 2 and sys.argv[1] in ['before', 'after']
assert digest(PROTOCOL) == PROTOCOL_SHA
frozen = {x['path']: x['sha256'] for x in load(PROTOCOL)['frozen_files']}
for name in ['supervise.py', 'fair_runner.py']:
    assert digest(ROOT / name) == frozen[name]
assert digest(ROOT / 'supervise_secondary_full.py') == SECONDARY_SUPERVISOR_SHA

if sys.argv[1] == 'before':
    assert not BEFORE.exists() and not REPORT.exists() and not any(p.exists() for p in SNAPSHOTS.values())
    p = point()
    exclusive_json(BEFORE, p)
    print(json.dumps({'status': 'LOCAL_BRACKET_STARTED', **p}))
    raise SystemExit(0)

end = point()
assert not REPORT.exists()
start = load(BEFORE)
wall_elapsed = (end['wall_ns'] - start['wall_ns']) / 1e9
mono_elapsed = (end['monotonic_ns'] - start['monotonic_ns']) / 1e9
assert 0 < wall_elapsed < 60 and abs(wall_elapsed - mono_elapsed) < 0.5
sources = {}
for arm, port, uuid, pid, ticks in [
    ('BASE', 30166, 'GPU-85eca9af-10f3-4279-d72a-05ff1ecdcd3a', 1194, '39931668'),
    ('FULL', 30989, 'GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a', 567, '42791005')]:
    path = SNAPSHOTS[arm]
    assert start['wall_ns'] <= path.stat().st_mtime_ns <= end['wall_ns']
    wrapper = load(path)
    assert wrapper['exit_code'] == 0 and not wrapper['stderr']
    s = json.loads(wrapper['stdout'])
    q = s['queue']
    assert s['host_port'] == port and s['gpu'].startswith(uuid + ',')
    assert s['protocol_sha256'] == PROTOCOL_SHA and all(s['critical_frozen_files_match'].values())
    assert s['reserved_log_matches'] and not s['unexpected_other_arm_directory']
    assert len(q['runs']) == 1 and q['runs'][0]['arm'] == arm and q['runs'][0]['pid'] == pid
    run = q['runs'][0]
    assert run['status'] == 'RUNNING' and run['start_ticks'] == ticks
    for expected_pid, expected_ticks in [(pid, ticks), (q['supervisor_pid'], q['supervisor_start_ticks'])]:
        identity = next(p for p in s['processes'] if p['pid'] == expected_pid)
        assert identity['live'] and identity['start_ticks'] == expected_ticks
    if arm == 'FULL':
        assert s['supervise_secondary_full.py_sha256'] == SECONDARY_SUPERVISOR_SHA
    remote = datetime.fromisoformat(s['observed_remote_utc']).timestamp()
    sources[arm] = {'port': port, 'snapshot_path': str(path), 'snapshot_sha256': digest(path),
                    'observed_remote_utc': s['observed_remote_utc'],
                    'remote_minus_local_seconds_bounds': [round(remote - end['wall_ns'] / 1e9, 3),
                                                           round(remote - start['wall_ns'] / 1e9, 3)],
                    'train_pid': pid, 'train_start_ticks': ticks,
                    'completed_epochs': s['arms'][arm]['receipt']['completed_epochs']}
    assert sources[arm]['remote_minus_local_seconds_bounds'][1] < 0

result = {'status': 'REMOTE_CLOCKS_BEHIND_LOCAL_AT_OBSERVATION',
          'scope': 'Two snapshots bracketed by this local computer wall clock; this is not an NTP calibration or proof of prior offsets.',
          'local_before': start, 'local_after': end,
          'local_wall_elapsed_seconds': round(wall_elapsed, 6),
          'local_monotonic_elapsed_seconds': round(mono_elapsed, 6),
          'offset_definition': 'remote observed wall time minus local wall time; inclusive bounds use the common before/after bracket and include communication and scheduling delay',
          'sources': sources, 'protocol_sha256': PROTOCOL_SHA,
          'static_timeout_evidence': {'BASE_supervisor_sha256': digest(ROOT / 'supervise.py'),
                                      'FULL_supervisor_sha256': digest(ROOT / 'supervise_secondary_full.py'),
                                      'finding': 'Both supervisors measure their per-arm 604800-second limit with time.monotonic(), not subtraction of local and remote wall timestamps.'},
          'snapshot_freshness_evidence': {'script_sha256': digest(ROOT / 'record_parallel_snapshot.py'),
              'finding': 'The current snapshot freshness guard compares local time with local receipt file modification time; remote observed UTC is recorded separately.'},
          'final_pair_audit_evidence': {'script_sha256': digest(ROOT / 'summarize_split_pair.py'),
              'finding': 'The final pair audit binds epochs, attempts, saved checkpoints and metrics; the inspected code does not use remote-versus-local wall-clock offsets as an endpoint criterion.'},
          'limitations': ['The present offset cannot be applied retroactively or assumed constant.',
                          'A static code reading does not prove whether wall-clock changes affected any other uninspected process or historical experiment.',
                          'The clock comparison alone does not verify model metrics or final training completion.'],
          'model_executions': 0, 'evaluator_executions': 0, 'clock_changes': 0, 'training_settings_changed': False}
exclusive_json(REPORT, result)
print(json.dumps({'status': result['status'], 'bounds': {k: v['remote_minus_local_seconds_bounds'] for k, v in sources.items()},
                  'local_bracket_seconds': result['local_wall_elapsed_seconds']}))
