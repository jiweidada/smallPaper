"""Run once after both fixed12 workers exit; source/process audit, no torch import."""
from pathlib import Path
import hashlib
import json
import sys
import time

P = Path(__file__).resolve().parent
PROTOCOL_SHA = 'da3ae8ac399483307eeb7e221627d0bdaef4222021dde468cb41459d674915a9'


def load(path):
    return json.loads(path.read_text('utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def process(pid, ticks):
    path = Path('/proc')/str(pid)/'stat'
    if not path.exists():
        return {'pid': pid, 'owned_live': False, 'reason': 'missing'}
    fields = path.read_text().split(') ', 1)[1].split()
    match = int(fields[19]) == ticks
    return {'pid': pid, 'start_ticks': int(fields[19]), 'identity_matches': match,
            'state': fields[0], 'owned_live': match and fields[0] not in ['Z', 'X']}


def main():
    assert sys.platform.startswith('linux') and P == Path('/root/uav_a56_20261102_v1')
    target = P/'terminal_frozen_input_audit_A56.json'
    assert not target.exists()
    assert sha(P/'protocol_FROZEN_v1.json') == PROTOCOL_SHA
    plan = load(P/'protocol_FROZEN_v1.json')
    final = load(P/'formal_exit.json')
    assert final['status'] == 'COMPLETED_BOTH_FIXED12' and final['post_Q_queue'] is None
    launch = load(P/'formal_launch_receipt.json')
    jobs = [process(launch['supervisor_PID'], launch['supervisor_start_ticks'])]
    for arm in ['M', 'Q']:
        record = next(r for r in final['records'] if r['arm'] == arm)
        assert record['child_exit_code'] == 0 and not record['timed_out']
        jobs.append(process(record['child_PID'], record['child_start_ticks']))
        result = load(P/f'arm_{arm}/result.json')
        assert result['status'] == 'COMPLETED' and result['epochs'] == 12
        assert result['optimizer_updates'] == 9708 and result['full_model_forwards'] == 10159
        assert result['protocol_sha256'] == PROTOCOL_SHA
    assert all(not j['owned_live'] for j in jobs), jobs
    own = {n: sha(P/n) == value for n, value in plan['files_sha256'].items()}
    external = {n: sha(Path(n)) == value for n, value in plan['external_sha256'].items()}
    assert len(own) == 8 and len(external) == 92 and all(own.values()) and all(external.values())
    cap = load(P/'capacity_result.json')
    assert cap['status'] == 'PASS' and cap['whole_model_forwards'] == 9
    assert cap['capacity_only_optimizer_updates'] == 6
    assert sha(P/'capacity_result.json') == plan['capacity_result_sha256']
    assert sha(P/'capacity_protocol_FROZEN_v1.json') == plan['capacity_protocol_sha256']
    assert sha(P/'capacity_exit.json') == plan['capacity_exit_sha256']
    endpoints = {a: {n: {'sha256': sha(P/f'arm_{a}/{n}'),
                          'bytes': (P/f'arm_{a}/{n}').stat().st_size}
                     for n in ['result.json', 'epoch12.pth', 'epoch12_EMA_metrics.json',
                               'epoch12_EMA_GT.json', 'epoch12_EMA_predictions.json']}
                 for a in ['M', 'Q']}
    result = {'status': 'PASS', 'scope': 'terminal owned-process and frozen-input audit only',
              'server_observation_unix': time.time(), 'protocol_sha256': PROTOCOL_SHA,
              'own_frozen_files': own, 'external_frozen_files': external,
              'owned_processes': jobs, 'fixed12_endpoints': endpoints,
              'A56_formal_forwards': 20318, 'A56_capacity_forwards': 9,
              'A56_formal_optimizer_updates': 19416, 'A56_capacity_optimizer_updates': 6,
              'new_detector_forwards_by_audit': 0, 'new_optimizer_updates_by_audit': 0,
              'auditor_sha256': sha(Path(__file__))}
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'status': 'PASS', 'own_files': len(own), 'external_files': len(external),
                      'owned_jobs_not_live': True, 'audit_sha256': sha(target)}))


if __name__ == '__main__':
    main()
