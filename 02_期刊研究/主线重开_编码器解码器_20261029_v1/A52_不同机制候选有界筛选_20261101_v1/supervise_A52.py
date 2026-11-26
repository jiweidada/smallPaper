from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def ticks(pid):
    try:
        return int(Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()[19])
    except FileNotFoundError:
        return None


if __name__ == '__main__':
    with (ROOT / 'supervisor_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    capacity = json.loads((ROOT / 'capacity_v2/capacity_result.json').read_text())
    assert capacity['status'] == 'PASS'
    save(ROOT / 'supervisor_identity.json', {'pid': os.getpid(), 'start_ticks': ticks(os.getpid()),
        'stages': ['M12', 'Q12'], 'B12_reused': '/root/uav_a49_20261101_v1/arm_B', 'auto_retry': False, 'capacity_repeated': False,
        'additional_training_queue': False})
    try:
        for arm in ['M', 'Q']:
            folder = ROOT / ('arm_' + arm)
            folder.mkdir(exist_ok=True)
            with (folder / 'stdout.log').open('x') as stdout, (folder / 'stderr.log').open('x') as stderr:
                child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'train_A52.py'), '--arm', arm],
                    cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
                identity = {'pid': child.pid, 'start_ticks': ticks(child.pid), 'arm': arm, 'unix': time.time()}
                save(folder / 'child_identity.json', identity)
                start = time.monotonic()
                while child.poll() is None:
                    save(ROOT / 'supervisor_progress.json', {'status': 'RUNNING', 'stage': arm + '12',
                        'child': identity, 'elapsed_stage_seconds': time.monotonic() - start,
                        'supervisor_pid': os.getpid()})
                    if time.monotonic() - start > 6 * 3600:
                        assert ticks(child.pid) == identity['start_ticks']
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL); child.wait()
                        raise TimeoutError('Own A52 stage timeout: ' + arm)
                    time.sleep(10)
                save(folder / 'exit.json', {'exit_code': child.returncode, 'identity': identity, 'unix': time.time()})
                if child.returncode:
                    raise RuntimeError(arm + ' failed with exit ' + str(child.returncode))
        rows = {a: json.loads((ROOT / ('arm_' + a) / 'result.json').read_text()) for a in ['M', 'Q']}
        rows['B'] = json.loads(Path('/root/uav_a49_20261101_v1/arm_B/result.json').read_text())
        assert all(r['status'] == 'COMPLETED' and r['epochs'] == 12 and r['optimizer_updates'] == 9708 for r in rows.values())
        for epoch in range(1, 13):
            flows = {a: json.loads((ROOT / ('arm_' + a) / ('epoch%02d_input_flow.json' % epoch)).read_text()) for a in ['M','Q']}
            flows['B'] = json.loads((Path('/root/uav_a49_20261101_v1/arm_B') / ('epoch%02d_input_flow.json' % epoch)).read_text())
            for key in ['order_sha256', 'first_eight_actual_input_sha256', 'first_eight_pre_forward_rng_sha256',
                        'all_batches_targets_including_ignore_tiles_sha256']:
                assert flows['B'][key] == flows['M'][key] == flows['Q'][key], (epoch, key)
        metrics = {a: r['endpoint_metrics'] for a, r in rows.items()}
        qb = metrics['Q']['APs'] - metrics['B']['APs']
        qm = metrics['Q']['APs'] - metrics['M']['APs']
        qa = metrics['Q']['AP'] - metrics['B']['AP']
        gates = {'Q_minus_B_APs_ge_0_5pp': qb >= .5, 'Q_minus_M_APs_ge_0_3pp': qm >= .3,
                 'Q_minus_B_AP_ge_minus_0_2pp': qa >= -.2}
        result = {'status': 'COMPLETED', 'metrics': metrics, 'gates': gates, 'pilot_pass': all(gates.values()),
            'Q_minus_B_APs_pp': qb, 'Q_minus_M_APs_pp': qm, 'Q_minus_B_AP_pp': qa,
            'decision': 'CANDIDATE_FOR_FURTHER_ASSESSMENT_NOT_NOVELTY_PROOF' if all(gates.values()) else 'RETIRE_EXACT_RECIPE_NO_RESCUE_SWEEP',
            'new_formal_model_forwards': sum(rows[a]['full_model_forwards'] for a in ['M','Q']),
            'B_reused_model_forwards_not_new': rows['B']['full_model_forwards'],
            'new_formal_optimizer_updates': sum(rows[a]['optimizer_updates'] for a in ['M','Q']),
            'B_reused_optimizer_updates_not_new': rows['B']['optimizer_updates'],
            'common_initialization_input_rng_including_ignore_metadata_checks': 'PASS',
            'additional_training_queued': False, 'scope': 'One-seed fixed12 allocation pilot, not significance or completed paper.'}
        save(ROOT / 'three_arm_result.json', result)
        save(ROOT / 'supervisor_progress.json', result)
    except BaseException as exc:
        result = {'status': 'FAILED_STOPPED_NO_RETRY', 'error': repr(exc), 'traceback': traceback.format_exc(),
                  'unix': time.time(), 'additional_training_queued': False}
        save(ROOT / 'supervisor_progress.json', result)
        raise
