from __future__ import annotations
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from runtime_common import R, digest, save_json


def ticks(pid):
    try:
        return int((__import__('pathlib').Path('/proc/%s/stat' % pid)).read_text().split(') ', 1)[1].split()[19])
    except FileNotFoundError:
        return None


def summarize():
    rows = {a: json.loads((R / ('arm_' + a) / 'result.json').read_text()) for a in ['B', 'M', 'Q']}
    assert all(v['status'] == 'COMPLETED' and v['epochs'] == 12 and v['optimizer_updates'] == 9708 for v in rows.values())
    input_checks = []
    for epoch in range(1, 13):
        e = {a: json.loads((R / ('arm_' + a) / ('epoch%02d_input_flow.json' % epoch)).read_text()) for a in rows}
        for a in ['M', 'Q']:
            assert e['B']['order_sha256'] == e[a]['order_sha256']
            assert e['B']['first_eight_actual_input_sha256'] == e[a]['first_eight_actual_input_sha256']
        assert e['M']['teacher_all_batches_flow_sha256'] == e['Q']['teacher_all_batches_flow_sha256']
        assert e['M']['teacher_matched_instances'] == e['Q']['teacher_matched_instances']
        input_checks.append({'epoch': epoch, 'sampler_order_and_first_eight_actual_inputs_same': True,
                             'all_M_Q_teacher_targets_and_instance_assignments_same': True})
    ini = {a: json.loads((R / ('arm_' + a) / 'initialization.json').read_text()) for a in rows}
    assert len({v['common_init_sha256'] for v in ini.values()}) == 1
    assert ini['M']['auxiliary_initial_tensor_digest'] == ini['Q']['auxiliary_initial_tensor_digest']
    for a in ['M', 'Q']:
        assert rows[a]['teacher_frozen_unchanged'] and rows[a]['inference_auxiliary_removed']
    assert rows['M']['teacher_state_start_digest'] == rows['Q']['teacher_state_start_digest']
    metrics = {a: rows[a]['endpoint_metrics'] for a in rows}
    qb = metrics['Q']['APs'] - metrics['B']['APs'];qm = metrics['Q']['APs'] - metrics['M']['APs']
    ap_change = metrics['Q']['AP'] - metrics['B']['AP']
    gates = {'Q_minus_B_APs_at_least_0_5pp': qb >= .5,
             'Q_minus_M_APs_at_least_0_3pp': qm >= .3,
             'overall_AP_drop_no_more_than_0_2pp': ap_change >= -.2}
    passed = all(gates.values())
    result = {'status': 'COMPLETED', 'arms': metrics, 'Q_minus_B_APs_pp': qb,
              'Q_minus_M_APs_pp': qm, 'Q_minus_B_AP_pp': ap_change,
              'Q_minus_B_small_AP75_pp': metrics['Q']['APs75'] - metrics['B']['APs75'],
              'Q_minus_M_small_AP75_pp': metrics['Q']['APs75'] - metrics['M']['APs75'],
              'latency_median_ms': {a: rows[a]['latency']['median_ms'] for a in rows},
              'practical_gates': gates, 'pilot_pass': passed,
              'decision': ('WORTH_NEXT_VALIDATION_NOT_ESTABLISHED_INNOVATION' if passed else
                           'STOP_DIRECT_QUERY_SUPERVISION_CLAIM_NO_RESCUE_SWEEP'),
              'interpretation': 'Single-seed12-epoch resource-priority pilot; not significance, mechanism or novelty proof.',
              'optimizer_updates': sum(r['optimizer_updates'] for r in rows.values()),
              'student_forwards': sum(r['full_model_forwards'] for r in rows.values()),
              'teacher_forwards': sum(r['teacher_forwards'] for r in rows.values()),
              'input_flow_verification': input_checks,
              'protocol_sha256': digest(R / 'execution_protocol_FROZEN.json'),
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save_json(R / 'three_arm_result.json', result)
    lines = ['VisDrone + ECDet-S 训练期蒸馏三组结果',
             '三组各12轮，固定12轮EMA，单seed，COCO-style，非官方VisDrone榜单分数。',
             '组别\tAP\tAPs\t小目标AP75']
    for a, m in metrics.items():
        lines.append('%s\t%.6f\t%.6f\t%.6f' % (a, m['AP'], m['APs'], m['APs75']))
    lines += ['Q-B APs：%+.6fpp' % qb, 'Q-M APs：%+.6fpp' % qm,
              '判断：' + ('达到下一阶段投入门槛，尚非贡献成立。' if passed else '未过预设门槛，停止直接query监督主张，不追加调参。'),
              'M/Q完整教师匹配与两类目标的逐批累计摘要一致；三组每轮前8个实际输入批次摘要一致。',
              '无新方向自动训练队列。']
    (R / '三组结果摘要.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main():
    with (R / 'supervisor_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    save_json(R / 'supervisor_started.json', {'pid': os.getpid(), 'start_ticks': ticks(os.getpid()),
        'started_unix': time.time(), 'protocol_sha256': digest(R / 'execution_protocol_FROZEN.json'),
        'queue': ['B', 'M', 'Q'], 'max_epochs_each': 12, 'timeout_seconds_per_arm': 20 * 3600,
        'auto_retry': False, 'new_direction_training_queued': False})
    try:
        for arm in ['B', 'M', 'Q']:
            folder = R / ('arm_' + arm);folder.mkdir(exist_ok=True)
            with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
                p = subprocess.Popen([sys.executable, '-u', str(R / 'train_arm.py'), '--arm', arm],
                                     cwd=str(R), stdout=stdout, stderr=stderr, start_new_session=True)
                started = time.monotonic()
                identity = {'pid': p.pid, 'start_ticks': ticks(p.pid), 'arm': arm, 'started_unix': time.time()}
                save_json(folder / 'child_identity.json', identity)
                while p.poll() is None:
                    if time.monotonic() - started > 20 * 3600:
                        os.killpg(p.pid, signal.SIGTERM)
                        try:
                            p.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL);p.wait()
                        raise TimeoutError('Own arm hard timeout: ' + arm)
                    progress = {}
                    if (folder / 'progress.json').exists():
                        progress = json.loads((folder / 'progress.json').read_text())
                    save_json(R / 'queue_progress.json', {'status': 'RUNNING', 'active_arm': arm,
                        'worker': identity, 'worker_progress': progress, 'supervisor_pid': os.getpid(),
                        'alive_checked_unix': time.time()})
                    time.sleep(30)
                save_json(folder / 'exit_receipt.json', {'exit_code': p.returncode, 'pid': p.pid,
                    'start_ticks': identity['start_ticks'], 'elapsed_seconds': time.monotonic() - started})
                if p.returncode != 0:
                    raise RuntimeError('Arm%s exited%s; no automatic retry or remaining queue' % (arm, p.returncode))
        result = summarize()
        save_json(R / 'queue_progress.json', {'status': 'COMPLETED', 'pilot_pass': result['pilot_pass'], 'next': result['decision']})
        save_json(R / 'supervisor_result.json', {'status': 'COMPLETED', 'exit_code': 0, 'pid': os.getpid(), 'result': str(R / 'three_arm_result.json')})
    except BaseException as exc:
        save_json(R / 'supervisor_result.json', {'status': 'FAILED', 'pid': os.getpid(), 'error': repr(exc),
            'traceback': traceback.format_exc(), 'automatic_retry': False, 'remaining_queue_started': False})
        save_json(R / 'queue_progress.json', {'status': 'FAILED', 'error': repr(exc), 'remaining_queue_started': False})
        raise


if __name__ == '__main__':
    main()
