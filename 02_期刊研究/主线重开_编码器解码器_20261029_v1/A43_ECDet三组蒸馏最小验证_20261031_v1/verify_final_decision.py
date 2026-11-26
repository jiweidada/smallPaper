"""Recompute the pre-fixed practical gates from independent endpoint metrics."""
import hashlib
import json
import math
from pathlib import Path
import statistics

D = Path(__file__).resolve().parent
V = D / 'result_verification_v1'
T = D / 'terminal_summary'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a, b):assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-10, (a, b)


def main():
    controls = read(V / 'B_M_Q_controls_verified.json')
    assert controls['status'] == 'PASS' and controls['formal_M_Q_teacher_pairing_verified']
    protocol_path = D / 'completed_results/B/execution_protocol_FROZEN.json'
    plan = read(protocol_path);thresholds = plan['practical_gates']
    assert plan['arms'] == ['B', 'M', 'Q'] and plan['epochs_per_group'] == 12
    assert plan['checkpoint_selection'] == 'fixed epoch12 EMA for every arm'
    assert thresholds == {'Q_minus_B_APs_pp_min': .5, 'Q_minus_M_APs_pp_min': .3, 'overall_AP_drop_max_pp': .2}
    metrics = {};rows = {};latency = {};inputs = {str(protocol_path.relative_to(D)): sha(protocol_path)}
    for arm in ['B', 'M', 'Q']:
        folder = D / 'completed_results' / arm / ('arm_' + arm)
        p = V / (arm + '_epoch12_independent_metrics.json')
        independent = read(p);assert independent['status'] == 'PASS'
        assert independent['all_precision_cells_max_abs_difference'] <= 1e-12
        assert independent['all_recall_cells_max_abs_difference'] <= 1e-12
        metrics[arm] = independent['metrics'];rows[arm] = read(folder / 'result.json')
        assert rows[arm]['status'] == 'COMPLETED' and rows[arm]['epochs'] == 12
        assert rows[arm]['optimizer_updates'] == 9708 and rows[arm]['full_model_forwards'] == 10159
        assert rows[arm]['teacher_forwards'] == (0 if arm == 'B' else 9708)
        assert read(folder / 'exit_receipt.json')['exit_code'] == 0
        for k in ['AP', 'APs', 'APs75']:
            near(metrics[arm][k], rows[arm]['endpoint_metrics'][k])
        samples = rows[arm]['latency']['samples_ms']
        assert len(samples) == 30 and all(math.isfinite(x) and x > 0 for x in samples)
        latency[arm] = statistics.median(samples)
        near(latency[arm], rows[arm]['latency']['median_ms'])
        for q in [p, folder / 'result.json', folder / 'exit_receipt.json']:
            inputs[str(q.relative_to(D))] = sha(q)
    qb = metrics['Q']['APs'] - metrics['B']['APs'];qm = metrics['Q']['APs'] - metrics['M']['APs']
    overall = metrics['Q']['AP'] - metrics['B']['AP']
    gates = {'Q_minus_B_APs_at_least_0_5pp': qb >= .5, 'Q_minus_M_APs_at_least_0_3pp': qm >= .3,
             'overall_AP_drop_no_more_than_0_2pp': overall >= -.2}
    passed = all(gates.values())
    decision = ('WORTH_NEXT_VALIDATION_NOT_ESTABLISHED_INNOVATION' if passed else
                'STOP_DIRECT_QUERY_SUPERVISION_CLAIM_NO_RESCUE_SWEEP')
    aggregate = read(T / 'three_arm_result.json');supervisor = read(T / 'supervisor_result.json')
    queue = read(T / 'queue_progress.json')
    assert aggregate['status'] == supervisor['status'] == queue['status'] == 'COMPLETED'
    assert supervisor['exit_code'] == 0 and aggregate['protocol_sha256'] == sha(protocol_path)
    near(qb, aggregate['Q_minus_B_APs_pp']);near(qm, aggregate['Q_minus_M_APs_pp'])
    near(overall, aggregate['Q_minus_B_AP_pp'])
    near(metrics['Q']['APs75'] - metrics['B']['APs75'], aggregate['Q_minus_B_small_AP75_pp'])
    near(metrics['Q']['APs75'] - metrics['M']['APs75'], aggregate['Q_minus_M_small_AP75_pp'])
    assert aggregate['practical_gates'] == gates and aggregate['pilot_pass'] == queue['pilot_pass'] == passed
    assert aggregate['decision'] == queue['next'] == decision
    assert aggregate['optimizer_updates'] == sum(r['optimizer_updates'] for r in rows.values()) == 29124
    assert aggregate['student_forwards'] == sum(r['full_model_forwards'] for r in rows.values()) == 30477
    assert aggregate['teacher_forwards'] == sum(r['teacher_forwards'] for r in rows.values()) == 19416
    for arm in ['B', 'M', 'Q']:
        for k in ['AP', 'APs', 'APs75']:near(aggregate['arms'][arm][k], metrics[arm][k])
        near(aggregate['latency_median_ms'][arm], latency[arm])
    for p in [V / 'B_M_Q_controls_verified.json', T / 'three_arm_result.json',
              T / 'supervisor_result.json', T / 'queue_progress.json']:
        inputs[str(p.relative_to(D))] = sha(p)
    output = {'status': 'PASS', 'source': 'independently recomputed local NumPy endpoints',
              'metrics': metrics, 'latency_median_ms': latency,
              'deltas_vs_B_pp': {a: {k: metrics[a][k] - metrics['B'][k] for k in ['AP', 'APs', 'APs75']} for a in ['M', 'Q']},
              'Q_minus_M_pp': {k: metrics['Q'][k] - metrics['M'][k] for k in ['AP', 'APs', 'APs75']},
              'frozen_thresholds': thresholds, 'practical_gates': gates, 'pilot_pass': passed, 'decision': decision,
              'supervisor_summary_independently_matched': True, 'formal_optimizer_updates': 29124,
              'formal_student_forwards': 30477, 'formal_teacher_forwards': 19416,
              'new_student_forwards': 0, 'new_teacher_forwards': 0, 'new_optimizer_updates': 0,
              'scope': 'Single-seed12epoch resource decision; not significance, generalization, semantic mechanism or novelty proof.',
              'inputs_sha256': inputs}
    (V / 'final_practical_gate_verification.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: output[k] for k in ['status', 'deltas_vs_B_pp', 'Q_minus_M_pp', 'practical_gates', 'pilot_pass', 'decision']}))


if __name__ == '__main__':main()
