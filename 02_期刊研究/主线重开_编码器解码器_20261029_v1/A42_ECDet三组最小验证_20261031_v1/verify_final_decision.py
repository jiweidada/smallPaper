"""Recalculate frozen practical gates from independently checked local endpoints.

This reads files only. It never imports a detector or performs inference/training.
The supervisor summary is used as a comparison target, not the metric source.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics

D = Path(__file__).resolve().parent
V = D / 'result_verification_v1'
T = D / 'terminal_summary'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual - expected) <= 1e-10, (actual, expected)


def main():
    controls = read(V / 'B_S_L_controls_verified.json')
    assert controls['status'] == 'PASS' and controls['arms'] == ['B', 'S', 'L']
    protocol_path = D / 'completed_results' / 'B' / 'execution_protocol_FROZEN.json'
    protocol = read(protocol_path)
    thresholds = protocol['practical_gates']
    # Fail rather than silently accept a changed protocol/endpoint/gate.
    assert protocol['arms'] == ['B', 'S', 'L']
    assert protocol['epochs_per_group'] == 12
    assert protocol['checkpoint_selection'] == 'fixed epoch12 EMA for every arm'
    assert thresholds == {
        'L_minus_B_small_AP75_pp_min': 0.5,
        'L_minus_S_small_AP75_pp_min': 0.3,
        'latency_overhead_max': 0.1,
        'overall_AP_drop_max_pp': 0.2,
    }
    rows, metrics, latencies = {}, {}, {}
    inputs = {str(protocol_path.relative_to(D)): sha(protocol_path)}
    for arm in ['B', 'S', 'L']:
        root = D / 'completed_results' / arm / ('arm_' + arm)
        rows[arm] = read(root / 'result.json')
        assert rows[arm]['status'] == 'COMPLETED'
        assert rows[arm]['epochs'] == 12
        assert rows[arm]['optimizer_updates'] == 9708
        assert rows[arm]['full_model_forwards'] == 10159
        assert read(root / 'exit_receipt.json')['exit_code'] == 0
        metric_path = V / (arm + '_epoch12_independent_metrics.json')
        independent = read(metric_path)
        assert independent['status'] == 'PASS'
        assert independent['all_precision_cells_max_abs_difference'] <= 1e-12
        assert independent['all_recall_cells_max_abs_difference'] <= 1e-12
        metrics[arm] = independent['metrics']
        for key in ['AP', 'AP50', 'AP75', 'APs', 'APs75', 'APm', 'APl', 'ARs']:
            near(metrics[arm][key], rows[arm]['endpoint_metrics'][key])
        samples = rows[arm]['latency']['samples_ms']
        assert len(samples) == 30
        assert all(math.isfinite(x) and x > 0 for x in samples)
        latencies[arm] = statistics.median(samples)
        near(latencies[arm], rows[arm]['latency']['median_ms'])
        for p in [metric_path, root / 'result.json', root / 'exit_receipt.json']:
            inputs[str(p.relative_to(D))] = sha(p)
    l_b = metrics['L']['APs75'] - metrics['B']['APs75']
    l_s = metrics['L']['APs75'] - metrics['S']['APs75']
    ap_l_b = metrics['L']['AP'] - metrics['B']['AP']
    latency_fraction = latencies['L'] / latencies['B'] - 1
    gates = {
        'L_minus_B_small_AP75_at_least_0_5pp': l_b >= thresholds['L_minus_B_small_AP75_pp_min'],
        'L_minus_S_small_AP75_at_least_0_3pp': l_s >= thresholds['L_minus_S_small_AP75_pp_min'],
        'same_device_latency_overhead_at_most_10pct': latency_fraction <= thresholds['latency_overhead_max'],
        'overall_AP_drop_no_more_than_0_2pp': ap_l_b >= -thresholds['overall_AP_drop_max_pp'],
    }
    passed = all(gates.values())
    decision = ('WORTH_NEXT_VALIDATION_NOT_ESTABLISHED_INNOVATION' if passed
                else 'STOP_THIS_LOCALIZATION_ONLY_MAINLINE_NO_RESCUE_SWEEP')
    aggregate = read(T / 'three_arm_result.json')
    supervisor = read(T / 'supervisor_result.json')
    queue = read(T / 'queue_progress.json')
    assert aggregate['status'] == supervisor['status'] == queue['status'] == 'COMPLETED'
    assert supervisor['exit_code'] == 0
    assert aggregate['protocol_sha256'] == sha(protocol_path)
    near(l_b, aggregate['L_minus_B_small_AP75_pp'])
    near(l_s, aggregate['L_minus_S_small_AP75_pp'])
    near(ap_l_b, aggregate['L_minus_B_AP_pp'])
    near(latency_fraction, aggregate['L_latency_fraction_over_B'])
    assert aggregate['practical_gates'] == gates
    assert aggregate['pilot_pass'] == queue['pilot_pass'] == passed
    assert aggregate['decision'] == queue['next'] == decision
    assert aggregate['optimizer_updates'] == sum(r['optimizer_updates'] for r in rows.values()) == 29124
    assert aggregate['full_model_forwards'] == sum(r['full_model_forwards'] for r in rows.values()) == 30477
    for a in ['B', 'S', 'L']:
        for k in ['AP', 'APs', 'APs75']:
            near(aggregate['arms'][a][k], metrics[a][k])
    for p in [V / 'B_S_L_controls_verified.json', T / 'three_arm_result.json',
              T / 'supervisor_result.json', T / 'queue_progress.json']:
        inputs[str(p.relative_to(D))] = sha(p)
    output = {
        'status': 'PASS', 'metric_source': 'independent local NumPy metric recalculation',
        'checkpoint_selection': protocol['checkpoint_selection'],
        'metrics': metrics, 'latency_median_ms': latencies,
        'deltas_vs_B_pp': {a: {k: metrics[a][k] - metrics['B'][k]
                              for k in ['AP', 'APs', 'APs75']} for a in ['S', 'L']},
        'L_minus_S_pp': {k: metrics['L'][k] - metrics['S'][k] for k in ['AP', 'APs', 'APs75']},
        'L_latency_fraction_over_B': latency_fraction,
        'frozen_thresholds': thresholds, 'practical_gates': gates,
        'pilot_pass': passed, 'decision': decision,
        'supervisor_summary_independently_matched': True,
        'formal_optimizer_updates': 29124, 'formal_full_model_forwards': 30477,
        'new_model_forwards_by_this_check': 0, 'new_optimizer_updates_by_this_check': 0,
        'statistics_scope': 'Single-seed, 12-epoch practical resource gate; no statistical significance, generalization, or novelty claim.',
        'metric_scope': protocol['metric'], 'inputs_sha256': inputs,
    }
    (V / 'final_practical_gate_verification.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: output[k] for k in ['status', 'deltas_vs_B_pp', 'L_minus_S_pp',
          'latency_median_ms', 'L_latency_fraction_over_B', 'practical_gates', 'pilot_pass', 'decision']}))


if __name__ == '__main__':
    main()
