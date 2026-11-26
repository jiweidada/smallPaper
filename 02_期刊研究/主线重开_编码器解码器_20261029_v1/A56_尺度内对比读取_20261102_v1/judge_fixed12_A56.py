"""Apply original project gates to locally verified fixed12 endpoints, not best epochs."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
PROTOCOL_SHA = 'da3ae8ac399483307eeb7e221627d0bdaef4222021dde468cb41459d674915a9'


def load(path):
    return json.loads(path.read_text('utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = P/'fixed_endpoint_judgment_A56.json'
    assert not destination.exists()
    protocol = P/'protocol_FROZEN_v1.json'
    assert sha(protocol) == PROTOCOL_SHA
    plan = load(protocol)
    audit_path = P/'terminal_frozen_input_audit_A56.json'
    audit = load(audit_path)
    assert audit['status'] == 'PASS' and audit['protocol_sha256'] == PROTOCOL_SHA
    assert all(audit['own_frozen_files'].values()) and all(audit['external_frozen_files'].values())
    assert all(not r['owned_live'] for r in audit['owned_processes'])
    baseline_folder = P.parent/'A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B'
    baseline_check = load(P.parent/'A49_下一主线有界筛选_20261101_v1/B_completed_local_verification.json')
    assert baseline_check['status'] == 'PASS'
    baseline = load(baseline_folder/'epoch12_EMA_metrics.json')
    assert baseline == load(baseline_folder/'result.json')['endpoint_metrics'] == baseline_check['endpoint_metrics']
    endpoints = {'B': baseline}
    sources = {str(baseline_folder/'epoch12_EMA_metrics.json'): sha(baseline_folder/'epoch12_EMA_metrics.json')}
    for arm in ['M', 'Q']:
        check_path = P/f'{arm}_completed_local_verification.json'
        check = load(check_path)
        assert check['status'] == 'PASS' and check['epochs'] == 12
        assert check['optimizer_updates'] == 9708 and check['full_model_forwards'] == 10159
        assert check['bundle_sha256'] == load(P/f'{arm}_bundle_ready.json')['sha256']
        folder = P/f'completed_{arm}_v1/arm_{arm}'
        metric_path = folder/'epoch12_EMA_metrics.json'
        assert sha(metric_path) == audit['fixed12_endpoints'][arm]['epoch12_EMA_metrics.json']['sha256']
        endpoints[arm] = load(metric_path)
        assert endpoints[arm] == check['endpoint_metrics']
        sources[str(check_path)] = sha(check_path)
        sources[str(metric_path)] = sha(metric_path)
    conventions = ['max_detections', 'evaluated_GT', 'ignored_valid_GT', 'area_convention', 'scope']
    assert all(endpoints[a][k] == baseline[k] for a in ['M', 'Q'] for k in conventions)
    metrics = ['AP', 'APs', 'AP75', 'APs75', 'AP50', 'APm', 'APl', 'ARs']
    differences = {name: {k: endpoints[x][k]-endpoints[y][k] for k in metrics}
                   for name, x, y in [('M_minus_B_pp', 'M', 'B'), ('Q_minus_B_pp', 'Q', 'B'),
                                       ('Q_minus_M_pp', 'Q', 'M')]}
    thresholds = plan['primary_gates_pp']
    gates = {'Q_minus_B_APs': differences['Q_minus_B_pp']['APs'] >= thresholds['Q_minus_B_APs_at_least'],
             'Q_minus_M_APs': differences['Q_minus_M_pp']['APs'] >= thresholds['Q_minus_M_APs_at_least'],
             'Q_minus_B_AP': differences['Q_minus_B_pp']['AP'] >= thresholds['Q_minus_B_AP_at_least']}
    passed = all(gates.values())
    result = {'phase': 'A56', 'status': 'FIXED12_VERIFIED', 'endpoint': 'epoch12 EMA',
              'protocol_sha256': PROTOCOL_SHA, 'terminal_audit_sha256': sha(audit_path),
              'endpoints': endpoints, **differences, 'thresholds_pp': thresholds,
              'gates': gates, 'all_gates_passed': passed,
              'decision': 'ADVANCE_EXPLORATORY_CANDIDATE_FOR_RESEARCH_SYNTHESIS' if passed
                          else 'RETIRED_EXACT_RECIPE_FIXED12',
              'goal_status': 'active', 'publication_novelty_cleared': False,
              'no_rescue_sweep': True, 'source_files_sha256': sources,
              'interpretation': 'One-seed fixed12 project resource gate, not statistical significance or publication acceptance. Complete mainline synthesis still requires nearest-method and evidence assessment.',
              'new_detector_forwards': 0, 'new_optimizer_updates': 0}
    with destination.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'decision': result['decision'], 'gates': gates,
                      'Q_minus_B_pp': differences['Q_minus_B_pp'],
                      'Q_minus_M_pp': differences['Q_minus_M_pp'],
                      'new_detector_forwards': 0}))


if __name__ == '__main__':
    main()
