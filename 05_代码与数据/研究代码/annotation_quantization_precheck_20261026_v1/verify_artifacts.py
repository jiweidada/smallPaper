"""Independent arithmetic and integrity audit; does not rerun the experiment."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'attempt02'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i+1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i+1+j)/2.
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def rho(a, b):
    x, y = average_ranks(a), average_ranks(b)
    xm, ym = math.fsum(x)/len(x), math.fsum(y)/len(y)
    numerator = math.fsum((u-xm)*(v-ym) for u, v in zip(x, y))
    denominator = math.sqrt(math.fsum((u-xm)**2 for u in x)*math.fsum((v-ym)**2 for v in y))
    return numerator/denominator


def main():
    result = json.loads((RUN/'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    assert digest(ROOT/'protocol.json') == digest(RUN/'protocol.json')
    for folder in (ROOT, RUN):
        assert (folder/'frozen_protocol.sha256').read_text().split()[0] == digest(folder/'protocol.json')
    artifact_manifest = json.loads((RUN/'delivery_manifest.json').read_text(encoding='utf-8'))
    for entry in artifact_manifest['files']:
        path = RUN/entry['path']
        assert path.stat().st_size == entry['size'] and digest(path) == entry['sha256']
    input_manifest = json.loads((RUN/'input_manifest.json').read_text(encoding='utf-8'))
    inputs = [input_manifest['annotation']] + input_manifest['cache_files']
    for entry in inputs:
        assert digest(Path(entry['path'])) == entry['sha256'], entry['path']
    with (RUN/'per_gt_targets.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4*4896
    checks = {}
    comparison_passes = []
    for case_name, case in result['cases'].items():
        selected = [r for r in rows if int(r['seed']) == case['seed'] and r['rule'] == case['rule']]
        assert len(selected) == 4896
        reference = [float(r['reference_target']) for r in selected]
        recomputed = {}
        for arm in protocol['arms']:
            column = 'expected12' if arm == 'expected' else arm
            target = [float(r[column]) for r in selected]
            errors = [t-g for t, g in zip(target, reference)]
            metrics = {'MSE': math.fsum(e*e for e in errors)/len(errors),
                       'MAE': math.fsum(abs(e) for e in errors)/len(errors),
                       'Spearman': rho(target, reference)}
            metrics['RMSE'] = math.sqrt(metrics['MSE'])
            for key, value in metrics.items():
                assert abs(value-case['metrics'][arm][key]) < 1e-12, (case_name, arm, key)
            recomputed[arm] = metrics
        for arm in ('point', 'center'):
            c = case['comparisons'][arm]
            relative = 1-recomputed['expected']['MSE']/recomputed[arm]['MSE']
            delta = recomputed['expected']['Spearman']-recomputed[arm]['Spearman']
            assert abs(relative-c['relative_MSE_reduction']) < 1e-12
            assert abs(delta-c['rho_advantage']) < 1e-12
            passed = (relative >= .05 and delta >= .001
                      and c['MSE_advantage_CI95'][0] > 0 and c['rho_advantage_CI95'][0] > 0)
            assert bool(passed) == c['passed']
            comparison_passes.append(passed)
        checks[case_name] = {'rows': len(selected), 'metric_max_tolerance': 1e-12,
                            'independent_average_tie_rank_correlation': True}
    assert not all(comparison_passes)
    assert result['sample_gate'] and result['numeric_gate']
    assert result['verdict'] == 'stop_current_expected_version'
    paired = [np.load(RUN/f'pairs_seed{seed}.npz') for seed in (1, 2)]
    for d in paired:
        assert (d['prediction'][:, 2:] > d['prediction'][:, :2]).all()
    assert np.array_equal(paired[0]['ids'], paired[1]['ids'])
    unused = input_manifest['unused_invalid_queries']
    invalid_count = sum(len(e['query_ids']) for e in unused)
    assert invalid_count == 7
    audit = {
        'status': 'ANALYZED; arithmetic and integrity independently checked, full experiment not rerun',
        'protocol_identical_between_attempts': True,
        'old_input_files_rehashed_unchanged': len(inputs),
        'successful_attempt_artifact_hashes_verified': len(artifact_manifest['files']),
        'per_gt_csv_rows': len(rows), 'metric_and_gate_checks': checks,
        'all_selected_predictions_positive_area': True,
        'unused_invalid_queries_recorded': invalid_count,
        'verdict_independently_confirmed': result['verdict'],
        'CI_note': 'CI signs and gate application checked; bootstrap itself not independently rerun.'
    }
    with (ROOT/'verification.json').open('x', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
