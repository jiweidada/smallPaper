"""Independent provenance, input/teacher pairing and terminal budget checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

D = Path(__file__).resolve().parent
V = D / 'result_verification_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''):h.update(x)
    return h.hexdigest()


def main(arms):
    assert arms[0] == 'B' and len(set(arms)) == len(arms)
    roots = {a: D / 'completed_results' / a for a in arms}
    metrics = {};init = {};results = {};protocols = {};gt = set();log_checks = {}
    for arm, root in roots.items():
        folder = root / ('arm_' + arm);result = read(folder / 'result.json');results[arm] = result
        assert result['status'] == 'COMPLETED' and result['epochs'] == 12
        assert result['optimizer_updates'] == 9708 and result['full_model_forwards'] == 10159
        assert result['teacher_forwards'] == (0 if arm == 'B' else 9708)
        assert result['inference_auxiliary_removed']
        assert read(folder / 'exit_receipt.json')['exit_code'] == 0
        cpu = read(V / (arm + '_epoch12_checkpoint_cpu_verification.json'))
        ig = read(V / (arm + '_epoch12_independent_GT_ignore.json'))
        met = read(V / (arm + '_epoch12_independent_metrics.json'))
        download = read(D / ('completed_arm_' + arm + '_download_verification.json'))
        assert all(item['status'] == 'PASS' for item in [cpu, ig, met, download])
        assert cpu['checkpoint_sha256'] == download['metadata']['epoch12_checkpoint_sha256']
        if arm != 'B':
            assert cpu['reconstructed_initial_auxiliary_matches_saved_digest']
            assert result['teacher_frozen_unchanged']
            assert result['teacher_state_start_digest'] == result['teacher_state_end_digest']
            assert set(cpu['auxiliary_optimizer_steps'].values()) == {9708}
        metrics[arm] = met['metrics'];gt.add(met['gt_sha256']);init[arm] = read(folder / 'initialization.json')
        protocols[arm] = sha(root / 'execution_protocol_FROZEN.json')
        plan = read(root / 'execution_protocol_FROZEN.json')
        with zipfile.ZipFile(D / ('completed_arm_' + arm + '_bundle.zip')) as z:
            manifest = json.loads(z.read('member_sha256.json'))
        for name, expected in plan['files_sha256'].items():
            if name == 'common_init_10class.pth':
                assert sha(D / name) == expected
            elif name == 'teacher_B_epoch12.pth':
                p = D.parent / 'A42_ECDet三组最小验证_20261031_v1/completed_results/B/arm_B/epoch12.pth'
                assert sha(p) == expected
            else:
                assert manifest[name] == expected, name
        samples = result['latency']['samples_ms']
        assert len(samples) == 30 and all(math.isfinite(x) and x > 0 for x in samples)
        assert statistics.median(samples) == result['latency']['median_ms']
        assert abs(statistics.fmean(samples) - result['latency']['mean_ms']) < 1e-9
        epochs = [json.loads(row) for row in (folder / 'epochs.jsonl').read_text().splitlines()]
        assert [e['epoch'] for e in epochs] == list(range(1, 13))
        assert all(e['optimizer_updates'] == 809 * e['epoch'] for e in epochs)
        assert all(math.isfinite(e[k]) for e in epochs for k in ['mean_native_loss', 'mean_distillation_loss'])
        sampled = []
        for line in (folder / 'stdout.log').read_text().splitlines():
            try:row = json.loads(line)
            except json.JSONDecodeError:continue
            if row.get('status') == 'TRAINING':
                assert all(math.isfinite(row[k]) for k in ['loss', 'native_loss', 'distillation_loss', 'grad_norm_before_clip'])
                sampled.append(row)
        assert sampled
        log_checks[arm] = {'sampled_progress_rows_finite': len(sampled),
                           'scope': 'Saved sampled scalar logs plus frozen per-step loss/global-gradient finite checks; not archived every gradient tensor.'}
    assert len(set(protocols.values())) == 1 and len(gt) == 1
    assert len({v['common_init_sha256'] for v in init.values()}) == 1
    assert all(v['all_base_tensors_equal'] and v['protocol_sha256'] == protocols[a] for a, v in init.items())
    assert all(v['native_inference_parameters'] == 9803112 for v in init.values())
    if 'M' in arms and 'Q' in arms:
        assert init['M']['auxiliary_initial_tensor_digest'] == init['Q']['auxiliary_initial_tensor_digest']
        assert init['M']['training_model_parameters'] == init['Q']['training_model_parameters'] == 9812520
        assert results['M']['teacher_state_start_digest'] == results['Q']['teacher_state_start_digest']
    flow = []
    for epoch in range(1, 13):
        data = {a: read(root / ('arm_' + a) / ('epoch%02d_input_flow.json' % epoch)) for a, root in roots.items()}
        for a, value in data.items():
            assert value['epoch'] == epoch and value['optimizer_updates'] == epoch * 809
            assert value['order_sha256'] == data['B']['order_sha256']
            assert len(value['first_eight_actual_input_sha256']) == 8
            assert value['first_eight_actual_input_sha256'] == data['B']['first_eight_actual_input_sha256']
            assert value['teacher_forward_count_cumulative'] == (0 if a == 'B' else epoch * 809)
        teacher_same = None
        if 'M' in arms and 'Q' in arms:
            assert data['M']['teacher_all_batches_flow_sha256'] == data['Q']['teacher_all_batches_flow_sha256']
            assert data['M']['first_eight_teacher_batch_sha256'] == data['Q']['first_eight_teacher_batch_sha256']
            assert data['M']['teacher_matched_instances'] == data['Q']['teacher_matched_instances']
            assert data['M']['teacher_unmatched_GT_due_to_query_limit'] == data['Q']['teacher_unmatched_GT_due_to_query_limit']
            teacher_same = True
        flow.append({'epoch': epoch, 'sampler_order_same': True, 'first_eight_actual_input_batches_same': True,
                     'all_batch_teacher_predictions_assignments_and_both_targets_same_M_Q': teacher_same})
    result = {'status': 'PASS', 'arms': arms, 'metrics': metrics,
              'frozen_protocol_same': True, 'common_initialization_same': True,
              'native_inference_parameters_each': 9803112, 'epoch_pairing': flow,
              'formal_M_Q_teacher_pairing_verified': 'M' in arms and 'Q' in arms,
              'finite_log_scope': log_checks,
              'deltas_vs_B_pp': {a: {k: metrics[a][k] - metrics['B'][k] for k in ['AP', 'APs', 'APs75']} for a in arms if a != 'B'},
              'input_scope': 'All sampler-order hashes and first8 actual input-batch digests per epoch; M/Q additionally full rolling stream of teacher predictions, assignments and both targets. Not all actual input pixels byte-compared.',
              'new_student_forwards': 0, 'new_teacher_forwards': 0, 'new_optimizer_updates': 0}
    path = V / ('_'.join(arms) + '_controls_verified.json')
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'arms', 'formal_M_Q_teacher_pairing_verified', 'deltas_vs_B_pp']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--arms', choices=['B', 'M', 'Q'], nargs='+', required=True)
    main(p.parse_args().arms)
