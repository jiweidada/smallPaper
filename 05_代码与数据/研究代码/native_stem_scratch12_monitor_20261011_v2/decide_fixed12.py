"""Accept only verified native epoch12 EMA results of both specified seeds."""
import argparse
from pathlib import Path
import sys
from verify_terminal_local import PASS, PROTOCOL, local_protocol, read, sha, write


def decision(report_paths):
    p = local_protocol()
    reports = [read(x) for x in report_paths]
    assert len(reports) == 2 and {r['identity']['seed'] for r in reports} == {1, 2}
    reports.sort(key=lambda x: x['identity']['seed'])
    for r in reports:
        assert r['status'] == PASS and r['fixed_endpoint'] == p['endpoint']
        assert r['identity']['arm'] == 'native_stem_scratch'
        assert r['identity']['implementation'] == p['implementation_sha256']
        assert r['verification_script_sha256'] == p['reader_files_sha256']['verify_terminal_local.py']
        assert r['local_reader_protocol_sha256'] == sha(PROTOCOL)
        assert r['all548_stage_arrays_reloaded_exact'] is True
        assert r['new_model_forwards'] == r['new_evaluator_calls'] == r['new_optimizer_updates'] == r['new_training_epochs'] == 0
        d = r['delta_AP_metrics_pp']
        assert d['AP'] == r['candidate_overall_percent']['AP'] - r['baseline_overall_percent']['AP']
        assert d['APs'] == r['candidate_overall_percent']['APs'] - r['baseline_overall_percent']['APs']
        assert r['native_gate_passed_this_seed'] == (d['APs'] > 0 and d['AP'] >= 0)
    passed = all(r['native_gate_passed_this_seed'] for r in reports)
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    return dict(schema='native_stem_scratch_fixed12_two_seed_decision_v1',
        status='VERIFIED_NATIVE_STEM_SCRATCH_REFERENCE12_COMPLETE', reference_control=True,
        both_seeds_passed=passed, acceptance=p['acceptance'], fixed_endpoint=p['endpoint'],
        reports=[dict(path=str(Path(x).resolve()), sha256=sha(x)) for x in report_paths],
        seeds=[dict(seed=r['identity']['seed'], baseline=r['baseline_overall_percent'],
            candidate=r['candidate_overall_percent'], delta_pp=r['delta_AP_metrics_pp'],
            gate_passed=r['native_gate_passed_this_seed']) for r in reports],
        decision_script_sha256=sha(__file__), local_reader_protocol_sha256=sha(PROTOCOL),
        new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0,
        claim_boundary='Both reference endpoints verified. Comparison versus cached pretrained BASE describes the initialization control. This reference is not promoted or rejected as a new method; full enhanced Stem must be compared separately to this reference and to existing BASE. No extension, retry, best epoch, significance or unique root claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed1-report', required=True)
    parser.add_argument('--seed2-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = decision([args.seed1_report, args.seed2_report])
    write(args.output, value)
    print(value['status'], flush=True)
