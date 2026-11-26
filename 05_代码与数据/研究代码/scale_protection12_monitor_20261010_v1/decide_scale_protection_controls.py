"""Apply the existing fixed12 gates to fully verified downloaded four-arm evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(root, seed, arm):
    folder = root / f'terminal_collection/seed{seed}_{arm}_v1'
    report_path = folder / 'verification_report_v1.json'
    if not report_path.exists():
        return None
    report = read(report_path)
    assert report['status'] == 'PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
    assert report['verification_script_sha256'] == sha(Path(__file__).parent / 'verify_terminal_local.py')
    assert report['identity']['seed'] == seed and report['identity']['arm'] == arm
    assert report['first100_rows_exact'] == 100 and report['all548_reloaded_stage_arrays_exact']
    packet = folder / 'extracted'
    receipt = read(packet / 'prepared/receipt.json')
    done = read(packet / 'run/training_completion.json')
    endpoint = read(packet / 'run/endpoint_epoch12/summary.json')
    replay = read(packet / 'candidate_reload_v1/verification.json')
    metadata_path = folder / f'terminal_{arm}_seed{seed}_v1_metadata.json'
    metadata = read(metadata_path)
    assert metadata['sha256'] == sha(folder / f'terminal_{arm}_seed{seed}_v1.zip') == report['archive_sha256']
    assert receipt['identity'] == done['identity'] == endpoint['identity'] == replay['identity'] == report['identity']
    assert done['completed_epochs'] == 12 and done['entered_epochs'] == list(range(12))
    assert endpoint['epoch_zero_based'] == 11 and endpoint['evaluated_weights'] == 'EMA'
    assert done['last_checkpoint_sha256'] == sha(packet / 'run/last.pth') == replay['checkpoint_sha256']
    assert replay['status'] == 'PASS_SAVED_CANDIDATE_RELOAD'
    assert replay['native_GT_hits_exact'] and replay['all548_stage_arrays_exact']
    assert replay['baseline_model_forwards'] == replay['optimizer_updates'] == replay['epochs_added'] == 0
    assert replay['stats_max_abs_fraction_difference'] <= 1e-8
    assert report['candidate_overall_percent'] == endpoint['overall_percent']
    assert report['final_training_counts']['attempts_total'] == 19404
    expected_gate = report['delta_AP_metrics_pp']['APs'] > 0 and report['delta_AP_metrics_pp']['AP'] >= 0
    assert expected_gate == report['native_gate_passed_this_seed'] == done['native_gate_passed']
    assert report['augmentation']['all_policy_decisions_exact']
    return dict(report=report, endpoint=endpoint, receipt=receipt,
        provenance=dict(verified_report_sha256=sha(report_path), archive_sha256=metadata['sha256'],
            metadata_sha256=sha(metadata_path), checkpoint_sha256=done['last_checkpoint_sha256'],
            endpoint_summary_sha256=sha(packet / 'run/endpoint_epoch12/summary.json'),
            replay_verification_sha256=sha(packet / 'candidate_reload_v1/verification.json')))


def decide(root, check_ready):
    root = Path(root).resolve()
    protocol = read(root / 'protocol.json')
    assert sha(root / 'protocol.json') == '4212196fe110725cd38e175218dc9d53ac34d242a391f5f2a6093a3e4e2ad542'
    assert protocol['prospective_training_budget']['endpoint'] == 'Fixed epoch12 native EMA, no best epoch selection or extension.'
    arms = {(seed, arm): validate(root, seed, arm) for seed in (1, 2) for arm in ('target_protect', 'random_protect')}
    missing = [f'seed{seed}_{arm}' for (seed, arm), evidence in arms.items() if evidence is None]
    if check_ready:
        assert 'torch' not in sys.modules
        return dict(status='ALL_FOUR_DOWNLOADED_ENDPOINTS_READY' if not missing else 'WAITING_VERIFIED_RANDOM_ENDPOINTS',
            missing=missing, validated=[f'seed{seed}_{arm}' for (seed, arm), evidence in arms.items() if evidence is not None],
            model_forwards=0, evaluator_calls=0, optimizer_updates=0, training_epochs=0,
            completed_decision=False)
    assert not missing, missing
    rows = {}
    for seed in (1, 2):
        target, random = arms[seed, 'target_protect'], arms[seed, 'random_protect']
        ti, ri = target['report']['identity'], random['report']['identity']
        for field in ('implementation', 'protocol_sha256', 'experiment_protocol_sha256',
                      'common_init_sha256', 'evaluator_binding', 'image_manifest_sha256',
                      'train_ann_sha256', 'val_ann_sha256', 'baseline_checkpoint_sha256',
                      'baseline_first100_sha256', 'training_profile_sha256', 'data_only_report_sha256',
                      'native_parameter_count', 'added_parameter_count'):
            assert ti[field] == ri[field], field
        assert ti['native_parameter_count'] == 10231997 and ti['added_parameter_count'] == 0
        target_metrics = target['endpoint']['overall_percent']
        random_metrics = random['endpoint']['overall_percent']
        assert target['report']['baseline_overall_percent'] == random['report']['baseline_overall_percent']
        native_pass = target['report']['native_gate_passed_this_seed']
        mechanism_pass = target_metrics['APs'] > random_metrics['APs']
        rows[str(seed)] = dict(target=target_metrics, random=random_metrics,
            baseline=target['report']['baseline_overall_percent'], target_native_gate_passed=native_pass,
            target_minus_random_AP_metrics_pp={key: target_metrics[key]-random_metrics[key] for key in target_metrics},
            target_APs_strictly_greater_than_random_APs=mechanism_pass,
            target_event_summary=target['report']['augmentation'], random_event_summary=random['report']['augmentation'],
            source_provenance=dict(target=target['provenance'], random=random['provenance']))
    retained = all(row['target_native_gate_passed'] and row['target_APs_strictly_greater_than_random_APs'] for row in rows.values())
    report = dict(status='RETAIN_SPECIFIED_TARGET_POLICY_FOR_FOLLOWUP' if retained else 'REJECT_SPECIFIED_TARGET_POLICY',
        protocol_sha256=sha(root / 'protocol.json'), decision_reader_sha256=sha(__file__), seeds=rows,
        fixed_epoch=12, evaluated_weights='EMA', mechanism_screen_passed=retained,
        random_native_gate_is_reported_not_required=True,
        gates='Target APs>same-seed BASE12 APs and target AP>=BASE12 AP for both seeds; target APs>same-seed random APs for both seeds. No target-v-random overallAP gate added.',
        scope='Bounded comparison of the two specified policies. Persistent membership, source GT density, scale and context differ; this does not isolate a pure pixel-size cause or establish root, uniqueness, significance or132-epoch efficacy.',
        new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0,
        automatic_retraining_or_extension=False)
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    with (root / 'fixed12_policy_decision_v1.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--check-ready', action='store_true')
    a = p.parse_args(); print(json.dumps(decide(a.root, a.check_ready)), flush=True)
