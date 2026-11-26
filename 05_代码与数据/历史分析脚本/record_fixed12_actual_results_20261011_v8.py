from pathlib import Path
import json
import hashlib
import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/reviewed_remote_tests_actual12_20261011_v8'
assert not OUT.exists()
OUT.mkdir()
evidence = {}


def read(relative):
    p = ROOT / relative
    evidence[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8'))


previous = read('outputs/reviewed_remote_tests_actual12_20261011_v7/report.json')
contract = read('outputs/active_stem_comparison_contract.json')
stem = read('outputs/stem_random_primary_comparison12_20261011_v1/report.json')
inputs = read('outputs/experiment_table_20261011_v1/table_inputs.json')
workbook = read('outputs/experiment_table_20261011_v1/workbook_validation.json')
rank = []
for seed in (1, 2):
    relative = f'outputs/p2_lowrank12_formal_collection12_20261011_v1/seed{seed}_p2_lowrank16/verification.json'
    d = read(relative)
    assert d['all548_stage_arrays_reloaded_exact'] and d['fixed_endpoint']['epochs'] == 12
    old = next(x for x in inputs['records'] if x['arm'] == 'p2_skip' and x['seed'] == seed)
    rank.append({'seed': seed, 'training_completed_epochs': 12, 'native_percent': d['candidate_overall_percent'],
        'delta_matched_pretrained_BASE_points': d['delta_AP_metrics_pp'], 'matched_pretrained_BASE_metric_gate': d['native_gate_passed_this_seed'],
        'delta_original_P2_points': {k: d['candidate_overall_percent'][k] - old['metrics'][k] for k in ('AP', 'APs')},
        'exact548_saved_EMA_replay_and_local_reader': True, 'local_evidence': relative,
        'Stem_initialization': 'Pretrained native Stem retained; not a fair random-Stem architecture efficacy comparison.'})
report = {
    'status': 'ALL_LISTED_ORIGINAL_TRAINING_COMPLETE12_RANDOM_STEM_MAIN_COMPARATOR_CORRECTED_NEW_RANK16_BOTH_LOCAL_REPLAY_VERIFIED',
    'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'previous_immutable_report_sha256': evidence['outputs/reviewed_remote_tests_actual12_20261011_v7/report.json'],
    'comparison_contract': contract,
    'human_Stem_random_baseline_comparison': stem['paired_results'],
    'both_Stem_APs_improve': stem['both_seeds_APs_improve'],
    'Stem_two_seed_joint_metric_gate_passed': stem['two_seed_joint_gate_passed'],
    'all_listed_training_completed12': len(inputs['records']) == 24 and all(r['epochs'] == 12 for r in inputs['records']),
    'new_rank16_P2_formal_results': rank,
    'rank16_both_matched_pretrained_BASE_metric_gates_passed': all(x['matched_pretrained_BASE_metric_gate'] for x in rank),
    'rank16_APs_higher_than_original_P2_both_seeds': all(x['delta_original_P2_points']['APs'] > 0 for x in rank),
    'original_P2_seed1_exact_replay_failure_preserved': True,
    'original_P2_seed1_primary_result_relabelled_as_replay_PASS': False,
    'Excel24_records_and_comparator_formulas_checked': workbook,
    'all_original_formal_packets_collected_and_verified': False,
    'new_rank_reader_revision': 'Read the actual AMP preflight as a byte-exact hash-bound external sidecar because the frozen collector omitted its nested folder; all native PR and548 exact replay checks are unchanged.',
    'current_goal_turn': 'progress', 'previous_goal_turn': 'progress', 'no_progress_streak': 0,
    'new_BASE_training_epochs': 0, 'new_BASE_forwards': 0, 'new_training_epochs': 0, 'new_model_forwards': 0,
    'root_or_novelty_proven': False, 'goal_complete': False,
    'pending': [
        'Collect and locally verify the already completed seed1 SiLU/attention first-closeout archives; no training or replay retry.',
        'Locally verify the two downloaded coefficient-error-matched diagnostic archives and collect the completed A direction outputs; no new parent or BASE forward.',
        'Recover remaining original Stem control formal evidence, retaining fixed12 log-only labels wherever full packets are missing.',
        'Use random native Stem as the main comparator for Stem structure. Analyze seed1 medium/large loss and seed2 gains using saved stages and native class/area PR.',
        'Review and implement a new falsifiable change only after that analysis. Do not repeat the failed warm-Stem or frequency/attention mechanisms or extend any completed configuration.',
    ],
    'evidence_files_sha256': evidence,
}
(OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(OUT / 'manifest.json').write_text(json.dumps({'report.json': hashlib.sha256((OUT / 'report.json').read_bytes()).hexdigest()}, indent=2) + '\n', encoding='utf-8')
state_folder = ROOT / 'outputs/stem_component12_20261011_v1'
state = {'status': report['status'], 'canonical_report': '../reviewed_remote_tests_actual12_20261011_v8/report.json',
    'canonical_report_sha256': hashlib.sha256((OUT / 'report.json').read_bytes()).hexdigest(),
    'current_goal_turn': 'progress', 'previous_goal_turn': 'progress', 'no_progress_streak': 0, 'goal_complete': False,
    'new_candidate_fixed12_result_available': True, 'both_new_rank16_formal_readers_passed': True,
    'human_random_Stem_main_comparator_persisted': True, 'all_listed_training_completed12': True,
    'all_original_formal_packets_collected_and_verified': False, 'pending': report['pending'], 'supersedes': 'execution_state_v48.json'}
assert not (state_folder / 'execution_state_v49.json').exists()
(state_folder / 'execution_state_v49.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'canonical_report': str(OUT / 'report.json'), 'rank16_matched_pretrained_BASE_gate': True, 'rank16_APs_above_original_P2': False, 'goal_complete': False}, ensure_ascii=False))
