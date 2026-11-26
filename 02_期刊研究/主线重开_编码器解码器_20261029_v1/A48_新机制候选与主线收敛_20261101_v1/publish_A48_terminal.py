"""Publish a verified terminal pilot while preserving prior signed state.

Requires three completed local bundles, judgment and fresh terminal observation.
Does not start/stop a GPU process or complete the broad research goal.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def replace_json(path, value):
    temp = path.with_name(path.name + '.A48_terminal.tmp')
    save_new(temp, value)
    os.replace(temp, path)


def verify_preserved(tree):
    if not isinstance(tree, dict):
        return 0
    if 'preserved_at' in tree and 'sha256' in tree:
        assert sha(tree['preserved_at']) == tree['sha256'], tree['preserved_at']
        return 1
    return sum(verify_preserved(v) for v in tree.values())


def own_live(identity):
    identity = identity or {}
    return (identity.get('present') and identity.get('same_start_ticks') and
            identity.get('owned_command') and identity.get('state') not in ['Z', 'X'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observer', required=True)
    args = parser.parse_args()
    observation_path = (ROOT / args.observer).resolve()
    assert observation_path.parent == ROOT
    raw = read(observation_path)
    assert raw['status'] == 'EXECUTED' and raw['exit'] == 0
    observed = json.loads(raw['stdout'])
    assert observed['observation_only']
    assert observed['three_arm_result']['status'] == observed['supervisor']['status'] == 'COMPLETED'
    assert not own_live(observed['supervisor_identity'])
    assert all(not own_live(observed['arms'][a]['identity']) for a in ['B', 'M', 'Q'])
    assert observed['frozen_file_checks']['all_pass'] and observed['frozen_file_checks']['count'] == 13
    protocol_sha = sha(ROOT / 'protocol_FROZEN.json')
    assert observed['protocol_sha256'] == protocol_sha
    judgment = read(ROOT / 'fixed_endpoint_judgment_A48.json')
    assert judgment['status'] == 'VALID_FIXED_ENDPOINT_COMPARISON'
    assert judgment['protocol_sha256'] == protocol_sha
    passed = judgment['all_frozen_gates_pass']
    assert passed == observed['three_arm_result']['pilot_pass']
    for a in ['B', 'M', 'Q']:
        verified = read(ROOT / (a + '_completed_local_verification.json'))
        assert verified['status'] == 'PASS' and verified['arm'] == a
        assert verified['endpoint_metrics'] == judgment['metrics_percent'][a] == observed['three_arm_result']['metrics'][a]
        assert sha(ROOT / (a + '_completed_bundle_v1.zip')) == verified['bundle_sha256']
    resource = judgment['resource_use']
    assert resource == {'formal_model_forwards': 30477, 'formal_optimizer_updates': 29124,
                        'formal_epochs': 36, 'capacity_model_forwards': 12, 'capacity_optimizer_updates': 9}
    # Validate everything before changing either canonical state file.
    previous = read(ROOT / '05_A48启动与历史保留校验回执.json')
    for name, digest in previous['current_files_sha256'].items():
        assert sha(name) == digest, name
    prior105 = verify_preserved(previous['previous_A47_all105_inputs_resolved'])
    prior122 = verify_preserved(previous['older_input_resolution'])
    assert prior105 == 105 and prior122 == 122
    interim_names = ['08_A48本轮交付校验.json', '10_续接进展校验_v1.json',
                     '13_B终点与继续验证回执.json', '17_M终点与继续验证回执.json',
                     'Q_epoch04_intermediate_review.json', 'Q_epoch08_intermediate_review.json']
    interim_checks = {}
    for name in interim_names:
        rows = read(ROOT / name)['files_sha256']
        for path, digest in rows.items():
            p = Path(path) if Path(path).is_absolute() else ROOT / path
            assert sha(p) == digest, (name, path)
        interim_checks[name] = {'signed_items_verified': len(rows), 'sha256': sha(ROOT / name)}
    assert (ROOT / 'A48_三组固定终点结论.txt').is_file()
    backup_source_paths = [ROOT.parents[1] / '文献_相关原始论文_20261004' / 'A48_backup_reading_registry_v1.json']
    assert backup_source_paths[0].is_file()
    for source in read(ROOT / 'backup_source_acquisition_v1.json'):
        path = Path(source['path'])
        assert sha(path) == source['sha256']
        backup_source_paths.append(path)
    receipt_path = ROOT / '21_A48终点与历史保留校验回执.json'
    before = ROOT / 'before_A48_terminal'
    assert not before.exists() and not receipt_path.exists()
    before.mkdir()
    remapped = {}
    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
        original = BASE / name
        saved = before / name
        shutil.copyfile(original, saved)
        assert sha(saved) == previous['current_files_sha256'][str(original)]
        remapped[str(original)] = {'preserved_at': str(saved), 'sha256': sha(saved)}
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    terminal_status = 'PILOT_GATES_PASSED_NOVELTY_UNCONFIRMED' if passed else 'RETIRED_EXACT_RECIPE_AFTER_FIXED12'
    report_path = ROOT / 'A48_三组固定终点结论.txt'
    evidence = {'phase': 'A48', 'status': terminal_status, 'goal_status': 'active',
                'mainline_confirmed': False, 'dataset': 'VisDrone2019-DET train6471/val548',
                'host': 'ECDet-S, pinned EdgeCrafter d8752b75c9beaecfcdd0e5cd207bae3b6946ae23',
                'candidate': 'Training-only per-GT-instance equal-mass encoder BN moments',
                'all_frozen_gates_pass': passed, 'metrics_percent': judgment['metrics_percent'],
                'deltas_pp': judgment['deltas_percentage_points'], 'gates': judgment['frozen_gates'],
                'new_formal_model_forwards': 30477, 'new_capacity_model_forwards': 12,
                'new_formal_optimizer_updates': 29124, 'new_capacity_optimizer_updates': 9,
                'new_formal_epochs': 36, 'all_three_bundles_verified': True,
                'full_12_epoch_common_flow_checks': True, 'no_owned_live_training_job': True,
                'new_training_automatically_queued': False, 'protocol_sha256': protocol_sha,
                'report': str(report_path), 'judgment': str(ROOT / 'fixed_endpoint_judgment_A48.json'),
                'terminal_observation': str(observation_path),
                'limits': judgment['limits']}
    question = ('Evaluate whether the positive task-specific normalization pilot has sufficient novelty and robustness.'
                if passed else 'Find a substantively different next problem and algorithm; do not retune the retired A48 recipe.')
    board = read(before / '01_候选判别表.json')
    board.update(latest_update_utc=utc, date='2026-11-01', goal_status='active',
                 status=terminal_status, mainline_validated=False, mainline_selected_for_pilot=passed,
                 selection_stage_complete=False,
                 current_priority_candidate='A48 实例等量统计（仅短程门槛通过，主线待定）' if passed else None,
                 current_priority_question=question, latest_report=str(report_path),
                 A48_terminal_result=evidence,
                 remaining_deliverable=('进一步评估最近工作差异与独立验证；不把门槛通过当创新成立。' if passed
                                        else '停止A48配方；继续选择有具体问题和可区分操作的不同机制候选，不把本次否定结果当总目标完成。'))
    matches = [row for row in board['candidates'] if row.get('id') == 'A48']
    assert len(matches) == 1
    matches[0].update(status=terminal_status, mainline_validated=False,
                      pilot_pass=passed, terminal_report=str(report_path))
    state = read(before / '15_当前状态与结果校验.json')
    state.update(recorded_utc=utc, goal_status='active', mainline_validated=False,
                 diagnostics_complete=True, diagnostics_complete_scope='A48 fixed12 pilot only; broad mainline goal is incomplete.',
                 latest_authoritative_phase='A48_FIXED12_TERMINAL', latest_report=str(report_path),
                 current_priority_question=question, current_priority_candidate=board['current_priority_candidate'],
                 mainline_selected_for_pilot=passed, current_native_job_terminal=True,
                 model_forwards=114452, training_updates=87926,
                 formal_detector_training_updates=87898, capacity_only_optimizer_updates=28,
                 detector_optimizer_updates_including_capacity=87926, formal_training_epochs=108,
                 training_update_scope='Inherited A47 total58793 updates including19capacity, plus A48 formal29124 and capacity9. Frozen teacher update counts unchanged.',
                 training_counts_terminal=True, counters_observed_lower_bounds_while_job_live=False,
                 counts_terminal=True, full_forward_cap_closed=True, cap=114452,
                 cap_scope='A48 ended at30477formal+12capacity forwards. This stage cap is closed, not a numeric goal-token budget.',
                 A48_terminal_result=evidence,
                 current_goal_turn={'classification': 'progress', 'goal_status': 'active',
                                    'mainline_confirmed': False, 'not_complete_just_because_one_candidate_tested': True})
    state.setdefault('authoritative_phase_history', []).append({'utc': utc, 'phase': 'A48_TERMINAL',
                    'status': terminal_status, 'mainline_confirmed': False})
    replace_json(BASE / '01_候选判别表.json', board)
    replace_json(BASE / '15_当前状态与结果校验.json', state)
    resolved = {name: remapped.get(name, {'preserved_at': name, 'sha256': digest})
                for name, digest in previous['current_files_sha256'].items()}
    assert verify_preserved(resolved) == len(previous['current_files_sha256'])
    names = ['fixed_endpoint_judgment_A48.json', 'A48_三组固定终点结论.txt',
             'publish_A48_terminal.py', 'judge_fixed_endpoint_A48.py', '19_等待训练期间的备选核查_非新主线.txt',
             'backup_method_screen_v1.json', 'backup_source_acquisition_v1.json'] + interim_names
    names += [a + suffix for a in ['B', 'M', 'Q'] for suffix in ['_completed_local_verification.json', '_bundle_ready.json']]
    paths = [ROOT / name for name in names] + [observation_path, BASE / '01_候选判别表.json', BASE / '15_当前状态与结果校验.json']
    paths += backup_source_paths
    output = {'status': 'PASS', 'goal_status': 'active', 'mainline_confirmed': False,
              'phase': 'A48_TERMINAL_PUBLISHED', 'candidate_status': terminal_status,
              'previous_A48_signed_current_inputs_resolved': resolved,
              'previous_A47_all105_inputs_resolved': previous['previous_A47_all105_inputs_resolved'],
              'older_input_resolution': previous['older_input_resolution'],
              'verified_prior_A47_input_count': prior105, 'verified_older_input_count': prior122,
              'intermediate_receipts_verified': interim_checks,
              'current_files_sha256': {str(p): sha(p) for p in paths},
              'cumulative_counts': {'model_forwards': 114452, 'optimizer_updates_including_capacity': 87926,
                                   'formal_optimizer_updates': 87898, 'capacity_updates': 28, 'formal_epochs': 108},
              'owned_jobs_terminal': True, 'new_detector_operations_by_publication': 0,
              'new_optimizer_jobs_by_publication': 0, 'next_candidate_automatically_queued': False,
              'previous_startup_receipt_sha256': sha(ROOT / '05_A48启动与历史保留校验回执.json')}
    save_new(receipt_path, output)
    pointer = read(ROOT / 'CURRENT_A48.json')
    pointer.update(terminal_judgment=str(ROOT / 'fixed_endpoint_judgment_A48.json'),
                   terminal_receipt=str(receipt_path), candidate_status=terminal_status,
                   next_action=question, goal_status='active', mainline_confirmed=False)
    replace_json(ROOT / 'CURRENT_A48.json', pointer)
    print(json.dumps({'status': 'PASS', 'goal_status': 'active', 'mainline_confirmed': False,
                      'candidate_status': terminal_status, 'old_A48_inputs_preserved': len(resolved),
                      'old_A47_inputs_preserved': prior105, 'older_inputs_preserved': prior122,
                      'counts': output['cumulative_counts']}, ensure_ascii=False))
