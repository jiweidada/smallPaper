"""Publish the new saved-output comparison without changing the active experiment."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
B = P.parent
D = B / 'A54_场景尺度约束候选_20261101_v1'
PARENT = P / 'A60_B4_B8_local_backup_preservation_receipt_v1.json'
SUPPLEMENT = P / 'A60_B12_archive_preparation_receipt_v1.json'
SNAP = P / 'before_A58_A60_baseline_context_publish_v1'
REPORT = P / '12_现有ECDet与当前生成式基线的可比性.txt'
HANDOFF = P / 'A60_turn_handoff_A58_A60_context_v1.json'
OBSERVATION = P / 'formal_live_observation_168_v1.json'
RECEIPT = P / 'A60_A58_A60_context_preservation_receipt_v1.json'
ROOTS = [B / '01_候选判别表.json', B / '15_当前状态与结果校验.json', P / 'CURRENT_A60.json']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def metadata(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def replace_json(path, value):
    temporary = path.with_name(path.name + '.A58_A60_context_tmp')
    write_new(temporary, value)
    temporary.replace(path)


def main():
    assert REPORT.is_file() and all(not x.exists() for x in [SNAP, HANDOFF, OBSERVATION, RECEIPT])
    assert all(not path.with_name(path.name + '.A58_A60_context_tmp').exists() for path in ROOTS)
    parent, supplement = read(PARENT), read(SUPPLEMENT)
    assert supplement['parent_receipt_sha256'] == sha(PARENT)
    for receipt in [parent, supplement]:
        for name, expected in receipt['files'].items():
            path = Path(name)
            assert path.is_file() and metadata(path) == expected, f'Prior evidence changed: {path}'
    for path in ROOTS:
        assert str(path) in parent['files']

    comparison = read(P / 'A58_A60_saved_baseline_comparison_v1.json')
    assert comparison['status'] == 'COMPLETED_EXPLORATORY_SAVED_B8_PIPELINE_COMPARISON'
    assert all(comparison['validation_GT_fields_identical'].values())
    assert comparison['new_forwards'] == comparison['new_updates'] == comparison['new_COCO_evaluations'] == 0
    assert comparison['new_matching_passes'] == 0
    archive = read(P / 'A60_turn_handoff_B12_archive_prepared_v1.json')
    assert not archive['archive_executed']
    receipt = read(D / 'A60_formal_observe_168.json')
    assert receipt['status'] == 'EXECUTED' and receipt['exit'] == 0
    obs = json.loads(receipt['stdout']); arm = obs['arms']['B']
    assert obs['supervisor']['identity_matched'] and obs['supervisor']['live'] and obs['source_binding']
    assert arm['process']['identity_matched'] and arm['process']['live']
    assert arm['completed_epochs'] == 9 and arm['progress']['epoch'] == 10
    assert not obs['arms']['M']['exists'] and not obs['arms']['Q']['exists']
    counts = arm['journal_counts']

    SNAP.mkdir()
    resolution = {}
    for path in ROOTS:
        target = SNAP / path.name
        shutil.copy2(path, target)
        assert metadata(target) == parent['files'][str(path)]
        resolution[str(path)] = {'resolved_path': str(target), **metadata(target)}
    write_new(OBSERVATION, obs)

    status = 'A60_A58_SAME_GT_CONTEXT_ORIGINAL_B10_LIVE'
    context = {
        'status': comparison['status'], 'report': str(REPORT),
        'result': str(P / 'A58_A60_saved_baseline_comparison_v1.json'),
        'scope': str(P / 'A58_A60_saved_baseline_comparison_scope_FROZEN_v1.json'),
        'scope_supplement': str(P / 'A58_A60_saved_baseline_comparison_scope_supplement_v1.json'),
        'same_validation_GT': True, 'A58_B8_EMA_AP': comparison['A58_B8_EMA_metrics']['AP'],
        'A58_B8_EMA_APs': comparison['A58_B8_EMA_metrics']['APs'],
        'A60_B8_raw_AP': comparison['A60_B8_raw_metrics']['AP'],
        'A60_B8_raw_APs': comparison['A60_B8_raw_metrics']['APs'],
        'A60_minus_A58_B8_AP_pp': comparison['A60_minus_A58_B8_pp_observed_pipeline_difference']['AP'],
        'A60_minus_A58_B8_APs_pp': comparison['A60_minus_A58_B8_pp_observed_pipeline_difference']['APs'],
        'A60_to_A58_registered_parameter_ratio': comparison['A60_to_A58_registered_parameter_ratio'],
        'directly_causally_matched': False, 'different_model_state': 'A58 EMA; A60 raw',
        'other_confounders': 'Different architecture, initialization, seed, optimizer, batch reduction and postprocessing.',
        'candidate_Q_tested': False, 'protocol_or_gate_changed': False,
        'meaning': 'Practical competitiveness risk; do not infer Q failure or add an exploratory stopping gate.',
    }
    current = read(ROOTS[2])
    current.update({
        'status': status, 'recorded_utc': obs['recorded_utc'], 'snapshot_is_live_lower_bound': True,
        'original_supervisor': obs['supervisor'], 'arms': obs['arms'], 'source_binding_verified': obs['source_binding'],
        'observed_formal_model_forwards': counts['started'], 'observed_formal_updates': counts['updates'],
        'completed_formal_epochs': arm['completed_epochs'],
        'global_observed_model_forwards': 276643 + counts['started'],
        'global_observed_all_optimizer_updates': 233606 + counts['updates'],
        'global_observed_formal_optimizer_updates': 233518 + counts['updates'],
        'global_completed_formal_epochs': 288 + arm['completed_epochs'],
        'previous_goal_turn_classification': 'verified wait',
        'current_goal_turn_classification': 'progress: practical ECDet-S baseline comparison from saved outputs',
        'A58_A60_saved_baseline_context': context,
        'latest_report': str(REPORT), 'report': str(REPORT), 'latest_handoff': str(HANDOFF),
        'next_readonly_receipt': 'A60_formal_observe_169.json',
    })
    handoff = {
        'status': status, 'goal_status': 'active', 'mainline_validated': False,
        'previous_goal_turn_classification': 'verified wait', 'this_goal_turn_classification': 'progress',
        'recorded_remote_utc': obs['recorded_utc'],
        'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
        'snapshot': str(OBSERVATION), 'canonical': str(ROOTS[2]), 'context': context,
        'original_supervisor': obs['supervisor'], 'original_B': arm['process'],
        'actual_journal_lower_bound': counts, 'B_completed_epochs': arm['completed_epochs'],
        'B_current_epoch': arm['progress']['epoch'], 'B12_gate_completed': False,
        'M_started': False, 'Q_started': False, 'candidate_efficacy_tested': False,
        'B12_archive_preparation_handoff': str(P / 'A60_turn_handoff_B12_archive_prepared_v1.json'),
        'archive_executed': False, 'training_protocol_modified': False,
        'bridge_session_id': 19671, 'next_readonly_receipt_hint': 'A60_formal_observe_169.json',
        'numbering_rule': 'Derive next receipt from actual files; hints can already be consumed.',
        'remaining_action': 'Observe original B12 learning gate and original M/Q queue; archive B12 once gate and worker-reaped records exist. Never change gates, relaunch or retry.',
        'risk': 'B8 complete-pipeline gap and parameter disadvantage are real observations but causally confounded and do not test Q.',
    }
    write_new(HANDOFF, handoff)
    board, state = read(ROOTS[0]), read(ROOTS[1])
    board.update({'status': status, 'latest_update_utc': obs['recorded_utc'], 'training_running': True,
                  'A60_execution': current, 'A60_A58_saved_baseline_context': context, 'latest_report': str(REPORT)})
    state.update({'status': status, 'recorded_utc': obs['recorded_utc'], 'latest_update_utc': obs['recorded_utc'],
                  'training_running': True, 'A60_execution': current, 'A60_A58_saved_baseline_context': context,
                  'latest_report': str(REPORT), 'model_forwards': current['global_observed_model_forwards'],
                  'training_updates': current['global_observed_all_optimizer_updates'],
                  'detector_optimizer_updates_including_capacity': current['global_observed_all_optimizer_updates'],
                  'formal_detector_training_updates': current['global_observed_formal_optimizer_updates'],
                  'formal_optimizer_updates_observed': current['global_observed_formal_optimizer_updates'],
                  'formal_training_epochs': current['global_completed_formal_epochs'],
                  'previous_goal_turn_classification': 'verified wait',
                  'current_goal_turn': 'PROGRESS_A58_A60_BASELINE_CONTEXT_AND_ORIGINAL_B10_LIVE'})
    for path, value in zip(ROOTS, [board, state, current]):
        replace_json(path, value)

    files = dict(parent['files'])
    for name, expected in supplement['files'].items():
        if name in files:
            assert files[name] == expected
        files[name] = expected
    scope = read(P / 'A58_A60_saved_baseline_comparison_scope_FROZEN_v1.json')
    scope2 = read(P / 'A58_A60_saved_baseline_comparison_scope_supplement_v1.json')
    new_paths = [*ROOTS, *[Path(x['resolved_path']) for x in resolution.values()], PARENT, SUPPLEMENT,
                 Path(__file__), REPORT, HANDOFF, OBSERVATION,
                 P / 'A58_A60_saved_baseline_comparison_v1.json',
                 P / 'A58_A60_saved_baseline_comparison_scope_FROZEN_v1.json',
                 P / 'A58_A60_saved_baseline_comparison_scope_supplement_v1.json',
                 P / 'compare_saved_A58_A60_baselines_v1.py',
                 *[Path(x['path']) for x in scope['data'].values()],
                 *[Path(x['path']) for x in scope2['data'].values()],
                 *[D / f'A60_formal_observe_{i:03d}.json' for i in range(166, 169)]]
    for path in new_paths:
        assert path.is_file(), str(path)
        files[str(path)] = metadata(path)
    output = {'status': status, 'goal_status': 'active', 'mainline_validated': False,
              'parents': [{'path': str(PARENT), 'sha256': sha(PARENT)},
                          {'path': str(SUPPLEMENT), 'sha256': sha(SUPPLEMENT)}],
              'parent_files_verified_before_publish': len(parent['files']),
              'supplement_files_verified_before_publish': len(supplement['files']),
              'parent_path_resolution': resolution, 'files': files,
              'scope': 'Saved-output context; no model forwards, updates, COCO rematch, protocol change or loss of prior evidence.'}
    write_new(RECEIPT, output)
    print(json.dumps({'status': status, 'parent_files_verified': len(parent['files']),
                      'supplement_files_verified': len(supplement['files']), 'receipt_files': len(files),
                      'B_current_epoch': arm['progress']['epoch'], 'A60_minus_A58_B8_APs_pp': context['A60_minus_A58_B8_APs_pp']}))


if __name__ == '__main__':
    main()
