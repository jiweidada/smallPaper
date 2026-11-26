"""Publish B12 saved-box coverage and the original live M snapshot."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
B = P.parent
D = B / 'A54_场景尺度约束候选_20261101_v1'
PARENT = P / 'A60_B12_M1_preservation_receipt_v1.json'
SNAP = P / 'before_B12_geometry_evidence_publish_v1'
REPORT = P / '14_B12框覆盖证据与主线边界.txt'
HANDOFF = P / 'A60_turn_handoff_B12_geometry_v1.json'
OBSERVATION = P / 'formal_live_observation_232_v1.json'
RECEIPT = P / 'A60_B12_geometry_evidence_preservation_receipt_v1.json'
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
    temporary = path.with_name(path.name + '.B12_geometry_publish_tmp')
    write_new(temporary, value)
    temporary.replace(path)


def main():
    assert REPORT.is_file() and all(not path.exists() for path in [SNAP, HANDOFF, OBSERVATION, RECEIPT])
    assert all(not path.with_name(path.name + '.B12_geometry_publish_tmp').exists() for path in ROOTS)
    parent = read(PARENT)
    for name, expected in parent['files'].items():
        path = Path(name)
        assert path.is_file() and metadata(path) == expected, f'Prior evidence changed: {path}'
    for path in ROOTS:
        assert str(path) in parent['files']
    result = read(P / 'saved_B12_geometry_coverage_v1/result.json')
    verified = read(P / 'B12_geometry_coverage_local_verification_v1.json')
    scope = P / 'saved_B12_geometry_coverage_scope_FROZEN_v1.json'
    assert result['status'] == 'COMPLETED_SAVED_B12_PAIRED_GT_GEOMETRY_COVERAGE'
    assert verified['status'] == 'PASS_SAVED_B12_PER_GT_ARRAYS_AND_PAIRED_AGGREGATION'
    assert result['scope_sha256'] == verified['scope_sha256'] == sha(scope)
    assert verified['result_sha256'] == sha(P / 'saved_B12_geometry_coverage_v1/result.json')
    assert all(result[key] == 0 for key in ['new_model_forwards', 'new_optimizer_updates', 'new_COCO_evaluations', 'new_image_reads'])
    assert result['GT_prediction_IoU_pairs'] == 6093964 and not result['training_protocol_or_gates_changed']
    receipt = read(D / 'A60_formal_observe_232.json')
    assert receipt['status'] == 'EXECUTED' and receipt['exit'] == 0
    obs = json.loads(receipt['stdout']); base, mid, candidate = (obs['arms'][x] for x in ['B', 'M', 'Q'])
    assert obs['supervisor']['identity_matched'] and obs['supervisor']['live'] and obs['source_binding']
    assert base['exit']['exit'] == 0 and base['completed_epochs'] == 12
    assert mid['process']['identity_matched'] and mid['process']['live'] and mid['progress']['epoch'] == 1
    assert mid['launch']['pid'] == 35385 and mid['launch']['start_ticks'] == '32308346'
    assert not candidate['exists'] and mid['completed_epochs'] == 0
    starts = sum(x['journal_counts']['started'] for x in obs['arms'].values())
    updates = sum(x['journal_counts']['updates'] for x in obs['arms'].values())
    epochs = sum(x['completed_epochs'] for x in obs['arms'].values())
    assert starts >= 78514 and updates >= 9708 and epochs == 12

    SNAP.mkdir()
    resolution = {}
    for path in ROOTS:
        target = SNAP / path.name
        shutil.copy2(path, target)
        assert metadata(target) == parent['files'][str(path)]
        resolution[str(path)] = {'resolved_path': str(target), **metadata(target)}
    write_new(OBSERVATION, obs)
    status = 'A60_B12_PAIRED_GEOMETRY_EVIDENCE_ORIGINAL_M1_LIVE'
    small, tiny = verified['groups']['small'], verified['groups']['very_tiny']
    evidence = {
        'status': result['status'], 'scope': str(scope),
        'result': str(P / 'saved_B12_geometry_coverage_v1/result.json'),
        'verification': str(P / 'B12_geometry_coverage_local_verification_v1.json'), 'report': str(REPORT),
        'small_GT': small['GT'], 'small_no_any_IoU_ge_05_percent_B4_B8_B12': small['uncovered_percent_B4_B8_B12'],
        'small_uncovered_all_three': small['uncovered_all_three'],
        'very_tiny_GT': tiny['GT'], 'very_tiny_no_any_IoU_ge_05_percent_B4_B8_B12': tiny['uncovered_percent_B4_B8_B12'],
        'very_tiny_uncovered_all_three': tiny['uncovered_all_three'],
        'same_GT_id_and_geometry_verified': True, 'optimistic_many_to_many_coverage': True,
        'is_AP_or_recall': False, 'clock_cause_proven': False, 'candidate_efficacy_proven': False,
        'CPU_new_passes': 1, 'IoU_pairs': result['GT_prediction_IoU_pairs'],
        'new_model_forwards': 0, 'new_updates': 0, 'new_COCO_evaluations': 0, 'new_image_reads': 0,
    }
    current = read(ROOTS[2])
    current.update({
        'status': status, 'recorded_utc': obs['recorded_utc'], 'snapshot_is_live_lower_bound': True,
        'original_supervisor': obs['supervisor'], 'arms': obs['arms'], 'source_binding_verified': obs['source_binding'],
        'observed_formal_model_forwards': starts, 'observed_formal_updates': updates, 'completed_formal_epochs': epochs,
        'global_observed_model_forwards': 276643 + starts,
        'global_observed_all_optimizer_updates': 233606 + updates,
        'global_observed_formal_optimizer_updates': 233518 + updates,
        'global_completed_formal_epochs': 288 + epochs,
        'B12_geometry_coverage_evidence': evidence,
        'scope': 'B12 saved-output coverage gap persists, but its cause and Q efficacy remain unproven; original M1 live.',
        'previous_goal_turn_classification': 'progress',
        'current_goal_turn_classification': 'progress: paired fixed12 saved-box coverage evidence',
        'latest_report': str(REPORT), 'report': str(REPORT), 'latest_handoff': str(HANDOFF),
        'next_readonly_receipt': 'A60_formal_observe_233.json',
    })
    handoff = {
        'status': status, 'goal_status': 'active', 'mainline_validated': False,
        'previous_goal_turn_classification': 'progress', 'this_goal_turn_classification': 'progress',
        'recorded_remote_utc': obs['recorded_utc'],
        'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
        'snapshot': str(OBSERVATION), 'canonical': str(ROOTS[2]), 'evidence': evidence,
        'original_supervisor': obs['supervisor'], 'original_M': mid['process'], 'M_launch': mid['launch'],
        'M_current_epoch': mid['progress']['epoch'], 'M_completed_epochs': mid['completed_epochs'],
        'M_journal_lower_bound': mid['journal_counts'], 'B_original_exit': base['exit'],
        'B12_metrics': current['B12_raw_metrics'], 'B12_gate_passed': current['B12_learning_gate_passed'],
        'B12_local_weight': current['B12_M1_milestone']['local_model_checkpoint'],
        'Q_started': False, 'candidate_efficacy_tested': False, 'training_protocol_modified': False,
        'bridge_session_id': 19671, 'next_readonly_receipt_hint': 'A60_formal_observe_233.json',
        'numbering_rule': 'Derive next receipt from actual files; old hints can already be consumed.',
        'CPU_saved_prediction_IoU_passes_A60_total': 3,
        'CPU_saved_prediction_IoU_pairs_A60_total': 11866829 + result['GT_prediction_IoU_pairs'],
        'remaining_action': 'Continue original M fixed12 and then original Q; archive terminal weights and verify full raw/noisy stream alignment. Never relaunch or repeat consumed saved-box passes.',
        'risk': 'B12 output coverage is weak but does not identify cause; comparison against ECDet-S is confounded, and CEOSOD full method and Q efficacy remain unresolved.',
    }
    write_new(HANDOFF, handoff)
    board, state = read(ROOTS[0]), read(ROOTS[1])
    board.update({'status': status, 'latest_update_utc': obs['recorded_utc'], 'training_running': True,
                  'A60_execution': current, 'A60_B12_geometry_evidence': evidence, 'latest_report': str(REPORT)})
    assert state['CPU_saved_prediction_IoU_analysis_passes_A60'] == 2
    assert state['CPU_saved_prediction_IoU_pairs_A60'] == 11866829
    state.update({'status': status, 'recorded_utc': obs['recorded_utc'], 'latest_update_utc': obs['recorded_utc'],
                  'training_running': True, 'A60_execution': current, 'A60_B12_geometry_evidence': evidence,
                  'latest_report': str(REPORT), 'model_forwards': current['global_observed_model_forwards'],
                  'training_updates': current['global_observed_all_optimizer_updates'],
                  'detector_optimizer_updates_including_capacity': current['global_observed_all_optimizer_updates'],
                  'formal_detector_training_updates': current['global_observed_formal_optimizer_updates'],
                  'formal_optimizer_updates_observed': current['global_observed_formal_optimizer_updates'],
                  'formal_training_epochs': current['global_completed_formal_epochs'],
                  'CPU_saved_prediction_IoU_analysis_passes_A60': 3,
                  'CPU_saved_prediction_IoU_pairs_A60': 11866829 + result['GT_prediction_IoU_pairs'],
                  'previous_goal_turn_classification': 'progress',
                  'current_goal_turn': 'PROGRESS_B12_SAVED_GEOMETRY_AND_ORIGINAL_M1_LIVE'})
    for path, value in zip(ROOTS, [board, state, current]):
        replace_json(path, value)
    files = dict(parent['files'])
    new_paths = [*ROOTS, *[Path(x['resolved_path']) for x in resolution.values()], PARENT,
                 Path(__file__), REPORT, HANDOFF, OBSERVATION, scope,
                 P / 'saved_B12_geometry_coverage_v1.py', P / 'B12_geometry_coverage_local_verification_v1.json',
                 *[x for x in (P / 'saved_B12_geometry_coverage_v1').rglob('*') if x.is_file()],
                 *[D / f'A60_formal_observe_{i:03d}.json' for i in range(230, 233)]]
    for path in new_paths:
        assert path.is_file(), str(path)
        files[str(path)] = metadata(path)
    output = {'status': status, 'goal_status': 'active', 'mainline_validated': False,
              'parent_receipt': str(PARENT), 'parent_receipt_sha256': sha(PARENT),
              'parent_files_verified_before_publish': len(parent['files']),
              'parent_path_resolution': resolution, 'files': files,
              'scope': 'B12 saved-output descriptive coverage and original live M; no replay, new update, gate change or loss of prior evidence.'}
    write_new(RECEIPT, output)
    print(json.dumps({'status': status, 'parent_files_verified': len(parent['files']),
                      'receipt_files': len(files), 'M_epoch': mid['progress']['epoch'],
                      'small_B12_no_any_iou05_percent': small['uncovered_percent_B4_B8_B12'][2]}))


if __name__ == '__main__':
    main()
