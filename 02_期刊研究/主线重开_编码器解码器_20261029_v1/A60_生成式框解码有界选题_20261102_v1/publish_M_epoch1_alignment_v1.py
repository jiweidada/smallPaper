"""Publish the original M epoch-one stream audit and prepared terminal backups."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
B = P.parent
D = B / 'A54_场景尺度约束候选_20261101_v1'
PARENT = P / 'A60_B12_geometry_evidence_preservation_receipt_v1.json'
SUPPLEMENT = P / 'A60_MQ_archive_preparation_receipt_v1.json'
SNAP = P / 'before_M_epoch1_alignment_publish_v1'
REPORT = P / '15_M第一轮输入流核验.txt'
HANDOFF = P / 'A60_turn_handoff_M_epoch1_v1.json'
OBSERVATION = P / 'formal_live_observation_239_v1.json'
RECEIPT = P / 'A60_M_epoch1_alignment_preservation_receipt_v1.json'
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
    temporary = path.with_name(path.name + '.M_epoch1_alignment_publish_tmp')
    write_new(temporary, value)
    temporary.replace(path)


def main():
    assert REPORT.is_file() and all(not path.exists() for path in [SNAP, HANDOFF, OBSERVATION, RECEIPT])
    assert all(not path.with_name(path.name + '.M_epoch1_alignment_publish_tmp').exists() for path in ROOTS)
    parent, supplement = read(PARENT), read(SUPPLEMENT)
    assert supplement['parent_receipt_sha256'] == sha(PARENT)
    for receipt in [parent, supplement]:
        for name, expected in receipt['files'].items():
            path = Path(name)
            assert path.is_file() and metadata(path) == expected, f'Prior evidence changed: {path}'
    for path in ROOTS:
        assert str(path) in parent['files']
    alignment = read(P / 'M_epoch1_raw_stream_verification_v1.json')
    preparation = read(P / 'A60_turn_handoff_MQ_archive_prepared_v1.json')
    assert alignment['status'] == 'PASS_M_EPOCH1_ACTUAL_RAW_STREAM_MATCHES_B'
    assert alignment['M_epoch1_images'] == 6471 and alignment['M_epoch1_updates'] == 809
    assert not alignment['M_vs_B_AP_available'] and not alignment['candidate_efficacy_tested']
    assert alignment['new_model_forwards'] == alignment['new_optimizer_updates'] == alignment['new_COCO_evaluations'] == 0
    assert not preparation['M_Q_archive_executed'] and not preparation['Q_started']
    receipt = read(D / 'A60_formal_observe_239.json')
    assert receipt['status'] == 'EXECUTED' and receipt['exit'] == 0
    obs = json.loads(receipt['stdout']); base, mid, candidate = (obs['arms'][x] for x in ['B', 'M', 'Q'])
    assert obs['supervisor']['identity_matched'] and obs['supervisor']['live'] and obs['source_binding']
    assert base['exit']['exit'] == 0 and base['completed_epochs'] == 12
    assert mid['process']['identity_matched'] and mid['process']['live'] and mid['progress']['epoch'] == 2
    assert mid['launch']['pid'] == alignment['M_original_pid'] and mid['launch']['start_ticks'] == alignment['M_start_ticks']
    assert not candidate['exists'] and mid['completed_epochs'] == 1
    assert mid['epochs'][0]['flow']['raw_stream_sha256'] == alignment['B_M_raw_stream_sha256']
    starts = sum(x['journal_counts']['started'] for x in obs['arms'].values())
    updates = sum(x['journal_counts']['updates'] for x in obs['arms'].values())
    epochs = sum(x['completed_epochs'] for x in obs['arms'].values())
    assert starts >= 78514 + 6471 and updates >= 9708 + 809 and epochs == 13

    SNAP.mkdir()
    resolution = {}
    for path in ROOTS:
        target = SNAP / path.name
        shutil.copy2(path, target)
        assert metadata(target) == parent['files'][str(path)]
        resolution[str(path)] = {'resolved_path': str(target), **metadata(target)}
    write_new(OBSERVATION, obs)
    status = 'A60_M_EPOCH1_RAW_STREAM_VERIFIED_ORIGINAL_M2_LIVE'
    evidence = {
        'status': alignment['status'], 'verification': str(P / 'M_epoch1_raw_stream_verification_v1.json'),
        'report': str(REPORT), 'M_epoch1_images': alignment['M_epoch1_images'],
        'M_epoch1_updates': alignment['M_epoch1_updates'], 'B_M_raw_stream_sha256': alignment['B_M_raw_stream_sha256'],
        'M_B_noisy_stream_differ': True, 'M_B_condition_stream_differ': True,
        'whole_first_epoch_actual_stream_hash_equal': True, 'all_12_epochs_verified': False,
        'M_AP_available': False, 'candidate_Q_tested': False,
        'scope': 'Original stored raw-input stream hash and counts; no model replay or independent image rehash.',
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
        'M_epoch1_raw_stream_alignment': evidence,
        'MQ_fixed12_archive_preparation': str(P / 'A60_turn_handoff_MQ_archive_prepared_v1.json'),
        'scope': 'B12 verified; M epoch1 raw stream matches B, noisy/condition differ as intended; M/Q efficacy untested.',
        'previous_goal_turn_classification': 'progress',
        'current_goal_turn_classification': 'progress: original M first full-epoch input alignment verified',
        'latest_report': str(REPORT), 'report': str(REPORT), 'latest_handoff': str(HANDOFF),
        'next_readonly_receipt': 'A60_formal_observe_240.json',
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
        'B12_geometry_coverage': current['B12_geometry_coverage_evidence'],
        'MQ_terminal_archive_preparation': str(P / 'A60_turn_handoff_MQ_archive_prepared_v1.json'),
        'Q_started': False, 'candidate_efficacy_tested': False, 'training_protocol_modified': False,
        'bridge_session_id': 19671, 'next_readonly_receipt_hint': 'A60_formal_observe_240.json',
        'numbering_rule': 'Derive next receipt from actual files; old hints can already be consumed.',
        'remaining_action': 'Observe original M at fixed4/8/12; after original M exit/reap run predeployed archive exactly once and verify locally. Original supervisor then runs Q; verify all 12 raw and M/Q noisy streams and frozen judgment.',
        'risk': 'B12 output coverage remains weak, but cause and Q efficacy are unproven; A58 cross-host gap is confounded and CEOSOD full method remains unread.',
    }
    write_new(HANDOFF, handoff)
    board, state = read(ROOTS[0]), read(ROOTS[1])
    board.update({'status': status, 'latest_update_utc': obs['recorded_utc'], 'training_running': True,
                  'A60_execution': current, 'A60_M_epoch1_raw_stream_alignment': evidence, 'latest_report': str(REPORT)})
    state.update({'status': status, 'recorded_utc': obs['recorded_utc'], 'latest_update_utc': obs['recorded_utc'],
                  'training_running': True, 'A60_execution': current, 'A60_M_epoch1_raw_stream_alignment': evidence,
                  'latest_report': str(REPORT), 'model_forwards': current['global_observed_model_forwards'],
                  'training_updates': current['global_observed_all_optimizer_updates'],
                  'detector_optimizer_updates_including_capacity': current['global_observed_all_optimizer_updates'],
                  'formal_detector_training_updates': current['global_observed_formal_optimizer_updates'],
                  'formal_optimizer_updates_observed': current['global_observed_formal_optimizer_updates'],
                  'formal_training_epochs': current['global_completed_formal_epochs'],
                  'previous_goal_turn_classification': 'progress',
                  'current_goal_turn': 'PROGRESS_M_EPOCH1_ALIGNMENT_AND_ORIGINAL_M2_LIVE'})
    for path, value in zip(ROOTS, [board, state, current]):
        replace_json(path, value)
    files = dict(parent['files'])
    for name, expected in supplement['files'].items():
        if name in files:
            assert files[name] == expected
        files[name] = expected
    new_paths = [*ROOTS, *[Path(x['resolved_path']) for x in resolution.values()], PARENT, SUPPLEMENT,
                 Path(__file__), REPORT, HANDOFF, OBSERVATION,
                 P / 'M_epoch1_raw_stream_verification_v1.json',
                 *[D / f'A60_formal_observe_{i:03d}.json' for i in range(237, 240)]]
    for path in new_paths:
        assert path.is_file(), str(path)
        files[str(path)] = metadata(path)
    output = {'status': status, 'goal_status': 'active', 'mainline_validated': False,
              'parents': [{'path': str(PARENT), 'sha256': sha(PARENT)},
                          {'path': str(SUPPLEMENT), 'sha256': sha(SUPPLEMENT)}],
              'parent_files_verified_before_publish': len(parent['files']),
              'supplement_files_verified_before_publish': len(supplement['files']),
              'parent_path_resolution': resolution, 'files': files,
              'scope': 'M1 saved input-stream summary and prepared terminal backup; no replay, new update, gate change or loss of prior evidence.'}
    write_new(RECEIPT, output)
    print(json.dumps({'status': status, 'parent_files_verified': len(parent['files']),
                      'supplement_files_verified': len(supplement['files']), 'receipt_files': len(files),
                      'M_current_epoch': mid['progress']['epoch'], 'M_epoch1_images': alignment['M_epoch1_images']}))


if __name__ == '__main__':
    main()
