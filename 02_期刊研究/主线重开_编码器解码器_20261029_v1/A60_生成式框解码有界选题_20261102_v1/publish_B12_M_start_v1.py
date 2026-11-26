"""Publish original B12 and live M after one-time local artifact verification."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
B = P.parent
D = B / 'A54_场景尺度约束候选_20261101_v1'
PARENT = P / 'A60_A58_A60_context_preservation_receipt_v1.json'
SNAP = P / 'before_B12_M1_publish_v1'
REPORT = P / '13_B组固定12轮终点与M组启动.txt'
HANDOFF = P / 'A60_turn_handoff_B12_M1_v1.json'
OBSERVATION = P / 'formal_live_observation_229_v1.json'
RECEIPT = P / 'A60_B12_M1_preservation_receipt_v1.json'
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
    temporary = path.with_name(path.name + '.B12_M1_publish_tmp')
    write_new(temporary, value)
    temporary.replace(path)


def main():
    assert REPORT.is_file() and all(not x.exists() for x in [SNAP, HANDOFF, OBSERVATION, RECEIPT])
    assert all(not path.with_name(path.name + '.B12_M1_publish_tmp').exists() for path in ROOTS)
    parent = read(PARENT)
    for name, expected in parent['files'].items():
        path = Path(name)
        assert path.is_file() and metadata(path) == expected, f'Prior evidence changed: {path}'
    for path in ROOTS:
        assert str(path) in parent['files']
    verification = read(P / 'B_fixed12_local_verification_v1.json')
    summary = read(P / 'A60_B12_M_start_summary_v1.json')
    alignment = read(P / 'B_M_first8_alignment_verified_v1.json')
    assert verification['status'] == 'PASS_ORIGINAL_B_FIXED12_ARTIFACTS_AND_LOCAL_MODEL_BACKUP'
    assert verification['gate']['passed'] and verification['new_model_forwards'] == 0
    assert verification['new_optimizer_updates'] == verification['new_COCO_evaluations'] == 0
    assert sha(Path(verification['checkpoint_path'])) == verification['checkpoint_sha256']
    assert alignment['status'] == 'PASS_B_M_COMMON_INITIALIZATION_AND_FIRST8_RAW_INPUTS'
    assert summary['baseline']['metrics'] == verification['metrics'] and not summary['candidate_efficacy_tested']
    exported = read(D / 'A60_B_fixed12_export_v1.json')
    assert exported['status'] == 'EXECUTED' and exported['exit'] == 0
    assert json.loads(exported['stdout'])['checkpoint_sha256'] == verification['checkpoint_sha256']
    transfer = read(D / 'A60_B_fixed12_download_v1.json')
    assert transfer['status'] == 'TRANSFERRED' and transfer['operation'] == 'get'
    obs_receipt = read(D / 'A60_formal_observe_229.json')
    assert obs_receipt['status'] == 'EXECUTED' and obs_receipt['exit'] == 0
    obs = json.loads(obs_receipt['stdout']); base, mid, candidate = (obs['arms'][x] for x in ['B', 'M', 'Q'])
    assert obs['supervisor']['identity_matched'] and obs['supervisor']['live'] and obs['source_binding']
    assert base['exit']['exit'] == 0 and base['completed_epochs'] == 12
    assert base['journal_counts'] == {'started': 78514, 'completed': 78514, 'updates': 9708}
    assert mid['process']['identity_matched'] and mid['process']['live'] and mid['progress']['epoch'] == 1
    assert mid['launch']['pid'] == alignment['original_M_pid'] and mid['launch']['start_ticks'] == alignment['M_start_ticks']
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
    status = 'A60_B_FIXED12_VERIFIED_ORIGINAL_M1_RUNNING'
    reference = summary['practical_reference']
    milestone = {
        'status': verification['status'], 'metrics': verification['metrics'], 'learning_gate': verification['gate'],
        'B12_minus_B8_pp': verification['B12_minus_B8_pp'], 'original_counts': verification['original_counts'],
        'local_verification': str(P / 'B_fixed12_local_verification_v1.json'),
        'local_model_checkpoint': verification['checkpoint_path'], 'checkpoint_sha256': verification['checkpoint_sha256'],
        'B_M_first8_alignment': str(P / 'B_M_first8_alignment_verified_v1.json'),
        'full_M_epoch_alignment_verified': False, 'Q_started': False, 'candidate_efficacy_tested': False,
        'A58_B12_AP': reference['A58_metrics']['AP'], 'A58_B12_APs': reference['A58_metrics']['APs'],
        'A60_minus_A58_B12_AP_pp': reference['A60_minus_A58_AP_pp'],
        'A60_minus_A58_B12_APs_pp': reference['A60_minus_A58_APs_pp'],
        'A60_model_latency_median_ms': summary['baseline']['latency_median_ms'],
        'A58_model_latency_median_ms': reference['A58_latency_median_ms'],
        'cross_host_comparison_causal': False, 'report': str(REPORT),
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
        'B12_learning_gate_completed': True, 'B12_learning_gate_passed': True,
        'B12_raw_metrics': verification['metrics'], 'B12_M1_milestone': milestone,
        'currently_running_arm': 'M', 'remaining_authorized_queue': ['Q'],
        'queue_conditions': 'B12 learning gate passed; original supervisor starts Q after successful fixed12 M and its integrity checks; no retry.',
        'next_endpoint': 'Original M fixed12 verification, then original Q fixed12 and frozen B/M/Q judgment.',
        'scope': 'B12 fixed raw metrics and local weight verified; M first8 inputs aligned, full M and Q untested.',
        'previous_goal_turn_classification': 'verified wait',
        'current_goal_turn_classification': 'progress: B12 fixed endpoint backed up and M first8 alignment verified',
        'latest_report': str(REPORT), 'report': str(REPORT), 'latest_handoff': str(HANDOFF),
        'next_readonly_receipt': 'A60_formal_observe_230.json',
    })
    handoff = {
        'status': status, 'goal_status': 'active', 'mainline_validated': False,
        'previous_goal_turn_classification': 'verified wait', 'this_goal_turn_classification': 'progress',
        'recorded_remote_utc': obs['recorded_utc'],
        'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
        'snapshot': str(OBSERVATION), 'canonical': str(ROOTS[2]), 'milestone': milestone,
        'original_supervisor': obs['supervisor'], 'original_M': mid['process'], 'M_launch': mid['launch'],
        'M_current_epoch': mid['progress']['epoch'], 'M_completed_epochs': mid['completed_epochs'],
        'M_journal_lower_bound': mid['journal_counts'], 'B_original_exit': base['exit'],
        'Q_started': False, 'candidate_efficacy_tested': False, 'training_protocol_modified': False,
        'bridge_session_id': 19671, 'next_readonly_receipt_hint': 'A60_formal_observe_230.json',
        'numbering_rule': 'Derive fresh number from actual receipts; hints may already be consumed.',
        'B12_export_completed_once': True, 'B12_backup_verified': verification['checkpoint_path'],
        'remaining_action': 'Observe original M, archive its fixed12 artifact after terminal, then observe original Q and frozen judgment. Do not relaunch, repeat B12 export, or change a gate.',
        'full_M_Q_raw_and_noisy_streams': 'M vs B raw stream per epoch and Q vs M noisy stream are enforced in original trainer/supervisor; M/Q full training remains unmeasured.',
        'risk': 'Q may improve within the host but the B12 complete pipeline trails the earlier ECDet-S result under confounded settings; closest CEOSOD full method remains unread.',
    }
    write_new(HANDOFF, handoff)
    board, state = read(ROOTS[0]), read(ROOTS[1])
    board.update({'status': status, 'latest_update_utc': obs['recorded_utc'], 'training_running': True,
                  'A60_execution': current, 'A60_B12_M1_milestone': milestone, 'latest_report': str(REPORT)})
    state.update({'status': status, 'recorded_utc': obs['recorded_utc'], 'latest_update_utc': obs['recorded_utc'],
                  'training_running': True, 'A60_execution': current, 'A60_B12_M1_milestone': milestone,
                  'latest_report': str(REPORT), 'model_forwards': current['global_observed_model_forwards'],
                  'training_updates': current['global_observed_all_optimizer_updates'],
                  'detector_optimizer_updates_including_capacity': current['global_observed_all_optimizer_updates'],
                  'formal_detector_training_updates': current['global_observed_formal_optimizer_updates'],
                  'formal_optimizer_updates_observed': current['global_observed_formal_optimizer_updates'],
                  'formal_training_epochs': current['global_completed_formal_epochs'],
                  'previous_goal_turn_classification': 'verified wait',
                  'current_goal_turn': 'PROGRESS_B12_VERIFIED_AND_ORIGINAL_M1_RUNNING'})
    for path, value in zip(ROOTS, [board, state, current]):
        replace_json(path, value)

    files = dict(parent['files'])
    new_paths = [*ROOTS, *[Path(x['resolved_path']) for x in resolution.values()], PARENT,
                 Path(__file__), REPORT, HANDOFF, OBSERVATION,
                 P / 'A60_B12_M_start_summary_v1.json', P / 'B_fixed12_local_verification_v1.json',
                 P / 'B_M_first8_alignment_verified_v1.json',
                 D / 'A60_B_fixed12_export_v1.json', D / 'A60_B_fixed12_download_v1.json',
                 D / 'A60_downloads/B_fixed12_export_ready_v1.json',
                 D / 'A60_downloads/B_fixed12_artifacts_bundle_v1.zip',
                 Path(verification['checkpoint_path']),
                 *[x for x in (P / 'fixed12_B_remote_v1').rglob('*') if x.is_file()],
                 *[D / f'A60_formal_observe_{i:03d}.json' for i in range(169, 230)]]
    for path in new_paths:
        assert path.is_file(), str(path)
        files[str(path)] = metadata(path)
    output = {'status': status, 'goal_status': 'active', 'mainline_validated': False,
              'parent_receipt': str(PARENT), 'parent_receipt_sha256': sha(PARENT),
              'parent_files_verified_before_publish': len(parent['files']),
              'parent_path_resolution': resolution, 'files': files,
              'scope': 'B12 saved artifacts and local weight verified; original M active; no replay, new update, gate change or loss of prior evidence.'}
    write_new(RECEIPT, output)
    print(json.dumps({'status': status, 'parent_files_verified': len(parent['files']),
                      'receipt_files': len(files), 'B_AP': verification['metrics']['AP'],
                      'B_APs': verification['metrics']['APs'], 'M_epoch': mid['progress']['epoch'],
                      'M_started_forwards_lower_bound': mid['journal_counts']['started']}))


if __name__ == '__main__':
    main()
