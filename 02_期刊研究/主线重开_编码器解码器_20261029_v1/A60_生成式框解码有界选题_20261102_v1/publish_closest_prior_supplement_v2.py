"""Publish the bounded nearest-prior supplement while original M remains live."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
B = P.parent
D = B / 'A54_场景尺度约束候选_20261101_v1'
PARENT = P / 'A60_M_epoch1_alignment_preservation_receipt_v1.json'
MASTER = B.parent.parent / '文献总库_20261101_v1/03_索引与核对/文献总库.json'
SNAP = P / 'before_closest_prior_supplement_v2_publish_v1'
REPORT = P / '16_生成式检测近作与创新边界.txt'
SUPPLEMENT = P / '02c_生成式检测近作定向补查_v1.json'
HANDOFF = P / 'A60_turn_handoff_closest_prior_v2_v1.json'
OBSERVATION = P / 'formal_live_observation_253_v1.json'
RECEIPT = P / 'A60_closest_prior_v2_preservation_receipt_v1.json'
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
    temporary = path.with_name(path.name + '.closest_prior_v2_tmp')
    write_new(temporary, value)
    temporary.replace(path)


def main():
    assert REPORT.is_file() and SUPPLEMENT.is_file()
    assert all(not path.exists() for path in [SNAP, HANDOFF, OBSERVATION, RECEIPT])
    assert all(not path.with_name(path.name + '.closest_prior_v2_tmp').exists() for path in ROOTS)
    parent = read(PARENT)
    for name, expected in parent['files'].items():
        path = Path(name)
        assert path.is_file() and metadata(path) == expected, f'Prior evidence changed: {path}'
    for path in ROOTS:
        assert str(path) in parent['files']
    supplement = read(SUPPLEMENT)
    assert supplement['status'] == 'BOUNDED_METHOD_LEVEL_PRIOR_SUPPLEMENT'
    assert [x['id'] for x in supplement['sources']] == ['DiffuDETR', 'ReDiffDet', 'DFM']
    assert not supplement['experiment_status']['protocol_or_gates_changed']
    assert all(supplement[key] == 0 for key in ['new_model_forwards', 'new_optimizer_updates', 'new_COCO_evaluations'])
    local = Path(supplement['sources'][2]['local_text'])
    assert sha(local) == supplement['sources'][2]['local_text_sha256']
    master = read(MASTER); assert len(master['records']) == 423
    master_at_read = {'path': str(MASTER), 'records': 423, 'sha256': sha(MASTER),
                      'scope': 'Read only; original remains editable and is not part of immutable receipt.'}
    receipt = read(D / 'A60_formal_observe_253.json')
    assert receipt['status'] == 'EXECUTED' and receipt['exit'] == 0
    obs = json.loads(receipt['stdout']); base, mid, candidate = (obs['arms'][x] for x in ['B', 'M', 'Q'])
    assert obs['supervisor']['identity_matched'] and obs['supervisor']['live'] and obs['source_binding']
    assert base['exit']['exit'] == 0 and base['completed_epochs'] == 12
    assert mid['process']['identity_matched'] and mid['process']['live'] and mid['progress']['epoch'] == 2
    assert mid['launch']['pid'] == 35385 and mid['launch']['start_ticks'] == '32308346'
    assert not candidate['exists'] and mid['completed_epochs'] == 1
    assert mid['epochs'][0]['flow']['raw_stream_sha256'] == base['epochs'][0]['flow']['raw_stream_sha256']
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
    status = 'A60_CLOSEST_PRIOR_V2_SUPPLEMENT_ORIGINAL_M2_LIVE'
    evidence = {
        'status': supplement['status'], 'report': str(REPORT), 'matrix': str(SUPPLEMENT),
        'sources': [{'id': x['id'], 'title': x['title'], 'venue': x['venue'],
                     'primary_url': x['primary_url'], 'read_scope': x['read_scope']} for x in supplement['sources']],
        'DiffuDETR_table6_vs_text_discrepancy': 'Table values 51.6 vs 50.9 imply +0.7 AP; body says +1.7 AP.',
        'limited_review_only': True, 'new_global_novelty_proven': False,
        'candidate_efficacy_tested': False, 'protocol_or_gates_changed': False,
        'literature_master_at_read': master_at_read,
        'meaning': 'Time-conditioned generative decoders and scale-conditioned spatial noise have prior art; A60 M/Q specific time-label comparison remains unvalidated and bounded by unread methods.',
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
        'closest_prior_supplement_v2': evidence,
        'scope': 'B12 verified, M1 raw stream aligned; selected primary prior art tightens novelty claims; M/Q efficacy remains untested.',
        'previous_goal_turn_classification': 'progress',
        'current_goal_turn_classification': 'progress: direct DiffuDETR/ReDiffDet/DFM prior boundaries checked',
        'latest_report': str(REPORT), 'report': str(REPORT), 'latest_handoff': str(HANDOFF),
        'next_readonly_receipt': 'A60_formal_observe_254.json',
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
        'M_epoch1_alignment': current['M_epoch1_raw_stream_alignment'],
        'MQ_terminal_archive_preparation': str(P / 'A60_turn_handoff_MQ_archive_prepared_v1.json'),
        'Q_started': False, 'candidate_efficacy_tested': False, 'training_protocol_modified': False,
        'bridge_session_id': 19671, 'next_readonly_receipt_hint': 'A60_formal_observe_254.json',
        'numbering_rule': 'Derive next receipt from actual files; old hints can already be consumed.',
        'remaining_action': 'Observe original M fixed4/8/12 and original Q; archive terminal artifacts once, verify 12 raw/noisy streams and frozen judgment. No source preparation, training restart, extra seed or change of gates.',
        'risk': 'DiffuDETR and ReDiffDet foreclose broad single-step/time-conditioned/scale-noise novelty claims; CEOSOD full method is unread, Q is untested, and B12 trails ECDet-S under confounded settings.',
    }
    write_new(HANDOFF, handoff)
    board, state = read(ROOTS[0]), read(ROOTS[1])
    board.update({'status': status, 'latest_update_utc': obs['recorded_utc'], 'training_running': True,
                  'A60_execution': current, 'A60_closest_prior_supplement_v2': evidence, 'latest_report': str(REPORT)})
    state.update({'status': status, 'recorded_utc': obs['recorded_utc'], 'latest_update_utc': obs['recorded_utc'],
                  'training_running': True, 'A60_execution': current, 'A60_closest_prior_supplement_v2': evidence,
                  'latest_report': str(REPORT), 'model_forwards': current['global_observed_model_forwards'],
                  'training_updates': current['global_observed_all_optimizer_updates'],
                  'detector_optimizer_updates_including_capacity': current['global_observed_all_optimizer_updates'],
                  'formal_detector_training_updates': current['global_observed_formal_optimizer_updates'],
                  'formal_optimizer_updates_observed': current['global_observed_formal_optimizer_updates'],
                  'formal_training_epochs': current['global_completed_formal_epochs'],
                  'previous_goal_turn_classification': 'progress',
                  'current_goal_turn': 'PROGRESS_CLOSEST_PRIORS_AND_ORIGINAL_M2_LIVE'})
    for path, value in zip(ROOTS, [board, state, current]):
        replace_json(path, value)
    files = dict(parent['files'])
    new_paths = [*ROOTS, *[Path(x['resolved_path']) for x in resolution.values()], PARENT,
                 Path(__file__), REPORT, SUPPLEMENT, HANDOFF, OBSERVATION,
                 *[D / f'A60_formal_observe_{i:03d}.json' for i in range(240, 254)]]
    for path in new_paths:
        assert path.is_file(), str(path)
        files[str(path)] = metadata(path)
    output = {'status': status, 'goal_status': 'active', 'mainline_validated': False,
              'parent_receipt': str(PARENT), 'parent_receipt_sha256': sha(PARENT),
              'parent_files_verified_before_publish': len(parent['files']),
              'parent_path_resolution': resolution, 'files': files,
              'scope': 'Bounded primary-source prior supplement and original live M; no model replay, new update, gate change or loss of prior evidence.'}
    write_new(RECEIPT, output)
    print(json.dumps({'status': status, 'parent_files_verified': len(parent['files']),
                      'receipt_files': len(files), 'M_current_epoch': mid['progress']['epoch'],
                      'new_primary_sources': 3, 'literature_master_records': len(master['records'])}))


if __name__ == '__main__':
    main()
