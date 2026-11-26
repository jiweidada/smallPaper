import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = BASE / 'A48_新机制候选与主线收敛_20261101_v1'
RECEIPT = HERE / '08_A49启动与历史保留校验回执.json'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def verify_map(node):
    count = 0
    if isinstance(node, dict):
        if set(['preserved_at', 'sha256']).issubset(node):
            if sha(Path(node['preserved_at'])) != node['sha256']:
                raise RuntimeError('Historical source changed: ' + node['preserved_at'])
            return 1
        for v in node.values():
            count += verify_map(v)
    elif isinstance(node, list):
        for v in node:
            count += verify_map(v)
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('observer_receipt')
    args = parser.parse_args()
    if RECEIPT.exists():
        raise RuntimeError('Already published. Do not republish or overwrite immutable receipt.')
    obs_path = HERE / args.observer_receipt
    obs_raw = read(obs_path)
    assert obs_raw['status'] == 'EXECUTED' and obs_raw['exit'] == 0
    observed = json.loads(obs_raw['stdout'])
    assert observed['observation_only']
    assert read(HERE / 'CURRENT_A49.json')['last_observation'] == str(obs_path)
    protocol_path = HERE / 'protocol_FROZEN.json'
    protocol = read(protocol_path)
    assert sha(protocol_path) == observed['protocol_sha256'] == 'd3a971a21d1c339052c8a7ef41fcab345882edc8893c4762c66dbbc5bc10f8f3'
    for n, expected in protocol['files_sha256'].items():
        assert sha(HERE / n) == expected, n
    old_receipt_path = OLD / '21_A48终点与历史保留校验回执.json'
    old_receipt = read(old_receipt_path)
    assert old_receipt['mainline_confirmed'] is False
    for s, expected in old_receipt['current_files_sha256'].items():
        assert sha(Path(s)) == expected, s
    historical_counts = {}
    for k in ['previous_A48_signed_current_inputs_resolved', 'previous_A47_all105_inputs_resolved', 'older_input_resolution']:
        historical_counts[k] = verify_map(old_receipt[k])

    before = HERE / 'before_A49_startup'
    before.mkdir(exist_ok=False)
    remapped = {}
    for s, expected in old_receipt['current_files_sha256'].items():
        source = Path(s)
        if source.parent == BASE and source.name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
            target = before / source.name
            shutil.copy2(source, target)
            assert sha(target) == expected
        else:
            target = source
        remapped[s] = {'preserved_at': str(target), 'sha256': expected}

    now = datetime.now(timezone.utc).isoformat()
    counts = observed['observed_counters_lower_bounds']
    prior = old_receipt['cumulative_counts']
    totals = {
        'model_forwards': prior['model_forwards'] + counts['model_forwards'],
        'optimizer_updates_including_capacity': prior['optimizer_updates_including_capacity'] + counts['formal_optimizer_updates'] + counts['capacity_optimizer_updates'],
        'formal_optimizer_updates': prior['formal_optimizer_updates'] + counts['formal_optimizer_updates'],
        'capacity_updates': prior['capacity_updates'] + counts['capacity_optimizer_updates'],
        'formal_epochs': prior['formal_epochs'] + counts['formal_epoch_sum']
    }
    live = read(HERE / 'CURRENT_A49.json')['owned_live_arms']
    assert live, 'Do not publish training-active status without verified live owned process.'
    question = '已知忽略区域下，decoder query读取范围的负监督有效性规则能否超过普通预测框覆盖忽略？'
    evidence = {
        'id': 'A49_B_KNOWN_IGNORE_SUPERVISION',
        'status': 'FIXED12_PILOT_ACTIVE_NOT_VALIDATED',
        'goal_status': 'active', 'mainline_confirmed': False,
        'dataset': protocol['dataset'], 'host': protocol['host'], 'question': question,
        'candidate_increment': 'Training-only native query sampling-support gating of unmatched negative classification; pre/encoder M fallback, matched rows and DN preserved.',
        'arms': {k: protocol[k] for k in ['B','M','Q']},
        'protocol': str(protocol_path), 'protocol_sha256': sha(protocol_path),
        'report': str(HERE / '07_定向调研与三组最小验证启动报告.txt'),
        'reading_registry': str(HERE / '05_增补检索与阅读范围_v1.json'),
        'annotation_audit': str(HERE / 'annotations_audit_v1' / 'result.json'),
        'capacity': {'status': 'PASS', 'model_forwards': 12, 'optimizer_updates': 9, 'extra_mask_core_reads': 12,
                     'repeated_by_formal_supervisor': False, 'scope': 'three selected batches, not all-data memory/efficacy proof'},
        'observation': str(obs_path), 'owned_live_arms': live,
        'observed_progress': {a: r['progress'] for a, r in observed['arms'].items()},
        'current_evaluations': {a: r['evaluations'] for a, r in observed['arms'].items()},
        'fixed_endpoint': 'B12 then M12 then Q12, same common initialization and fixed12EMA; no automatic retry or rescue sweep',
        'gates_pp': protocol['gates_pp'],
        'new_phase_counters_lower_bounds': counts,
        'limits': protocol['known_risks'],
        'old_route_boundaries': 'A45_B2_A held/untrained; explicit class0 here, no missingness estimator. No A42 readout/A43 teacher/A44 quota/A46 class dominance/A47 jitter/A48 normalization restart.',
        'candidate_selection_status': 'A49_B selected only for bounded pilot; A49_A Delta inspiration held, abstract-only, untrained.'
    }
    board_path = BASE / '01_候选判别表.json'
    board = read(board_path)
    board.update({'latest_update_utc': now, 'status': 'A49_FIXED12_PILOT_ACTIVE',
                  'goal_status': 'active', 'mainline_validated': False,
                  'current_priority_candidate': evidence['id'], 'current_priority_question': question,
                  'mainline_selected_for_pilot': True, 'pilot_validation_status': evidence['status'],
                  'latest_report': evidence['report'], 'selection_stage_complete': True,
                  'selection_stage_scope': 'A49 candidate selection only, broad mainline search incomplete.',
                  'current_host_candidate': 'ECDet-S originalBN fixed d8752b75c9beaecfcdd0e5cd207bae3b6946ae23',
                  'dataset_candidate': protocol['dataset'], 'A49_execution': evidence})
    board.setdefault('historical_model_candidates_before_A49', board.get('model_candidates'))
    board['model_candidates'] = ['ECDet-S, current controlled pilot host; not a confirmed final paper baseline']
    board.setdefault('historical_reference_before_A49', board.get('existing_reference'))
    board['existing_reference'] = 'Fresh B49 baseline; A48 fixed12 terminal scores preserved but not reused as B49.'
    board['constraints']['new_experiments_this_turn'] = 'A49 one capacity check12forwards/9updates, then three same-protocol12epoch arms; actual lower bounds in A49_execution.'
    board['remaining_deliverable'] = 'Finish A49 fixed12 comparison and select/retire the exact recipe; a negative does not finish the mainline goal.'
    board['next_decision_protocol'] = str(protocol_path)
    board['next_exploratory_question'] = question
    board['candidates'].append({'id': evidence['id'], 'status': evidence['status'], 'question': question,
                                'increment_approved': 'bounded experiment only, no novelty/efficacy confirmation',
                                'nearest': ['SparseDet selected methods', 'BRL selected methods', 'PUD-DETR pinned selected code, paper unverified', 'ordinary known-region box-overlap ignoring'],
                                'execution': str(HERE / 'CURRENT_A49.json')})
    dump(board_path, board)

    state_path = BASE / '15_当前状态与结果校验.json'
    state = read(state_path)
    if 'goal_completion_tool_receipt' in state:
        state['historical_goal_completion_tool_receipt_before_A49'] = state.pop('goal_completion_tool_receipt')
    state.update({'recorded_utc': now, 'goal_status': 'active', 'mainline_validated': False,
                  'current_phase': 'A49', 'latest_authoritative_phase': 'A49_FIXED12_PILOT_ACTIVE',
                  'current_priority_candidate': evidence['id'], 'current_priority_question': question,
                  'current_user_task_phase': 'A49', 'current_user_task_status': evidence['status'],
                  'latest_report': evidence['report'], 'A49_execution': evidence,
                  'diagnostics_complete': False, 'diagnostics_complete_scope': 'A49 training active; old completed stages remain historical.',
                  'model_forwards': totals['model_forwards'], 'training_updates': totals['optimizer_updates_including_capacity'],
                  'detector_optimizer_updates_including_capacity': totals['optimizer_updates_including_capacity'],
                  'formal_detector_training_updates': totals['formal_optimizer_updates'],
                  'capacity_only_optimizer_updates': totals['capacity_updates'],
                  'formal_training_epochs': totals['formal_epochs'],
                  'counts_terminal': False, 'training_counts_terminal': False,
                  'counters_observed_lower_bounds_while_job_live': True,
                  'current_native_job_terminal': False, 'full_forward_cap_closed': False,
                  'cap': 148941, 'cap_scope': 'Cumulative planned ceiling if all A49 arms finish; current counters are observed lower bounds.',
                  'mainline_selected_for_pilot': True,
                  'current_goal_turn': {'classification': 'verified_wait', 'goal_status': 'active', 'mainline_confirmed': False,
                                        'not_complete_just_because_one_candidate_tested': True, 'observer': str(obs_path)},
                  'full_model_forwards_scope': 'A48 terminal114452 + A49 observed whole-model forwards; extra single-channel mask core reads are separately counted.',
                  'training_update_scope': 'A48 terminal87926 incl28capacity + A49 observed formal updates +9capacity. Live counts are lower bounds.',
                  'formal_training_epochs_scope': 'A48 terminal108 + A49 completed epoch sum; no partial epoch counted as complete.'})
    state.setdefault('authoritative_history', []).append({'phase': 'before_A49', 'preserved_state': str(before / state_path.name),
        'previous_receipt': str(old_receipt_path), 'previous_receipt_sha256': sha(old_receipt_path),
        'scope': 'Previous A48 terminal and older linked history; no old verdict changed.'})
    dump(state_path, state)
    assert verify_map(remapped) == len(remapped)
    for k in ['previous_A48_signed_current_inputs_resolved', 'previous_A47_all105_inputs_resolved', 'older_input_resolution']:
        assert verify_map(old_receipt[k]) == historical_counts[k]

    signed = [board_path, state_path, protocol_path, obs_path]
    names = ['00_A49问题与边界.json', '01_检索原始记录_v1.json', '02_核心方法矩阵_v1.json',
             '03_新候选选择与证据边界.txt', '04_最小验证方案_草案.json',
             '05_增补检索与阅读范围_v1.json', '06_检索记录索引_v1.json',
             '07_定向调研与三组最小验证启动报告.txt', 'source_acquisition_v1.json',
             'formal_sources_upload_v1.json', 'formal_launch_response_v1.json',
             'capacity_prelaunch_revision_v1.json', 'capacity_protocol_FROZEN_v1.json',
             'math_and_loss_check_v1.json', 'annotation_audit_response_v1.json',
             'annotations_audit_v1/result.json', 'annotations_audit_v1/image_rows.json',
             'observe_A49.py', 'update_current_A49.py', 'finalize_research_registry_A49.py',
             Path(__file__).name]
    signed.extend(HERE / n for n in names)
    signed.extend(HERE / n for n in protocol['files_sha256'])
    signed.extend(sorted((HERE / 'search_read_logs').glob('*.json')))
    registry = HERE.parents[1] / '文献_相关原始论文_20261004' / 'A49_阅读范围与来源核验_20261101_v1.json'
    signed.append(registry)
    files = {str(p): sha(p) for p in signed}
    receipt = {
        'status': 'PASS', 'phase': 'A49_FIXED12_STARTUP_PUBLISHED',
        'goal_status': 'active', 'mainline_confirmed': False,
        'previous_A48_terminal_receipt': str(old_receipt_path),
        'previous_A48_terminal_receipt_sha256': sha(old_receipt_path),
        'previous_A48_terminal_current_inputs_resolved': remapped,
        'previous_linked_history_counts_verified': historical_counts,
        'previous_linked_history_access': 'Follow unchanged A48 receipt maps; A48 startup23, A47 inputs105, older122 verified at publication.',
        'current_files_sha256': files, 'cumulative_counts_observed_lower_bounds': totals,
        'A49_snapshot_counters': counts, 'owned_live_arms': live,
        'future_live_reference': str(HERE / 'CURRENT_A49.json'),
        'mutable_pointer_is_not_signed': True,
        'new_detector_operations_by_publication': 0,
        'new_optimizer_jobs_by_publication': 0,
        'next_candidate_automatically_queued': False,
        'scope': 'File preservation, fixed execution inputs and authentic observed status; not AP reproduction or confirmed novelty.'
    }
    dump(RECEIPT, receipt)
    print(json.dumps({'status': 'PASS', 'signed_current_files': len(files),
                      'preserved_A48_terminal_files': len(remapped), 'linked_history_counts': historical_counts,
                      'cumulative_counts_lower_bounds': totals, 'live': live, 'mainline_confirmed': False}, ensure_ascii=False))

if __name__ == '__main__':
    main()
