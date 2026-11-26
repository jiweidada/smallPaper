from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone

P = Path(__file__).resolve().parent
B = P.parent
BRIDGE = B/'A54_场景尺度约束候选_20261101_v1'
ID = 'A56_LEVEL_MATCHED_CONTRAST_READ'


def load(path):
    return json.loads(path.read_text('utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(receipt_name):
    receipt = load(BRIDGE/receipt_name)
    assert receipt['status'] == 'EXECUTED' and receipt['exit'] == 0
    snap = json.loads(receipt['stdout'])
    assert snap['protocol_sha256'] == sha(P/'protocol_FROZEN_v1.json')
    now = datetime.now(timezone.utc).isoformat()
    snapshot_name = receipt_name.replace('_rpc', '').replace('.json', '_parsed.json')
    assert not (P/snapshot_name).exists()
    save(P/snapshot_name, snap)
    completed = 0
    formal_forwards = formal_updates = 0
    for arm, value in snap['arms'].items():
        progress = value['result'] or value['progress']
        completed_here = len(value['epochs'])
        if progress:
            formal_forwards += progress.get('full_model_forwards', 0)
            formal_updates += progress.get('optimizer_updates', 0)
            completed_here = max(completed_here, progress.get('epochs', progress.get('completed_epochs', 0)))
            if progress.get('status') == 'VALIDATING':
                completed_here = max(completed_here, progress['epoch'])
        completed += completed_here
    assert formal_updates <= 19416 and formal_forwards <= 20318
    terminal = bool(snap['exit'])
    if terminal:
        status = snap['exit']['status']
    elif any(r['arm'] == 'Q' and r['status'] == 'RUNNING' for r in (snap['processes'] or {}).get('records', [])):
        status = ('A56_Q_RUNNING_M_FIXED12_COMPLETE' if snap['arms']['Q']['progress']
                  else 'A56_Q_INITIALIZING_M_FIXED12_COMPLETE')
    elif snap['arms']['M']['result'] and snap['arms']['M']['result']['status'] == 'COMPLETED':
        status = 'A56_M_FIXED12_COMPLETE_Q_PENDING_SUPERVISOR'
    else:
        status = 'A56_M_RUNNING_Q_QUEUED'
    evidence = {'phase': 'A56', 'recorded_utc': now, 'client_date': '2026-11-02',
                'status': status, 'goal_status': 'active', 'mainline_validated': False,
                'novelty_cleared': False, 'candidate_selected_for_pilot': True,
                'protocol_sha256': snap['protocol_sha256'], 'remote_directory': '/root/uav_a56_20261102_v1',
                'snapshot': str(P/snapshot_name), 'capacity_whole_forwards': 9,
                'capacity_only_optimizer_updates': 6, 'formal_whole_forwards_observed': formal_forwards,
                'formal_updates_observed': formal_updates, 'formal_epochs_completed': completed,
                'counts_terminal': terminal, 'counts_observed_lower_bounds_while_live': not terminal,
                'max_formal_whole_forwards': 20318, 'max_formal_updates': 19416,
                'CPU_synthetic_native_attention_calls': 7, 'CPU_synthetic_component_optimizer_steps': 4,
                'CPU_component_steps_are_not_detector_updates': True,
                'report': str(P/'03_冻结12轮验证与运行状态.txt'), 'post_Q_queue': None}
    verified = {}
    for arm in ['M', 'Q']:
        verification = P/f'{arm}_completed_local_verification.json'
        if verification.exists():
            value = load(verification)
            assert value['status'] == 'PASS' and value['epochs'] == 12 and value['optimizer_updates'] == 9708
            assert value['bundle_sha256'] == load(P/f'{arm}_bundle_ready.json')['sha256']
            verified[arm] = {'status': 'PASS', 'file': str(verification), 'sha256': sha(verification),
                             'endpoint_metrics': value['endpoint_metrics'],
                             'optimizer_step_histogram': value['optimizer_step_histogram'],
                             'new_detector_forwards': 0, 'new_optimizer_updates': 0}
    evidence['locally_verified_fixed12_arms'] = sorted(verified)
    if 'M' in verified and (P/'10_M12结果与Q启动核验.txt').exists():
        evidence['report'] = str(P/'10_M12结果与Q启动核验.txt')
    state = {'phase': 'A56', **evidence, 'selected_for_formal_training': True,
             'terminal_verified_arms': verified,
             'current_observation': snap, 'latest_raw_receipt': str(BRIDGE/receipt_name)}
    save(P/'CURRENT_A56.json', state)
    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
        path = B/name
        current = load(path)
        current.update({'status': status, 'goal_status': 'active', 'mainline_validated': False,
                        'current_priority_candidate': ID, 'candidate_selected_for_pilot': True,
                        'mainline_selected_for_pilot': False,
                        'mainline_selected_for_pilot_scope': 'A56 is an exploratory candidate, not a confirmed research mainline',
                        'latest_report': evidence['report'], 'A56_execution': evidence})
        for arm, verification in verified.items():
            current[f'A56_{arm}_terminal_verification'] = verification
        if name.startswith('01_'):
            entries = current['candidates']
            row = next((x for x in entries if x['id'] == ID), None)
            if row is None:
                row = {'id': ID, 'name': '保留原生读取的尺度内对比残差',
                       'old_references': ['D008', 'D015', 'D025', 'A35', 'A42', 'K_ERROR_SPECIFIC_SPATIAL_RESPONSE_CANCELLATION'],
                       'distinct_increment': 'match the negative read mass to the unchanged positive read separately at each level; retain native path',
                       'new_basis': 'explicit operator difference, live native integration and matched M/Q control; no established task-causal failure',
                       'novelty_cleared': False, 'mainline_validated': False}
                entries.append(row)
            row.update({'status': status, 'selected_for_pilot': True, 'protocol': str(P/'protocol_FROZEN_v1.json')})
        else:
            before = load(P/'before_A56'/name)
            current.update({'phase': 'A56', 'model_forwards': before['model_forwards']+9+formal_forwards,
                            'cap': before['model_forwards']+9+20318,
                            'formal_detector_training_updates': before['formal_detector_training_updates']+formal_updates,
                            'capacity_only_optimizer_updates': before['capacity_only_optimizer_updates']+6,
                            'detector_optimizer_updates_including_capacity': before['training_updates']+6+formal_updates,
                            'training_updates': before['training_updates']+6+formal_updates,
                            'formal_training_epochs': before['formal_training_epochs']+completed,
                            'counts_terminal': terminal, 'training_counts_terminal': terminal,
                            'counters_observed_lower_bounds_while_job_live': not terminal,
                            'current_native_job_terminal': terminal, 'full_forward_cap_closed': terminal,
                            'formal_training_epochs_scope': 'exact216 through A55 plus completed A56 epochs; temporary/component steps excluded',
                            'full_model_forwards_scope': 'exact205939 through A55 plus A56 capacity9 and observed formal forwards; current counts are lower bounds while live'})
        save(path, current)
    save(P/'prior_canonical_resolution.json',
         {'reason': 'Mutable canonical files advance; old A55 receipts resolve these exact prior bytes through before_A56.',
          'files': {str(B/name): {'snapshot': str(P/'before_A56'/name),
                                 'sha256': sha(P/'before_A56'/name)}
                    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']}})
    print(json.dumps({'status': status, 'formal_forwards_observed': formal_forwards,
                      'formal_updates_observed': formal_updates, 'epochs_completed': completed,
                      'counts_terminal': terminal}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    main(parser.parse_args().receipt)
