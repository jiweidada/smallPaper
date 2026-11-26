from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt')
    args = parser.parse_args()
    path = (ROOT / args.receipt).resolve()
    assert path.parent == ROOT
    raw = json.loads(path.read_text(encoding='utf-8'))
    assert raw['status'] == 'EXECUTED' and raw['exit'] == 0
    observed = json.loads(raw['stdout'])
    assert observed['observation_only']
    live = []
    for arm, row in observed['arms'].items():
        ident = row['identity'] or {}
        if ident.get('present') and ident.get('same_start_ticks') and ident.get('owned_command') and ident.get('state') not in ['Z', 'X']:
            live.append({'arm': arm, 'pid': ident['pid'], 'start_ticks': ident['expected_start_ticks']})
    pairs = {}
    for left, right in [('M', 'B'), ('Q', 'B'), ('Q', 'M')]:
        lm, rm = observed['arms'][left]['evaluations'], observed['arms'][right]['evaluations']
        pairs[left + '_minus_' + right] = {e: {k: lm[e][k] - rm[e][k] for k in ['AP', 'APs', 'APs75']}
                                           for e in sorted(set(lm) & set(rm), key=int)}
    pointer = {'goal_status': 'active', 'mainline_confirmed': False, 'mutable_reference_only': True,
               'stage': 'FIXED12_PILOT_ACTIVE' if live else 'INSPECT_TERMINAL_OR_UNRESOLVED',
               'candidate_selected_for_pilot': 'A49_B_KNOWN_IGNORE_SUPERVISION',
               'last_observation': str(path), 'snapshot': observed, 'same_epoch_deltas_pp': pairs,
               'owned_live_arms': live, 'bridge_session_if_alive': 35948,
               'remote_root': '/root/uav_a49_20261101_v1',
               'observer_command': '/opt/conda/envs/mamba/bin/python /root/uav_a49_20261101_v1/observe_A49.py',
               'formal_protocol': str(ROOT / 'protocol_FROZEN.json'),
               'next_action': 'Poll the same owned job. Keep B/M/Q each12 fixed; compare fixed12EMA. No threshold/site/seed rescue, no restart from timeout/lock alone.'}
    (ROOT / 'CURRENT_A49.json').write_text(json.dumps(pointer, indent=2, ensure_ascii=False), encoding='utf-8')
    compact = {a: {'status': (r['result'] or r['progress'] or {}).get('status', 'NOT_STARTED'),
                    'completed_epochs': r['completed_epochs'],
                    'latest_progress': {k: v for k, v in (r['progress'] or {}).items() if k in
                        ['status', 'epoch', 'microbatch', 'optimizer_updates', 'loss', 'grad_norm', 'extra_mask_core_reads']},
                    'evaluations': {e: {k: m[k] for k in ['AP', 'APs', 'APs75']} for e, m in r['evaluations'].items()}}
               for a, r in observed['arms'].items()}
    print(json.dumps({'classification': 'verified_wait' if live else 'inspect_terminal_or_unresolved',
                     'live': live, 'arms': compact, 'same_epoch_deltas_pp': pairs,
                     'counters': observed['observed_counters_lower_bounds']}))
