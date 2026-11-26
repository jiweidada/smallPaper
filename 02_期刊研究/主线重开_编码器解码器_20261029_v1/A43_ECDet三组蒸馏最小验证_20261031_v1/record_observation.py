"""Record a real SSH observation without touching frozen training or canonicals."""
import argparse
import datetime
import json
from pathlib import Path

D = Path(__file__).resolve().parent
LOG = D.parent.parent / '前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl'


def main(receipt_id):
    matches = []
    with LOG.open(encoding='utf-8') as f:
        for line in f:
            try:r = json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('id') == receipt_id and r.get('action') == 'exec':matches.append(r)
    assert len(matches) == 1 and matches[0]['exit_code'] == 0
    observation = json.loads(matches[0]['stdout'])
    def alive(info):
        return bool(info.get('identity_matches') and info.get('state') not in ['Z', 'X'])
    running = [a for a in ['B', 'M', 'Q'] if alive(observation['processes'].get(a, {}))]
    supervisor_alive = alive(observation['processes'].get('supervisor', {}))
    terminal = observation['terminal'].get('supervisor_result.json')
    if terminal:
        classification = 'terminal_observation_requires_verification'
    elif running:
        assert supervisor_alive
        classification = 'verified_wait_live_frozen_job'
    elif supervisor_alive:
        classification = 'verified_wait_live_supervisor_transition'
    else:
        raise AssertionError('Neither a verified live process nor a terminal receipt; inspect actual state')
    student = sum(v.get('full_model_forwards', 0) or 0 for v in observation['arms'].values())
    teacher = sum(v.get('teacher_forwards', 0) or 0 for v in observation['arms'].values())
    updates = sum(v.get('optimizer_updates', 0) or 0 for v in observation['arms'].values())
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'goal_status': 'active', 'classification': classification, 'receipt_id': receipt_id,
              'source_receipt_log': str(LOG), 'live_observation': observation,
              'active_frozen_workers_verified': running, 'supervisor_confirmed_live': supervisor_alive,
              'A43_student_forwards_observed_lower_bound': 8 + student,
              'A43_teacher_forwards_observed_lower_bound': 4 + teacher,
              'architecture_cumulative_full_forwards_observed_lower_bound': 34010 + 8 + 4 + student + teacher,
              'architecture_cumulative_optimizer_updates_observed_lower_bound': 29664 + 5 + updates,
              'formal_A43_optimizer_updates_observed_lower_bound': updates,
              'new_model_forwards_by_observer': 0, 'new_optimizer_updates_by_observer': 0,
              'blocking_condition': None, 'training_restarted': False,
              'next': 'Continue existing frozen queue, verify completed endpoints, no new direction training or rescue sweep.'}
    (D / 'goal_live_latest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with (D / 'goal_verified_wait_journal.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(result) + '\n')
    print(json.dumps({'classification': classification, 'running': running,
                      'formal_updates_observed': updates, 'goal_status': 'active'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--receipt-id', required=True)
    main(p.parse_args().receipt_id)
