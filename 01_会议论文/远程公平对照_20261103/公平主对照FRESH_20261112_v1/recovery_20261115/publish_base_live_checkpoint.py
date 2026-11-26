"""Publish a live BASE continuation and the paired export with FULL already archived."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import psutil

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
MONITOR = RUN / 'monitor_goal_20261112'
DATA = ROOT / '实验数据汇总_20261108/fresh_pair_20261112'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def ref(path):
    return {'file': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def replace(path, content):
    stage = path.with_name(path.name + '.base_live.partial')
    with stage.open('x', encoding='utf-8') as stream:
        stream.write(content)
    stage.replace(path)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--window', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--exec-session-id', type=int, required=True)
    args = parser.parse_args()
    assert all(x.replace('_', '').isalnum() for x in (args.window, args.checkpoint))
    folder = MONITOR / args.window
    identity = read(folder / 'observer_identity.json')
    process = psutil.Process(identity['pid'])
    assert abs(process.create_time() - identity['created_timestamp']) < .001 and identity['arguments']['base_only']
    obs = read(folder / 'latest.json')
    now = datetime.now(timezone.utc)
    assert not obs['issues'] and (now - datetime.fromisoformat(obs['observed_local_utc'])).total_seconds() <= 120
    base, = obs['summary']
    assert base['arm'] == 'BASE_FRESH' and base['port'] == 30009 and base['processes_live_now'] and base['backup_client_live_now']
    export_path = DATA / '逐轮导出核对_FRESH.json'
    export = read(export_path)
    rounds = export['accepted_rounds']
    assert rounds['FULL_FRESH'] == 320 and rounds['BASE_FRESH'] <= base['backup_accepted_round']
    assert ref(DATA / '逐轮指标_FRESH.csv')['sha256'] == export['csv']['sha256']
    index = read(DATA / 'run_index.json')
    assert index['accepted_rounds'] == rounds
    full = index['full_terminal_audit']
    assert ref(Path(full['file'])) == full
    state_path = RUN.parent / '模块消融/current_execution_state.json'
    state = read(state_path)
    assert state['goal_status'] == 'active' and state['fresh_pair']['FULL_FRESH']['remote_terminal_state_verified']
    record_path = MONITOR / (args.checkpoint + '_checkpoint.json')
    with record_path.open('x', encoding='utf-8') as stream:
        stream.write(dump({'recorded_local_utc': now.isoformat(), 'live_snapshot': obs,
                           'observer_identity': identity, 'accepted_data_export': ref(export_path),
                           'accepted_rounds': rounds, 'full_terminal_audit': full,
                           'goal_status': 'active', 'underlying_runs_complete': False, 'training_actions': 0}))
    observer = {'exec_session_id': args.exec_session_id, 'identity': identity, 'folder': str(folder)}
    prior = state.get('active_monitor_observer')
    if prior and prior != observer:
        state.setdefault('monitor_observer_history', []).append(prior)
    state['fresh_pair']['BASE_FRESH'].update(status=base['state'], completed_rounds=base['completed_rounds'],
        current_round_as_observed=base['current_round'], current_actual_epoch=base['current_round']-1,
        current_batch_as_observed=base['batch'], optimizer_step_attempts_as_observed=base['attempts'],
        backup_accepted_round=base['backup_accepted_round'],
        per_round_backup_verified=base['backup_accepted_round']>=base['completed_rounds'],
        last_observed_local_utc=obs['observed_local_utc'])
    state.update(active_monitor_observer=observer, latest_monitor_goal_checkpoint=str(record_path),
        latest_monitor_goal_checkpoint_sha256=ref(record_path)['sha256'], latest_fresh_pair_data_export=str(export_path),
        latest_fresh_pair_data_export_sha256=ref(export_path)['sha256'],observed_local_utc=now.isoformat())
    replace(state_path, dump(state))
    divider='以下为历史记录。\n\n'
    note=('【当前目标为持续监控两组新主对照】\n'
          f'BASE_FRESH 已验收 {rounds["BASE_FRESH"]}/320 轮，续训运行中。FULL_FRESH 的 320 轮和终点已核验归档。\n'
          f'两组共导出 {export["csv"]["rows"]} 条已验收记录。监控记录见 {record_path.name}。\n\n')
    for path in (ROOT / '当前修订入口.txt', RUN / '运行状态与数据入口.txt'):
        previous = path.read_text(encoding='utf-8-sig')
        historical = previous if previous.startswith('【新主对照状态】') else previous.partition(divider)[2]
        replace(path,note+divider+historical)
    print(json.dumps({'accepted_rounds':rounds,'rows':export['csv']['rows'],'observer':identity['pid']},ensure_ascii=False))


if __name__ == '__main__':
    main()
