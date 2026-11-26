"""Record a point-in-time read-only monitoring checkpoint and accepted data prefix."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import psutil
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FOLDER=HERE/'monitor_goal_20261112'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--window',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--exec-session-id',type=int);p.add_argument('--inspection',type=Path);p.add_argument('--gpu-observation',type=Path)
p.add_argument('--base-outage', type=Path, help='Verified BASE disconnect and local checkpoint record.')
p.add_argument('--base-stopped', type=Path, help='Verified BASE interruption and saved continuation anchor.')
a=p.parse_args()
assert all(x.replace('_','').isalnum() for x in [a.window,a.checkpoint])
window=FOLDER/a.window;identity=read(window/'observer_identity.json');latest=read(window/'latest.json')
outage = read(a.base_outage) if a.base_outage is not None else None
stopped = read(a.base_stopped) if a.base_stopped is not None else None
assert outage is None or stopped is None
rows = {row['arm']: row for row in latest['summary']}
assert set(rows) == {'BASE_FRESH', 'FULL_FRESH'}
if stopped is not None:
    base = next(x for x in stopped['observations'] if x['arm'] == 'BASE_FRESH')
    assert base['port'] == 30344 and base['matching_remote_acceptance']
    assert not base['continuation']['processes'] and not base['staged_evidence']['launch_lock']
    assert not rows['BASE_FRESH'].get('processes_live_now')
    assert rows['FULL_FRESH']['processes_live_now'] and rows['FULL_FRESH']['backup_client_live_now']
    assert all(issue.endswith(':BASE_FRESH') or issue.startswith('BASE_FRESH:') for issue in latest['issues'])
elif outage is None:
    assert not latest['issues'] and all(row.get('processes_live_now', False) for row in rows.values())
else:
    assert outage['arm'] == 'BASE_FRESH' and outage['remote_terminal_state_verified'] is False
    assert outage['port'] == rows['BASE_FRESH']['port']
    assert rows['BASE_FRESH']['state'] == 'OBSERVATION_UNAVAILABLE_NOT_TERMINAL'
    assert rows['FULL_FRESH']['processes_live_now'] and rows['FULL_FRESH']['backup_client_live_now']
    assert set(latest['issues']) <= {'REPEATED_OBSERVATION_FAILURE:BASE_FRESH'}
    assert outage['local_backup_reverification']['status'] == 'BASE_LAST_ACCEPTED_STAGE_REVERIFIED'
observation_age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(latest['observed_local_utc'])).total_seconds()
assert 0 <= observation_age <= 120, 'Live observation is too old to publish a monitoring checkpoint'
process=psutil.Process(identity['pid'])
assert abs(process.create_time()-identity['created_timestamp'])<0.001
assert any('watch_pair_window.py' in arg for arg in process.cmdline())
snapshot=FOLDER/(a.checkpoint+'_live.json');assert not snapshot.exists()
export_path=ROOT/'实验数据汇总_20261108/fresh_pair_20261112/逐轮导出核对_FRESH.json'
export=read(export_path);now=dt.datetime.now(dt.timezone.utc).isoformat()
inspection=None
if a.inspection is not None:
    inspection=read(a.inspection)
    assert inspection['csv_sha256']==export['csv']['sha256'] and inspection['accepted_rounds']==export['accepted_rounds']
    assert inspection['model_forwards']==inspection['evaluator_runs']==inspection['training_actions']==0
if outage is not None:
    assert outage['local_backup_reverification']['round'] == export['accepted_rounds']['BASE_FRESH']
snapshot.write_text(json.dumps(latest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
record={'recorded_local_utc':now,'goal_objective':'监控','goal_status':'active','classification':'verified_wait',
        'live_snapshot':{'file':str(snapshot),'sha256':sha(snapshot)},'observer_identity':identity,'observer_live_now':True,
        'accepted_data_export':{'file':str(export_path),'sha256':sha(export_path),'accepted_rounds':export['accepted_rounds']},
        'underlying_runs_complete':False,'new_training_actions':0,'model_forwards':0,'evaluator_runs':0,
        'next_action':'Continue waiting on the same live observation session. A completed observation window is not completion of either underlying training run.'}
if outage is not None:
    record['base_outage'] = {'file': str(a.base_outage.resolve()), 'sha256': sha(a.base_outage)}
    record['next_action'] = 'Continue observing FULL and checking for BASE reconnection. Verify endpoint, process identities and checkpoints after renewal.'
if stopped is not None:
    assert export['accepted_rounds']['BASE_FRESH'] == base['local_accepted_rounds']
    record['base_stopped'] = {'file': str(a.base_stopped.resolve()), 'sha256': sha(a.base_stopped)}
    record['next_action'] = 'Continue observing the same live FULL continuation and accepted backups. BASE is staged at the verified round165 boundary, awaiting a decision on the additional replay.'
if inspection is not None:
    record['accepted_metrics_storage_inspection']={'file':str(a.inspection.resolve()),'sha256':sha(a.inspection)}
if a.gpu_observation is not None:
    gpu_observation=read(a.gpu_observation)
    assert {x['port'] for x in gpu_observation['observations']}=={30715,30048}
    assert all(x['exit_code']==0 and x['observation']['gpus']['returncode']==x['observation']['apps']['returncode']==0 for x in gpu_observation['observations'])
    record['supplementary_gpu_occupancy']={'file':str(a.gpu_observation.resolve()),'sha256':sha(a.gpu_observation)}
record_path=FOLDER/(a.checkpoint+'_checkpoint.json');assert not record_path.exists()
record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state_path=HERE.parent/'模块消融/current_execution_state.json'
shutil.copyfile(state_path,FOLDER/(a.checkpoint+'_previous_execution_state.json'))
state=read(state_path)
if 'supplementary_gpu_occupancy' in record:
    state['latest_fresh_pair_gpu_occupancy_observation']=record['supplementary_gpu_occupancy']
if a.exec_session_id is not None:
    previous=state.get('active_monitor_observer')
    if previous:state.setdefault('monitor_observer_history',[]).append(previous)
    state['active_monitor_observer']={'exec_session_id':a.exec_session_id,'identity':identity,'folder':str(window)}
state.update(observed_local_utc=now,goal_status='active',goal_objective='监控',last_goal_turn_classification='verified_wait',
             operational_work_status='MONITORING_CONFIRMED_LIVE_FRESH_PAIR; ACCEPTED_BACKUPS_AND_CSV_ADVANCING; NO_NEW_EXPERIMENTS',
             latest_fresh_pair_observation=str(snapshot),latest_fresh_pair_observation_sha256=sha(snapshot),
             latest_monitor_goal_checkpoint=str(record_path),latest_monitor_goal_checkpoint_sha256=sha(record_path),
             latest_fresh_pair_data_export=str(export_path),latest_fresh_pair_data_export_sha256=sha(export_path),
             next_action=record['next_action'])
for row in latest['summary']:
    if stopped is not None and row['arm'] == 'BASE_FRESH':
        continue
    if outage is not None and row['arm'] == 'BASE_FRESH':
        prior = outage['last_live_observation']
        last_live = prior['summary']
        state['fresh_pair'][row['arm']].update(
            status=row['state'], completed_rounds=last_live['completed_rounds'],
            current_round_as_observed=last_live['current_round'], current_actual_epoch=last_live['current_round']-1,
            current_batch_as_observed=last_live['batch'], optimizer_step_attempts_as_observed=last_live['attempts'],
            backup_accepted_round=export['accepted_rounds']['BASE_FRESH'], per_round_backup_verified=True,
            last_observed_local_utc=prior['observed_local_utc'],
            last_observation_attempt_local_utc=latest['observed_local_utc'], observation_error=row['error'],
            remote_terminal_state_verified=False)
        continue
    state['fresh_pair'][row['arm']].update(status=row['state'],completed_rounds=row['completed_rounds'],current_round_as_observed=row['current_round'],
                                         current_actual_epoch=row['current_round']-1 if row['current_round'] is not None else None,
                                         current_batch_as_observed=row['batch'],optimizer_step_attempts_as_observed=row['attempts'],
                                         backup_accepted_round=row['backup_accepted_round'],
                                         per_round_backup_verified=row['backup_accepted_round']>=row['completed_rounds'],
                                         last_observed_local_utc=latest['observed_local_utc'])
    state['fresh_pair'][row['arm']].pop('observation_error', None)
    state['fresh_pair'][row['arm']].pop('last_observation_attempt_local_utc', None)
if outage is not None:
    state['operational_work_status'] = 'BASE_SSH_UNAVAILABLE_AWAITING_OPERATOR_RENEWAL; FULL_TRAINING_AND_BACKUPS_LIVE'
    state['latest_base_ssh_outage'] = record['base_outage']
if stopped is not None:
    state['operational_work_status'] = 'FULL_CONTINUATION_LIVE; BASE_REPLAY_AWAITING_AUTHORIZATION'
    state['goal_blocking_audit_consecutive_turns'] = 0
    full_obs=next(x['observation'] for x in latest['raw_remote_observations'] if x['port']==rows['FULL_FRESH']['port'])
    supervisor=full_obs['records']['supervisor.json']
    state['fresh_pair']['FULL_FRESH'].update(worker_pid=supervisor['pid'],supervisor_pid=supervisor['supervisor_pid'],
                                          continuation_started=True,remote_terminal_state_verified=False)
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
parts = []
for row in latest['summary']:
    arm = row['arm']
    if stopped is not None and arm == 'BASE_FRESH':
        parts.append(f"BASE 已验收 {base['local_accepted_rounds']}/320 轮，等待额外重算授权")
    elif outage is not None and arm == 'BASE_FRESH':
        parts.append(f"BASE SSH 暂不可用，最后校验的本地断点为第 {export['accepted_rounds'][arm]} 轮，待续期后核对远端状态")
    elif row['backup_accepted_round'] >= row['completed_rounds']:
        parts.append(f"{arm} 已完成并备份 {row['completed_rounds']}/320 轮，训练正在运行")
    else:
        parts.append(f"{arm} 已完成 {row['completed_rounds']}/320 轮，已校验备份 {row['backup_accepted_round']} 轮")
summary = '。'.join(parts)
note = ('【当前目标为持续监控两组新主对照】\n' + summary + '。\n'
        f"已导出 {export['csv']['rows']} 条已验收记录。\n"
        f'监控记录见 {record_path.name}。\n\n')
entry=ROOT/'当前修订入口.txt';previous=entry.read_text(encoding='utf-8-sig')
shutil.copyfile(entry,FOLDER/(a.checkpoint+'_previous_current_entry.txt'))
divider='以下为历史记录。\n\n'
body=previous.split(divider,1)[-1] if previous.startswith('【当前目标') else previous
entry.write_text(note+divider+body,encoding='utf-8')
readable=HERE/'运行状态与数据入口.txt';old=readable.read_text(encoding='utf-8-sig')
shutil.copyfile(readable,FOLDER/(a.checkpoint+'_previous_readable_status.txt'))
body=old.split(divider,1)[-1] if old.startswith('【当前目标') else old
readable.write_text(note+divider+body,encoding='utf-8')
print(json.dumps({'classification':'verified_wait','goal_status':'active','progress':summary,'exported_rows':export['csv']['rows'],'observer_pid':identity['pid'],'snapshot_sha256':sha(snapshot)},ensure_ascii=False))
