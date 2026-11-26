from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FOLDER=HERE/'monitor_goal_20261112'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
now=dt.datetime.now(dt.timezone.utc).isoformat()
latest=FOLDER/'window001/latest.json';observed=read(latest)
assert observed['classification']=='verified_wait' and not observed['issues']
assert all(r['processes_live_now'] and r['backup_client_live_now'] for r in observed['summary'])
snapshot=FOLDER/'initial_verified_live_snapshot.json'
assert not snapshot.exists();shutil.copyfile(latest,snapshot)
contract={
    'goal_objective':'监控','goal_status':'active','recorded_local_utc':now,
    'scope':'Monitor the existing authorized BASE_FRESH on30715 and FULL_FRESH on30048 through their bounded320-round runs and accepted local backups. No new group, preflight, inference, evaluator run, tuning, restart, or paid resource purchase.',
    'success_evidence_required':[
        'Both exact recorded supervisors report COMPLETED with exit0 and worker receipts completed_epochs320/517440 update attempts.',
        'All320 epoch records per arm have verified local manifest/ACK/remote accepted receipts and complete CPU-verified checkpoint payloads.',
        'Terminal actual-epoch319 EMA and same-row AP-best secondary endpoints have separately traceable weights, predictions, evaluator arrays and metrics.',
        'The two fresh trajectories and collected results are indexed separately from the historical four-arm evidence.',
    ],
    'failure_handling':'Collect the specific failure/unknown state and report it. A transient observation timeout is not a terminal job and never authorizes a restart. Preserve other live work; user decides any new experiment or resume.',
    'observation_method':'Read exact PID/start-tick/boot-ID/protocol bindings, training counters and logs, GPU state, local backup client PID/create-time/command and acceptance status. Bounded observation windows are not new training and do not end the underlying jobs.',
    'initial_live_evidence':{'file':str(snapshot),'sha256':sha(snapshot)},
    'observer_identity':read(FOLDER/'window001/observer_identity.json'),
    'active_exec_session_id':37020,'active_observation_window':str(FOLDER/'window001'),
    'next_window_targets':{'BASE_FRESH':20,'FULL_FRESH':16},
    'new_training_actions':0,'model_forwards':0,'evaluator_runs':0,'no_scheduled_automation_created':True,
    'paper_submission_readiness_is_not_the_completion_claim_of_this_monitoring_goal':True,
}
path=FOLDER/'monitoring_contract.json';assert not path.exists();path.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state_path=HERE.parent/'模块消融/current_execution_state.json';shutil.copyfile(state_path,FOLDER/'current_execution_state_before_monitor_goal_start.json')
state=read(state_path)
state.update(observed_local_utc=now,goal_status='active',goal_objective='监控',goal_tool_status_note='User created the active monitoring goal after clearing the earlier goal; no replacement goal created by the assistant.',
             operational_work_status='MONITORING_CONFIRMED_LIVE_FRESH_PAIR; BACKUP_CHAINS_ADVANCING; NO_NEW_EXPERIMENTS',
             goal_blocking_audit_consecutive_turns=0,last_goal_turn_classification='verified_wait',
             monitoring_contract=str(path),monitoring_contract_sha256=sha(path),
             active_monitor_observer={'exec_session_id':37020,'identity':contract['observer_identity'],'folder':str(FOLDER/'window001')},
             latest_fresh_pair_observation=str(snapshot),latest_fresh_pair_observation_sha256=sha(snapshot),
             next_action='Wait on the same live observation handle and verify next accepted-round milestone. Continue the monitoring goal until both runs and backups are terminally audited; never restart because an observation window ends.')
for row in observed['summary']:
    state['fresh_pair'][row['arm']].update(status=row['state'],completed_rounds=row['completed_rounds'],current_round_as_observed=row['current_round'],
            current_batch_as_observed=row['batch'],optimizer_step_attempts_as_observed=row['attempts'],backup_accepted_round=row['backup_accepted_round'],
            last_observed_local_utc=observed['observed_local_utc'])
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
entry=ROOT/'当前修订入口.txt';shutil.copyfile(entry,FOLDER/'当前修订入口_before_monitor_goal_start.txt')
summary='；'.join(r['arm']+'已接受'+str(r['completed_rounds'])+'/320轮' for r in observed['summary'])
note=('【当前目标：监控两组正在运行的新主对照】\n'+summary+'。训练进程和本地备份客户端身份已实时核实，备份链随已完成轮次推进。保持既有配置，不新增实验，不自动重启。\n'
      '监控合同：远程公平对照_20261103/公平主对照FRESH_20261112_v1/monitor_goal_20261112/monitoring_contract.json\n'
      '此入口是点时摘要，最新状态应读取当前监控窗口或重新连接对应进程核验。\n\n以下为历史记录。\n\n')
entry.write_text(note+entry.read_text(encoding='utf-8-sig'),encoding='utf-8')
print(json.dumps({'goal_status':'active','classification':'verified_wait','summary':observed['summary'],'monitoring_contract_sha256':sha(path),'session_id':37020},ensure_ascii=False))
