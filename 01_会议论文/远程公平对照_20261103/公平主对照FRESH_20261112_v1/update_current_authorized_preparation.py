from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
auth=HERE/'user_authorization_20261112.json'
assert json.loads(auth.read_text(encoding='utf-8'))['training_authorized'] is True
state_path=HERE.parent/'模块消融'/'current_execution_state.json'
backup=HERE/'current_execution_state_before_fresh_authorization.json'
assert not backup.exists();shutil.copyfile(state_path,backup)
state=json.loads(state_path.read_text(encoding='utf-8-sig'))
now=dt.datetime.now(dt.timezone.utc).isoformat()
state['historical_deferred_training_episode']=state.get('current_deferral_blocking_episode')
state.update(observed_local_utc=now,goal_status=None,goal_tool_status_note='User cleared goal; current explicit experiment request continues as ordinary work. No new goal created.',
             operational_work_status='NEW_FRESH_PAIR_AUTHORIZED; DEVICES_CONFIRMED_IDLE; INPUT_AND_DRIVER_PREPARATION_IN_PROGRESS; NOT_YET_TRAINING',
             goal_blocking_audit_consecutive_turns=0,last_goal_turn_classification='ordinary_task_after_user_cleared_goal',
             current_fresh_pair_authorization=str(auth),current_fresh_pair_authorization_sha256=sha(auth),
             next_action='Finish input/driver/preflight/backup gates and launch the two explicitly authorized fresh arms. Do not add experimental groups or alter scientific settings.')
state['pending_additional_training_proposal'].update(status='AUTHORIZED_PREPARING_NOT_YET_TRAINING',training_authorized=True,approval_received=True,response_received=True,
                                                     latest_authorization=str(auth),latest_authorization_sha256=sha(auth))
state['current_pending_decision_episode'].update(status='SUPERSEDED_BY_EXPLICIT_START_INSTRUCTION',unanswered=False)
state['fresh_pair']={name:{'status':'PREPARING_NOT_TRAINING','port':port,'GPU':gpu,'target_rounds':320} for name,port,gpu in [
    ('BASE_FRESH',30715,'GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a'),('FULL_FRESH',30048,'GPU-85eca9af-10f3-4279-d72a-05ff1ecdcd3a')]}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
entry=ROOT/'当前修订入口.txt';shutil.copyfile(entry,HERE/'当前修订入口_before_fresh_authorization.txt')
note=('【当前新指令：已授权运行BASE_FRESH/FULL_FRESH，正在准备，尚未训练】\n'
      '用户在提供30715和30048两个入口后明确要求“去跑实验”，已替代此前暂缓决定。两台各一张空闲24GiB RTX3090已确认；先核验冻结输入、数据、实际环境、首batch预检及逐轮备份条件，再各运行320轮。旧稿和旧实验不覆盖。用户随后清除了goal卡片，本次按普通任务继续执行，不另建目标。\n'
      '本次目录：远程公平对照_20261103/公平主对照FRESH_20261112_v1\n\n以下为历史记录。\n\n')
entry.write_text(note+entry.read_text(encoding='utf-8-sig'),encoding='utf-8')
print(json.dumps({'training_authorized':True,'training_started':False,'goal_card_cleared':True,'current_state_sha256':sha(state_path)},ensure_ascii=False))
