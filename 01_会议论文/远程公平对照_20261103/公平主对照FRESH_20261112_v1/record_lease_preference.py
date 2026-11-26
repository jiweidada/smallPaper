from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
FOLDER=HERE/'monitor_goal_20261112'
out=FOLDER/'user_lease_expiry_handling.json'
assert not out.exists()
record={
    'recorded_local_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
    'source':'User instruction during active monitoring',
    'user_words':'如果 ssh 连接不上就是 显卡到期了  我再去续就行了',
    'handling':'If repeated SSH observations fail, inform the user that the lease may have expired under their stated operating rule. Record the last confirmed round and locally accepted full checkpoint, and let the user renew the server.',
    'cause_scope':'User-provided operational explanation; a connection failure alone does not independently prove lease expiry or terminal training state.',
    'after_reconnection':'Verify the actual endpoint, GPU, project volume, exact process identities and checkpoint state before deciding the next step.',
    'no_new_training_or_restart_authorization_in_this_message':True,
    'do_not_restart_on_observation_timeout':True,
}
out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state_path=HERE.parent/'模块消融/current_execution_state.json'
shutil.copyfile(state_path,FOLDER/'current_execution_state_before_lease_preference.json')
state=json.loads(state_path.read_text(encoding='utf-8-sig'))
state['operator_lease_expiry_preference']={'file':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'lease_handling_saved':True,'user_handles_renewal':True,'training_actions':0}))
