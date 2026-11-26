from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
proposal=ROOT/'修订记录_20261103'/'剩余证据与最小补强方案_20261112'/'proposed_BASE_FULL_FRESH_protocol.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(proposal)=='fa4a0f72bec33b54218ba0948908f087f790c69b061108feff61e4bf3a6b33f7'
now=dt.datetime.now(dt.timezone.utc).isoformat()
record={"status":"AUTHORIZED_PREPARATION_NOT_YET_TRAINING","observed_utc":now,
        "user_instructions":["去跑实验","ssh root@172.17.120.160 -p30048 [credential omitted] 去跑实验"],
        "authorization_scope":"Previously reviewed BASE_FRESH and FULL_FRESH,320 rounds each. Two already provided compatible idle GPUs; frozen recipe; no new groups, tuning or paid resource purchases.",
        "proposal":{"file":str(proposal),"sha256":sha(proposal)},"ports":[30715,30048],
        "supersedes_prior_user_deferral":True,"training_authorized":True,"training_started":False,
        "goal_card":"User cleared the goal after activating it. Continue the explicit experiment request as ordinary task work; do not create a new goal.",
        "gates_before_launch":"Frozen file/data/environment/initial state/input hashes, isolated driver review, first-batch no-step preflight, persistent storage and per-round backup/ACK.",
        "local_disk_free_GiB":{str(p):shutil.disk_usage(p).free/2**30 for p in ['C:/','D:/']},
        "new_forward_or_training_in_this_record":False}
out=HERE/'user_authorization_20261112.json'
assert not out.exists()
out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({"status":record['status'],"proposal_sha256":sha(proposal),"local_disk_free_GiB":record['local_disk_free_GiB']},ensure_ascii=False))
