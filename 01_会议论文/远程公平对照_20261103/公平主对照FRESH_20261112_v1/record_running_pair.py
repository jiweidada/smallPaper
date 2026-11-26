from pathlib import Path, PurePosixPath
import concurrent.futures
import datetime as dt
import hashlib
import io
import json
import shutil
import tarfile
from ssh_control import connect
from stage_and_preflight import SPECS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p):return {'file':str(p.resolve()),'sha256':sha(p)}

live_path=HERE/'live_pair_first_updates.json'
live=json.loads(live_path.read_text(encoding='utf-8'))
for row in live['remote']:
    receipt=row['records']['run/receipt.json']
    assert receipt['status']=='RUNNING' and receipt['optimizer_step_attempts']>100
    assert row['actual_first100_rows']==100 and all(x['live'] for x in row['processes'].values())
assert all(x['pid_alive'] and x['command_matches'] and not x['stderr_tail'] for x in live['local_backup_clients'])
expected_path=HERE.parent/'preflight/BASE.first100.json'
if expected_path.is_file():expected=json.loads(expected_path.read_text(encoding='utf-8'))
else:
    bundle=HERE.parent/'secondary_frozen_inputs.tar.gz'
    if not bundle.is_file():bundle=next(HERE.parent.glob('**/secondary_frozen_inputs.tar.gz'))
    with tarfile.open(bundle,'r:gz') as tar:expected=json.load(tar.extractfile('preflight/BASE.first100.json'))
def fetch_inputs(port):
    volume,arm,python,gpu=SPECS[port]
    remote=PurePosixPath('/'+volume)/'mpsma_fresh_pair_20261112_v1'/arm/'run/actual_first100.jsonl'
    client=connect(port)
    try:
        with client.open_sftp() as sftp:
            with sftp.open(str(remote),'rb') as f:payload=f.read()
        rows=[json.loads(x) for x in payload.splitlines() if x]
        assert rows==expected and len(rows)==100
        target=HERE/f'{arm}.actual_first100.jsonl';assert not target.exists();target.write_bytes(payload)
        return {'arm':arm,'port':port,'remote_source':str(remote),'native_first100':rec(target),'all100_rows_equal_archived_reference':True}
    finally:client.close()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:inputs=list(pool.map(fetch_inputs,[30715,30048]))
now=dt.datetime.now(dt.timezone.utc).isoformat()
manifest={
    'material_passport':{'origin_skill':'academic-research-suite / experiment-agent','origin_mode':'run','verification_status':'RUNNING_NOT_COMPLETED','scope':'Two newly authorized fresh trajectories, not final results.'},
    'observed_utc':now,'status':'BOTH_FRESH_ARMS_RUNNING','user_authorization':rec(HERE/'user_authorization_20261112.json'),
    'live_process_and_gpu_evidence':rec(live_path),'actual_training_first100':inputs,
    'arms':[{'arm':r['arm'],'port':r['port'],'protocol':rec(HERE/f"execution_protocol_{r['port']}.json"),
             'worker_pid':r['records']['run/receipt.json']['pid'],'supervisor':r['records']['supervisor.json'],
             'progress_as_observed':r['records']['run/receipt.json'],'gpu_as_observed':r['gpu']} for r in live['remote']],
    'shared_conditions':{'seed':0,'batch':4,'precision':'FP32; AMP/TF32 disabled','rounds_per_arm':320,'updates_per_round':1617,'validation_images':548,'input_pixels':[640,640]},
    'local_backup_root':'D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1','backup_limits_GiB':{'BASE_FRESH':45,'FULL_FRESH':75},
    'backup_dependency':'Each next round requires verified local D-drive payloads and ACK. This computer and network connection must remain available.',
    'clock_note':'Remote and local wall clocks differ. Preserve both observed timestamps, use process identities and native monotonic timeout/progress counters; do not infer cross-machine elapsed time from the raw dates.',
    'results_handling':'Record fresh pair separately from the four completed historical arms. No new fixed-round endpoint exists yet; do not replace manuscript tables during training.',
    'paper_revision':40,'submission_ready':False,'new_goal_created':False,
}
target=HERE/'running_pair_manifest.json';assert not target.exists();target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
data_dir=ROOT/'实验数据汇总_20261108'/'fresh_pair_20261112';data_dir.mkdir(exist_ok=True)
(data_dir/'run_index.json').write_text(json.dumps({'status':'RUNNING_NO_FINAL_ENDPOINTS','manifest':rec(target),'separate_from_completed_four_arm_snapshot':True,'arms':['BASE_FRESH','FULL_FRESH'],'final_endpoint_rows':0},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state_path=HERE.parent/'模块消融/current_execution_state.json'
shutil.copyfile(state_path,HERE/'current_execution_state_before_recording_running_pair.json')
state=json.loads(state_path.read_text(encoding='utf-8'))
state.update(observed_local_utc=now,operational_work_status='BASE_FRESH_AND_FULL_FRESH_RUNNING_FROM_ARCHIVED_INITIAL_STATES; BOTH_FIRST100_INPUTS_MATCH; PER_ROUND_BACKUP_CLIENTS_LIVE',
             latest_fresh_pair_manifest=str(target),latest_fresh_pair_manifest_sha256=sha(target),
             next_action='Verify first complete round, CPU payload validation and ACK; then continue the authorized bounded runs with their standalone supervisors/backup clients. No additional groups or retries.')
state['pending_additional_training_proposal'].update(status='AUTHORIZED_AND_RUNNING',training_started=True)
state['fresh_pair']={r['arm']:{'status':'RUNNING','port':r['port'],'worker_pid':r['records']['run/receipt.json']['pid'],
                              'supervisor_pid':r['records']['supervisor.json']['supervisor_pid'],'completed_rounds':0,'target_rounds':320,
                              'optimizer_step_attempts_as_observed':r['records']['run/receipt.json']['optimizer_step_attempts'],
                              'protocol_sha256':r['records']['run/receipt.json']['protocol_sha256']} for r in live['remote']}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
entry=ROOT/'当前修订入口.txt';shutil.copyfile(entry,HERE/'当前修订入口_before_recording_running_pair.txt')
note=('【当前状态：新BASE_FRESH/FULL_FRESH已并行训练，目标各320轮】\n'
      '30715运行BASE_FRESH，30048运行FULL_FRESH，各使用一张RTX3090。两组均从归档初始权重开始，seed0、batch4、FP32。各首batch预检通过，实际训练前100批与冻结参考逐项一致；后台逐轮备份客户端存活、初始文件载荷校验及ACK已通过，正在等待首次完整轮次备份验收。\n'
      '新实验数据单列：实验数据汇总_20261108/fresh_pair_20261112/run_index.json。原四组完成数据和第40版论文不覆盖，新末轮结果尚未产生。逐轮备份依赖本机D盘，训练期间本机须保持开机联网。goal卡片由用户清除，本次仍按普通实验任务执行。\n'
      '运行证据：远程公平对照_20261103/公平主对照FRESH_20261112_v1/running_pair_manifest.json\n\n以下为历史记录。\n\n')
entry.write_text(note+entry.read_text(encoding='utf-8-sig'),encoding='utf-8')
print(json.dumps({'status':'BOTH_FRESH_ARMS_RUNNING','actual_first100_equal':True,'manifest':rec(target),'data_index':str(data_dir/'run_index.json'),'paper_changed':False},ensure_ascii=False))
