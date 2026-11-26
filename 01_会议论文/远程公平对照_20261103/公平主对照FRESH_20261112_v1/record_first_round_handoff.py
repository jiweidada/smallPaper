from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rec=lambda p:{'file':str(p.resolve()),'sha256':sha(p)}
live_path=HERE/'live_pair_first_round_check.json';live=read(live_path)
export_path=ROOT/'实验数据汇总_20261108/fresh_pair_20261112/逐轮导出核对_FRESH.json';export=read(export_path)
assert all(v>=1 for v in export['accepted_rounds'].values())
for r in live['remote']:
    receipt=r['records']['run/receipt.json']
    assert receipt['status']=='RUNNING' and receipt['completed_epochs']>=1 and receipt['actual_loop_epoch']>=1
    assert all(v['live'] and v['identity_matches'] for v in r['processes'].values())
for b in live['local_backup_clients']:
    assert b['pid_alive'] and b['command_matches'] and not b['stderr_tail']
    assert b['status']['status']=='EPOCH_BACKUP_VERIFIED' and b['status']['epoch']>=0
paper=ROOT/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
assert sha(paper)=='0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c'
now=dt.datetime.now(dt.timezone.utc).isoformat()
receipt_path=HERE/'first_round_execution_handoff.json'
assert not receipt_path.exists()
receipt={'status':'BOTH_NEW_RUNS_ACTIVE_FIRST_COMPLETE_ROUND_BACKUP_VERIFIED','observed_local_utc':now,
         'live_evidence':rec(live_path),'verified_new_rows':rec(export_path),'running_manifest':rec(HERE/'running_pair_manifest.json'),
         'completed_rounds_as_observed':{r['arm']:r['records']['run/receipt.json']['completed_epochs'] for r in live['remote']},
         'training_continues':True,'target_rounds_per_arm':320,'final_endpoints_available':False,
         'first_checkpoint_validation':'Actual evaluated EMA, full next-state checkpoint, optimizer/EMA/warmup/RNG state, prediction/evaluator-array bindings and local manifest/ACK/remote accepted receipts checked by the separate CPU-only validator before round2.',
         'standalone_supervisors_and_backup_clients_verified_live':True,
         'local_backup_dependency':'D-drive client needs this Windows computer awake and connected; each following round is gated on verified local backup.',
         'bounds':{'groups':2,'rounds_per_arm':320,'hard_timeout_seconds_per_arm':604800,'max_full_validation_passes_total':640,'automatic_training_retry':False,'max_backup_GiB_total':120},
         'preparation_incidents':[
             {'issue':'Python3.12 AST-unparse quote form incompatible with Python3.10','model_forwards_at_failure':0,'fix':'Only quote spelling; same AST. Actual remote3.10 compile and the one authorized forward/backward succeeded afterward.','evidence':[rec(HERE/f'actual_python310_compile_{p}.json') for p in [30715,30048]]},
             {'issue':'DPAPI user context differed for approved D-drive background process','training_started_at_failure':False,'fix':'Separate local encrypted credentials created under the actual backup process identity. First failed bootstrap logs retained; replacement backup clients passed probe and first full checkpoint.'},
         ],
         'actual_local_payload_reader_torch':'2.6.0+cpu; read-only tensor/state checks, no forward or evaluator',
         'manuscript':rec(paper),'revision':40,'submission_ready':False,
         'goal_card':'User cleared goal; no replacement goal or scheduled automation created.'}
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
status_path=HERE/'运行状态与数据入口.txt'
text=(
    'BASE_FRESH / FULL_FRESH 新主对照运行记录\n\n'
    '当前已确认：30715运行BASE_FRESH，30048运行FULL_FRESH，各使用一张RTX3090 24GiB。两组已完成首轮训练、548图验证、完整状态本地备份和回传确认，并进入第2轮。训练与后台备份进程的PID、启动标识和命令均已核对存活。该状态是首轮验收时的点时记录。\n\n'
    '统一设置：从归档初始权重开始，seed0，batch4，640×640，FP32，AMP和TF32关闭，各320轮，每轮1617次更新和一次548图验证。两台实际软件版本相同；与历史清单有差异的wheel、tomli和packaging三个构建工具版本已如实记录，科学依赖版本匹配。初始共享554个张量及实际训练前100批输入均已核验。\n\n'
    '备份：D:/MPSMA_DEIM_experiment_backups/BASE_FULL_FRESH_20261112_v1，BASE45GiB和FULL75GiB合计120GiB封顶。每轮先校验实际评价状态、下一轮状态、优化器/EMA/warmup/随机状态、预测和评价数组的绑定，再回传ACK。训练期间本机须保持开机联网。监督进程每组7天硬超时，不自动重试，不调参或增加实验组。\n\n'
    '新数据入口：实验数据汇总_20261108/fresh_pair_20261112/run_index.json。逐轮指标_FRESH.csv目前只导出两组已接受的首轮记录；训练过程中的原生记录持续保存于各组D盘backups/epochs目录，后续导出需以这些接受凭据为准。旧四组完成数据单独保留。\n\n'
    '论文仍为第40版；新固定320轮端点尚未产生，暂不替换论文结果，不宣称已达到投稿要求。本次只补完整配置主对照，不隔离MSCA容量效应。\n\n'
    '首轮证据：first_round_execution_handoff.json；运行命令、预检、协议及准备失败记录均保留在本目录。用户已清除goal卡片，本次按普通实验任务执行，未另建目标。\n')
status_path.write_text(text,encoding='utf-8')
state_path=HERE.parent/'模块消融/current_execution_state.json'
shutil.copyfile(state_path,HERE/'current_execution_state_before_first_round_handoff.json')
state=read(state_path)
state.update(observed_local_utc=now,operational_work_status='BOTH_FRESH_ARMS_RUNNING; FIRST_FULL_ROUND_AND_CPU_BACKUP_ACK_CHAIN_VERIFIED; NEW_DATA_SEPARATELY_INDEXED',
             latest_fresh_pair_observation=str(live_path),latest_fresh_pair_observation_sha256=sha(live_path),
             latest_fresh_pair_handoff=str(receipt_path),latest_fresh_pair_handoff_sha256=sha(receipt_path),
             latest_fresh_pair_data_export=str(export_path),latest_fresh_pair_data_export_sha256=sha(export_path),
             next_action='The two authorized bounded jobs continue under their standalone supervisors and per-round backup clients. When checking again, verify these exact live process identities and accepted backup prefix; never restart on an observation timeout. Import completed endpoints only after320 accepted rounds in both arms.')
for r in live['remote']:
    raw=r['records']['run/receipt.json']
    state['fresh_pair'][r['arm']].update(completed_rounds=raw['completed_epochs'],current_actual_epoch=raw['actual_loop_epoch'],
                                       current_batch_as_observed=raw.get('batch'),optimizer_step_attempts_as_observed=raw['optimizer_step_attempts'],
                                       per_round_backup_verified=True,last_observed_local_utc=now)
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
entry=ROOT/'当前修订入口.txt';shutil.copyfile(entry,HERE/'当前修订入口_before_first_round_handoff.txt')
note=('【当前：两组新主对照已开跑，首轮完整训练/验证/本地备份链已核验】\n'
      '30715→BASE_FRESH；30048→FULL_FRESH。两组均已从初始权重完成第1轮并进入第2轮，目标各320轮。新实验数据已单独导出，旧结果和第40版稿件保留。后台逐轮备份正在运行，本机需要保持开机联网。\n'
      '可读运行记录：远程公平对照_20261103/公平主对照FRESH_20261112_v1/运行状态与数据入口.txt\n'
      '新数据：实验数据汇总_20261108/fresh_pair_20261112/run_index.json\n\n以下为历史记录。\n\n')
entry.write_text(note+entry.read_text(encoding='utf-8-sig'),encoding='utf-8')
assert sha(paper)=='0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c'
print(json.dumps({'status':receipt['status'],'receipt':rec(receipt_path),'human_status':rec(status_path),'accepted_rounds':export['accepted_rounds'],'training_continues':True,'manuscript_changed':False},ensure_ascii=False))
