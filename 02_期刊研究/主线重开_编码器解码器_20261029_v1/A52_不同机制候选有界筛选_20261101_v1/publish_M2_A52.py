from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
A49=BASE/'A49_下一主线有界筛选_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
fields=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']
checks={}
for e in [1,2]:
    native=load(A49/'completed_B_v1/arm_B'/('epoch%02d_input_flow.json'%e))
    m=load(ROOT/'M_startup'/('epoch%02d_input_flow.json'%e))
    assert native['optimizer_updates']==m['optimizer_updates']==809*e
    checks[str(e)]={k:native[k]==m[k] for k in fields};assert all(checks[str(e)].values())
init=load(ROOT/'M_startup/initialization.json');exact=load(ROOT/'new_module_untrained_initial_state_receipt.json')
assert init['all_native_initial_tensors_identical'] and init['scratch_no_capacity_weights']
assert init['new_module_initialization_sha256']==exact['tensor_sha256']=='69da2b6c3fd9b4c7fa879d807880542cb3a31243004895a67502d8c418829bd0'
assert sha(ROOT/'new_module_untrained_initial_state.pth')==exact['file_sha256']
obsp=ROOT/'observer_09_M1_verified.json';o=json.loads(load(obsp)['stdout'])
p=o['arms']['M']['progress'];assert p['completed_epochs']==2 and p['optimizer_updates']==1719
assert o['arms']['M']['process']['live'] and o['supervisor_process']['live'] and all(o['local_frozen_files_unchanged'].values())
counts={'whole_model_forwards':165294+p['full_model_forwards'],'formal_optimizer_updates':136438+p['optimizer_updates'],
 'capacity_only_updates':49,'all_optimizer_updates':136487+p['optimizer_updates'],'formal_epochs':168+p['completed_epochs']}
verify=ROOT/'10_M首两轮输入流与初始化核验.json';assert not verify.exists()
save(verify,{'status':'PASS','M_completed_epochs_verified':2,'flow_checks':checks,
 'M_common_initialization':init,'Q_formal_initialization_not_yet_observed':True,
 'exact_remote_untrained_state_receipt':str(ROOT/'new_module_untrained_initial_state_receipt.json'),
 'CPU_cross_version_recreation':str(ROOT/'local_module_init_cross_version_v1.json'),
 'archive_utility_preparation':str(ROOT/'result_tool_preparation_receipt_v2.json'),
 'source_files_unchanged':True,'detector_forwards_added_by_verification':0,'optimizer_updates_added_by_verification':0,
 'scope':'Recorded input/RNG/target flow and declared tensor initialization; not repeated full-model inference or an AP result.'})
current=load(ROOT/'CURRENT_A52.json');prior=Path(current['receipt']);assert sha(prior)==current['receipt_sha256']
old=load(prior);archive=ROOT/'before_M2_update';archive.mkdir(exist_ok=False);resolved={}
for n in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    original=BASE/n;h=old['current_files_sha256'][str(original)];assert sha(original)==h
    dest=archive/n;shutil.copy2(original,dest);assert sha(dest)==h
    resolved[str(original)]={'preserved_path':str(dest),'sha256':h}
report=ROOT/'12_M两轮进展.txt';receipt=ROOT/'11_M两轮进展与保留回执.json'
assert not report.exists() and not receipt.exists()
report.write_text('A52候选训练进行，主线未确认。M已完成2轮，当前第3轮101/809批；Q尚未正式启动。\n'
 '原版B12复用，M同矩高斯关系，Q离散边际关系；正式各最多12轮，固定12EMA与原冻结三项门槛。\n'
 'M前两轮顺序、每轮首8批实际输入/RNG、全轮目标流均与原版匹配；公共初始化一致，容量权重未用于正式训练。\n'
 '容量含最初eval检查错误共10前向/6临时更新，原错误与修复均保留。\n'
 '本地旧torch与服务器torch的相同seed线性层初始化存在最大1.49e−8字节级差异；后续CPU归档比较采用服务器精确未训练小模块状态，其SHA与M正式启动声明一致。没有修改训练源码、权重或容差，没有为此重跑检测器。\n'
 '真实loss %.6f，峰值%.4fGiB；目前没有M4或Q的AP结论。\n'%(p['loss'],p['peak_allocated_bytes']/1024**3)
 +'累计实际下界'+json.dumps(counts,ensure_ascii=False)+'。\n'
 '总上限185612整模型前向、155903含容量更新、192正式轮；B复用不重复累计。\n'
 '文献/方法/风险见04、07；单seed12与多次使用的val548不能当统计显著、独立测试或完成论文。\n'
 '在M/Q固定终点前不新增候选或额外训练队列。\n',encoding='utf-8')
board=load(BASE/'01_候选判别表.json');state=load(BASE/'15_当前状态与结果校验.json')
update={'observation':str(obsp),'M_completed_epochs_verified':2,'M_optimizer_update_lower_bound':1719,
 'M_first_two_epoch_flow_verified':True,'Q_not_started_at_observation':True,'verification':str(verify),'report':str(report)}
now=datetime.now(timezone.utc).isoformat()
board.update(latest_report=str(report),latest_update_utc=now)
board['A52_execution'].update(latest_milestone=update,first_epoch_input_flow_verified=True)
state.update(recorded_utc=now,latest_report=str(report),model_forwards=counts['whole_model_forwards'],
 formal_detector_training_updates=counts['formal_optimizer_updates'],capacity_only_optimizer_updates=49,
 training_updates=counts['all_optimizer_updates'],detector_optimizer_updates_including_capacity=counts['all_optimizer_updates'],
 formal_training_epochs=counts['formal_epochs'],counts_terminal=False,training_counts_terminal=False,
 counters_observed_lower_bounds_while_job_live=True)
state['A52_execution'].update(latest_milestone=update,first_epoch_input_flow_verified=True)
save(BASE/'01_候选判别表.json',board);save(BASE/'15_当前状态与结果校验.json',state)
signed=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',verify,report,
 ROOT/'new_module_untrained_initial_state_receipt.json',ROOT/'new_module_untrained_initial_state.pth',
 ROOT/'local_module_init_cross_version_v1.json',ROOT/'result_tool_preparation_receipt_v2.json']
save(receipt,{'status':'PASS','phase':'A52_M2','goal_status':'active','mainline_confirmed':False,
 'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),
 'previous_signed_current_inputs_resolved':resolved,'current_files_sha256':{str(f):sha(f) for f in signed},
 'observed_counts_lower_bounds':counts,'new_detector_or_optimizer_operations_from_publication':0,
 'no_extra_training_queue':True})
current.update(receipt=str(receipt),receipt_sha256=sha(receipt),latest_report=str(report),latest_observation=str(obsp),
 first_epoch_input_flow_verified=True,first_two_epoch_flow_verified=True,M_completed_epochs_at_observation=2,
 M_observed_optimizer_update_lower_bound=1719)
save(ROOT/'CURRENT_A52.json',current)
print(json.dumps({'status':'M_FIRST_TWO_EPOCH_FLOWS_VERIFIED','receipt_sha256':sha(receipt),'counts_lower_bounds':counts,'mainline_confirmed':False}))

