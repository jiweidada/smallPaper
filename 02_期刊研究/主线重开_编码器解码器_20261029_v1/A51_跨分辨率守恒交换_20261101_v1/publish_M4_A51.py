from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
A49=BASE/'A49_下一主线有界筛选_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
prior=ROOT/'08_A51启动与历史保留校验回执.json'
assert sha(prior)=='3f4e0142cd0cb3f4f08afba3f16520b1cdec8fc770b5298063009ccefb01de58'
receipt=load(prior);resolved={};archive=ROOT/'before_A51_M4';archive.mkdir(exist_ok=True)
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    p=BASE/name;h=receipt['current_files_sha256'][str(p)];assert sha(p)==h
    q=archive/name;assert not q.exists();shutil.copy2(p,q);assert sha(q)==h
    resolved[str(p)]={'preserved_path':str(q),'sha256':h}
ob=load(ROOT/'observer_11_M4.json');assert ob['exit']==0
obs=json.loads(ob['stdout'])
assert all(obs['frozen_files_unchanged'].values())
assert obs['supervisor_process']['live'] and obs['supervisor_process']['identity_matches']
assert obs['arms']['M']['process']['live'] and obs['arms']['M']['process']['identity_matches']
checks={};keys=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']
for e in range(1,5):
    local=ROOT/('M_startup' if e==1 else 'M4_inputs')/('epoch%02d_input_flow.json'%e)
    m=load(local);b=load(A49/'completed_B_v1/arm_B'/('epoch%02d_input_flow.json'%e))
    current={k:m[k]==b[k] for k in keys};assert all(current.values())
    assert m['optimizer_updates']==809*e
    checks[str(e)]={'checks':current,'M_input_log_sha256':sha(local)}
m4=load(ROOT/'M4_inputs/epoch04_EMA_metrics.json')
b4=load(A49/'completed_B_v1/arm_B/epoch04_EMA_metrics.json')
assert m4==obs['arms']['M']['metrics']['4']
for k in ['max_detections','evaluated_GT','ignored_valid_GT','area_convention','scope']:assert m4[k]==b4[k],k
completed=load(ROOT/'M4_inputs/completed_epochs.json')
event4=next(x for x in completed['events'] if x['epoch']==4)
assert event4['EMA_validation']==m4
evaluation={'B4':b4,'M4':m4,'M_minus_B_AP_pp':m4['AP']-b4['AP'],
 'M_minus_B_APs_pp':m4['APs']-b4['APs'],'M_minus_B_APs75_pp':m4['APs75']-b4['APs75'],
 'Q4':None,'scientific_verdict':'INTERIM_CONTROL_ONLY_NO_FIXED12_OR_Q_EFFECT_CLAIM'}
checkpath=ROOT/'17_M4与首四轮数据流核验_v1.json';assert not checkpath.exists()
save(checkpath,{'status':'PASS','arm':'M','validated_epoch':4,'optimizer_updates_at_M4':3236,
 'flow_checks':checks,'evaluation':evaluation,'frozen_source_files':obs['frozen_files_unchanged'],
 'no_detector_rerun':True,'new_model_forwards_by_verification':0,'new_optimizer_updates_by_verification':0})
report=ROOT/'18_M4阶段解释_v1.txt';assert not report.exists()
report.write_text('A51普通交换对照M已完成第4轮验证，训练继续，Q尚未开始。总目标进行，主线未确认。\n'
 '同轮EMA：B AP22.365629/APs12.537371；M AP22.480718/APs12.589967。\n'
 'M−B：AP+0.115089pp，APs+0.052596pp，small AP75+'+format(evaluation['M_minus_B_APs75_pp'],'.6f')+'pp。\n'
 '这只是普通交换对照的早期描述，不是Q的增量，也不是固定12终点/统计显著/论文创新结论。\n'
 '首4轮逐轮顺序、首8批实际输入、首8批前向前RNG、全轮目标流SHA与原版B完全一致，M4评估GT数/面积/ignore/maxDets口径一致，5个冻结源文件未变。核验没有新增模型前向或优化器更新。\n'
 '真实训练峰值约12.86GiB，仍在24GB单卡范围。初始稀疏batch8容量3.34GiB只代表容量样本，不能用来代替真实训练峰值。\n'
 '理论和先例阅读补充：FMRNet的IEEE/Crossref登记已确认，出版社PDF尚未取得；mHC-DEIM仍只有作者摘要。Q可写成带质量矩阵的连接扩散更新；扩散、正交传输与Cayley都归已有基础。M/Q共有细网格块残差保持，Q相对M独有的是共同交换量约束。详见13、14、16。\n'
 '继续冻结方案：M12后Q12，不以4/8轮最佳点做结论；Q−B APs≥0.5pp、Q−M APs≥0.3pp、Q−B AP≥−0.2pp。未过则淘汰该具体配方，不自动扩容或救援扫描。\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
counts={'new_model_forwards_observed_lower_bound':9,'new_optimizer_updates_observed_lower_bound':6,
 'new_formal_updates_observed_lower_bound':0,'new_completed_formal_epochs':0}
for a in ['M','Q']:
    r=obs['arms'][a]['result'] or obs['arms'][a]['progress'] or {}
    counts['new_model_forwards_observed_lower_bound']+=r.get('full_model_forwards',0)
    u=r.get('optimizer_updates',0)
    counts['new_optimizer_updates_observed_lower_bound']+=u
    counts['new_formal_updates_observed_lower_bound']+=u
    counts['new_completed_formal_epochs']+=u//809
model_count=144957+counts['new_model_forwards_observed_lower_bound']
optim_count=117059+counts['new_optimizer_updates_observed_lower_bound']
board_path=BASE/'01_候选判别表.json';state_path=BASE/'15_当前状态与结果校验.json'
board=load(board_path);state=load(state_path)
milestone={'phase':'A51_M4','goal_status':'active','mainline_confirmed':False,'report':str(report),
 'verification':str(checkpath),'observation':str(ROOT/'observer_11_M4.json'),'evaluation':evaluation,'counts':counts,
 'latest_reading_registry':str(ROOT/'14_核心文献矩阵与阅读范围_v2.json')}
board.update(latest_report=str(report),latest_update_utc=now,A51_M4_milestone=milestone,goal_status='active',mainline_validated=False)
board['A51_execution']['latest_milestone']=milestone
state.update(recorded_utc=now,latest_report=str(report),A51_M4_milestone=milestone,goal_status='active',
 model_forwards=model_count,formal_detector_training_updates=117022+counts['new_formal_updates_observed_lower_bound'],
 detector_optimizer_updates_including_capacity=optim_count,training_updates=optim_count,
 formal_training_epochs=144+counts['new_completed_formal_epochs'],capacity_only_optimizer_updates=43,
 cap=165284,full_forward_cap_closed=False,counts_terminal=False,counters_observed_lower_bounds_while_job_live=True,
 current_phase='A51_M12_TRAINING_AFTER_FIRST_VALIDATION',latest_authoritative_phase='A51_M4_VERIFIED')
save(board_path,board);save(state_path,state)
out=ROOT/'19_M4里程碑与历史保留回执.json';assert not out.exists()
signed=[board_path,state_path,checkpath,report,ROOT/'13_理论归属与直接检测先例补充_v1.txt',
 ROOT/'14_核心文献矩阵与阅读范围_v2.json',ROOT/'16_共同设计与唯一增量边界_v1.txt']
save(out,{'status':'PASS','phase':'A51_M4','goal_status':'active','mainline_confirmed':False,
 'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':resolved,
 'current_files_sha256':{str(p):sha(p) for p in signed},'observed_lower_bounds':{'whole_model_forwards':model_count,'optimizer_including_capacity':optim_count},
 'forward_ceiling':165284,'owned_processes_verified_live':True,'no_extra_training_queue':True})
current=load(ROOT/'CURRENT_A51.json')
current.update(receipt=str(out),receipt_sha256=sha(out),latest_observation=str(ROOT/'observer_11_M4.json'),
 latest_report=str(report),M_completed_epochs_at_observation=counts['new_completed_formal_epochs'],
 M_observed_optimizer_update_lower_bound=obs['arms']['M']['progress']['optimizer_updates'],
 first_four_epoch_flow_checks=str(checkpath))
save(ROOT/'CURRENT_A51.json',current)
print(json.dumps({'status':'M4_VERIFIED_AND_PUBLISHED','evaluation':evaluation,'counts':counts,
 'receipt_sha256':sha(out),'goal_status':'active','mainline_confirmed':False}))
