"""Publish verified terminal pilot evidence; broad research goal remains active pending assessment."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    judgment=load(ROOT/'fixed_endpoint_judgment_A51.json')
    assert judgment['status']=='FIXED12_REVIEW_COMPLETE' and judgment['new_model_forwards_this_phase']==20327
    assert judgment['cumulative_model_forwards']==165284 and judgment['cumulative_optimizer_updates_including_capacity']==136481
    audit=load(ROOT/'terminal_frozen_input_audit.json');assert audit['status']=='PASS' and not audit['owned_jobs_live']
    assert all(audit['local_frozen_files'].values()) and all(audit['external_frozen_inputs'].values())
    verify={a:load(ROOT/(a+'_completed_local_verification.json')) for a in ['M','Q']}
    assert all(x['status']=='PASS' and x['epochs']==12 and x['optimizer_updates']==9708 for x in verify.values())
    current=load(ROOT/'CURRENT_A51.json');prior=Path(current['receipt']);assert sha(prior)==current['receipt_sha256']
    old=load(prior)
    report=ROOT/'A51_三组固定终点结论.txt';receipt=ROOT/'terminal_A51_preservation_receipt.json'
    assert not report.exists() and not receipt.exists()
    metrics=judgment['metrics'];passed=judgment['pilot_pass']
    lines=['A51原版/普通独立门控交换/共同交换候选固定12轮已全部结束；B沿用已核验原版，M/Q为本轮新训练。',
       '本文件关闭A51实验阶段，不关闭持续寻找主线的总目标。',
       '固定12轮EMA，统一VisDrone train6471/val548、640stretch、seed20261101、共同初始化及评估口径：']
    for arm in ['B','M','Q']:
        r=metrics[arm];lines.append('%s: AP %.9f，APs %.9f，small AP75 %.9f。'%(arm,r['AP'],r['APs'],r['APs75']))
    lines+=['Q−B APs %+.9fpp；Q−M APs %+.9fpp；Q−B整体AP %+.9fpp。'%(judgment['Q_minus_B_APs_pp'],judgment['Q_minus_M_APs_pp'],judgment['Q_minus_B_AP_pp']),
       '冻结门槛：Q−B APs≥0.5pp、Q−M APs≥0.3pp、Q−B AP≥−0.2pp。逐项'+json.dumps(judgment['gates'],ensure_ascii=False),
       ('三项通过：进入主线价值与最近方法差异的综合判断；本次短验证本身不证明新颖性或统计显著。' if passed else '三项未全通过：淘汰这一具体共同交换配方，不扩容、改门槛、改层位或换seed救援；继续找实质不同机制。'),
       'M/Q均exit0、每组12轮9708更新、10159整模型前向；原始完整结果已下载并逐文件SHA校验。CPU核验覆盖raw/EMA键/形状/类型/有限性、4个BN计数、新参数变化和490个优化器状态；不是重新运行模型。',
       '三组记录的全部12轮顺序、首8批实际输入/RNG及全轮目标流一致；M/Q新增模块初始化一致，容量检查权重未用于正式训练。所有协议列出的本地与外部输入SHA终点核查通过，原监督器和两子进程均无存活的所属身份。',
       '本轮新增20327整模型前向（容量9、正式20318）、19422优化器更新（容量6、正式19416）、24正式轮。累计165284整模型前向、136481含容量更新、168正式轮；原版B复用，不重复累计。CPU归档/裁决无新增模型或优化器操作。',
       '所报AP/APs为冻结COCO式指标，非官方VisDrone挑战分数；官方协议复核仅为条件路线准备，尚未执行。相同seed一次短训练不足以主张统计显著、跨数据集泛化或部署性能。',
       'M/Q共有块残差保持，Q相对M独有的是共同交换量约束；守恒是给定通道映射下的特征代数，不是语义信息或能量守恒。正交传输、Cayley、扩散与受约束连接有既有基础；直接检测先例全文获取与阅读范围见13/14/20。',
       '原A42/A43停止、A44撤回、A48/A49终点、A50未训练筛选、旧方向阳性与受限阴性均保留。本结论不否定所有跨尺度融合、正交对齐、扩散或小目标方法。',
       'A51没有追加训练队列；总目标保持active，主线是否可保留需结合本次真实结果和研究范围判断。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=ROOT/'before_A51_terminal';archive.mkdir(exist_ok=False);resolved={}
    for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        p=BASE/name;h=old['current_files_sha256'][str(p)];assert sha(p)==h
        q=archive/name;shutil.copy2(p,q);assert sha(q)==h
        resolved[str(p)]={'preserved_path':str(q),'sha256':h}
    board_path=BASE/'01_候选判别表.json';state_path=BASE/'15_当前状态与结果校验.json'
    board=load(board_path);state=load(state_path);now=datetime.now(timezone.utc).isoformat()
    status='A51_PILOT_PASS_RESEARCH_ASSESSMENT_PENDING' if passed else 'A51_RETIRED_FIXED12_GATES_FAILED'
    entry={**judgment,'report':str(report),'terminal_input_audit':str(ROOT/'terminal_frozen_input_audit.json'),
           'archive_verifications':{a:str(ROOT/(a+'_completed_local_verification.json')) for a in ['M','Q']}}
    board.update(status=status,goal_status='active',mainline_validated=False,selection_stage_complete=False,
       mainline_selected_for_pilot=passed,pilot_validation_status=status,latest_report=str(report),latest_update_utc=now,
       A51_terminal_result=entry,selection_stage_scope='A51 execution complete; broad mainline research goal active, not completed by a negative or by gates alone.')
    board['A51_execution'].update(state='TERMINAL_NO_OWNED_JOBS',latest_milestone=entry)
    if not passed:
        board['current_priority_candidate']=None
        board['current_priority_question']='寻找与已停止方案有实质差别的下一主线；不得只改本配方的门槛、seed、层位或名称。'
    for row in board['candidates']:
        if row.get('id')=='A51_CROSS_RESOLUTION_PAIRED_EXCHANGE':row.update(status=status,terminal_result=entry)
    state.update(recorded_utc=now,goal_status='active',mainline_validated=False,selection_stage_complete=False,
       mainline_selected_for_pilot=passed,current_phase='A51_TERMINAL_RESEARCH_GOAL_ACTIVE',latest_authoritative_phase='A51_TERMINAL_VERIFIED',
       current_priority_candidate=board.get('current_priority_candidate'),current_priority_question=board.get('current_priority_question'),
       latest_report=str(report),A51_terminal_result=entry,model_forwards=165284,cap=165284,
       formal_detector_training_updates=136438,capacity_only_optimizer_updates=43,training_updates=136481,
       detector_optimizer_updates_including_capacity=136481,formal_training_epochs=168,counts_terminal=True,
       training_counts_terminal=True,training_update_scope='A51 terminal exact totals; includes43 lifetime capacity updates. Reused B12 not added again.',
       full_forward_cap_closed=True,counters_observed_lower_bounds_while_job_live=False,current_native_job_terminal=True)
    save(board_path,board);save(state_path,state)
    signed=[board_path,state_path,report,ROOT/'fixed_endpoint_judgment_A51.json',ROOT/'terminal_frozen_input_audit.json',
            ROOT/'M_completed_local_verification.json',ROOT/'Q_completed_local_verification.json',ROOT/'23_初始化来源与随机种子说明_v1.txt']
    save(receipt,{'status':'PASS','phase':'A51_TERMINAL','goal_status':'active','mainline_confirmed':False,
       'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':resolved,
       'current_files_sha256':{str(p):sha(p) for p in signed},'pilot_pass':passed,
       'cumulative_counts':{'whole_model_forwards':165284,'optimizer_including_capacity':136481,'formal_epochs':168},
       'owned_jobs_live':False,'new_training_queued':False,'publication_added_model_operations':0})
    current.update(status=status,goal_status='active',mainline_confirmed=False,receipt=str(receipt),receipt_sha256=sha(receipt),
       latest_report=str(report),owned_jobs_live=False,M_terminal_verified=True,Q_terminal_verified=True,
       M_completed_epochs_at_observation=12,Q_completed_epochs_at_observation=12,
       do_not_run_old_live_updaters_after_terminal_publication=True,
       next='Assess pilot and nearest prior if passed; otherwise select substantively different mechanism. Broad goal remains active.')
    save(ROOT/'CURRENT_A51.json',current)
    print(json.dumps({'status':status,'pilot_pass':passed,'goal_status':'active','mainline_confirmed':False,'receipt_sha256':sha(receipt)}))


if __name__=='__main__':main()
