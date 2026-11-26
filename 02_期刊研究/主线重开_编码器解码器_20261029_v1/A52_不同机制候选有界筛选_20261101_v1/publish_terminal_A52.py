"""Publish completed A52 evidence only after archive verification and an independent judge."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    j=load(ROOT/'fixed_endpoint_judgment_A52.json')
    assert j['status']=='FIXED12_REVIEW_COMPLETE' and j['new_model_forwards_this_phase']==20328
    assert j['cumulative_model_forwards']==185612 and j['cumulative_optimizer_updates_including_capacity']==155903
    assert j['cumulative_formal_epochs']==192
    audit=load(ROOT/'terminal_frozen_input_audit.json')
    assert audit['status']=='PASS' and not audit['owned_jobs_live']
    assert all(audit['local_frozen_files'].values()) and all(audit['external_frozen_inputs'].values())
    vv={a:load(ROOT/(a+'_completed_local_verification.json')) for a in ['M','Q']}
    assert all(x['status']=='PASS' and x['epochs']==12 and x['optimizer_updates']==9708 for x in vv.values())
    assert all(x['optimizer_step_histogram']=={'9708':491} for x in vv.values())
    cap=load(ROOT/'capacity_v2/capacity_result.json');oldcap=load(ROOT/'capacity_v1_failed_archive/capacity_result.json')
    assert cap['status']=='PASS' and cap['whole_model_forwards']==9 and cap['capacity_only_optimizer_updates']==6
    assert oldcap['whole_model_forwards']==1 and oldcap['capacity_only_optimizer_updates']==0
    current=load(ROOT/'CURRENT_A52.json');prior=Path(current['receipt'])
    assert sha(prior)==current['receipt_sha256']
    old=load(prior);passed=j['pilot_pass'];metrics=j['metrics']
    report=ROOT/'A52_三组固定终点结论.txt';receipt=ROOT/'terminal_A52_preservation_receipt.json'
    assert not report.exists() and not receipt.exists()
    lines=['A52原版/同矩高斯关系/离散边际关系的固定12轮比较已结束。总目标仍active，主线确认待研究综合判断。',
      '数据VisDrone train6471/val548，640stretch、batch8、共同10类初始化、训练seed20261101、模块seed20261102。B沿用核验的A49原版12；M/Q为本轮新训练。B与val此前已可见，不是所有结果未知的预注册。']
    for a in ['B','M','Q']:
        m=metrics[a];lines.append('%s：AP %.9f，APs %.9f，small AP75 %.9f。'%(a,m['AP'],m['APs'],m['APs75']))
    lines+=['Q−B APs %+.9fpp；Q−M APs %+.9fpp；Q−B AP %+.9fpp。'%(j['Q_minus_B_APs_pp'],j['Q_minus_M_APs_pp'],j['Q_minus_B_AP_pp']),
      '冻结三门槛：Q−B APs≥0.5pp，Q−M APs≥0.3pp，Q−B AP≥−0.2pp。逐项'+json.dumps(j['gates'],ensure_ascii=False),
      ('全部通过：作为有短程证据的主线候选进入价值综合判断；本次尚不证明新颖性、统计显著、独立测试泛化或论文完成。' if passed else
       '未全部通过：淘汰本离散边际关系配方；不改频段、系数、位置、层数、seed或训练终点救援。继续找有实质区别的机制。'),
      'M/Q各12轮、9708正式更新、10159整模型前向、exit0，完整归档已下载核验。CPU核查覆盖raw/EMA键/形状/类型/有限性、原生结构性anchor缓冲例外、4个BN计数、新参数变化及491优化器状态step9708；没有重新运行检测器。',
      '三组12轮记录的顺序、每轮首8批实际输入/RNG与全轮目标流匹配。M/Q原生初始化相同、新模块精确状态SHA一致，未用容量检查权重初始化。',
      '本地CPU和服务器torch重建线性层存在约1.49e−8差异；终点比对使用已归档的服务器精确未训练模块，SHA与正式启动声明匹配。没有改变正式训练源文件、权重或放宽核验容差。',
      '最初容量v1在原版eval输出检查处报KeyError，1前向0更新；原代码/错误保留。隔离v2仅修复检查，9前向6临时更新通过。当前阶段容量合计10前向、6更新，不漏记失败操作。',
      '本阶段新增20328整模型前向（容量10、正式20318），19422含容量更新（正式19416、容量6），24正式轮。累计185612整模型前向，155854正式更新、49容量更新，含容量155903更新，192正式轮。B复用不重复累计。',
      '终点审计核实所属监督器及M/Q进程不再存活，协议所列全部本地与外部输入SHA未变；归档与裁决没有新增模型或优化器操作。',
      'M/Q新增参数均6480，输出关系维数和位置相同；离散与高斯的运算成本不完全相同。时延仅为各自eager前向记录，不作部署或配对提速结论。',
      '比较支持范围仅为完整离散边际描述相对同矩高斯的增量；尚未隔离支撑格点/初始参考几何、对角质量信息及跨query辨别的贡献。四条边际不构成校准的联合有效框后验，也不是同一对象概率。',
      'Fourier期望、积分位置编码与关系注意力有既有基础，不声称新原理。mip-NeRF/MDS/Relation/EASE/DETRNN等实际阅读范围及Relation-DETR+、Exact-NeRF摘要限制见07；方法查重不是全球新颖性保证。',
      'AP采用冻结COCO式原图clippedGT面积small≤1024、class0 union IoA、maxDets300，非官方VisDrone挑战分数。val548多次参与选题，一次seed12不能当独立测试或统计显著。',
      '没有Q之后的自动训练队列。旧A51等精确裁决、D007/D048等阳性、未训练与受限阴性均保留；不将此结论扩展为整个分布/关系建模领域无效。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=ROOT/'before_A52_terminal';archive.mkdir(exist_ok=False);resolved={}
    for n in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        p=BASE/n;h=old['current_files_sha256'][str(p)];assert sha(p)==h
        q=archive/n;shutil.copy2(p,q);assert sha(q)==h
        resolved[str(p)]={'preserved_path':str(q),'sha256':h}
    board=load(BASE/'01_候选判别表.json');state=load(BASE/'15_当前状态与结果校验.json');now=datetime.now(timezone.utc).isoformat()
    status='A52_PILOT_PASS_RESEARCH_ASSESSMENT_PENDING' if passed else 'A52_RETIRED_FIXED12_GATES_FAILED'
    entry={**j,'report':str(report),'terminal_input_audit':str(ROOT/'terminal_frozen_input_audit.json'),
           'archive_verifications':{a:str(ROOT/(a+'_completed_local_verification.json')) for a in ['M','Q']}}
    board.update(status=status,goal_status='active',mainline_validated=False,mainline_selected_for_pilot=False,
      candidate_selected_for_pilot=passed,selection_stage_complete=False,latest_report=str(report),latest_update_utc=now,A52_terminal_result=entry)
    board['A52_execution'].update(state='TERMINAL_NO_OWNED_JOBS',latest_milestone=entry)
    if not passed:
        board['current_priority_candidate']=None
        board['current_priority_question']='继续寻找有实质区别的主线；不通过频段、层位、系数、seed或名称救援A52。'
    for row in board['candidates']:
        if row.get('id')=='A52_DISTRIBUTIONAL_PAIR_GEOMETRY':row.update(status=status,terminal_result=entry)
    state.update(recorded_utc=now,status=status,goal_status='active',mainline_validated=False,mainline_selected_for_pilot=False,
      candidate_selected_for_pilot=passed,selection_stage_complete=False,current_phase='A52_TERMINAL_RESEARCH_GOAL_ACTIVE',
      latest_authoritative_phase='A52_TERMINAL_VERIFIED',current_priority_candidate=board.get('current_priority_candidate'),
      current_priority_question=board.get('current_priority_question'),latest_report=str(report),A52_terminal_result=entry,
      model_forwards=185612,cap=185612,formal_detector_training_updates=155854,capacity_only_optimizer_updates=49,
      training_updates=155903,detector_optimizer_updates_including_capacity=155903,formal_training_epochs=192,
      counts_terminal=True,training_counts_terminal=True,full_forward_cap_closed=True,current_native_job_terminal=True,
      counters_observed_lower_bounds_while_job_live=False,training_update_scope='A52 terminal exact totals, including failed eval-guard forward and all49 capacity updates; B reused not counted twice.')
    save(BASE/'01_候选判别表.json',board);save(BASE/'15_当前状态与结果校验.json',state)
    signed=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',report,ROOT/'fixed_endpoint_judgment_A52.json',
      ROOT/'terminal_frozen_input_audit.json',ROOT/'M_completed_local_verification.json',ROOT/'Q_completed_local_verification.json',
      ROOT/'13_比较范围与归因限制_v1.txt']
    save(receipt,{'status':'PASS','phase':'A52_TERMINAL','goal_status':'active','mainline_confirmed':False,
      'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':resolved,
      'current_files_sha256':{str(p):sha(p) for p in signed},'pilot_pass':passed,'owned_jobs_live':False,'new_training_queued':False,
      'cumulative_counts':{'whole_model_forwards':185612,'optimizer_including_capacity':155903,'formal_epochs':192},'publication_added_model_operations':0})
    current.update(status=status,goal_status='active',mainline_confirmed=False,receipt=str(receipt),receipt_sha256=sha(receipt),
      latest_report=str(report),owned_jobs_live=False,M_terminal_verified=True,Q_terminal_verified=True,
      M_completed_epochs_at_observation=12,Q_completed_epochs_at_observation=12,
      do_not_run_old_live_updaters_after_terminal_publication=True,
      next='Assess defensible incremental mainline if passed; otherwise select a substantively different mechanism. Broad goal remains active.')
    save(ROOT/'CURRENT_A52.json',current)
    print(json.dumps({'status':status,'pilot_pass':passed,'goal_status':'active','mainline_confirmed':False,'receipt_sha256':sha(receipt)}))
if __name__=='__main__':main()

