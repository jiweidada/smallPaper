"""Publish M12 and Q startup only after M archive/CPU verification and Q epoch1 logs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;A49=BASE/'A49_下一主线有界筛选_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observation',required=True);args=parser.parse_args()
    current=load(ROOT/'CURRENT_A52.json')
    assert not current.get('do_not_run_old_live_updaters_after_terminal_publication',False)
    v=load(ROOT/'M_completed_local_verification.json')
    assert v['status']=='PASS' and v['epochs']==12 and v['optimizer_updates']==9708
    assert v['optimizer_step_histogram']=={'9708':491} and v['model_parameters']==9809592
    mf=ROOT/'completed_M_v1/arm_M';bf=A49/'completed_B_v1/arm_B'
    m=load(mf/'result.json');b=load(bf/'result.json')
    assert m['full_model_forwards']==10159 and m['endpoint_metrics']==v['endpoint_metrics']
    assert load(mf/'exit.json')['exit_code']==0
    mi=load(mf/'initialization.json');qi=load(ROOT/'Q_startup/initialization.json')
    assert qi['new_module_initialization_sha256']==mi['new_module_initialization_sha256']=='69da2b6c3fd9b4c7fa879d807880542cb3a31243004895a67502d8c418829bd0'
    assert qi['common_init_sha256']==mi['common_init_sha256']=='2c6e0f9c6765fc7ba069c2292e0884bb86924d2232a952499fcd251cc57b47e6'
    assert qi['all_native_initial_tensors_identical'] and qi['scratch_no_capacity_weights'] and qi['seed']==20261101
    assert qi['protocol_sha256']==mi['protocol_sha256']=='c13b61bdbc09d81a051108f7a28516f14ae3a3521a81d8247574129ca413d027'
    fields=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']
    checks={}
    for e in range(1,13):
        f=load(mf/('epoch%02d_input_flow.json'%e));ref=load(bf/('epoch%02d_input_flow.json'%e))
        assert f['optimizer_updates']==ref['optimizer_updates']==809*e
        checks[str(e)]={k:f[k]==ref[k] for k in fields};assert all(checks[str(e)].values())
    q1=load(ROOT/'Q_startup/epoch01_input_flow.json');b1=load(bf/'epoch01_input_flow.json')
    qchecks={k:q1[k]==b1[k] for k in fields}
    assert all(qchecks.values()) and q1['optimizer_updates']==809
    obsfile=ROOT/args.observation;wire=load(obsfile);assert wire['exit']==0
    o=json.loads(wire['stdout']);assert all(o['local_frozen_files_unchanged'].values())
    assert o['supervisor_process']['live'] and o['supervisor_process']['identity_matches']
    assert o['arms']['Q']['process']['live'] and o['arms']['Q']['process']['identity_matches']
    assert o['arms']['M']['exit']['exit_code']==0 and not o['arms']['M']['process']['live']
    qp=o['arms']['Q']['progress']
    assert qp['status'] in ('TRAINING','VALIDATING') and qp['optimizer_updates']>=809
    mb=m['endpoint_metrics'];bb=b['endpoint_metrics']
    for k in ['max_detections','evaluated_GT','ignored_valid_GT','area_convention','scope']:assert mb[k]==bb[k]
    delta={k:mb[k]-bb[k] for k in ['AP','APs','APs75','ARs']}
    requirements={'APs_at_least':max(bb['APs']+.5,mb['APs']+.3),'AP_at_least':bb['AP']-.2,
       'Q_minus_B_APs_pp_at_least':.5,'Q_minus_M_APs_pp_at_least':.3}
    record={'status':'M12_ARCHIVE_AND_STATE_VERIFIED_Q_RUNNING','goal_status':'active','mainline_confirmed':False,
       'metrics':{'B12':bb,'M12':mb},'M_minus_B_pp':delta,'M_formal_optimizer_updates':9708,
       'M_full_model_forwards':10159,'M_new_parameters':6480,'M_optimizer_steps':v['optimizer_step_histogram'],
       'M_payload_files_verified':v['payload_files_verified'],'M_bundle_sha256':v['bundle_sha256'],
       'M12_recorded_flow_checks_against_B':checks,'Q_initialization_verified':True,
       'Q1_recorded_flow_checks_against_B':qchecks,'Q_exact_primary_requirements':requirements,
       'scientific_verdict':'Q_FIXED12_PENDING_NO_EFFICACY_OR_NOVELTY_CLAIM',
       'verification_added_detector_forwards':0,'verification_added_optimizer_updates':0}
    verify=ROOT/'M12_终点与Q初始化核验.json';report=ROOT/'M12_结果与Q启动.txt';receipt=ROOT/'M12_终点与Q启动保留回执.json'
    assert all(not p.exists() for p in [verify,report,receipt])
    save(verify,record)
    lines=['A52高斯关系对照M完成固定12，离散边际关系候选Q实际运行；主线尚未确认。',
      '固定12EMA：B AP%.9f/APs%.9f；M AP%.9f/APs%.9f。'%(bb['AP'],bb['APs'],mb['AP'],mb['APs']),
      'M−B：AP%+.9fpp，APs%+.9fpp，small AP75%+.9fpp。'%(delta['AP'],delta['APs'],delta['APs75']),
      'M是同矩高斯关系对照；候选Q仍待固定12。三门槛和主指标不变，不挑选中间最好轮次。',
      'M exit0，9708更新、10159整模型前向、完整%d个payload已下载核验。raw/EMA状态与4个BN计数、新参数变化、491个优化器状态step9708均通过。'%v['payload_files_verified'],
      'M计时%.2f分钟，allocated峰值%.4fGiB；batch1 eager BF16前向中位%.3fms，含关系模块，未含decode/postprocess，不是机载或配对提速结论。'%(m['elapsed_seconds']/60,m['peak_allocated_bytes']/1024**3,m['latency']['median_ms']),
      'M全部12轮记录顺序、每轮首8批实际输入/RNG、全轮目标流同B；Q原生与新增模块初始化同M，Q首轮同类记录也匹配B。容量权重未用于正式训练。',
      'Q固定12必须同时满足APs≥%.12f、AP≥%.12f，以及Q−M APs≥0.3pp；具体比较使用未四舍五入原始值。'%(requirements['APs_at_least'],requirements['AP_at_least']),
      'COCO式原图GT面积/ignore/maxDets300口径，非官方挑战分数。val548多次可见，一次seed12不足以统计显著、独立测试或完成论文。',
      '归因限制见13：完整离散边际还含支撑格点、对角质量与周期结构，不单独证明跨query实例辨别或校准后验。最近期刊扩展的摘要/元数据与未获全文范围见15。',
      '本次核验新增检测器前向0、优化器更新0；唯一队列为本轮Q12，无额外训练或候选。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    prior=Path(current['receipt']);assert sha(prior)==current['receipt_sha256'];old=load(prior)
    archive=ROOT/'before_M_terminal_canonical';archive.mkdir(exist_ok=False);resolved={}
    for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        p=BASE/name;h=old['current_files_sha256'][str(p)];assert sha(p)==h
        q=archive/name;shutil.copy2(p,q);assert sha(q)==h
        resolved[str(p)]={'preserved_path':str(q),'sha256':h}
    qup=qp['optimizer_updates'];qfw=qp['full_model_forwards']
    counts={'whole_model_forwards':165294+10159+qfw,'formal_optimizer_updates':136438+9708+qup,
       'capacity_only_updates':49,'all_optimizer_updates':136487+9708+qup,'formal_epochs':168+12+qup//809}
    board=load(BASE/'01_候选判别表.json');state=load(BASE/'15_当前状态与结果校验.json');now=datetime.now(timezone.utc).isoformat()
    milestone={**record,'report':str(report),'verification':str(verify),'observation':str(obsfile)}
    board.update(status='ACTIVE_A52_Q12_TRAINING_MAINLINE_NOT_CONFIRMED',goal_status='active',mainline_validated=False,
       latest_report=str(report),latest_update_utc=now,A52_M_terminal_verification=milestone)
    board['A52_execution'].update(state='M12_COMPLETE_VERIFIED_Q12_TRAINING',latest_milestone=milestone)
    for row in board['candidates']:
        if row.get('id')=='A52_DISTRIBUTIONAL_PAIR_GEOMETRY':row.update(status='Q12_TRAINING_M12_COMPLETE_NOT_VALIDATED',latest_milestone=milestone)
    state.update(recorded_utc=now,goal_status='active',mainline_validated=False,current_phase='A52_Q12_TRAINING',
       latest_authoritative_phase='A52_M12_VERIFIED_Q_STARTUP',latest_report=str(report),A52_M_terminal_verification=milestone,
       model_forwards=counts['whole_model_forwards'],cap=185612,formal_detector_training_updates=counts['formal_optimizer_updates'],
       capacity_only_optimizer_updates=49,training_updates=counts['all_optimizer_updates'],
       detector_optimizer_updates_including_capacity=counts['all_optimizer_updates'],formal_training_epochs=counts['formal_epochs'],
       counts_terminal=False,training_counts_terminal=False,full_forward_cap_closed=False,current_native_job_terminal=False,
       counters_observed_lower_bounds_while_job_live=True)
    save(BASE/'01_候选判别表.json',board);save(BASE/'15_当前状态与结果校验.json',state)
    signed=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',verify,report,
       ROOT/'M_completed_local_verification.json',ROOT/'Q_startup/initialization.json',ROOT/'Q_startup/epoch01_input_flow.json',
       ROOT/'new_module_untrained_initial_state_receipt.json',ROOT/'13_比较范围与归因限制_v1.txt',ROOT/'15_最近工作期刊扩展核查范围_v1.txt']
    save(receipt,{'status':'PASS','phase':'A52_M12_Q_STARTUP','goal_status':'active','mainline_confirmed':False,
       'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':resolved,
       'current_files_sha256':{str(p):sha(p) for p in signed},'observed_counts_lower_bounds':counts,'whole_forward_cap':185612,
       'M_terminal_exit0':True,'Q_actual_process':o['arms']['Q']['process'],'no_extra_training_queue':True,'publication_added_model_operations':0})
    current.update(status='M12_COMPLETED_Q12_TRAINING_B12_REUSED',goal_status='active',mainline_confirmed=False,
       receipt=str(receipt),receipt_sha256=sha(receipt),latest_report=str(report),latest_observation=str(obsfile),
       M_completed_epochs_at_observation=12,M_observed_optimizer_update_lower_bound=9708,M_terminal_verified=True,
       M_recorded_flow_verified_through_epoch=12,Q_pid=o['arms']['Q']['process']['pid'],
       Q_start_ticks=o['supervisor']['child']['start_ticks'],Q_not_started_at_observation=False,
       Q_completed_epochs_at_observation=qup//809,Q_observed_optimizer_update_lower_bound=qup,
       Q_first_epoch_flow_verified=True,next='Observe Q4/8/12; no new method or queue before fixed12 terminal judgment.')
    save(ROOT/'CURRENT_A52.json',current)
    print(json.dumps({'status':'M12_VERIFIED_Q_RUNNING_PUBLISHED','M_minus_B_pp':delta,'Q_requirements':requirements,
       'observed_counts_lower_bounds':counts,'receipt_sha256':sha(receipt),'goal_status':'active','mainline_confirmed':False}))
if __name__=='__main__':main()

