from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
A49=BASE/'A49_下一主线有界筛选_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    verified=load(ROOT/'M_completed_local_verification.json')
    assert verified['status']=='PASS' and verified['epochs']==12 and verified['optimizer_updates']==9708
    assert verified['optimizer_step_histogram']=={'9708':490}
    mfolder=ROOT/'completed_M_v1/arm_M';bfolder=A49/'completed_B_v1/arm_B'
    m=load(mfolder/'result.json');b=load(bfolder/'result.json')
    assert m['full_model_forwards']==10159 and load(mfolder/'exit.json')['exit_code']==0
    assert m['endpoint_metrics']==verified['endpoint_metrics']
    mi=load(mfolder/'initialization.json');qi=load(ROOT/'Q_startup/initialization.json')
    assert qi['new_module_initialization_sha256']==mi['new_module_initialization_sha256']
    assert qi['common_init_sha256']==mi['common_init_sha256']
    assert qi['all_native_initial_tensors_identical'] and qi['scratch_no_capacity_weights'] and qi['seed']==20261101
    fields=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']
    checks={}
    for e in range(1,13):
        f=load(mfolder/('epoch%02d_input_flow.json'%e));ref=load(bfolder/('epoch%02d_input_flow.json'%e))
        check={k:f[k]==ref[k] for k in fields};assert all(check.values())
        assert f['optimizer_updates']==e*809
        checks[str(e)]=check
    q1=load(ROOT/'Q_startup/epoch01_input_flow.json');b1=load(bfolder/'epoch01_input_flow.json')
    q_checks={k:q1[k]==b1[k] for k in fields};assert all(q_checks.values()) and q1['optimizer_updates']==809
    obs=json.loads(load(ROOT/'observer_20_M_verified_Q_active.json')['stdout'])
    assert obs['supervisor_process']['live'] and obs['supervisor_process']['identity_matches']
    assert obs['arms']['Q']['process']['live'] and obs['arms']['Q']['process']['identity_matches']
    assert obs['arms']['M']['exit']['exit_code']==0 and not obs['arms']['M']['process']['live']
    assert all(obs['frozen_files_unchanged'].values())
    qm=obs['arms']['Q']['progress'];assert qm['status']=='TRAINING' and qm['optimizer_updates']>=809
    mb=m['endpoint_metrics'];bb=b['endpoint_metrics']
    delta={k:mb[k]-bb[k] for k in ['AP','APs','APs75','ARs']}
    required_APs=max(bb['APs']+.5,mb['APs']+.3);required_AP=bb['AP']-.2
    terminal={'status':'M12_ARCHIVE_AND_STATE_VERIFIED_Q_RUNNING','goal_status':'active','mainline_confirmed':False,
       'metrics':{'B12':bb,'M12':mb},'M_minus_B_pp':delta,
       'M_formal_optimizer_updates':9708,'M_full_model_forwards':10159,'M_new_parameters':73746,
       'M_optimizer_steps':verified['optimizer_step_histogram'],'M_payload_files_verified':verified['payload_files_verified'],
       'M_bundle_sha256':verified['bundle_sha256'],'M12_flow_checks_against_B':checks,
       'Q_initialization_verified':True,'Q1_recorded_flow_checks_against_B':q_checks,
       'Q_exact_primary_requirements':{'APs_at_least':required_APs,'AP_at_least':required_AP,
              'Q_minus_B_APs_pp_at_least':.5,'Q_minus_M_APs_pp_at_least':.3},
       'scientific_verdict':'Q_FIXED12_PENDING_NO_NOVELTY_OR_EFFICACY_CLAIM',
       'verification_added_model_forwards':0,'verification_added_optimizer_updates':0}
    checkpath=ROOT/'M12_终点与Q初始化核验.json';assert not checkpath.exists();save(checkpath,terminal)
    report=ROOT/'M12_结果与Q启动.txt';assert not report.exists()
    lines=['A51普通交换对照M已完整结束12轮，候选Q实际运行。总目标进行，主线未确认。',
      '固定12轮EMA：B AP%.6f/APs%.6f；M AP%.6f/APs%.6f。'%(bb['AP'],bb['APs'],mb['AP'],mb['APs']),
      'M−B：AP%+.6fpp，APs%+.6fpp，small AP75%+.6fpp。'%(delta['AP'],delta['APs'],delta['APs75']),
      'M是普通独立门控交换对照；这些结果不代表共同交换候选Q有效，也不把secondary smallAP75改成主结局。',
      'M exit0，正式9708更新、10159整模型前向，完整43文件结果包已下载校验。raw/EMA张量键/形状/类型/有限性和BN更新计数核验通过，490个优化器状态均到9708。',
      'M训练及预设验证计时%.2f分钟，PyTorch allocated峰值%.3fGiB；batch1 eager BF16模型前向中位数%.3fms，包含新模块，未包含decode/postprocess，不是机载/TensorRT时延。'%(m['elapsed_seconds']/60,m['peak_allocated_bytes']/1024**3,m['latency']['median_ms']),
      'M全部12轮记录的顺序/首8批实际输入与RNG/全轮目标流同B；Q基础及新增模块初始化同M，Q首轮同类输入流检查也已通过。',
      'Q继续其唯一12轮运行，不追加训练。按精确数据，Q必须同时满足APs≥%.9f、整体AP≥%.9f，才能满足冻结三门槛；最终由固定12轮脚本判定。'%(required_APs,required_AP),
      '当前COCO式指标与官方VisDrone协议分开；官方复核仅为达标后的条件路线，未执行。最近方法全文阅读范围与理论归属限度保持13/14/16/20/22记录。',
      '本次CPU归档/核验新增模型前向0、优化器更新0。Q尚无固定12结论，总目标未完成。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    current=load(ROOT/'CURRENT_A51.json');prior=Path(current['receipt']);assert sha(prior)==current['receipt_sha256']
    old=load(prior);archive=ROOT/'before_M_terminal_canonical';archive.mkdir(exist_ok=False);resolved={}
    for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        p=BASE/name;expected=old['current_files_sha256'][str(p)];assert sha(p)==expected
        q=archive/name;shutil.copy2(p,q);assert sha(q)==expected
        resolved[str(p)]={'preserved_path':str(q),'sha256':expected}
    q_updates=qm['optimizer_updates'];q_fw=qm['full_model_forwards']
    model_count=144957+9+10159+q_fw;opt_count=117059+6+9708+q_updates
    board_path=BASE/'01_候选判别表.json';state_path=BASE/'15_当前状态与结果校验.json'
    board=load(board_path);state=load(state_path);now=datetime.now(timezone.utc).isoformat()
    milestone={**terminal,'report':str(report),'verification':str(checkpath),
               'observation':str(ROOT/'observer_20_M_verified_Q_active.json')}
    board.update(status='ACTIVE_A51_Q12_TRAINING_MAINLINE_NOT_CONFIRMED',goal_status='active',mainline_validated=False,
       latest_report=str(report),latest_update_utc=now,A51_M_terminal_verification=milestone)
    board['A51_execution'].update(state='M12_COMPLETE_VERIFIED_Q12_TRAINING',latest_milestone=milestone)
    for row in board['candidates']:
        if row.get('id')=='A51_CROSS_RESOLUTION_PAIRED_EXCHANGE':
            row['status']='Q12_TRAINING_M12_COMPLETE_NOT_VALIDATED';row['latest_milestone']=milestone
    state.update(recorded_utc=now,goal_status='active',mainline_validated=False,
       current_phase='A51_Q12_TRAINING',latest_authoritative_phase='A51_M12_VERIFIED_Q_STARTUP',
       latest_report=str(report),A51_M_terminal_verification=milestone,
       model_forwards=model_count,cap=165284,formal_detector_training_updates=117022+9708+q_updates,
       capacity_only_optimizer_updates=43,training_updates=opt_count,detector_optimizer_updates_including_capacity=opt_count,
       formal_training_epochs=144+12+q_updates//809,counts_terminal=False,training_counts_terminal=False,
       full_forward_cap_closed=False,counters_observed_lower_bounds_while_job_live=True,current_native_job_terminal=False)
    save(board_path,board);save(state_path,state)
    receipt=ROOT/'M12_终点与Q启动保留回执.json';assert not receipt.exists()
    signed=[board_path,state_path,checkpath,report,ROOT/'M_completed_local_verification.json',ROOT/'local_module_init_consistency_v1.json',
            ROOT/'Q_startup/initialization.json',ROOT/'Q_startup/epoch01_input_flow.json']
    save(receipt,{'status':'PASS','phase':'A51_M12_Q_STARTUP','goal_status':'active','mainline_confirmed':False,
       'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':resolved,
       'current_files_sha256':{str(p):sha(p) for p in signed},
       'observed_lower_bounds':{'whole_model_forwards':model_count,'optimizer_including_capacity':opt_count},
       'forward_ceiling':165284,'M_terminal_exit0':True,'Q_actual_process':obs['arms']['Q']['process'],
       'no_extra_training_queue':True,'new_detector_operations_by_publication':0})
    current.update(status='M12_COMPLETED_Q12_TRAINING_B12_REUSED',goal_status='active',mainline_confirmed=False,
       receipt=str(receipt),receipt_sha256=sha(receipt),latest_report=str(report),latest_observation=str(ROOT/'observer_20_M_verified_Q_active.json'),
       M_completed_epochs_at_observation=12,M_observed_optimizer_update_lower_bound=9708,M_terminal_verified=True,
       Q_pid=obs['arms']['Q']['process']['pid'],Q_start_ticks=obs['supervisor']['child']['start_ticks'],
       Q_completed_epochs_at_observation=q_updates//809,Q_observed_optimizer_update_lower_bound=q_updates,
       Q_first_epoch_flow_verified=True,next='Observe Q fixed4/8/12; no new method/queue before actual terminal judgment.')
    save(ROOT/'CURRENT_A51.json',current)
    print(json.dumps({'status':'M12_VERIFIED_Q_RUNNING_PUBLISHED','M_minus_B_pp':delta,
       'Q_requirements':terminal['Q_exact_primary_requirements'],'observed_forward_lower_bound':model_count,
       'observed_optimizer_lower_bound':opt_count,'receipt_sha256':sha(receipt),'goal_status':'active','mainline_confirmed':False}))


if __name__=='__main__':main()
