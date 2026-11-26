"""Publish a frozen intermediate observation without changing scientific endpoints."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
A49=BASE/'A49_下一主线有界筛选_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
def flow_path(arm,e):
    complete=(A49 if arm=='B' else ROOT)/('completed_'+arm+'_v1')/('arm_'+arm)/('epoch%02d_input_flow.json'%e)
    if complete.exists():return complete
    if arm=='M':folder='M_startup' if e==1 else ('M4_inputs' if e<=4 else 'M8_inputs')
    else:folder='Q_startup' if e==1 else ('Q4_inputs' if e<=4 else 'Q8_inputs')
    return ROOT/folder/('epoch%02d_input_flow.json'%e)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['M','Q'],required=True)
    parser.add_argument('--epoch',type=int,choices=[4,8],required=True);parser.add_argument('--observation',required=True)
    args=parser.parse_args();arm=args.arm;epoch=args.epoch;tag=arm+str(epoch)
    current=load(ROOT/'CURRENT_A51.json');prior=Path(current['receipt'])
    assert sha(prior)==current['receipt_sha256'];old=load(prior)
    observation=(ROOT/args.observation).resolve();assert observation.is_relative_to(ROOT.resolve())
    raw=load(observation);assert raw['exit']==0;obs=json.loads(raw['stdout'])
    assert obs['supervisor_process']['live'] and obs['supervisor_process']['identity_matches']
    assert all(obs['frozen_files_unchanged'].values())
    metric=load(ROOT/(tag+'_inputs')/('epoch%02d_EMA_metrics.json'%epoch))
    b=load(A49/'completed_B_v1/arm_B'/('epoch%02d_EMA_metrics.json'%epoch))
    assert metric==obs['arms'][arm]['metrics'][str(epoch)]
    for key in ['max_detections','evaluated_GT','ignored_valid_GT','area_convention','scope']:assert metric[key]==b[key],key
    fields=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']
    checks={}
    for e in range(1,epoch+1):
        p=flow_path(arm,e);f=load(p);ref=load(flow_path('B',e))
        match={k:f[k]==ref[k] for k in fields};assert all(match.values())
        assert f['epoch']==e and f['optimizer_updates']==809*e
        checks[str(e)]={'checks':match,'source':str(p),'sha256':sha(p)}
    native_init=load(ROOT/'M_startup/initialization.json')
    this_init=load(ROOT/(arm+'_startup')/'initialization.json')
    assert this_init['new_module_initialization_sha256']==native_init['new_module_initialization_sha256']
    assert this_init['all_native_initial_tensors_identical'] and this_init['scratch_no_capacity_weights']
    m=load(ROOT/('M'+str(epoch)+'_inputs')/('epoch%02d_EMA_metrics.json'%epoch))
    evaluation={'B':b,arm:metric,'arm_minus_B_AP_pp':metric['AP']-b['AP'],
       'arm_minus_B_APs_pp':metric['APs']-b['APs'],'Q_fixed12_verdict':'NOT_AVAILABLE_NOT_INFERRED_FROM_INTERIM'}
    if arm=='Q':
        evaluation['M']=m;evaluation['Q_minus_M_APs_pp']=metric['APs']-m['APs']
    counts={'new_model_forwards_lower_bound':9,'new_optimizer_updates_lower_bound':6,
            'new_formal_updates_lower_bound':0,'completed_new_formal_epochs':0}
    for a in ['M','Q']:
        r=obs['arms'][a]['result'] or obs['arms'][a]['progress'] or {}
        u=r.get('optimizer_updates',0)
        counts['new_model_forwards_lower_bound']+=r.get('full_model_forwards',0)
        counts['new_optimizer_updates_lower_bound']+=u;counts['new_formal_updates_lower_bound']+=u
        counts['completed_new_formal_epochs']+=u//809
    forward_count=144957+counts['new_model_forwards_lower_bound'];optimizer_count=117059+counts['new_optimizer_updates_lower_bound']
    assert forward_count<=165284 and optimizer_count<=136481
    archive=ROOT/('before_'+tag+'_canonical');archive.mkdir(exist_ok=False);resolved={}
    for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        p=BASE/name;h=old['current_files_sha256'][str(p)];assert sha(p)==h
        q=archive/name;shutil.copy2(p,q);assert sha(q)==h
        resolved[str(p)]={'preserved_path':str(q),'sha256':h}
    verification=ROOT/('interim_'+tag+'_verification.json');assert not verification.exists()
    save(verification,{'status':'PASS','arm':arm,'epoch':epoch,'flow_checks_against_B':checks,
      'initialization_check':'same base initialization and same M/Q module initialization hash',
      'frozen_files_unchanged':obs['frozen_files_unchanged'],'evaluation':evaluation,
      'verification_added_model_forwards':0,'verification_added_optimizer_updates':0})
    report=ROOT/('interim_'+tag+'_report.txt');assert not report.exists()
    lines=['A51 '+tag+'中期记录。总目标进行，主线未确认。',
      '同轮EMA：B AP%.6f/APs%.6f；%s AP%.6f/APs%.6f。'%(b['AP'],b['APs'],arm,metric['AP'],metric['APs']),
      '%s−B：AP%+.6fpp，APs%+.6fpp。'%(arm,evaluation['arm_minus_B_AP_pp'],evaluation['arm_minus_B_APs_pp']),
      '这是固定中期观察，不挑选最佳轮次，不代替固定12主结局或统计显著性。',
      '首%d轮的顺序、首8批实际输入/RNG、全轮目标流与B匹配；相同评估GT/面积/ignore/maxDets，冻结源文件未变。'%epoch,
      'Q固定12三项门槛仍为：APs相对B≥0.5pp、相对M≥0.3pp，整体AP相对B≥−0.2pp。',
      '指标为当前冻结COCO式分数；官方协议与未过滤预测复核仅作条件路线准备，详见22。',
      'M/Q共有块残差保持，Q独有增量是共同交换量约束；不声称新扩散原理，详见13/16。',
      '当前真实进程与状态以'+observation.name+'为准；本次发布不新增模型、优化器更新或训练队列。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    milestone={'phase':'A51_'+tag,'goal_status':'active','mainline_confirmed':False,'report':str(report),
      'verification':str(verification),'observation':str(observation),'evaluation':evaluation,'counts':counts}
    board_path=BASE/'01_候选判别表.json';state_path=BASE/'15_当前状态与结果校验.json'
    board=load(board_path);state=load(state_path);now=datetime.now(timezone.utc).isoformat()
    board.update(latest_report=str(report),latest_update_utc=now,goal_status='active',mainline_validated=False)
    board['A51_'+tag+'_milestone']=milestone;board['A51_execution']['latest_milestone']=milestone
    state.update(recorded_utc=now,latest_report=str(report),goal_status='active',mainline_validated=False,
      model_forwards=forward_count,training_updates=optimizer_count,detector_optimizer_updates_including_capacity=optimizer_count,
      formal_detector_training_updates=117022+counts['new_formal_updates_lower_bound'],capacity_only_optimizer_updates=43,
      formal_training_epochs=144+counts['completed_new_formal_epochs'],cap=165284,
      full_forward_cap_closed=False,counts_terminal=False,training_counts_terminal=False,counters_observed_lower_bounds_while_job_live=True,
      current_phase='A51_'+obs['supervisor']['stage']+'_RUNNING',latest_authoritative_phase='A51_'+tag+'_VERIFIED')
    state['A51_'+tag+'_milestone']=milestone
    save(board_path,board);save(state_path,state)
    receipt=ROOT/('interim_'+tag+'_preservation_receipt.json');assert not receipt.exists()
    signed=[board_path,state_path,report,verification,ROOT/'20_核心参考文献_版本与阅读范围.bib',ROOT/'22_指标口径与条件复核路线_v1.txt']
    save(receipt,{'status':'PASS','phase':'A51_'+tag,'goal_status':'active','mainline_confirmed':False,
      'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':resolved,
      'current_files_sha256':{str(p):sha(p) for p in signed},
      'observed_lower_bounds':{'whole_model_forwards':forward_count,'optimizer_including_capacity':optimizer_count},
      'forward_ceiling':165284,'live_processes_verified':True,'new_extra_training_queue':False})
    current.update(receipt=str(receipt),receipt_sha256=sha(receipt),latest_report=str(report),latest_observation=str(observation),
       latest_verified_interim=tag,**{arm+'_completed_epochs_at_observation':obs['arms'][arm]['progress'].get('completed_epochs',epoch),
       arm+'_observed_optimizer_update_lower_bound':obs['arms'][arm]['progress']['optimizer_updates']})
    save(ROOT/'CURRENT_A51.json',current)
    print(json.dumps({'status':tag+'_VERIFIED_AND_PUBLISHED','evaluation':evaluation,'counts':counts,
      'receipt_sha256':sha(receipt),'goal_status':'active','mainline_confirmed':False},ensure_ascii=False))


if __name__=='__main__':main()
