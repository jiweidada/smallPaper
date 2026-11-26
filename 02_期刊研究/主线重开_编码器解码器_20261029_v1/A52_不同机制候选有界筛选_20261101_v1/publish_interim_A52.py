"""Publish a fixed4/8 interim snapshot only. No detector or optimizer calls."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
A49=BASE/'A49_下一主线有界筛选_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def locate(arm,name):
    if arm=='B':return A49/'completed_B_v1/arm_B'/name
    folders=[ROOT/('completed_'+arm+'_v1')/('arm_'+arm)]
    folders += [ROOT/(arm+'_startup'),ROOT/(arm+'4_inputs'),ROOT/(arm+'8_inputs')]
    found=[p/name for p in folders if (p/name).exists()]
    assert found,(arm,name)
    assert len({sha(p) for p in found})==1,(arm,name,'local duplicates differ')
    return found[0]
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['M','Q'],required=True)
    p.add_argument('--epoch',type=int,choices=[4,8],required=True);p.add_argument('--observation',required=True);a=p.parse_args()
    prefix='interim_'+a.arm+str(a.epoch)
    verify=ROOT/(prefix+'_verification.json');report=ROOT/(prefix+'_report.txt');receipt=ROOT/(prefix+'_preservation_receipt.json')
    assert all(not x.exists() for x in [verify,report,receipt])
    current=load(ROOT/'CURRENT_A52.json');assert not current.get('do_not_run_old_live_updaters_after_terminal_publication',False)
    obsfile=ROOT/a.observation;wire=load(obsfile);assert wire['status']=='EXECUTED' and wire['exit']==0
    o=json.loads(wire['stdout']);assert all(o['local_frozen_files_unchanged'].values())
    assert o['supervisor_process']['live'] and o['supervisor_process']['identity_matches']
    assert o['arms'][a.arm]['process']['live'] and o['arms'][a.arm]['process']['identity_matches']
    assert o['arms'][a.arm]['metrics'][str(a.epoch)] is not None
    compare=['B','M'] if a.arm=='M' else ['B','M','Q']
    metrics={arm:load(locate(arm,'epoch%02d_EMA_metrics.json'%a.epoch)) for arm in compare}
    assert metrics[a.arm]==o['arms'][a.arm]['metrics'][str(a.epoch)]
    flow={}
    keys=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']
    for epoch in range(1,a.epoch+1):
        logs={arm:load(locate(arm,'epoch%02d_input_flow.json'%epoch)) for arm in compare}
        assert all(x['optimizer_updates']==809*epoch for x in logs.values())
        flow[str(epoch)]={k:len({json.dumps(x[k],sort_keys=True) for x in logs.values()})==1 for k in keys}
        assert all(flow[str(epoch)].values())
    init={arm:load(locate(arm,'initialization.json')) for arm in compare if arm!='B'}
    for arm,item in init.items():
        assert item['all_native_initial_tensors_identical'] and item['scratch_no_capacity_weights']
        assert item['common_init_sha256']=='2c6e0f9c6765fc7ba069c2292e0884bb86924d2232a952499fcd251cc57b47e6'
        assert item['new_module_initialization_sha256']=='69da2b6c3fd9b4c7fa879d807880542cb3a31243004895a67502d8c418829bd0'
        assert item['protocol_sha256']=='c13b61bdbc09d81a051108f7a28516f14ae3a3521a81d8247574129ca413d027'
    for arm in compare[1:]:
        for k in ['max_detections','evaluated_GT','ignored_valid_GT','area_convention','scope']:
            assert metrics[arm][k]==metrics['B'][k],(arm,k)
    delta={arm:{k:metrics[arm][k]-metrics['B'][k] for k in ['AP','APs','APs75']} for arm in compare[1:]}
    qm=None if a.arm=='M' else metrics['Q']['APs']-metrics['M']['APs']
    counters={'whole_model_forwards':165294,'formal_optimizer_updates':136438,'capacity_only_updates':49,'all_optimizer_updates':136487,'formal_epochs':168}
    for arm in ['M','Q']:
        pp=o['arms'][arm]['progress']
        if pp:
            counters['whole_model_forwards']+=pp['full_model_forwards'];counters['formal_optimizer_updates']+=pp['optimizer_updates']
            counters['all_optimizer_updates']+=pp['optimizer_updates'];counters['formal_epochs']+=pp['optimizer_updates']//809
    record={'status':'INTERIM_VERIFIED_NOT_FIXED12_DECISION','arm':a.arm,'epoch':a.epoch,
      'goal_status':'active','mainline_confirmed':False,'metrics':metrics,'minus_B_pp':delta,'Q_minus_M_APs_pp':qm,
      'recorded_flow_checks':flow,'formal_initializations':init,'observation':str(obsfile),
      'source_hashes_unchanged':True,'observed_counts_lower_bounds':counters,'new_model_or_optimizer_operations':0}
    save(verify,record)
    lines=['A52 '+a.arm+str(a.epoch)+'中期观察；总目标active，主线未确认。','固定同轮EMA、相同COCO式原图GT面积/ignore/maxDets300口径；非官方挑战分数。']
    for arm,row in metrics.items():lines.append('%s: AP %.9f，APs %.9f，small AP75 %.9f。'%(arm,row['AP'],row['APs'],row['APs75']))
    for arm,row in delta.items():lines.append('%s−B: AP %+.9fpp，APs %+.9fpp。'%(arm,row['AP'],row['APs']))
    if qm is not None:lines.append('Q−M APs %+.9fpp。'%qm)
    lines+=['前%d轮记录顺序、首8批实际输入/RNG、全轮目标流匹配，原生初始化一致、容量权重未用于训练。'%a.epoch,
      '最终仍以固定12EMA三项门槛：Q−B APs≥0.5pp，Q−M APs≥0.3pp，Q−B AP≥−0.2pp；不选中期最佳轮次。',
      '完整离散边际与高斯矩压缩的比较不单独排除格点、对角质量或周期结构解释；详见13。无校准后验、新编码原理、统计显著或论文完成声明。',
      '真实进程以'+a.observation+'为准；当前计数为运行中下界。无新增候选、训练队列、模型或优化器操作。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    prior=Path(current['receipt']);assert sha(prior)==current['receipt_sha256'];old=load(prior)
    archive=ROOT/('before_'+a.arm+str(a.epoch)+'_canonical');archive.mkdir(exist_ok=False);resolved={}
    for n in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        original=BASE/n;h=old['current_files_sha256'][str(original)];assert sha(original)==h
        dest=archive/n;shutil.copy2(original,dest);assert sha(dest)==h
        resolved[str(original)]={'preserved_path':str(dest),'sha256':h}
    board=load(BASE/'01_候选判别表.json');state=load(BASE/'15_当前状态与结果校验.json');now=datetime.now(timezone.utc).isoformat()
    board.update(latest_report=str(report),latest_update_utc=now)
    board['A52_execution'].update(latest_milestone=record)
    state.update(latest_report=str(report),recorded_utc=now,model_forwards=counters['whole_model_forwards'],
      formal_detector_training_updates=counters['formal_optimizer_updates'],capacity_only_optimizer_updates=49,
      training_updates=counters['all_optimizer_updates'],detector_optimizer_updates_including_capacity=counters['all_optimizer_updates'],
      formal_training_epochs=counters['formal_epochs'],counts_terminal=False,training_counts_terminal=False,
      counters_observed_lower_bounds_while_job_live=True)
    state['A52_execution'].update(latest_milestone=record)
    save(BASE/'01_候选判别表.json',board);save(BASE/'15_当前状态与结果校验.json',state)
    signed=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',verify,report,ROOT/'13_比较范围与归因限制_v1.txt',
      ROOT/'15_最近工作期刊扩展核查范围_v1.txt',ROOT/'RelationDETRplus_metadata_verification_v1.json',ROOT/'RelationDETRplus_fulltext_access_v1.json']
    save(receipt,{'status':'PASS','phase':'A52_'+a.arm+str(a.epoch),'goal_status':'active','mainline_confirmed':False,
      'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':resolved,
      'current_files_sha256':{str(p):sha(p) for p in signed},'observed_counts_lower_bounds':counters,'publication_added_model_operations':0})
    current.update(receipt=str(receipt),receipt_sha256=sha(receipt),latest_report=str(report),latest_observation=str(obsfile),
      latest_verified_interim=a.arm+str(a.epoch))
    for arm in ['M','Q']:
        progress=o['arms'][arm]['progress']
        if progress:
            current[arm+'_completed_epochs_at_observation']=progress['optimizer_updates']//809
            current[arm+'_observed_optimizer_update_lower_bound']=progress['optimizer_updates']
    current['first_'+str(a.epoch)+'_epoch_recorded_flow_verified_for_'+a.arm]=True
    save(ROOT/'CURRENT_A52.json',current)
    print(json.dumps({'status':record['status'],'arm':a.arm,'epoch':a.epoch,'metrics':metrics,'minus_B_pp':delta,'Q_minus_M_APs_pp':qm,'receipt_sha256':sha(receipt)}))
if __name__=='__main__':main()

