"""Publish only after B12 archive verification and an authentic M startup observation."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RECEIPT = HERE / '12_B终点与M启动校验回执.json'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('observer');args=ap.parse_args()
    assert not RECEIPT.exists(), 'Immutable milestone already published.'
    raw=read(HERE/args.observer);assert raw['status']=='EXECUTED' and raw['exit']==0
    o=json.loads(raw['stdout']);b=o['arms']['B'];m=o['arms']['M'];mi=m['identity'] or {}
    assert b['result']['status']=='COMPLETED' and b['result']['epochs']==12
    assert b['exit']['exit_code']==0 and not b['identity']['present']
    assert mi.get('present') and mi.get('same_start_ticks') and mi.get('owned_command') and mi.get('state') not in ['Z','X']
    assert o['supervisor_identity']['present'] and o['supervisor_identity']['same_start_ticks'] and o['supervisor_identity']['owned_command']
    transfer=read(HERE/'B_completed_bundle_download_v2_bounded.json');assert transfer['status']=='TRANSFERRED'
    verified=read(HERE/'B_completed_local_verification.json')
    assert verified['status']=='PASS' and verified['epochs']==12 and verified['optimizer_updates']==9708
    assert verified['endpoint_metrics']==b['result']['endpoint_metrics']
    assert verified['extra_mask_core_reads']==0 and verified['native_inference_no_mask_reads']
    init=read(HERE/'M_startup'/'initialization.json')
    assert init['common_init_sha256']==b['result']['initialization_sha256']
    assert init['all_initial_tensors_identical'] and init['model_parameters']==9803112 and init['seed']==20261101
    protocol=HERE/'protocol_FROZEN.json'
    assert sha(protocol)==init['protocol_sha256']==b['result']['protocol_sha256']=='d3a971a21d1c339052c8a7ef41fcab345882edc8893c4762c66dbbc5bc10f8f3'
    assert all(r['all_pass'] for r in o['common_epoch_checks']['M_vs_B'])
    flow_b=read(HERE/'completed_B_v1'/'arm_B'/'epoch01_input_flow.json')
    flow_m=read(HERE/'M_startup'/'epoch01_input_flow.json')
    flow_keys=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256',
               'all_batches_targets_including_ignore_tiles_sha256']
    first_epoch_checks={k:flow_b[k]==flow_m[k] for k in flow_keys}
    assert all(first_epoch_checks.values())
    assert flow_b['optimizer_updates']==flow_m['optimizer_updates']==809
    previous=HERE/'08_A49启动与历史保留校验回执.json';prev=read(previous)
    for p,expected in prev['current_files_sha256'].items():
        assert sha(Path(p))==expected,p
    before=HERE/'before_A49_B_terminal';before.mkdir(exist_ok=False)
    resolved={}
    for s,expected in prev['current_files_sha256'].items():
        p=Path(s)
        if p.parent==BASE and p.name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
            dest=before/p.name;shutil.copy2(p,dest);assert sha(dest)==expected
        else:dest=p
        resolved[s]={'preserved_at':str(dest),'sha256':expected}

    c=o['observed_counters_lower_bounds']
    totals={'model_forwards':114452+c['model_forwards'],
            'formal_optimizer_updates':87898+c['formal_optimizer_updates'],
            'capacity_updates':28+c['capacity_optimizer_updates'],
            'optimizer_updates_including_capacity':87926+c['formal_optimizer_updates']+c['capacity_optimizer_updates'],
            'formal_epochs':108+c['formal_epoch_sum']}
    metrics=verified['endpoint_metrics']
    now=datetime.now(timezone.utc).isoformat()
    milestone={'phase':'A49_B12_TERMINAL_M_ACTIVE','goal_status':'active','mainline_confirmed':False,
        'B_completed_exit0':True,'B_metrics':metrics,'B_local_integrity':str(HERE/'B_completed_local_verification.json'),
        'B_archive_manifest':str(HERE/'completed_B_v1'/'bundle_manifest.json'),
        'M_initialization':str(HERE/'M_startup'/'initialization.json'),'M_live_identity':mi,
        'M_progress':m['progress'],'M_completed_epoch_flow_checks':o['common_epoch_checks']['M_vs_B'],
        'M_first_epoch_independent_local_flow_checks':first_epoch_checks,
        'M_first_epoch_omitted_query_loss_row_occurrences':flow_m['epoch_omitted_query_loss_rows'],
        'Q_status':(o['arms']['Q']['progress'] or {}).get('status','NOT_STARTED'),
        'authoritative_observation':str(HERE/args.observer),'cumulative_counts_lower_bounds':totals,
        'scope':'B endpoint and M startup only; no candidate efficacy or novelty verdict. CPU archive/state integrity is not independent AP reproduction.'}
    lines=['A49 原版B12终点与M启动记录','',
        '主线仍未确认，目标active；当前只完成原版B，不能比较M/Q增量。',
        '数据：VisDrone2019-DET train6471/val548；算法：ECDet-S原生BN。',
        '固定第12轮EMA：AP %.8f，APs %.8f，APs75 %.8f。'%(metrics['AP'],metrics['APs'],metrics['APs75']),
        '统一COCO式评估，maxDets300、原图面积和class0并集过滤；非官方VisDrone成绩。',
        'B共12轮、9708更新、10159完整模型前向，退出0；完整包已下载并逐文件/权重状态校验。',
        '本地CPU校验未重新执行模型预测，不声称独立复现AP。',
        'M同初始化/seed/参数/协议核对通过；已完成轮次的数据流检查见JSON，尚未完成轮次不提前宣称通过。',
        'M当前进程PID%d/start_ticks%d；Q状态%s。'%(mi['pid'],mi['expected_start_ticks'],milestone['Q_status']),
        '仍由原监督器依次完成M12/Q12，无额外训练队列。固定12EMA门槛不变，不根据早期曲线改阈值或加轮。',
        '最新可更新观察见CURRENT_A49.json；本记录为不可变里程碑快照。']
    note=HERE/'11_B12终点与M启动记录.txt'
    with note.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        p=BASE/name;x=read(p)
        x.update({'goal_status':'active','mainline_validated':False,'latest_report':str(note),
                  'A49_B_terminal_verification':milestone})
        if name.startswith('01'):
            x['latest_update_utc']=now;x['status']='A49_B12_TERMINAL_M_ACTIVE'
            x['pilot_validation_status']='B_COMPLETED_M_ACTIVE_Q_PENDING'
        else:
            x.update({'recorded_utc':now,'current_phase':'A49','latest_authoritative_phase':'A49_B12_TERMINAL_M_ACTIVE',
                      'model_forwards':totals['model_forwards'],'training_updates':totals['optimizer_updates_including_capacity'],
                      'detector_optimizer_updates_including_capacity':totals['optimizer_updates_including_capacity'],
                      'formal_detector_training_updates':totals['formal_optimizer_updates'],
                      'capacity_only_optimizer_updates':totals['capacity_updates'],
                      'formal_training_epochs':totals['formal_epochs'],'counts_terminal':False,
                      'training_counts_terminal':False,'counters_observed_lower_bounds_while_job_live':True,
                      'current_user_task_status':'B_COMPLETED_M_ACTIVE_Q_PENDING',
                      'current_goal_turn':{'classification':'progress_and_verified_wait','goal_status':'active',
                                           'mainline_confirmed':False,'observer':str(HERE/args.observer)}})
        x['A49_execution']['startup_snapshot_preserved_in']=str(before/name)
        x['A49_execution']['architecture_reserve_reading_supplement']=str(HERE/'09_架构备选补读与阅读范围_v1.json')
        x['A49_execution']['candidate_selection_status']='A49_B controlled pilot active; A49_A Delta selected methods now read, still held/untrained with no distinct UAV operation selected.'
        x['A49_execution'].update({'status':'B_COMPLETED_M_ACTIVE_Q_PENDING','observation':str(HERE/args.observer),
                                  'owned_live_arms':[{'arm':'M','pid':mi['pid'],'start_ticks':mi['expected_start_ticks']}],
                                  'current_evaluations':{a:r['evaluations'] for a,r in o['arms'].items()},
                                  'observed_progress':{a:r['progress'] for a,r in o['arms'].items()},
                                  'new_phase_counters_lower_bounds':c})
        assert sha(p)==prev['current_files_sha256'][str(p)],'Canonical record changed while preparing publication.'
        save(p,x)

    for row in resolved.values():assert sha(Path(row['preserved_at']))==row['sha256']
    names=['11_B12终点与M启动记录.txt','B_completed_local_verification.json','B_completed_bundle_download_v2_bounded.json',
           'B_bundle_ready.json','M_startup/initialization.json','M_startup/epoch01_input_flow.json','09_架构备选补读与阅读范围_v1.json',
           '10_架构备选补读结论.txt','result_tools_preparation_v1.json','result_backup_storage_preflight_v1.json',
           'B_completed_bundle_download_v1.json','bounded_transfer_recovery_preparation_v1.json',
           'remote_A49_bounded_transfers.py','update_current_bounded_A49.py',
           Path(__file__).name,args.observer]
    signed=[HERE/n for n in names]+[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json']
    receipt={'status':'PASS','phase':milestone['phase'],'goal_status':'active','mainline_confirmed':False,
             'previous_goal_turn_classification':'verified_wait','this_turn_classification':'progress_and_verified_wait',
             'previous_A49_startup_receipt':str(previous),'previous_A49_startup_receipt_sha256':sha(previous),
             'previous_A49_signed_current_inputs_resolved':resolved,
             'current_files_sha256':{str(p):sha(p) for p in signed},'milestone':milestone,
             'new_detector_operations_by_publication':0,'new_optimizer_jobs_by_publication':0,
             'new_candidate_automatically_queued':False}
    save(RECEIPT,receipt)
    print(json.dumps({'status':'PASS','B_fixed12_metrics':metrics,'M_live_identity':mi,
                      'M_flow_checks':milestone['M_completed_epoch_flow_checks'],
                      'cumulative_lower_bounds':totals,'mainline_confirmed':False},ensure_ascii=False))
