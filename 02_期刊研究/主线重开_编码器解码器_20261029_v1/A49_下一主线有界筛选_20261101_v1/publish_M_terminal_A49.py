"""Publish M12 only after local archive verification and authentic Q startup."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
REPORT=ROOT/'16_M终点与Q启动校验回执.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('observer');args=ap.parse_args()
    assert not REPORT.exists()
    observation=ROOT/args.observer;raw=read(observation)
    assert raw['status']=='EXECUTED' and raw['exit']==0
    o=json.loads(raw['stdout']);m=o['arms']['M'];q=o['arms']['Q'];qi=q['identity'] or {}
    assert m['result']['status']=='COMPLETED' and m['exit']['exit_code']==0 and not m['identity']['present']
    assert qi.get('present') and qi.get('same_start_ticks') and qi.get('owned_command') and qi.get('state') not in ['Z','X']
    si=o['supervisor_identity'];assert si['present'] and si['same_start_ticks'] and si['owned_command']
    verified={a:read(ROOT/(a+'_completed_local_verification.json')) for a in ['B','M']}
    for a,v in verified.items():assert v['status']=='PASS' and v['epochs']==12 and v['optimizer_updates']==9708
    assert read(ROOT/'M_completed_bundle_download_v1_bounded.json')['status']=='TRANSFERRED'
    assert verified['M']['endpoint_metrics']==m['result']['endpoint_metrics']
    initial=read(ROOT/'Q_startup'/'initialization.json')
    assert initial['all_initial_tensors_identical'] and initial['model_parameters']==9803112 and initial['seed']==20261101
    assert initial['common_init_sha256']==m['result']['initialization_sha256']
    protocol=ROOT/'protocol_FROZEN.json';expected='d3a971a21d1c339052c8a7ef41fcab345882edc8893c4762c66dbbc5bc10f8f3'
    assert sha(protocol)==initial['protocol_sha256']==m['result']['protocol_sha256']==expected
    bfolder=ROOT/'completed_B_v1'/'arm_B';mfolder=ROOT/'completed_M_v1'/'arm_M'
    assert read(bfolder/'initialization.json')==read(mfolder/'initialization.json')==initial
    assert sha(bfolder/'epoch12_EMA_GT.json')==sha(mfolder/'epoch12_EMA_GT.json')
    stream_keys=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256',
                 'all_batches_targets_including_ignore_tiles_sha256']
    checks=[]
    for epoch in range(1,13):
        b=read(bfolder/('epoch%02d_input_flow.json'%epoch));control=read(mfolder/('epoch%02d_input_flow.json'%epoch))
        row={k:b[k]==control[k] for k in stream_keys};assert all(row.values())
        assert b['optimizer_updates']==control['optimizer_updates']==epoch*809
        checks.append({'epoch':epoch,**row})
    if (q['progress'] or {}).get('status')=='TRAINING':
        assert q['progress']['extra_mask_core_reads']==4*q['progress']['optimizer_updates']
    qchecks=o['common_epoch_checks']['Q_vs_B']
    assert len(qchecks)==q['completed_epochs'] and all(r['all_pass'] for r in qchecks)
    metrics={a:v['endpoint_metrics'] for a,v in verified.items()}
    delta={k:metrics['M'][k]-metrics['B'][k] for k in ['AP','APs','APs75']}
    secondary={}
    for a in ['B','M']:
        path=ROOT/'postprocess_ignore_strata'/('arm_'+a)/'result.json'
        if path.exists():
            result=read(path);assert result['status']=='COMPLETED'
            secondary[a]={'path':str(path),'COCO_evaluate_attempts':result['COCO_evaluate_attempts'],
                          'cached_precision_max_abs_difference':result['groups']['full548']['cached_precision_max_abs_difference'],
                          'cached_recall_max_abs_difference':result['groups']['full548']['cached_recall_max_abs_difference']}
    prior_path=ROOT/'12_B终点与M启动校验回执.json';prior=read(prior_path)
    for s,expected_sha in prior['current_files_sha256'].items():assert sha(Path(s))==expected_sha,s
    before=ROOT/'before_A49_M_terminal';before.mkdir(exist_ok=False);remap={}
    for s,expected_sha in prior['current_files_sha256'].items():
        p=Path(s)
        if p.parent==BASE and p.name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
            dest=before/p.name;shutil.copy2(p,dest);assert sha(dest)==expected_sha
        else:dest=p
        remap[s]={'preserved_at':str(dest),'sha256':expected_sha}
    c=o['observed_counters_lower_bounds']
    totals={'model_forwards':114452+c['model_forwards'],'formal_optimizer_updates':87898+c['formal_optimizer_updates'],
            'capacity_updates':28+c['capacity_optimizer_updates'],
            'optimizer_updates_including_capacity':87926+c['formal_optimizer_updates']+c['capacity_optimizer_updates'],
            'formal_epochs':108+c['formal_epoch_sum']}
    milestone={'phase':'A49_B_M12_TERMINAL_Q_ACTIVE','goal_status':'active','mainline_confirmed':False,
       'B_M_fixed12_metrics':metrics,'M_minus_B_fixed12_pp':delta,'M12_independent_local_stream_checks':checks,
       'endpoint_GT_identical':True,'Q_initialization_identical':True,'Q_owned_live_identity':qi,'Q_progress':q['progress'],
       'Q_completed_epoch_stream_checks':qchecks,
       'observer':str(observation),'cumulative_counts_lower_bounds':totals,
       'secondary_CPU_metric_recalculation':secondary,
       'scope':'Completed B/M control comparison and Q startup only. No fixed12 Q endpoint, candidate gate verdict or novelty confirmation.'}
    note=ROOT/'15_M12终点与Q启动记录.txt'
    lines=['A49 B/M固定12轮结果与Q启动','',
           '数据VisDrone2019-DET；算法ECDet-S原生BN；问题是已知忽略区域的query负监督有效性。',
           'B/M两组已完成12轮且退出0，完整权重/日志均本地校验。Q仍在运行，主线未确认，目标active。',
           '固定第12轮EMA：','组别\tAP(%)\tAPs(%)\tAPs75(%)']
    for a in ['B','M']:
        r=metrics[a];lines.append('%s\t%.8f\t%.8f\t%.8f'%(a,r['AP'],r['APs'],r['APs75']))
    lines+=['M−B（百分点）：AP %+.8f，APs %+.8f，APs75 %+.8f。'%(delta['AP'],delta['APs'],delta['APs75']),
            '两组12轮输入顺序、前8批实际输入/RNG和全轮含ignore的目标流逐轮一致，评估GT字节一致；Q同初始化/seed/参数/协议。',
            'Q当前身份PID%d/start_ticks%d；按原监督器完成Q12，无追加训练队列。'%(qi['pid'],qi['expected_start_ticks']),
            'Q训练的额外mask核读取在TRAINING快照中核对为每次更新4次；它不是额外完整模型前向。',
            '普通忽略区规则属于已有处理；M的结果无论正负均不代表Q增量或创新成立。',
            '固定主门槛仍是Q−B APs>=0.5pp、Q−M APs>=0.3pp、Q−B AP>=−0.2pp。',
            '不以最佳轮次或辅助分组收益挽救未达主门槛的候选。',
            '统一COCO式评估、maxDets300、原图GT面积及class0并集过滤，非官方VisDrone成绩。',
            '已有预测的CPU重算状态见16回执；重算不等于重跑检测器。',
            '最新进度见可更新CURRENT_A49.json，本报告为不可变终点/启动快照。']
    with note.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    now=datetime.now(timezone.utc).isoformat()
    for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        path=BASE/name;x=read(path)
        x.update({'goal_status':'active','mainline_validated':False,'latest_report':str(note),
                  'A49_M_terminal_verification':milestone})
        if name.startswith('01'):
            x.update({'latest_update_utc':now,'status':'A49_B_M12_TERMINAL_Q_ACTIVE','pilot_validation_status':'B_M_COMPLETED_Q_ACTIVE'})
        else:
            x.update({'recorded_utc':now,'current_phase':'A49','latest_authoritative_phase':'A49_B_M12_TERMINAL_Q_ACTIVE',
                      'model_forwards':totals['model_forwards'],'cap':144941,
                      'cap_scope':'Derived fixed12 cumulative endpoint114452+12+30477=144941; current counts are observed lower bounds.',
                      'cap_arithmetic_correction':str(ROOT/'forward_cap_arithmetic_correction_v1.json'),
                      'training_updates':totals['optimizer_updates_including_capacity'],
                      'detector_optimizer_updates_including_capacity':totals['optimizer_updates_including_capacity'],
                      'formal_detector_training_updates':totals['formal_optimizer_updates'],'capacity_only_optimizer_updates':totals['capacity_updates'],
                      'formal_training_epochs':totals['formal_epochs'],'counts_terminal':False,'training_counts_terminal':False,
                      'counters_observed_lower_bounds_while_job_live':True,'current_user_task_status':'B_M_COMPLETED_Q_ACTIVE',
                      'CPU_saved_prediction_COCO_evaluate_calls_A49':sum(r['COCO_evaluate_attempts'] for r in secondary.values()),
                      'current_goal_turn':{'classification':'progress_and_verified_wait','goal_status':'active','mainline_confirmed':False,'observer':str(observation)}})
        x['A49_execution'].update({'status':'B_M_COMPLETED_Q_ACTIVE','observation':str(observation),
                  'owned_live_arms':[{'arm':'Q','pid':qi['pid'],'start_ticks':qi['expected_start_ticks']}],
                  'current_evaluations':{a:r['evaluations'] for a,r in o['arms'].items()},
                  'observed_progress':{a:r['progress'] for a,r in o['arms'].items()},'new_phase_counters_lower_bounds':c})
        assert sha(path)==prior['current_files_sha256'][str(path)]
        save(path,x)
    for r in remap.values():assert sha(Path(r['preserved_at']))==r['sha256']
    names=['15_M12终点与Q启动记录.txt','M_completed_local_verification.json','M_bundle_ready.json',
           'M_completed_bundle_download_v1_bounded.json','Q_startup/initialization.json','forward_cap_arithmetic_correction_v1.json',args.observer,Path(__file__).name]
    files=[ROOT/n for n in names]+[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json']
    files += [Path(r['path']) for r in secondary.values()]
    result={'status':'PASS','phase':milestone['phase'],'goal_status':'active','mainline_confirmed':False,
       'classification':'progress_and_verified_wait','previous_goal_turn_classification':'verified_wait',
       'previous_B_milestone_receipt':str(prior_path),'previous_B_milestone_receipt_sha256':sha(prior_path),
       'previous_B_milestone_current_inputs_resolved':remap,'current_files_sha256':{str(p):sha(p) for p in files},
       'milestone':milestone,'new_detector_operations_by_publication':0,'new_optimizer_jobs_by_publication':0,
       'new_candidate_automatically_queued':False}
    save(REPORT,result)
    print(json.dumps({'status':'PASS','M_minus_B_fixed12_pp':delta,'Q_live':qi,'secondary_CPU_checks':secondary,
                      'cumulative_lower_bounds':totals,'mainline_confirmed':False},ensure_ascii=False))
