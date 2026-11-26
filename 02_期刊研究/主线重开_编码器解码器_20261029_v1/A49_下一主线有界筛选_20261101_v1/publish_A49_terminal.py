"""Publish a completed A49 comparison; does not complete the broad research goal."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
RECEIPT=ROOT/'18_A49终点与历史保留校验回执.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def alive(i):return bool(i and i.get('present') and i.get('same_start_ticks') and i.get('owned_command') and i.get('state') not in ['Z','X'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('observer');args=ap.parse_args()
    assert not RECEIPT.exists()
    raw=read(ROOT/args.observer);assert raw['status']=='EXECUTED' and raw['exit']==0
    o=json.loads(raw['stdout']);assert not alive(o['supervisor_identity'])
    assert all(not alive(r['identity']) for r in o['arms'].values())
    assert o['frozen_file_checks']=={'count':12,'all_pass':True,'failed':[]}
    protocol_path=ROOT/'protocol_FROZEN.json';protocol=read(protocol_path)
    assert sha(protocol_path)==o['protocol_sha256']=='d3a971a21d1c339052c8a7ef41fcab345882edc8893c4762c66dbbc5bc10f8f3'
    for name,expected in protocol['files_sha256'].items():assert sha(ROOT/name)==expected,name
    j=read(ROOT/'fixed_endpoint_judgment_A49.json')
    assert j['status']=='VALID_FIXED_ENDPOINT_COMPARISON' and j['protocol_sha256']==sha(protocol_path)
    remote=o['three_arm_result'];assert remote['status']=='COMPLETED' and remote['additional_training_queued'] is False
    assert remote['metrics']==j['metrics_percent']
    assert remote['pilot_pass']==j['all_frozen_gates_pass']
    assert remote['Q_minus_B_APs_pp']==j['deltas_percentage_points']['Q_minus_B']['APs']
    assert remote['Q_minus_M_APs_pp']==j['deltas_percentage_points']['Q_minus_M']['APs']
    assert remote['Q_minus_B_AP_pp']==j['deltas_percentage_points']['Q_minus_B']['AP']
    for arm,row in o['arms'].items():
        assert row['result']['status']=='COMPLETED' and row['exit']['exit_code']==0
        v=read(ROOT/(arm+'_completed_local_verification.json'));assert v['status']=='PASS'
        assert v['endpoint_metrics']==j['metrics_percent'][arm]
    for arm in ['M','Q']:
        rows=o['common_epoch_checks'][arm+'_vs_B']
        assert len(rows)==12 and all(x['all_pass'] for x in rows)

    phase_counts=o['observed_counters_lower_bounds']
    assert phase_counts=={'model_forwards':30489,'formal_optimizer_updates':29124,'capacity_optimizer_updates':9,
                          'formal_epoch_sum':36,'extra_GPU_mask_core_reads':38844}
    old_receipt=BASE/'A48_新机制候选与主线收敛_20261101_v1'/'21_A48终点与历史保留校验回执.json'
    old=read(old_receipt)['cumulative_counts']
    totals={'model_forwards':old['model_forwards']+phase_counts['model_forwards'],
            'formal_optimizer_updates':old['formal_optimizer_updates']+phase_counts['formal_optimizer_updates'],
            'capacity_updates':old['capacity_updates']+phase_counts['capacity_optimizer_updates'],
            'optimizer_updates_including_capacity':old['optimizer_updates_including_capacity']+phase_counts['formal_optimizer_updates']+phase_counts['capacity_optimizer_updates'],
            'formal_epochs':old['formal_epochs']+phase_counts['formal_epoch_sum']}
    assert totals=={'model_forwards':144941,'formal_optimizer_updates':117022,'capacity_updates':37,
                   'optimizer_updates_including_capacity':117059,'formal_epochs':144}
    secondary={}
    for arm in ['B','M','Q']:
        path=ROOT/'postprocess_ignore_strata'/('arm_'+arm)/'result.json'
        if path.exists():
            r=read(path);assert r['status']=='COMPLETED'
            assert r['model_forwards']==r['optimizer_updates']==0
            secondary[arm]={'path':str(path),'COCO_evaluate_attempts':r['COCO_evaluate_attempts'],
                            'full_cached_precision_max_diff':r['groups']['full548']['cached_precision_max_abs_difference'],
                            'full_cached_recall_max_diff':r['groups']['full548']['cached_recall_max_abs_difference'],
                            'groups':{name:row['metrics_percent'] for name,row in r['groups'].items()}}
    assert sum(r['COCO_evaluate_attempts'] for r in secondary.values())<=9
    passed=j['all_frozen_gates_pass']
    decision='CANDIDATE_FIXED12_GATES_PASSED_MAINLINE_REVIEW_PENDING' if passed else 'RETIRED_EXACT_RECIPE_AFTER_FIXED12'
    next_action=('Assess whether this positive pilot and inspected closest-work differences support a concrete paper research line; no automatic longer training.'
                 if passed else 'Select a substantively different next mechanism; no threshold/site/seed/host-only rescue of A49.')
    milestone={'phase':'A49_FIXED12_TERMINAL','status':decision,'goal_status':'active','mainline_confirmed':False,
       'dataset':protocol['dataset'],'host':protocol['host'],'candidate':protocol['Q'],
       'all_frozen_gates_pass':passed,'metrics_percent':j['metrics_percent'],
       'deltas_pp':j['deltas_percentage_points'],'gates':j['frozen_gates'],
       'formal_model_forwards':30477,'capacity_model_forwards':12,'formal_optimizer_updates':29124,
       'capacity_optimizer_updates':9,'formal_epochs':36,'extra_GPU_mask_core_reads':38844,
       'local_bundles_verified':True,'all12_epoch_control_checks':True,'no_owned_live_training_job':True,
       'additional_training_queued':False,'protocol_sha256':sha(protocol_path),
       'judgment':str(ROOT/'fixed_endpoint_judgment_A49.json'),'terminal_observation':str(ROOT/args.observer),
       'report':str(ROOT/'A49_三组固定终点结论.txt'),'secondary_descriptive_analysis':secondary,
       'cumulative_counts_terminal':totals,'limits':j['limits'],'next_action':next_action}
    note=ROOT/'17_A49全程验收与解释.txt'
    lines=['A49 完整阶段验收与解释','',
       '数据：VisDrone2019-DET train6471/val548；算法：ECDet-S原生BN。',
       '具体问题：已知class0忽略场下，query采样支持的负监督规则能否超越普通预测框覆盖忽略。',
       '原文/源码依据与旧路线区别见07调研报告、02矩阵、05与09阅读范围及统一PDF库；仍不声称穷尽已有方法。',
       '这里的采样支持是几何代理，不代表完整视觉感受野、未标注目标概率或因果贡献。',
       '三组各12轮已正常完成，完整归档/状态校验和数据流核对通过。完整分数与三项固定门槛见A49_三组固定终点结论.txt。',
       '实际裁决：'+decision,
       '研究总目标继续active，当前终点裁决不等于论文创新已经成立，也不因一项否定结果结束广义目标。',
       '所有主门槛同时通过后才作主线评估；未通过则停止此配方，辅助分组不能挽救主门槛。',
       '新增正式完整模型前向30477、容量前向12；正式优化29124、容量优化9；额外单通道mask核读取38844。',
       '累计完整模型前向144941、正式优化117022、容量优化37、总优化117059、正式训练轮数144。',
       '此前摘要中的累计前向上限148941是加法错误，已保留原记录并更正为114452+12+30477=144941；实际执行协议和已观察计数始终按各组12轮记录。',
       '归档/权重状态校验不等于重跑模型预测；若有CPU重算，它只验证已存预测的指标实现。',
       '已有预测的附加COCO重算调用数：'+str(sum(r['COCO_evaluate_attempts'] for r in secondary.values()))+'；无新增模型前向或优化。',
       '附加图像分组只作描述：AP不具可加性，不能按GT数加权还原整体AP，组间组成差异也不证明因果。',
       '所有评估是maxDets300、原图GT面积和class0并集过滤的COCO式口径，非官方VisDrone成绩。',
       '没有自动重试、加轮或额外训练队列。下一步：'+next_action]
    with note.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')

    previous=ROOT/'16_M终点与Q启动校验回执.json';prior=read(previous)
    for s,expected in prior['current_files_sha256'].items():assert sha(Path(s))==expected,s
    before=ROOT/'before_A49_terminal';before.mkdir(exist_ok=False);remap={}
    for s,expected in prior['current_files_sha256'].items():
        p=Path(s)
        if p.parent==BASE and p.name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
            dest=before/p.name;shutil.copy2(p,dest);assert sha(dest)==expected
        else:dest=p
        remap[s]={'preserved_at':str(dest),'sha256':expected}
    now=datetime.now(timezone.utc).isoformat()
    for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        path=BASE/name;x=read(path)
        x.update({'goal_status':'active','mainline_validated':False,'latest_report':milestone['report'],
                  'A49_terminal_result':milestone,'current_priority_candidate':'A49_B_KNOWN_IGNORE_SUPERVISION' if passed else None,
                  'current_priority_question':next_action,'mainline_selected_for_pilot':passed})
        if name.startswith('01'):
            x.update({'latest_update_utc':now,'status':decision,'pilot_validation_status':decision,
                      'remaining_deliverable':next_action,'next_decision_protocol':None})
        else:
            x.update({'recorded_utc':now,'current_phase':'A49','latest_authoritative_phase':'A49_FIXED12_TERMINAL',
               'model_forwards':totals['model_forwards'],'cap':totals['model_forwards'],
               'cap_scope':'A49 exact fixed12 phase closed at derived cumulative144941. Older caps are historical.',
               'training_updates':totals['optimizer_updates_including_capacity'],
               'detector_optimizer_updates_including_capacity':totals['optimizer_updates_including_capacity'],
               'formal_detector_training_updates':totals['formal_optimizer_updates'],'capacity_only_optimizer_updates':totals['capacity_updates'],
               'formal_training_epochs':totals['formal_epochs'],'counts_terminal':True,'training_counts_terminal':True,
               'counters_observed_lower_bounds_while_job_live':False,'current_native_job_terminal':True,'full_forward_cap_closed':True,
               'diagnostics_complete':True,'diagnostics_complete_scope':'A49 exact fixed12 pilot only; broad research goal remains incomplete.',
               'current_user_task_status':decision,'CPU_saved_prediction_COCO_evaluate_calls_A49':sum(r['COCO_evaluate_attempts'] for r in secondary.values()),
               'current_goal_turn':{'classification':'progress','goal_status':'active','mainline_confirmed':False,'observer':str(ROOT/args.observer)}})
        x['A49_execution'].update({'status':decision,'observation':str(ROOT/args.observer),'owned_live_arms':[],
            'current_evaluations':{a:r['evaluations'] for a,r in o['arms'].items()},
            'observed_progress':{a:r['progress'] for a,r in o['arms'].items()},'new_phase_counters_lower_bounds':phase_counts,
            'terminal_result':str(ROOT/'fixed_endpoint_judgment_A49.json')})
        assert sha(path)==prior['current_files_sha256'][str(path)]
        save(path,x)
    for row in remap.values():assert sha(Path(row['preserved_at']))==row['sha256']
    files=[ROOT/n for n in ['A49_三组固定终点结论.txt','fixed_endpoint_judgment_A49.json',
             '17_A49全程验收与解释.txt','forward_cap_arithmetic_correction_v1.json',args.observer,Path(__file__).name]]
    files += [BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json']
    files += [ROOT/(a+'_completed_local_verification.json') for a in ['B','M','Q']]
    files += [Path(v['path']) for v in secondary.values()]
    receipt={'status':'PASS','phase':'A49_FIXED12_TERMINAL_PUBLISHED','goal_status':'active','mainline_confirmed':False,
       'previous_M_milestone_receipt':str(previous),'previous_M_milestone_receipt_sha256':sha(previous),
       'previous_M_signed_current_inputs_resolved':remap,'current_files_sha256':{str(p):sha(p) for p in files},
       'milestone':milestone,'cumulative_counts':totals,'owned_jobs_terminal':True,
       'new_detector_operations_by_publication':0,'new_optimizer_jobs_by_publication':0,'next_candidate_automatically_queued':False}
    save(RECEIPT,receipt)
    pointer=read(ROOT/'CURRENT_A49.json')
    pointer.update({'goal_status':'active','mainline_confirmed':False,'stage':'A49_FIXED12_TERMINAL',
                    'owned_live_arms':[],'terminal_receipt':str(RECEIPT),'next_action':next_action})
    save(ROOT/'CURRENT_A49.json',pointer)
    print(json.dumps({'status':'PASS','candidate_status':decision,'all_primary_gates_pass':passed,
                      'cumulative_counts':totals,'mainline_confirmed':False,'goal_status':'active'},ensure_ascii=False))
