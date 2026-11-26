"""Three actual seed2 user endpoints and new relative-class evidence; preserve failures."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def validated(folder):
    for n,h in read(folder/'manifest.json')['files'].items():assert sha(folder/n)==h,n
    return read(folder/'report.json')
inputs=read(ROOT/'outputs/user_backbone_seed2_review12_20261011_v2_inputs.json');observed={}
for family,seeds in inputs.items():
    observed[family]={}
    for s,v in seeds.items():
        b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256']
        observed[family][s]=json.loads(b)
margin=validated(ROOT/'outputs/user_class_margin_geometry12_20261011_v1')
assert margin['run_tags']==['BASE12','backbone_silu','p2_skip','backbone_attention'] and margin['all_GT']==38759 and margin['small_GT']==26586
for n,h in read(ROOT/'英文小论文/code/user_class_margin_geometry12_20261011_v1/protocol.json')['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
assert margin['source_sha256']==sha(ROOT/'英文小论文/code/user_class_margin_geometry12_20261011_v1/analyze.py')
base_endpoint=ROOT/'outputs/scale_protection12_20261010_v1/terminal_collection/seed2_target_protect_v1/extracted/baseline_eval/endpoint_epoch12'
base=read(base_endpoint/'summary.json')
with np.load(base_endpoint/'precision_recall.npz',allow_pickle=False) as z:bp=z['precision'][:,:,:,1,2].copy()
assert np.all(bp>=0) and abs(bp.mean()*100-base['native_stats'][3]*100)<1e-12
arms={};sources={};counts={};trainers=[];collectors=[]
for seed,q in observed['observations'].items():
    assert q['code_SHA_verified'] and q['queue']['same_live'] and not q['failures']
    counts[seed]={}
    c=observed['collectors'][seed];assert c['remote4_SHA_verified']
    for arm,j in q['jobs'].items():
        counts[seed][arm]=j['completed_epochs'];assert j['failure'] is None and j['first100']['all100_rows_exact']
        co=c['jobs'][arm];assert co['failure'] is None and co['start_receipt_source_exact']
        if j['completed_epochs']<12:
            assert seed=='1' and j['trainer']['same_live'] and j['controller']['same_live'] and co['same_live']
            trainers.append(dict(seed=1,arm=arm,pid=j['trainer']['child_pid'],start_ticks=j['trainer']['child_start_ticks']))
            collectors.append(dict(seed=1,arm=arm,pid=co['pid'],start_ticks=co['start_ticks']))
        else:assert seed=='2' and not j['trainer']['same_live'] and not j['controller']['same_live'] and co['completed'] and co['waiter_completed'] and not co['same_live']
for arm in ('backbone_silu','p2_skip','backbone_attention'):
    folder=ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed2_{arm}_v1';v=read(folder/'primary_verification_v1.json')
    assert v['status']=='PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS' and v['files_verified']==617 and v['actual_stage_images']==548
    assert v['all_collected_files_verified'] and v['all_ZIP_entries_CRC_verified'] and not v['saved_EMA_independent_replay_verified'] and not v['formal_terminal_reader_passed']
    stage=ROOT/f'outputs/user_backbone_primary_analysis12_20261011_v2/seed2_{arm}';order=ROOT/f'outputs/user_backbone_native_order12_20261011_v2/seed2_{arm}'
    S,O=validated(stage),validated(order);assert S['identity']==v['identity'] and S['evaluated_weights']=='EMA' and S['completed_epochs']==12
    assert O['total_precision_cells_reconstructed']==20200 and O['native_PR_max_abs_difference']==0
    assert S['all548_images'] and S['all38759_GT'] and S['all10_categories']
    endpoint=Path(v['downloaded_packet'])/'run/endpoint_epoch12'
    with np.load(endpoint/'precision_recall.npz',allow_pickle=False) as z:cp=z['precision'][:,:,:,1,2].copy()
    assert abs(cp.mean()*100-S['candidate12']['APs'])<1e-12
    classes=[dict(class_id=c['class_id'],name=c['name'],BASE_APs=c['BASE_APs'],candidate_APs=c['warm_APs'],delta_APs_points=c['delta_APs_points'],contribution_to_total_APs_points=c['contribution_to_total_APs_points']) for c in S['class_APs_arithmetic']]
    assert abs(sum(c['contribution_to_total_APs_points'] for c in classes)-S['delta_points']['APs'])<1e-12
    init=read(Path(v['downloaded_packet'])/'bundle/cpu_initialization.json')['initialization']
    groups={t:{group:{k:val for k,val in S['outcomes'][t][group].items() if k in ('GT','native_all_area_TP_gained','native_all_area_TP_lost','native_all_area_TP_net','supported_GT_by_boundary','gained_TP_first_observed_geometric_support_change','gained_TP_unchanged_three_boundary_support','lost_TP_candidate_outcomes')} for group in ('COCO_small','COCO_small_input_under4')} for t in ('0.5','0.75')}
    for t in ('0.5','0.75'):
        row=margin['comparisons'][arm]['COCO_small'][t]
        assert row['native_TP_gained']==groups[t]['COCO_small']['native_all_area_TP_gained'] and row['native_TP_lost']==groups[t]['COCO_small']['native_all_area_TP_lost']
    arms[arm]=dict(BASE12=S['BASE12'],candidate12=S['candidate12'],delta_points=S['delta_points'],native_gate_passed_seed2=S['native_gate_passed'],class_APs_arithmetic=classes,
        APs_delta_contributions_by_IoU=[dict(IoU=.5+i*.05,contribution_points=float((cp-bp)[i].mean()*10)) for i in range(10)],initialization=init,small_cohorts=groups,
        relative_class_margins=margin['comparisons'][arm],relative_class_summary=margin['runs'][arm]['groups'],files_verified=617,stages_verified=548,
        independent_saved_EMA_replay_verified=False,formal_terminal_reader_passed=False,other_seed_endpoint_pending=True,training_extension_or_retry_allowed=False)
    for f in (folder/'primary_verification_v1.json',stage/'manifest.json',order/'manifest.json'):sources[f.relative_to(ROOT).as_posix()]=sha(f)
assert arms['p2_skip']['native_gate_passed_seed2'] and not arms['backbone_silu']['native_gate_passed_seed2'] and not arms['backbone_attention']['native_gate_passed_seed2']
attention=arms['backbone_attention'];assert all(c['delta_APs_points']<0 for c in attention['class_APs_arithmetic'])
assert attention['initialization']['exact_shared_tensors']==714 and attention['initialization']['removed_native_tensors']==80 and attention['initialization']['new_state_tensors']==90
assert attention['initialization']['extra_parameters']==3798006 and attention['initialization']['parameters']==14030003
P=margin['comparisons']['p2_skip'];ratios={t:dict(gained_TP=P['COCO_small'][t]['native_TP_gained'],BASE_no_final_geometric_support=P['COCO_small'][t]['gained_TP_BASE_no_final_support'],fraction=P['COCO_small'][t]['gained_TP_BASE_no_final_support']/P['COCO_small'][t]['native_TP_gained'],APs_contribution_claim=False) for t in ('0.5','0.75')}
assert ratios['0.75']['gained_TP']==1197 and ratios['0.75']['BASE_no_final_geometric_support']==1096
wait=observed['seed2_subspace_wait']['2'];assert wait['waiter']['same_live'] and wait['wait_failure'] is None and wait['GPU_dispatch'] is None and wait['GPU_completion'] is None
subspace_code=ROOT/'英文小论文/code/p2_subspace_diagnostic12_20261011_v1';assert wait['source_SHA_verified']=={p.name:sha(p) for p in subspace_code.iterdir() if p.is_file()}
assert wait['waiter']['pid']==32667 and wait['waiter']['start_ticks']=='257659599'
old=observed['seed2_component_replays']['2'];assert all(v['candidate_reload_v1/verification.json']['status']=='PASS_SAVED_CANDIDATE_RELOAD' and v['candidate_reload_v1/verification.json']['stats_difference']==0 and v['candidate_reload_v1/verification.json']['all548_exact'] and v['candidate_reload_v1/verification.json']['native_GT_exact'] and v['candidate_reload_v1/completion.json']['exit_code']==0 for v in old['jobs'].values())
dest=ROOT/'outputs/user_backbone_seed2_review12_20261011_v2';dest.mkdir(exist_ok=False)
for f,seeds in inputs.items():
    for s,v in seeds.items():
        with (dest/f'{f}_seed{s}.json').open('xb') as stream:stream.write(v['payload'].encode())
write(dest/'transport_proofs.json',{f:{s:v['transport'] for s,v in seeds.items()} for f,seeds in inputs.items()})
result=dict(status='THREE_ORIGINAL_USER_SEED2_FIXED12_ENDPOINTS_AND_FOUR_RUN_RELATIVE_CLASS_EVIDENCE_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    arms=arms,source_proofs=sources,new_attention_archive_downloaded_and617_files_verified=True,new_attention_saved_stage_scientific_analysis=True,new_attention_native_order_scientific_analysis=True,
    native_PR_cells_pair_checks=60600,distinct_run_PR_cells=40400,native_PR_max_abs_error=0,relative_class_four_saved_runs_report='../user_class_margin_geometry12_20261011_v1/report.json',relative_class_report_sha256=sha(ROOT/'outputs/user_class_margin_geometry12_20261011_v1/report.json'),P2_gained_TP_BASE_no_final_support_ratios=ratios,
    P2_direction='Prioritize geometric information and localization: most newly matched smallGT atIoU.75 lacked final BASE geometry, while correct-top1 existence is nearly unchanged in the jointly geometry-supported stratum. These are outcome-conditioned descriptions, not causal APs shares. car and van behave differently; retain all10 categories.',
    attention_failure='Seed2 AP andAPs decline and all10 classAPs decline. Initial smallGT geometric support improves, but first-layerpre and final support decline. Attention arm replaced five full aggregations, removed80 native pretrained states, initialized90 new states and added37.11891percent parameters; cannot attribute failure uniquely to attention operation.',
    class_margin_limit=margin['scope'],completed_epochs=counts,training_observed_UTC={s:q['UTC'] for s,q in observed['observations'].items()},live_actual_training_handles=trainers,live_primary_collectors=collectors,
    P2_native_subspace_GPU_started=False,P2_native_subspace_waiter=wait['waiter'],P2_native_subspace_wait_observed_UTC=wait['UTC'],P2_native_subspace_wait_predecessors=wait['wait_live']['predecessors'],original_seed2_serial_EMA_closeout_pending=True,
    older_component_seed2_four_saved_EMA_replays_observed_PASS=True,older_component_observed_UTC=old['UTC'],older_formal_terminal_packets_not_yet_downloaded=True,corrupted_large_observer_payload_rejected='../remote_closeout_observer_envelope_rejection12_20261011_v1.json',
    next_action='Complete original seed1 fixed12 and all owned savedEMA closeouts. Run queued P2 rank16 runtime interventions after exact predecessors stop and parent replayPASS. Use per-class geometry/score evidence to decide fresh training; do not extend failed attention orSiLU.',
    root_proven=False,novelty_proven=False,two_seed_efficacy_proven=False,new_training_epochs=0,new_optimizer_updates=0,new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_native_matching_calls=0,existing_process_signals=0,analysis_source_sha256=sha(__file__))
write(dest/'report.json',result)
lines=['原始用户模块：B服务器三组固定第12轮结果与问题定位','已下载并核验每组617个文件、548张阶段输出。原有基线被复用，没有重跑基线。正式保存EMA重放收尾仍在队列；A三组尚未完成。','','配置 | AP | APs | ΔAP | ΔAPs']
for arm,a in arms.items():lines.append(f"{arm} | {a['candidate12']['AP']:.6f} | {a['candidate12']['APs']:.6f} | {a['delta_points']['AP']:+.6f} | {a['delta_points']['APs']:+.6f}")
lines+=['','小GT的几何支持（不是APs分解）：','配置 | IoU | 初始参考框覆盖Δ | 首层pre覆盖Δ | 最终覆盖Δ | 原生真阳性净变化']
for arm,a in arms.items():
    for t in ('0.5','0.75'):
        r=a['small_cohorts'][t]['COCO_small'];b=r['supported_GT_by_boundary']
        lines.append(f"{arm} | {t} | {b['encoder_initial_boxes']['delta']:+d} | {b['pre_after_first_layer_boxes']['delta']:+d} | {b['final_boxes']['delta']:+d} | {r['native_all_area_TP_net']:+d}")
lines+=['','P2定位方向：IoU0.50新增1483个小GT真阳性，其中955个基线最终没有几何支持；IoU0.75新增1197个，其中1096个基线没有几何支持（91.5622%）。这是新增TP样本的组成，不是APs贡献率。',
    '在两组都有几何支持的同一批小GT上，正确类别top1支持的数量：IoU0.50为11449→11518，IoU0.75为4026→4024。car在IoU0.50为4577→4577；van为91→106。不能把P2收益统一编成“分类表达变强”。',
    'P2仍有1991个IoU0.50小GT原生TP，其所有几何支持查询的真实类别分数都低于其他类别。原生多标签输出仍可正确命中，因此top1错误不能单独当成分类检测错误；这里只是已有输出描述，没有重新匹配或测量top1过滤后的AP。',
    '', '注意力配置：AP -2.5852、APs -0.8039，10类APs全部下降；小GT初始覆盖+314，首层pre -857，最终 -999。它替换了五个完整聚合模块，丢弃80个预训练状态，初始化90个新状态，增加约37.12%参数。这是结构、初始化和容量一起变化的实验，不能证明某种注意力机制本身无效。',
    '', '当前：A SiLU11/P2 11/注意力7；B三组12。P2秩16对照的真实CPU检查已通过，GPU完整验证尚未启动，等现有收尾与P2原权重重载校验完成。B旧组件四组保存EMA重放刚观察到PASS，队列正在推进。',
    '下一步：完成双种子固定终点；检验P2局部均值与最强16线性方向的运行时对照；根据几何/得分证据决定下一次训练。当前没有新增训练或延长失败配置。全部原始负结果、权重和日志保留。']
with (dest/'三组第12轮结果与方向审查.txt').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
main=ROOT/'outputs/stem_component12_20261011_v1';state=read(main/'execution_state_v40.json')
state.update(status='ACTIVE_THREE_SEED2_USER_ENDPOINTS_AND_GEOMETRY_CONDITIONED_CLASS_EVIDENCE',supersedes='execution_state_v40.json',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,evidence='../user_backbone_seed2_review12_20261011_v2/report.json',original_user_current_completed_epochs=counts,original_user_actual_trainers_verified_live=trainers,
    seed2_original_attention_native_gate_failed=True,seed2_original_attention_extended_or_retried=False,relative_class_four_saved_runs_verified=True,P2_runtime_subspace_GPU_started=False,P2_runtime_subspace_GPU_waiter=wait['waiter'],changed_next_action=result['next_action'],root_or_novelty_proven=False,
    remote_result_observer_liveness_scope='Three seed1 original trainers and three readonly collectors live at recordedUTC; seed2 allthree original trainers/controllers and primarycollectors stopped positively. NewP2 wait32667/257659599 verifiedlive19:32UTC. OldB component34131 verifiedrunning at19:28 before four savedEMA PASS snapshots19:30; formal packets/replay user endpoints not yet consumed.')
write(main/'execution_state_v41.json',state)
wrapper=read(ROOT/'outputs/original_stem12_20261011_v1/execution_state_v43.json');wrapper.update(status=state['status'],supersedes='execution_state_v43.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v41.json',goal_complete=False)
write(ROOT/'outputs/original_stem12_20261011_v1/execution_state_v44.json',wrapper)
user=read(ROOT/'outputs/user_backbone12_20261011_v1/execution_state_v17.json');user.update(status='THREE_SEED2_PRIMARY_ENDPOINTS_VERIFIED_THREE_SEED1_TRAINERS_LIVE',supersedes='execution_state_v17.json',evidence='../user_backbone_seed2_review12_20261011_v2/report.json',completed_epochs=counts,live_actual_training_handles=trainers,seed2_original_attention_native_gate_failed=True,relative_class_geometry_review=result['relative_class_four_saved_runs_report'],goal_complete=False)
write(ROOT/'outputs/user_backbone12_20261011_v1/execution_state_v18.json',user)
primary=read(ROOT/'outputs/user_backbone_primary_collection12_20261011_v2/execution_state_v2.json');primary.update(status='THREE_SEED2_PRIMARY_PACKAGES_VERIFIED_THREE_SEED1_COLLECTORS_LIVE',supersedes='execution_state_v2.json',evidence='../user_backbone_seed2_review12_20261011_v2/report.json',live_collectors=collectors,goal_complete=False)
write(ROOT/'outputs/user_backbone_primary_collection12_20261011_v2/execution_state_v3.json',primary)
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
print(json.dumps(dict(status=result['status'],epochs=counts,attention_delta=attention['delta_points'],P2_gained_TP_ratios=ratios,Main_state='execution_state_v41.json',goal_complete=False)),flush=True)
