"""Join both actual live tests with the completed six-record Stem analysis."""
from pathlib import Path
import datetime,hashlib,json

R=Path(__file__).resolve().parent.parent
evidence={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def record(p):
    p=Path(p);evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p));return read(p)
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def captures(p):
    data=record(p);out={}
    for k,v in data.items():
        b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256'];out[k]=json.loads(b)
    return out
def verify_analysis(folder):
    m=record(folder/'manifest.json')
    for n,h in m['files'].items():assert sha(folder/n)==h,n
    return len(m['files'])

prior=R/'outputs/reviewed_remote_tests_actual12_20261011_v4'
old=record(prior/'report.json');m=record(prior/'manifest.json')
assert sha(prior/'report.json')==m['report_sha256']
for n,v in m['files'].items():assert sha(R/n)==v['sha256'],n
traces=captures(R/'outputs/reviewed_remote_tests_actual12_20261011_v5_training_inputs.json')
dispatch=captures(R/'outputs/reviewed_remote_tests_actual12_20261011_v5_remaining_formal_inputs.json')
low=R/'outputs/p2_lowrank12_20261011_v3'
source=record(low/'source_review.json')
assert {p.name:sha(p) for p in (R/'英文小论文/code/p2_lowrank12_20261011_v3').iterdir()}==source['files']
training=[]
for seed in (1,2):
    previous=old['new_lowrank_actual_training'][seed-1]
    x=traces[f'turn5_actual_rank16_native_same_epoch_v2_seed{seed}']
    assert x['seed']==seed and x['candidate_same_live'] and x['queue_same_live'] and x['controller_same_live']
    assert x['failure'] is None and x['queue_failure'] is None
    assert x['completed_epochs']==2 and len(x['same_epoch_metrics'])==2
    assert x['audit']['actual_optimizer_steps']>previous['audit']['actual_optimizer_steps']>0
    assert x['first100']['all100_rows_exact'] and x['candidate_handle']['automatic_retry'] is False
    for row in x['same_epoch_metrics']:assert row['decision_scope']=='descriptive_only_fixed12_required'
    training.append(dict(previous,observed_remote_UTC=x['UTC'],completed_epochs=x['completed_epochs'],audit=x['audit'],
        actual_handles=dict(queue=x['queue_handle'],supervisor=x['controller_handle'],trainer=x['candidate_handle']),
        exact_owner_handles_live=True,source_log_paths=x['source_log_paths'],source_log_sha256=x['source_log_sha256'],
        same_epoch_metrics=x['same_epoch_metrics'],observation_scope='Snapshot only, not a fixed12 efficacy verdict'))

six=R/'outputs/native_selection_analysis12_20261011_v5/actual_six'
file_count=verify_analysis(six);analysis=record(six/'report.json')
assert file_count==49 and analysis['seeds']==[1,2]
assert analysis['new_model_forwards']==analysis['new_evaluator_calls']==analysis['new_training_epochs']==0
categoryfolder=R/'outputs/six_Stem_category_APs12_20261011_v1'
category=record(categoryfolder/'report.json');cm=record(categoryfolder/'manifest.json')
assert sha(categoryfolder/'report.json')==cm['report_sha256']
for n,v in cm['files'].items():assert sha(R/n)==v['sha256'],n
assert category['all_six_formal_fixed12_and_exact_saved_EMA_reloads_verified']
assert category['saved_model_replay_exact_requirement_unchanged']
common={};common_counts={}
stems=[]
for seed in (1,2):
    f=R/f'outputs/native_selection_common_location12_20261011_v3/seed{seed}'
    common_counts[str(seed)]=verify_analysis(f);j=record(f/'report.json')
    assert j['seed']==seed and j['images']==548 and j['small_GT']==26586
    assert j['all_three_EMA_anchor_buffers_equal'] and j['actual_original_TopK_locations']
    assert j['new_model_forwards']==j['new_evaluator_calls']==j['new_training_epochs']==0
    common[str(seed)]=dict(images=j['images'],small_GT=j['small_GT'],source_report_sha256=sha(f/'report.json'),
        level_selection_counts=j['level_selection_counts'],comparisons={n:dict(COCO_small=v['groups']['COCO_small'],
            common_selected_locations_mean=v['common_selected_locations_mean'],
            median_common_site_bbox_offset_RMS_difference=v['median_common_site_bbox_offset_RMS_difference']) for n,v in j['comparisons'].items()})
    for arm in ('original_stem','native_stem_scratch','highfreq_rgb'):
        formal=record(R/f'outputs/stem_fixed12_seed{seed}_formal_collection12_20261011_v1/seed{seed}_{arm}/verification.json')
        packet=record(R/f'outputs/native_selection_collection12_20261011_v2/seed{seed}_{arm}/packet_verification.json')
        assert formal['all548_stage_arrays_reloaded_exact'] and formal['first100_inputs_and_pre_forward_RNG_exact']
        assert packet['status']=='PASS_DOWNLOADED_SELECTION_DIAGNOSTIC_PACKET_HASHES_AND_BINDINGS' and packet['files_verified']==1126
        assert packet['identity']==formal['identity'] and packet['native_terminal_manifest_sha256']==formal['terminal_manifest_sha256']
        d=analysis['descriptions'][f'{arm}_seed{seed}']['stages']
        assert d['AP_percent']==formal['candidate_overall_percent']
        counts=d['counts']['COCO_small'];support={}
        for threshold,v in counts.items():
            patterns=v['geometric_support_patterns'];assert sum(patterns.values())==26586
            support[threshold]=dict(initial=v['first_support_at_initial'],
                pre=sum(z for k,z in patterns.items() if k[1]=='1'),
                final=sum(z for k,z in patterns.items() if k[2]=='1'),
                native_all_area_TP_filtered_small_GT=v['native_all_area_TP_filtered_GT'],
                no_geometry_at_any_saved_boundary=v['no_support_at_any_observed_boundary'],
                final_geometry_without_native_TP=v['final_geometry_without_native_TP'])
        stems.append(dict(seed=seed,arm=arm,candidate12=formal['candidate_overall_percent'],
            delta_against_same_seed_pretrained_BASE12=formal['delta_AP_metrics_pp'],
            native_gate_passed_this_seed=formal['native_gate_passed_this_seed'],
            exact_saved_EMA548_reloaded=True,native8400_packet_files_verified=1126,small_GT_geometry_support=support))

queue=dispatch['turn4_A_two_remaining_formal_actual_dispatch_seed1']
controllerpath=R/'outputs/user_remaining_formal_closeout12_20261011_v1.py'
evidence[controllerpath.relative_to(R).as_posix()]=dict(bytes=controllerpath.stat().st_size,sha256=sha(controllerpath))
assert sha(controllerpath)==queue['source_sha256']
remaining=traces['turn5_actual_rank16_native_same_epoch_v2_seed1']['remaining_formal_queue']
assert all(remaining['handle'][key]==queue[key] for key in ('pid','start_ticks','source_sha256','training_epochs','BASE_forwards'))
assert remaining['same_live'] and remaining['failure'] is None
assert queue['training_epochs']==queue['BASE_forwards']==queue['process_signals']==0
assert remaining['phase']['phase']=='waiting_exact_rank16_owners_and_actual_GPU_idle'

out=R/'outputs/reviewed_remote_tests_actual12_20261011_v5';out.mkdir(exist_ok=False)
r=dict(old)
r.update(status='BOTH_REVIEWED_RANK16_P2_TESTS_ACTUAL_EPOCH2_AND_ALL_SIX_STEM_ANALYSES_COMPLETE',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_immutable_report_sha256=sha(prior/'report.json'),
    new_lowrank_actual_training=training,
    six_Stem_native_selection_geometry_completed=dict(status=analysis['status'],verified_analysis_files=49,
        formal_fixed12_and_exact_reload=stems,source_report_sha256=sha(six/'report.json'),
        common_physical_location_analysis_both_seeds_complete=True,common_location_verified_files=common_counts,
        common_physical_location_description=common),
    six_Stem_native_category_AP_arithmetic=category,
    A_remaining_SiLU_attention_unattempted_formal_closeout_queue=dict(dispatch=queue,actual_live_observation=remaining,
        scope='Each existing saved12 candidate gets its first independent closeout once after all current exact rank16 owners end and GPU becomes idle; no training, BASE inference or failed P2 replay retry.'),
    current_lowrank_interpretation=dict(frozen_live_source_files_changed=False,
        intervention='Replace the unrestricted P2 restore matrix with a learned rank16 factorization, with common pretrained native tensors preserved.',
        simultaneous_changes=['Effective rank constraint','Parameterization and optimization/gradient path','Parameter count and initial branch construction'],
        correction_to_frozen_protocol_wording='The phrase only rank constraint differs is too broad as a causal claim. Final AP cannot alone isolate pure rank regularization or establish discarded directions as redundant.',
        original_P2_comparison_required=True,BASE_acceptance_criterion_unchanged=True,pure_rank_causal_effect_proven=False),
    current_scientific_conclusions=[
        'Enhanced Stem minus random native Stem: seed1 AP -1.1672918373/APs +0.1582066690, seed2 AP +1.7979219984/APs +1.8295223735. Whole-Stem overall improvement is inconsistent and both enhanced endpoints fail pretrained BASE12.',
        'Highpass x-avg_pool3(x) versus same-parameter raw RGB: APs improves +0.4757623873/+0.9985955905 with 8/9 improved small-class APs, and initial/final geometry support improves in both seeds. Seed1 overall AP still drops -0.1193533151; this control does not isolate mean/amplitude/background/frequency causes.',
        'At shared physical TopK locations, initial boxes also differ. Selection-change and initial-regression-change both occur; spatial identity is not query or feature identity, and the upstream source is not isolated.',
        'Native small AP is an equal-weight class/IoU PR statistic at maxDets100. All-area TP flags filtered by small GT, nearby selection counts and geometry populations are not arithmetic APs contributions.',
        'Rank16 actual epoch2 is mixed: seed1 APs +0.4052634688 versus same-epoch BASE, seed2 -0.2223068455; no fixed12 verdict.'],
    new_lowrank_fixed12_efficacy_proven=False,two_seed_P2_weight_intervention_complete=False,
    two_seed_formal_efficacy_complete=False,root_or_novelty_proven=False,goal_complete=False,
    current_goal_turn='progress',no_progress_streak=0,
    pending=['Finish both actually running new rank16 configurations through the fixed12 EMA and first independent saved-weight reload; do not extend or retry failed training.',
        'Download and verify both new fixed12 logs/weights/native results/stages. Report both same-seed BASE12 and originalP2 deltas, with losses as well as gains.',
        'Allow the reviewed A SiLU/attention first formal-closeout queue to run once after current exact owners finish and actual GPU becomes idle.',
        'A old P2 strict saved-EMA replay mismatch remains failed. Its two fixed-weight controls were never run; any future separate reference design must retain that failure and must not reuse the failed closeout namespace.',
        'Use the completed six Stem outputs to choose a mechanism with distinguishable competing predictions; highpass and whole-Stem differences do not establish a unique root or novelty.'])
for x in r['A_three_native8400_diagnostic_packets_downloaded_verified']:x['offline_all548_tensor_geometry_consumer_pending']=False
write(out/'report.json',r)
lines=['审查后两台远程服务器均已实际运行 rank16 P2；当前第2轮完成，正在继续训练，最多12轮。',
    '种子1：AP10.467838 / APs5.674179；同轮 BASE AP10.370407 / APs5.268915；原P2 AP10.831738 / APs5.827115。',
    '种子2：AP11.296967 / APs6.016536；同轮 BASE AP11.430800 / APs6.238843；原P2 AP10.956307 / APs5.964604。',
    '这是同轮过程读数，不能替代固定第12轮EMA。最终要求两种子APs高于各自BASE12且总体AP不下降。',
    '新增基线训练/前向为0；两台前100批输入与前向前RNG完全一致；当前真实优化更新3800/4700次。',
    '', '六组Stem原生选择/几何/类别APs分析已完成并核验。增强Stem=用户原增强模块；随机原Stem=原生模块随机初始化。']
for seed, comparisons in category['pairs'].items():
    for name,x in comparisons.items():
        lines.append(f'种子{seed} {name}: ΔAP{x["AP"]["native_delta_points"]:+.6f} / ΔAPs{x["APs"]["native_delta_points"]:+.6f}；小类别提升{x["APs"]["improved_classes"]}/10。')
lines+=['','目前能支持：高频输入相对同参数RGB输入，两个种子小类别精度及初始/最终几何支持均改善。',
    '目前不能支持：整个增强Stem稳定提高总体AP，或高频就是唯一根因。种子1总体AP下降，而且两种子增强Stem仍未通过预训练BASE12门槛。',
    '共享物理TopK位置上的初始框也变了；选择位置与初始回归同时变化，不能将全部收益归到覆盖增加。',
    '小目标命中数不是APs贡献；原生评价每图每类maxDets100，模型输出300项。类别贡献只作指标算术分解。',
    '', 'rank16修改是学习式矩阵分解，同时改变秩、参数量和优化路径；不得写成已隔离的纯低秩因果效应。',
    'A旧P2严格EMA重放失败继续保留；没有重跑旧训练或失败重放。',
    'A已有SiLU/注意力12轮训练的首次独立核验已部署队列，等待新rank16所有实际进程结束且GPU空闲后各执行一次。',
    '代码、原生产物、类别分解和共享位置分区的SHA均已核对；新的最终12轮提升、唯一根因和创新均未证明。']
(out/'审查后实验与问题分析.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(out/'manifest.json',dict(files=evidence,report_sha256=sha(out/'report.json'),text_sha256=sha(out/'审查后实验与问题分析.txt'),producer_sha256=sha(__file__)))
state=dict(status=r['status'],canonical_report='../reviewed_remote_tests_actual12_20261011_v5/report.json',canonical_report_sha256=sha(out/'report.json'),
    current_goal_turn='progress',previous_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    all_six_Stem_selector_geometry_category_analyses_complete=True,new_lowrank_actual_training=training,
    A_remaining_user_first_closeout_queue_actually_live=True,A_original_P2_exact_replay_failure_preserved=True,
    two_seed_P2_weight_intervention_complete=False,two_seed_formal_efficacy_proven=False,root_or_novelty_proven=False,
    new_BASE_forwards=0,new_BASE_training_epochs=0,failed_training_retried=False,pending=r['pending'])
for family,num,previous in [('stem_component12_20261011_v1',46,45),('original_stem12_20261011_v1',49,48),
    ('user_backbone12_20261011_v1',23,22),('user_backbone_primary_collection12_20261011_v2',8,7),
    ('p2_subspace_diagnostic12_20261011_v1',6,5),('p2_lowrank12_20261011_v3',2,1)]:
    write(R/f'outputs/{family}/execution_state_v{num}.json',dict(state,supersedes=f'execution_state_v{previous}.json'))
for family in ('native_selection_analysis12_20261011_v5','native_selection_common_location12_20261011_v3','six_Stem_category_APs12_20261011_v1'):
    write(R/f'outputs/{family}/execution_state_v1.json',state)
print(json.dumps(dict(status=r['status'],completed_epochs=[x['completed_epochs'] for x in training],six_actual_Stem_analysis_complete=True,
    actual_optimizer_updates=[x['audit']['actual_optimizer_steps'] for x in training],BASE_forwards=0,goal_complete=False),ensure_ascii=False))
