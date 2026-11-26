"""Bind completed single-seed scientific analyses to reviewed actual remote tests."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence={}
def record(p):
    p=Path(p);evidence[p.relative_to(ROOT).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p));return read(p)
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def bound_manifest(folder):
    m=record(folder/'manifest.json')
    for n,h in m['files'].items():
        if isinstance(h,dict):h=h['sha256']
        assert sha(folder/n)==h,(folder,n)
    return record(folder/'report.json')

previous=ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v1'
old=record(previous/'report.json');old_manifest=record(previous/'manifest.json')
assert sha(previous/'report.json')==old_manifest['report_sha256']
for n,v in old_manifest['files'].items():assert sha(ROOT/n)==v['sha256'],n
assert old['goal_complete'] is False
captures=record(ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v2_inputs.json');captured={}
for n,v in captures.items():
    raw=v['payload'].encode();assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256']
    captured[n]=json.loads(raw)
A=captured['current_A_reviewed_experiments_status12']
assert A['seed']==1 and A['user']['backbone_attention']['completed_epochs']==11 and A['user']['backbone_attention']['trainer_live']
assert not A['native_selection']['failure'] and not A['P2']['failure']
fixtures=record(ROOT/'outputs/native_selection_analysis12_20261011_v4_CPU/verification.json')
buffers=record(ROOT/'outputs/native_selection_analysis12_20261011_v4_CPU/checkpoint_buffer_fixtures.json')
descriptions={}
for arm in ('original_stem','native_stem_scratch','highfreq_rgb'):
    d=bound_manifest(ROOT/f'outputs/native_selection_seed2_analysis12_20261011_v1/{arm}')
    assert d['status']=='SEED2_SINGLE_ARM_ACTUAL_SELECTOR_AND_SAVED_GEOMETRY_DESCRIPTION'
    descriptions[arm]=dict(status=d['status'],report_sha256=evidence[f'outputs/native_selection_seed2_analysis12_20261011_v1/{arm}/report.json']['sha256'])
common=bound_manifest(ROOT/'outputs/native_selection_seed2_common_location12_20261011_v2')
old_common=record(ROOT/'outputs/native_selection_seed2_common_location12_20261011_v1/report.json')
metadata_failure=record(ROOT/'outputs/common_location_manifest12_20261011_v1_failure.json')
assert metadata_failure['status']=='V1_METADATA_SELF_HASH_REJECTED_NO_MODEL_FAILURE'
assert {k:v for k,v in common.items() if k not in ('source_sha256','protocol_sha256')}=={k:v for k,v in old_common.items() if k not in ('source_sha256','protocol_sha256')}
for name in ('native_stem_scratch_partitions.npz','highfreq_rgb_partitions.npz'):
    with np.load(ROOT/'outputs/native_selection_seed2_common_location12_20261011_v1'/name,allow_pickle=False) as x, np.load(ROOT/'outputs/native_selection_seed2_common_location12_20261011_v2'/name,allow_pickle=False) as y:
        assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
assert common['status']=='SEED2_SHARED_PHYSICAL_LOCATION_INITIAL_REFERENCE_AND_NATIVE_TP_PARTITIONS'
assert common['images']==548 and common['small_GT']==26586 and common['all_three_EMA_anchor_buffers_equal']
assert common['new_model_forwards']==common['new_evaluator_calls']==common['new_training_epochs']==0
P2=bound_manifest(ROOT/'outputs/p2_subspace_geometry_class12_20261011_v1')
assert P2['status']=='ACTUAL_P2_RANK16_INTERVENTIONS_GEOMETRY_AND_RELATIVE_CLASS_COHORTS'
assert P2['images']==548 and P2['small_GT']==26586
assert P2['new_model_forwards']==P2['new_BASE_forwards']==P2['new_evaluator_calls']==P2['new_native_matching_calls']==P2['new_training_epochs']==0
for family in ('native_selection_analysis12_20261011_v4','native_selection_seed2_analysis12_20261011_v1','native_selection_seed2_common_location12_20261011_v2','p2_subspace_geometry_class12_20261011_v1'):
    folder=ROOT/'英文小论文/code'/family;spec=record(folder/'protocol.json')
    if 'reader_sha256' in spec:
        name='run.py' if (folder/'run.py').exists() else 'analyze.py';assert sha(folder/name)==spec['reader_sha256']
    for n,h in spec.get('dependencies_sha256',{}).items():assert sha(ROOT/n)==h,n

stem_small={n:c['groups']['COCO_small'] for n,c in common['comparisons'].items()}
p2_small={n:c['COCO_small'] for n,c in P2['comparisons'].items()}
assert stem_small['native_stem_scratch_to_original_stem']['0.5']['initial_support_gained']==2910
assert stem_small['highfreq_rgb_to_original_stem']['0.5']['initial_gains_with_full_common_selected_location_support']==1756
assert p2_small['svd16_norm']['0.5']['supported_by_boundary']['encoder_initial_boxes']['delta']==370
assert p2_small['svd16_norm']['0.75']['gained_TP_parent_no_final_support']==285
out=ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v2';out.mkdir(exist_ok=False)
report=dict(status='REVIEWED_REMOTE_TESTS_WITH_ACTUAL_SINGLE_SEED_FAILURE_LOCATION_ANALYSES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    previous_immutable_report_sha256=sha(previous/'report.json'),user_fixed12_results=old['user_fixed12_results'],seed1_current_remote_observation=A,
    seed2_three_individual_selection_consumers=descriptions,seed2_common_physical_location_small_GT_comparisons=stem_small,
    seed2_original_selection_level_counts=common['level_selection_counts'],seed2_P2_rank16_small_GT_geometry_and_class_comparisons=p2_small,
    seed2_P2_fixed_weight_runtime_percent={k:v['overall_percent'] for k,v in P2['runs'].items()},
    strict_six_record_EMA_bound_consumer_CPU_review_passed=True,strict_six_record_actual_two_seed_analysis_complete=False,
    metadata_manifest_v1_failure_preserved=True,corrected_manifest_v2_all_saved_array_joins_recomputed=True,corrected_numeric_arrays_and_all_cohort_counts_exactly_equal_v1=True,
    single_seed_findings=[
      'Enhanced Stem selects nearby grid sites for647 fewer smallGT than random original Stem but supplies1687 more initial IoU50-supported smallGT. Simple nearby-site count does not explain the observed benefit.',
      'Compared with HF_RGB,1756 of2884 newly initial-IoU50-supported smallGT also have support from locations selected by both models. Same physical location and EMA prior, changed regressed initial box. This localizes some observable gains before Decoder without isolating backbone/Encoder/regression-head learning.',
      'The rank16 norm-matched SVD P2 fixed-weight intervention adds370 initial and261 final IoU50-supported smallGT. Among319 new IoU75 nativeTP,285 lacked final geometric support in the parent. Small relative-class improvements also occur; changes are not exclusively geometric.',
      'The mean-direction control loses2459 final IoU75-supported smallGT;2788 of2854 lost nativeTP lack final geometric support in that intervention. Equal parameter-weight norm does not match feature energy or approximation error.'
    ],
    limitations=['Single seed descriptive analyses and fixed-weight interventions, no trained low-rank mechanism or two-seed robustness proved.',
      'No cross-model query-slot identity; only GT identity and physical grid-location joins.',
      'Native all-area GT matching stratified by size does not decompose APs. Counts are not contributions to AP.',
      'No dense8400 regressed boxes: nearby-site selection is a proxy, not dense proposal recall.',
      'EMA buffer drift explains failed nominal-grid verification, not AP variation.'],
    next_training_hypothesis='Constrain the P2 bridge to a learned rank16 spatial mixing while retaining zero-output initialization. A repeated fixed-weight result is necessary before committing a new training protocol; this is a hypothesis, not an innovation or efficacy claim.',
    pending=['A authorized attention11/12 must finish at fixed12; do not extend or restart.',
      'A native-owned formal replays and then three native selection diagnostics remain pending.',
      'A already-reviewed two P2 runtime controls: actual CPU passed, exact previous-owner barrier remains live.',
      'Only after both-seed runtime controls and failure-location evidence review decide the next distinct12-epoch configuration.'],
    new_training_configurations_started_this_turn=0,new_model_forwards_in_local_analyses=0,new_BASE_forwards=0,new_BASE_training_epochs=0,
    existing_process_signals=0,failed_training_retried=False,root_or_novelty_proven=False,two_seed_formal_efficacy_complete=False,
    goal_complete=False,current_goal_turn='progress',no_progress_streak=0)
write(out/'report.json',report)
lines=['远程实测与问题定位：A远程UTC '+A['UTC'],'','固定第12轮EMA；复用既有BASE12；没有重跑基线。','服务器/改动 | AP | APs | ΔAP | ΔAPs']
for r in report['user_fixed12_results']:
    lines.append(f"{'A' if r['seed']==1 else 'B'}/{r['arm']} | {r['candidate12']['AP']:.4f} | {r['candidate12']['APs']:.4f} | {r['delta_points']['AP']:+.4f} | {r['delta_points']['APs']:+.4f}")
lines+=['','B：P2同一第12轮EMA权重上的完整548图对照。','原P2 AP14.2248/APs9.0044；局部均值16方向 AP11.0691/APs5.8166；最强16学习方向 AP14.2927/APs9.1255。','两个干预只改变restore权重，同秩16、同权重范数；未匹配特征能量和算子近似误差。','','错误首次出现位置：','最强16方向相对原P2：初始参考框多覆盖370个小目标(IoU50)，第一层后多342个，最终多261个。','IoU75新增319个原生TP中，285个在原P2没有达到该IoU的最终框；分类相对次序也有变化。','局部均值对照：IoU75最终覆盖少2459个；丢失2854个原生TP中2788个已无合格最终框。','因此下一步应优先验证P2早期学习到的空间混合是否改善定位；不能把所有提升归因于分类或注意力。','','增强Stem：与随机原Stem比，附近有入选位置的小目标少647个，初始IoU50框支持却多1687个。','与高频RGB对照比，新增2884个初始框支持，其中1756个在双方都选中的同一物理网格位置已有更准确初始框。','说明部分收益在Decoder前就出现，尚不能区分主干、Encoder或回归头的学习贡献。','增强Stem仍低于有预训练的BASE12；高于随机原Stem不能替代对正式基线的提升。','','这些是B单种子实际产物分析及固定权重干预，不是新的两种子训练成功，也不是APs贡献分解。','A注意力11/12；原有任务继续，达到12即停止。A同样两组P2对照已审查部署并通过CPU，等待前序正式重放和诊断。','新训练方向暂定：学习到的低秩空间混合；先等A同方案复核，再确定从共同初始化开始、每配置最多12轮的新训练。']
(out/'审查与远程实测结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(out/'manifest.json',dict(files=evidence,report_sha256=sha(out/'report.json'),text_sha256=sha(out/'审查与远程实测结果.txt'),producer_sha256=sha(__file__)))
state=dict(status='ACTIVE_ACTUAL_SINGLE_SEED_FAILURE_LOCATION_ANALYSIS_AND_PENDING_REPLICATE',current_goal_turn='progress',previous_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    canonical_report='../reviewed_remote_tests_actual12_20261011_v2/report.json',canonical_report_sha256=sha(out/'report.json'),latest_A_training=A,
    B_three_selection_descriptive_consumers_passed=True,B_common_location_initial_reference_analysis_completed=True,B_P2_geometry_class_analysis_completed=True,
    full_six_record_actual_analysis_complete=False,seed1_P2_controls_CPU_passed=True,seed1_P2_controls_GPU_status='waiting_exact_prior_owner_and_positive_saved_EMA',
    new_training_configurations=0,new_BASE_forwards=0,new_BASE_training_epochs=0,original_training_source_files_changed=False,existing_process_signals=0,
    two_seed_formal_efficacy_proven=False,root_or_novelty_proven=False,pending=report['pending'])
for family,number,prev in [('stem_component12_20261011_v1',43,42),('original_stem12_20261011_v1',46,45),('user_backbone12_20261011_v1',20,19),('user_backbone_primary_collection12_20261011_v2',5,4),('p2_subspace_diagnostic12_20261011_v1',3,2)]:
    write(ROOT/f'outputs/{family}/execution_state_v{number}.json',dict(state,supersedes=f'execution_state_v{prev}.json'))
print(json.dumps(dict(status=report['status'],A_attention_epochs=A['user']['backbone_attention']['completed_epochs'],new_training_configurations=0,goal_complete=False)))
