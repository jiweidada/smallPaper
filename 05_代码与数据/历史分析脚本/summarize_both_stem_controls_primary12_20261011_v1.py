from pathlib import Path
import ast,hashlib,json

ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=ROOT/'outputs/stem_both_controls_primary_review12_20261011_v1';root.mkdir(exist_ok=False)
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
native=ROOT/'英文小论文/code/p2_candidate_seed1_full132_20261007_v1/engine/backbone/hgnetv2.py'
assert sha(native)=='5e107a7b387bcdffab8f834b9c9d5df3e4c848079d62eda5bf2a0c22088bb0da'
tree=ast.parse(native.read_text(encoding='utf8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StemBlock')
init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
strides={}
for n in ast.walk(init):
    if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call):
        for target in n.targets:
            if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self':
                kw={k.arg:ast.literal_eval(k.value) for k in n.value.keywords if k.arg=='stride'}
                if 'stride' in kw:strides[target.attr]=kw['stride']
assert strides==dict(stem1=2,stem2a=1,stem2b=1,pool=1,stem3=2,stem4=1)
rows=[]
for seed in (1,2):
    for arm in ('attention_spatial_mean','early_final_downsample'):
        primary=ROOT/f'outputs/stem_component_primary_collection_20261011_v2/seed{seed}_{arm}_v1'
        v=read(primary/'primary_verification_v1.json');packet=Path(v['downloaded_packet'])
        assert v['status']=='PASS_COMPONENT_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS' and v['files_verified']==607
        assert v['actual_stage_images']==548 and v['all_collected_files_verified'] and not v['formal_terminal_reader_passed']
        assert v['verification_script_sha256']==sha(ROOT/'英文小论文/code/stem_component_primary_verify12_20261011_v2/verify_primary.py')
        pre=read(packet/'screen/preflight.json')
        assert pre['seed']==seed and pre['arm']==arm and pre['all_initial_stem_tensors_match_full']
        assert pre['initialization']['exact_shared_tensors']==754 and pre['initialization']['parameters']==10237322
        analysis=ROOT/f'outputs/stem_remaining_primary_analysis12_20261011_v1/seed{seed}_{arm}'
        for n,h in read(analysis/'manifest.json')['files'].items():assert sha(analysis/n)==h
        d=read(analysis/'report.json');assert d['analysis_source_sha256']==sha(ROOT/'英文小论文/code/stem_remaining_primary_analysis12_20261011_v1/analyze.py')
        assert d['seed']==seed and d['arm']==arm and d['all10_categories_retained'] and d['all38759_GT']
        assert not d['score_mismatch_images'] and d['native_TP_proxy_conflicts']==0
        cohorts={}
        for name in ('COCO_small','COCO_small_input_under4'):
            cohorts[name]={}
            for t in ('0.5','0.75'):
                c=d['outcomes'][t][name]
                cohorts[name][t]=dict(GT=c['GT'],supported_GT_by_boundary=c['supported_GT_by_boundary'],
                    control_native_all_area_TP=c['control']['counts']['native_TP'],full_native_all_area_TP=c['full_enhanced']['counts']['native_TP'],
                    native_all_area_TP_gained=c['native_all_area_TP_gained'],native_all_area_TP_lost=c['native_all_area_TP_lost'],native_all_area_TP_net=c['native_all_area_TP_net'])
        rows.append(dict(seed=seed,arm=arm,control_scope=d['control_scope'],control_AP_percent=d['control_AP_percent'],full_AP_percent=d['full_AP_percent'],
            full_minus_control_points=d['full_minus_control_pp'],cohorts=cohorts,ranking_ambiguous_images=d['ambiguous_images'],
            identical_initial_Stem_tensors=True,parameters=pre['initialization']['parameters'],
            source_analysis_manifest_sha256=sha(analysis/'manifest.json'),source_analysis_report_sha256=sha(analysis/'report.json'),
            source_primary_verification_sha256=sha(primary/'primary_verification_v1.json'),source_primary_manifest_sha256=v['primary_manifest_sha256'],checkpoint_sha256=v['last_checkpoint_sha256']))
randomroot=ROOT/'outputs/stem_random_primary_collection12_20261011_v1'
for n,h in read(randomroot/'analysis_summary_manifest.json')['files'].items():assert sha(randomroot/n)==h
random=read(randomroot/'analysis_summary.json')
assert random['source_sha256']==sha(ROOT/'英文小论文/code/stem_random_primary_analysis12_20261011_v1/summarize.py')
assert len(random['seeds'])==2 and random['verified_new_primary_files_each']==606
for seed in (1,2):
    p=ROOT/f'outputs/stem_primary_collection_20261011_v1/seed{seed}_native_stem_scratch_v1/extracted/code/native_backbone_source.txt'
    assert sha(p)==sha(native)
late=[r for r in rows if r['arm']=='early_final_downsample'];attention=[r for r in rows if r['arm']=='attention_spatial_mean']
assert len(late)==len(attention)==2
assert all(r['full_minus_control_points']['AP']>0 and r['full_minus_control_points']['APs']>0 for r in late)
assert attention[0]['full_minus_control_points']['APs']<0 and attention[1]['full_minus_control_points']['APs']>0
report=dict(status='BOTH_SEEDS_ATTENTION_AND_DOWNSAMPLE_PRIMARY12_PACKETS_VERIFIED_AND_SAME_GT_ANALYZED',controls=rows,
    random_native_vs_enhanced_summary='../stem_random_primary_collection12_20261011_v1/analysis_summary.json',
    random_native_summary_sha256=sha(randomroot/'analysis_summary.json'),native_Stem_code_audit=dict(path=str(native),sha256=sha(native),strides=strides,
        finding='Native HGNetv2 already performs its second stride2 after branch concatenation. Enhanced delayed-versus-early fusion tests additional high-resolution processing within its own design; it does not by itself identify the cause of enhanced-versus-native gain.'),
    attention_spatial_variation_benefit_consistent_across_seeds=False,delayed_second_downsample_net_primary_AP_and_APs_positive_both_seeds=True,
    all_control_initial_Stem_states_identical_to_full=True,all_control_parameters_identical_to_full=True,
    sole_anti_aliasing_or_information_preservation_cause_proven=False,independent_saved_EMA_replay_pending=True,formal_terminal_reader_passed=False,
    other_seed_control_endpoint_pending=False,all10_categories_retained=True,cohort_matching='Same GT IDs, not cross-run query or box IDs; small cohorts use all-area native matching flags.',
    first_observed_boundary='Only selected300 REGRESSED encoder boxes; location selection and box regression are both upstream of this boundary.',
    changed_next_action='Prioritize pretrained-preserving evaluation and original-user SiLU/P2 tests. Treat delayed high-resolution fusion as an internally supported design choice, not an established advantage over native Stem. Do not claim spatial attention gives consistent gain. Use exact selection diagnostics to separate locations, selected-box regression and final ranking before selecting a new mechanism.',
    new_primary_packets_downloaded_and_verified_this_turn=4,all_four_component_primary_packets_now_verified=True,
    new_BASE_forwards=0,new_model_forwards_from_analysis=0,new_evaluator_calls_from_analysis=0,new_optimizer_updates_from_analysis=0,new_training_epochs_from_analysis=0,
    root_proven=False,novelty_proven=False,goal_complete=False,source_sha256=sha(__file__))
write(root/'report.json',report)
lines=['Stem两种子消融审查：第12轮EMA主记录','',
    '两项控制的两种子完整产物已下载并核验；每组607个文件、548张验证图像。独立保存EMA重载复验仍未完成。',
    '所有控制的初始Stem张量、参数数量均与完整增强版本一致；各组均保留754个非Stem共享预训练状态。','',
    '完整增强 − 控制，单位为AP百分点：','种子 | 控制 | ΔAP | ΔAPs']
for r in rows:lines.append(f"{r['seed']} | {r['arm']} | {r['full_minus_control_points']['AP']:+.6f} | {r['full_minus_control_points']['APs']:+.6f}")
lines += ['', '空间注意力：收益方向不一致。种子1完整版本相对空间均值控制的小目标IoU0.5初始覆盖+12，但第一层之后−400、最终框−330，最终正确检出−160；不能只用初始覆盖讲故事。',
    '下采样位置：完整版本相对提前下采样，在两个种子的AP/APs、小目标IoU0.5/0.75覆盖和正确检出数上均有正差值。它支持“增强Stem内先做更多融合再做第二次下采样”这一设计选择。',
    '代码审查边界：原始HGNetv2同样在分支拼接后执行第二次下采样。上述控制不是原始Stem，因此不能直接把增强相对原始的全部提升归因于渐进下采样；移动stride也会改变感受野和计算位置，不能唯一归因于抗混叠。','',
    '原始随机Stem对照：完整增强相对随机原始的ΔAP/ΔAPs分别为种子1 −1.167292/+0.158207、种子2 +1.797922/+1.829522。两种子APs上升，但总体AP方向不一致。',
    '种子1的原生small面积专用评估在IoU0.5下TP从9083降为8951，但小目标AP提高0.278024；AP、TP数量和几何覆盖须分开分析。完整类别/IoU贡献见随机对照报告。',
    '完整增强仍低于两个种子的预训练BASE12，尚未通过目标验收。保留预训练并加入增强残差的Warm版本、原始SiLU/P2/主干聚合替换继续按每配置最多12轮测试。',
    '空间均值控制仍有可学习注意力及每图平均增益，并非删除整个注意力模块；所有10类别、IoU0.5/0.75、极小目标子群和排序模糊项都保留。',
    '下一步：先看保留预训练的实验和原始SiLU/P2终点；用真实dense位置/TopK选择与选中框回归诊断继续定位，不从300个回归框倒推唯一位置选择问题。']
note=root/'两种子Stem消融结论.txt'
with note.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
write(root/'manifest.json',dict(files={p.name:sha(p) for p in (root/'report.json',note)},source_sha256=sha(__file__)))
print(json.dumps(dict(status=report['status'],controls=[dict(seed=r['seed'],arm=r['arm'],delta=r['full_minus_control_points']) for r in rows],native_post_concat_downsample_confirmed=True,goal_complete=False)))
