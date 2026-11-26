from pathlib import Path
import hashlib
import json
from pypdf import PdfReader

ROOT=Path.cwd()
OUT=ROOT/'outputs/stem_new_controls_primary_collection12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


rows=[]
for arm in ('attention_spatial_mean','early_final_downsample'):
    folder=ROOT/f'outputs/stem_component_primary_collection_20261011_v2/seed2_{arm}_v1'
    v=read(folder/'primary_verification_v1.json');p=Path(v['downloaded_packet'])
    assert v['status']=='PASS_COMPONENT_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'
    assert v['all_collected_files_verified'] and v['actual_stage_images']==548 and v['files_verified']==607
    assert v['saved_EMA_independent_replay_verified'] is False and v['formal_terminal_reader_passed'] is False
    assert v['verification_script_sha256']==sha(ROOT/'英文小论文/code/stem_component_primary_verify12_20261011_v2/verify_primary.py')
    pre=read(p/'screen/preflight.json')
    assert pre['status']=='CONTROL_SHAPE_GRADIENT_AND_INITIALIZATION_PASS'
    assert pre['seed']==2 and pre['arm']==arm and pre['all_initial_stem_tensors_match_full']
    assert pre['initialization']['exact_shared_tensors']==754 and pre['initialization']['only_missing_prefix']=='backbone.stem.'
    assert pre['full_model_forwards']==pre['optimizer_updates']==pre['baseline_forwards']==0
    root=ROOT/f'outputs/stem_remaining_primary_analysis12_20261011_v1/seed2_{arm}'
    report=read(root/'report.json')
    for n,h in read(root/'manifest.json')['files'].items():assert sha(root/n)==h
    assert report['analysis_source_sha256']==sha(ROOT/'英文小论文/code/stem_remaining_primary_analysis12_20261011_v1/analyze.py')
    assert report['seed']==2 and report['arm']==arm and report['all10_categories_retained']
    assert report['saved_EMA_independent_replay_verified'] is False and report['formal_terminal_reader_passed'] is False
    assert not report['causal_root_proven'] and not report['novelty_proven']
    small={};ultra={}
    for t in ('0.5','0.75'):
        for name,target in (('COCO_small',small),('COCO_small_input_under4',ultra)):
            r=report['outcomes'][t][name]
            target[t]=dict(GT=r['GT'],supported_GT_by_boundary=r['supported_GT_by_boundary'],
                control_native_all_area_TP=r['control']['counts']['native_TP'],
                full_native_all_area_TP=r['full_enhanced']['counts']['native_TP'],
                native_all_area_TP_net=r['native_all_area_TP_net'],
                gained=r['native_all_area_TP_gained'],lost=r['native_all_area_TP_lost'],
                gained_control_outcomes=r['gained_TP_control_outcomes'],
                gained_first_observed_geometric_support_change=r['gained_TP_first_observed_geometric_support_change'])
    rows.append(dict(seed=2,arm=arm,control_scope=report['control_scope'],control_AP_percent=report['control_AP_percent'],
        full_AP_percent=report['full_AP_percent'],full_minus_control_pp=report['full_minus_control_pp'],
        small=small,under4_small=ultra,ranking_ambiguous_images=report['ambiguous_images'],
        all_initial_stem_tensors_match_full=True,retained_shared_pretrained_tensors=754,
        source_verification_sha256=sha(folder/'primary_verification_v1.json'),
        source_primary_manifest_sha256=v['primary_manifest_sha256'],source_checkpoint_sha256=v['last_checkpoint_sha256'],
        source_module_initialization_preflight_sha256=sha(p/'screen/preflight.json'),
        source_analysis_report_sha256=sha(root/'report.json'),source_analysis_manifest_sha256=sha(root/'manifest.json')))
pairs=read(OUT/'existing_ultratiny_pairs.json')
assert pairs['analysis_source_sha256']==sha(ROOT/'英文小论文/code/stem_remaining_primary_analysis12_20261011_v1/read_existing_ultratiny_pairs.py')
assert len(pairs['seeds'])==2 and pairs['all10_classes_retained_including_empty_groups']
paper=ROOT/'英文小论文/文献_相关原始论文_20261004/14_DEIM_CVPR2025_improved-matching.pdf'
assert sha(paper)=='271cceb7e212f69760d5f49f1ba4b9c607c7e6016d6aa12f555cd979cca83ef8'
doc=PdfReader(paper);assert len(doc.pages)==10 and doc.metadata.title=='DEIM: DETR with Improved Matching for Fast Convergence'
assert 'Matchability' in doc.pages[4].extract_text()
native=ROOT/'outputs/stem_native_supervision_audit12_20261011_v1'
for n,h in read(native/'manifest.json')['files'].items():assert sha(native/n)==h
report=dict(status='NEW_STOPPED_SEED2_PRIMARY12_PACKETS_VERIFIED_AND_SAME_GT_CONTROL_ANALYSIS_COMPLETED',
    analyses=rows,both_seed_existing_native_vs_enhanced_under4=pairs['seeds'],
    under4_pair_report_sha256=sha(OUT/'existing_ultratiny_pairs.json'),
    literature=dict(title=doc.metadata.title,publisher_metadata_URL='https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html',
        local_existing_PDF=str(paper),local_PDF_sha256=sha(paper),read_pages=[1,3,4,5],
        publisher_PDF_fetch_status='403; existing local PDF reused, no repeated search/download',
        learning='DEIM separates sparse O2O supervision from low-quality matches introduced while increasing positives, and validates Dense O2O and MAL. Apply that problem-decomposition discipline: assess coverage and score/retention separately. This is a methodological analogy, not evidence that supervision is our unique cause.',
        already_in_our_native_code='Active MAL uses matched-box IoU**1.5. A generic missing-IoU-supervision claim is contradicted by the actual source.',
        native_supervision_audit_sha256=sha(native/'report.json')),
    changed_next_action='Keep both old controls at the fixed12 endpoint. For later exact-EMA diagnostics separately assess input-under4 GT at IoU50/75, dense location score/selection and selected-box regression. Compare initialization-aware WarmStem and original-user SiLU/P2/aggregation tests against cached same-seed BASE12. Do not infer unique TopK, Stem or classification cause from selected-box support or replace the actual BASE12 gate with a weaker cold-Stem comparison.',
    unique_root_proven=False,novelty_proven=False,formal_BASE12_gate_changed=False,
    other_seed_control_fixed12_still_pending=True,independent_saved_EMA_replay_pending=True,
    new_BASE_epochs=0,new_BASE_forwards=0,new_model_forwards_here=0,new_evaluator_calls_here=0,
    new_optimizer_updates_here=0,new_training_epochs_here=0,goal_complete=False)
write(OUT/'analysis_summary.json',report)
note='''新完成的Stem消融：实际产物与问题定位

种子2的空间注意力均值对照、提前下采样对照均已取回到本地。每组607份文件、548张验证图的阶段数组、原生第12轮EMA主记录和权重均已通过哈希/CRC/身份验证。独立保存EMA重载验证仍待完成。两组对照的初始Stem张量均与完整增强Stem相同，且同样保留754份共享预训练张量；不是在不相同的随机初始化之间随意对比。

固定12轮主记录（AP / APs，百分数）：
完整增强Stem：7.9081 / 5.1606
空间注意力图改为每图空间均值：7.3830 / 4.4432
将第二次stride2前移到融合卷积：6.0622 / 3.9750
完整版本分别高0.5251 / 0.7174和1.8459 / 1.1856个百分点。这些只覆盖种子2；另一种子尚未到第12轮。

同样26,586个小目标，IoU0.5的几何覆盖GT数：
                           初始所选300个回归框    第一层更新后    最终查询框
完整增强Stem                         8029             11000          12189
空间注意力均值对照                   6985             10249          11301
提前下采样对照                       5207              9024          10374
相同GT的原生全尺寸匹配正确检出数，完整版本相对两组分别净增660和1542。
几何覆盖会允许多个GT共用一个框，因此不能把覆盖GT数当作正确检出数或APs。
最早可观察的差异已出现在初始300个回归框；该节点已经包含位置选择和框回归两步，仍不能唯一断定哪个环节负责。

需要保留的负结果：输入短边不足4像素的5,096个小目标，在IoU0.75时，完整版本TP=26，空间注意力均值对照TP=30，提前下采样对照TP=20。完整版本没有在所有细分任务上改善。
同一子群，已保存的预训练原始BASE12与直接替换增强Stem的TP：
                       IoU0.5（BASE / 增强）    IoU0.75（BASE / 增强）
种子1                         899 / 355                    72 / 25
种子2                         793 / 529                    59 / 26
原始BASE的初始框在IoU0.75也仅覆盖2个GT/种子，随后解码器能恢复一部分；直接替换增强Stem的初始框为0，最终恢复仍少于BASE。两者还存在原生Stem预训练被移除的差异，因此这不是纯结构因果估计。

从DEIM（CVPR2025，现有本地PDF已阅读）借鉴的是问题分解：先区分监督数量与低质量匹配，再分别验证对应干预。我们的代码已经采用MAL及IoU的1.5次方分类目标；不能以“缺乏质量监督”为理由重复增加同类损失。

下一步：等待另一种子的同终点结果，检查保留预训练的增强Stem以及原始SiLU/P2/主干聚合替换；精确EMA诊断中单列不足4像素目标和IoU0.75，继续区分选入率、入选框回归、最终分类保留。全部10类、模糊排序情况和负结果均保留。最终仍按两种子APs严格超过原始BASE12、总体AP不下降判定。
''' 
(OUT/'新消融产物与定位说明.txt').write_text(note,encoding='utf8')
write(OUT/'analysis_summary_manifest.json',dict(files={n:sha(OUT/n) for n in ('analysis_summary.json','新消融产物与定位说明.txt','existing_ultratiny_pairs.json')},
    summary_source_sha256=sha(__file__)))
print(json.dumps(dict(status=report['status'],verified_packets=2,files_each=607,stage_images_each=548,
    seed2_initial_small50_support={'full':8029,'mean_control':6985,'early_control':5207},
    under4_75_TP={'BASE':59,'full':26,'mean_control':30,'early_control':20},other_seed12_pending=True,saved_EMA_replay_pending=True)))
