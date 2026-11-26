"""Test literal stage explanations against already saved actual outputs."""
from pathlib import Path
import hashlib,json,datetime,math
import numpy as np
R=Path(__file__).resolve().parent.parent;OUT=R/'outputs/Stem_stage_hypotheses12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence={}
def record(p):
    p=Path(p);evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p));return read(p)
def verify(folder):
    m=record(folder/'manifest.json')
    for n,h in m['files'].items():assert sha(folder/n)==h,n
def values(path):
    evidence[path.relative_to(R).as_posix()]=dict(bytes=path.stat().st_size,sha256=sha(path))
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
six=R/'outputs/native_selection_analysis12_20261011_v5/actual_six';verify(six);analysis=record(six/'report.json')
cf=R/'outputs/six_Stem_category_APs12_20261011_v1';cat=record(cf/'report.json');cm=record(cf/'manifest.json');assert sha(cf/'report.json')==cm['report_sha256']
rows=[]
for seed in (1,2):
    commonroot=R/f'outputs/native_selection_common_location12_20261011_v3/seed{seed}';verify(commonroot)
    common=record(commonroot/'report.json');v=common['comparisons']['highfreq_rgb_to_original_stem']['groups']['COCO_small']
    full=values(six/f'original_stem_seed{seed}/per_GT.npz');rgb=values(six/f'highfreq_rgb_seed{seed}/per_GT.npz')
    for k in ('annotation_id','category_id','image_id','original_area'):assert np.array_equal(full[k],rgb[k])
    small=full['original_area']<=1024;assert int(small.sum())==26586
    gained75=small&~rgb['native_TP'][:,5]&full['native_TP'][:,5]
    noinitial=gained75&(full['max_iou'][:,0]<.75)
    assert int(gained75.sum())==v['0.75']['native_TP_gained']
    assert int(noinitial.sum())==v['0.75']['gained_native_TP_by_full_initial_source']['none']
    assert np.all(full['max_iou'][noinitial,2]>=.75)
    pre=noinitial&(full['max_iou'][:,1]>=.75)
    final=noinitial&(full['max_iou'][:,1]<.75)
    assert np.array_equal(pre|final,noinitial)
    first_stage_counts=dict(saved_pre_boundary=int(pre.sum()),saved_final_boundary=int(final.sum()))
    initial_gain=v['0.5']['initial_support_gained'];shared_gain=v['0.5']['initial_gains_with_full_common_selected_location_support']
    initial_net=int((small&(full['max_iou'][:,0]>=.5)).sum())-int((small&(rgb['max_iou'][:,0]>=.5)).sum())
    final_net=int((small&(full['max_iou'][:,2]>=.5)).sum())-int((small&(rgb['max_iou'][:,2]>=.5)).sum())
    assert initial_net==v['0.5']['initial_support_gained']-v['0.5']['initial_support_lost']
    arithmetic=cat['pairs'][str(seed)]['highfreq_rgb_to_original_stem']
    rows.append(dict(seed=seed,small_GT=26586,AP_delta_points=arithmetic['AP']['native_delta_points'],APs_delta_points=arithmetic['APs']['native_delta_points'],
        initial_IoU50_support_net_change=initial_net,final_IoU50_support_net_change=final_net,
        initial_IoU50_support_gained=initial_gain,initial_support_gains_at_shared_selected_sites=shared_gain,
        shared_site_fraction_of_initial_support_gains=shared_gain/initial_gain,
        native_all_area_small_GT_TP75_gained=int(gained75.sum()),gained_TP75_without_full_initial75=int(noinitial.sum()),
        fraction_gained_TP75_requiring_later_refinement=int(noinitial.sum())/int(gained75.sum()),
        first_saved_boundary_reaching75_among_these_gains=first_stage_counts,
        spatial_join_scope='A shared physical feature site is not the same feature vector or query identity; the actual EMA priors at these sites match exactly.'))
reasoning=R/'outputs/sod_problem_reasoning_20261010/paper_reasoning.txt'
evidence[reasoning.relative_to(R).as_posix()]=dict(bytes=reasoning.stat().st_size,sha256=sha(reasoning))
papers=[]
for identifier,path,lesson in [
    ('DEIM_CVPR2025','英文小论文/文献_相关原始论文_20261004/14_DEIM_CVPR2025_improved-matching.pdf',
     'Transfer the separation of quantity and quality: actual TopK sites, geometric support and native PR quality are separate measurements; Dense O2O/MAL already belong to our baseline.'),
    ('SET_CVPR2025','英文小论文/文献_相关原始论文_20261004/55_SET_CVPR2025.pdf',
     'Transfer directional information controls and foreground/background distinction. A raw highpass-versus-RGB difference does not by itself isolate useful frequency information or its location.')]:
    p=R/path;evidence[path]=dict(bytes=p.stat().st_size,sha256=sha(p));papers.append(dict(id=identifier,local_file=str(p),sha256=sha(p),migration_inference=lesson))
OUT.mkdir(exist_ok=False)
report=dict(status='TWO_SEED_SAVED_STAGE_EXPLANATIONS_CHECKED_WITH_LATER_REFINEMENT_COHORTS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    comparison='Highpass x-avg_pool3(x) minus same-parameter raw RGB control within the enhanced Stem',actual_rows=rows,
    hypotheses=[
      dict(statement='All observed differences consist only of changed selected spatial locations.',outcome='Contradicted as a literal description: same-site regression offsets differ and both seeds have initial gains supported at shared selected sites.',
           boundary='This is not a controlled isolation of backbone versus encoder versus bbox-head parameters; their learned features all differ.'),
      dict(statement='All gained native TP75 were already supported by the enhanced initial boxes at IoU75.',outcome='Contradicted: 826/937 and 1053/1239 gained small-GT TP75 had no initial75 support and do have final75 support.',
           boundary='This proves later geometric refinement is required for those particular observed predictions; it does not prove modified decoder weights caused the cross-model gain or exclude better initial50 features.'),
      dict(statement='The difference is only a final confidence-ranking change, with all saved box support unchanged.',outcome='Contradicted as a literal output description: initial50 and final50 geometric support increase in both seeds.',
           boundary='This does not isolate how much APs comes from ranking versus geometry; matching, false positives and the whole PR curve also change.'),
      dict(statement='The highpass control establishes a unique useful-high-frequency root.',outcome='Unproven: the input substitution also changes amplitude/mean/boundary content and trained downstream features.',
           boundary='Do not add another frequency module or claim novelty merely from the two positive APs deltas.'),
      dict(statement='A learned rank16 P2 bridge improves by eliminating irrelevant directions.',outcome='Pending: the new two-seed12 runs change rank, parameterization, parameter count and optimization. SVD16-minus-DC16 controls only conditional direction sensitivity of an already trained model.',
           boundary='Weight norm matching does not match feature/output energy or operator error. The failed seed1 parent replay remains failed, and no fresh parent/BASE forward is allowed.')],
    source_papers_read_and_previously_verified=papers,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,new_optimizer_updates=0,
    root_proven=False,novelty_proven=False,formal_training_efficacy_proven=False,
    next_decision='Keep both rank16 configurations through fixed12. Use saved outputs plus the two previously unattempted direction controls to test selection versus geometric-refinement predictions. If formal gates fail, stop those configurations at12 and retain losses; do not extend them.',
    metric_scope='Native all-area matching TP flags filtered to small GT are observational cohorts, not native small-area APs arithmetic contributions.')
with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['Stem的改进最早在哪里表现出来：六组实际产物后的两种子检验','',
       '比较：同参数增强Stem中的高通输入 x−avg_pool3(x)，与原始RGB输入。这里只读已核验产物，没有新增模型/评价/训练。','']
for x in rows:
    lines.append(f'种子{x["seed"]}：APs{x["APs_delta_points"]:+.6f}，AP{x["AP_delta_points"]:+.6f}。')
    lines.append(f'初始IoU≥0.5支持净增{x["initial_IoU50_support_net_change"]}；初始新获支持{x["initial_IoU50_support_gained"]}中，共同入选位置提供支持{x["initial_support_gains_at_shared_selected_sites"]}（{x["shared_site_fraction_of_initial_support_gains"]:.2%}）。')
    c=x['first_saved_boundary_reaching75_among_these_gains']
    lines.append(f'新增最终TP75共{x["native_all_area_small_GT_TP75_gained"]}；其中{x["gained_TP75_without_full_initial75"]}在初始框没有IoU75支持（{x["fraction_gained_TP75_requiring_later_refinement"]:.2%}），后续首次在已保存pre边界达到75：{c["saved_pre_boundary"]}，在已保存final边界达到75：{c["saved_final_boundary"]}。\n')
lines+=['可以排除的字面解释：“变化只有入选位置”“新增最终75命中初始已全部覆盖”“只有最终分数变化而框没变”。',
    '尚不能定位的原因：Stem、主干、encoder、回归头、decoder之间的参数因果贡献；最终APs不能由支持/命中人数换算。',
    '同物理位置不是同特征/同query；初始75缺支持却最终达到75，只证明该预测需要后续细化，不证明改了decoder才带来跨模型收益。',
    '', '从DEIM学习：数量与质量分开测量，不能把更多位置与更好监督/更高AP当成同件事。',
    '从SET学习：用有方向的删除/替换检验，并区分目标和背景；现有高通对照仍没有隔离幅度与频率。',
    '', '下一步：现有rank16两种子固定12轮；A此前未执行的SVD16/DC16方向对照各一次；保留原严格重放失败。',
    'rank16训练如果失败，不加轮、不重试、不解释成已证明去冗余；根据真实阶段损失继续选择下一机制。']
(OUT/'问题解释检验.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
with (OUT/'manifest.json').open('x',encoding='utf8') as f:json.dump(dict(files=evidence,report_sha256=sha(OUT/'report.json'),text_sha256=sha(OUT/'问题解释检验.txt'),producer_sha256=sha(__file__)),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=report['status'],rows=rows,new_model_forwards=0),ensure_ascii=False))
