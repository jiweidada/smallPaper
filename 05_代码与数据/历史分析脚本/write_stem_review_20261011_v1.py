"""Human-readable review derived from actual primary endpoints and GT joins."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report_path=ROOT/'outputs/stem_primary_joint_analysis12_20261011_v1/report.json'
r=read(report_path)
comparison_path=ROOT/'outputs/stem_component12_20261011_v1/same_epoch_random_native_comparison_v7.json'
pairs=read(comparison_path)['current_S_pairs']
progress_path=ROOT/'outputs/stem_component12_20261011_v1/progress_v26.json'
progress=read(progress_path)
out=ROOT/'outputs/stem_review_20261011_v1';out.mkdir(exist_ok=False)
lines=['本轮 Stem 实验结果与问题定位（2026-10-11）','',
'一、已完成的真实实验','完整增强 Stem 的两个种子均完成第 12 轮。以下采用训练时 EMA 验证；独立读取最后权重的复验仍待原有协调程序完成。基线复用既有结果。']
for row in r['seeds']:
    a=row['native_primary_AP_percent'];b=row['cached_BASE12_AP_percent']
    lines.append(f"种子 {row['seed']}：基线 AP={b['AP']:.3f}，APs={b['APs']:.3f}；完整增强 AP={a['AP']:.3f}，APs={a['APs']:.3f}。")
lines+=['','二、增强与随机原始 Stem 的同轮比较','两边仅 Stem 随机初始化，Stem 外共享既有权重；这项比较仍未到两个种子的第 12 轮。']
for x in pairs:
    lines.append(f"种子 {x['seed']}，第 {x['epoch']} 轮：随机 AP={x['random_native_AP']:.3f} / APs={x['random_native_APs']:.3f}；增强 AP={x['enhanced_AP']:.3f} / APs={x['enhanced_APs']:.3f}；增益 AP={x['delta_AP_pp']:+.3f} / APs={x['delta_APs_pp']:+.3f} 个百分点。")
lines+=['不能从一个种子的正结果认定稳定提升，也不能因没有超过带预训练 Stem 的基线就认定结构完全无效。','',
'三、分支作用已有的终点数字（仍为训练时验证）']
full=next(x for x in progress['rows'] if x['seed']==2 and x['arm']=='original_stem' and x['epoch']==12)
for arm,label in (('detail_lowres','细节分支降到低分辨率'),('highfreq_rgb','高频分支输入换成 RGB')):
    x=next(x for x in progress['rows'] if x['seed']==2 and x['arm']==arm and x['epoch']==12)
    lines.append(f"种子 2，相对“{label}”对照：完整增强 AP 增加 {full['AP']-x['AP']:.3f}，APs 增加 {full['APs']-x['APs']:.3f} 个百分点。")
lines+=['这些是当前完整结构下的条件贡献；另一种子及剩余注意力、下采样对照尚未全部完成。早下采样对照也会改变后续有效感受野，不能只归因于采样时机。','',
'四、从真实产物发现的问题','已取回两组最后权重及每组 548 张验证图的阶段数组。全部 38,759 个有效 GT 按标注 ID、图像 ID、类别和面积对齐。COCO-small 共 26,586 个；下述计数为原生 all-area IoU=0.5 TP 标志筛选到小目标，不能直接分解 APs。']
for row in r['seeds']:
    x=row['transitions']['0.5']['COCO_small'];lost=x['BASE_native_TP_lost']
    c=x['lost_BASE_native_TP_primary_outcome'];none=c['miss_never_supported_at_three_boundaries']
    lines.append(f"种子 {row['seed']}：基线检出而增强漏检的小目标 {lost} 个；其中 {none} 个（{100*none/lost:.2f}%）在三个记录边界都没有 IoU≥0.5 的框覆盖。")
    lines.append(f"另外，早期覆盖但最终丢失 {c['miss_earlier_supported_lost_at_final']} 个；最终有覆盖却未通过真实类别 Top300/分数筛选 {c['miss_final_supported_no_true_class_after_top300_score_filter']} 个；被每类 Top100 移除 {c['miss_final_supported_true_class_removed_by_class100']} 个；真实类别框保留但未原生匹配 {c['miss_final_supported_retained_true_class_no_native_TP']} 个；排序重建不确定 {c['miss_uncertain_ranking_reconstruction']} 个。")
lines+=['最终分数序列与原生记录在全部 548 张图上相符，两组均无 TP/几何代理冲突；接近排序边界等不确定案例继续保留。','',
'五、结论与下一步','现有最早记录是 decoder 输入的已选 300 个初始框，覆盖差异已经在此出现。尚不能分开判定是主干/encoder 特征、TopK 的位置选择还是初始框回归造成的。',
'原生模型先从 8400 个位置的分类分数中选 300 个，再对这 300 个位置回归框。原生执行中没有“8400 个回归框”，不能伪造一个密集框覆盖率来定位问题。',
'继续完成授权的随机 Stem 与各组件对照；固定每配置最多 12 轮。原有协调程序完成独立权重复验并确认 GPU 空闲后，执行已准备的原生位置选择记录，核对真实选择索引、目标邻域分数以及选中位置的初始回归。',
'初始化衔接是另一个待检验解释：现有增强替换了原 Stem 的预训练参数及 BN 统计，其他 754 个张量相同。保留预训练原 Stem、将用户增强作为逐渐学习的残差补充是一个可检验候选；本轮尚未实现或训练它，不能宣称根因、创新或有效提升。',
'第 5 轮起原配方开始允许 Mosaic/MixUp 等策略，完整增强及随机对照的策略相同；曲线同期回落不能单独用来诊断 Stem 出错。',
'不重跑基线训练，不延长失败配置，不用最好轮代替固定第 12 轮。两种子 APs 均提升且总体 AP 不下降的原始目标仍未达成。']
target=out/'本轮结果和下一步.txt'
with target.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
with (out/'evidence.json').open('x',encoding='utf8') as f:
    json.dump(dict(status='REVIEW_DERIVED_FROM_ACTUAL_PRIMARY_RECORDS',sources={
      str(p.relative_to(ROOT)):sha(p) for p in (report_path,comparison_path,progress_path)},
      review_sha256=sha(target),model_forwards=0,evaluator_calls=0,optimizer_updates=0,
      saved_EMA_independent_replay_verified=False,goal_complete=False),f,indent=2)
print(str(target))

