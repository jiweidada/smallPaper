import json,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
t=json.loads((ROOT/'topk_membership_frozen/02_results.json').read_text(encoding='utf8'))
old=json.loads((ROOT.parents[1]/'英文小论文/data/10_来源分配冻结诊断_20261020_v1/01_冻结协议与资产.json').read_text(encoding='utf8'))
assert t['protocol']['imported_diagnostic_sha256']==old['runner_sha256']
lines=['Material Passport: academic-research-suite / experiment-agent; frozen CPU audit; exploratory, not independently replicated.',
       'top-k排序与成员替换的冻结核查；两seed各development00/01；原损失与历史Adam，一次克隆虚拟端点，不是实际训练轨迹。',
       'seed | ordered_changed/2400 | missing_members/2400 | retained_rank_changed | membership retention']
for seed in [1,2]:
    rows=[x for x in t['rows'] if x['seed']==seed]
    c={k:sum(x['counts'][k] for x in rows) for k in ['ordered_slots_changed','members_lost','retained_rank_changed','slots']}
    assert c['ordered_slots_changed']==c['members_lost']+c['retained_rank_changed']
    lines.append(f"{seed} | {c['ordered_slots_changed']}/{c['slots']} | {c['members_lost']}/{c['slots']} | {c['retained_rank_changed']} | {1-c['members_lost']/c['slots']:.6%}")
    for stage in ['encoder','final']:
        for size in ['shortside_lt4','shortside_ge4']:
            r=[v for x in rows for v in x['GT_best_site_retention'] if v['stage']==stage and v['size']==size]
            n=sum(v['baseline_IoU_ge_0_5_GT'] for v in r);lost=sum(v['best_source_site_lost'] for v in r)
            lines.append(f"seed{seed} {stage} {size}: 原maxIoU>=.5的GT曝光数{n}，其基点最佳source site换出数{lost}；这不是最终GT丢失数。")
lines += ['旧data10基点topk_indices、全参数梯度范数、裁剪系数、虚拟位移范数、fixed/native跨批几何端点均重现（原始标量相等或指标差<1e-12）。',
          '同成员逆序控制：1200个ordered槽全部变化，但inverse-permute后框最大误差5.36e-7至2.77e-6，logit误差2.36e-5至9.49e-5。实际learn_query_content=False。不是比特级等变，不能冒充AP复现。',
          '结论：原77–84%是旧八批有序槽统计，新两批是86.71%/77.29%；新两批实际成员仅1.58%/.79%换出，绝大部分为集合内重排。不支持从有序槽数推出严重候选集合流失。已覆盖tiny的最佳site在这些输入均未换出，但tiny分母仅6/6（encoder）与22/16（final），不能推广所有GT/训练阶段。',
          '模型/检查点SHA前后不变，CPU2线程BelowNormal，未使用GPU/远端/监测；20次模型前向、4次全参数求导、真实训练步0。完整索引和每图原始表保存于topk_membership_frozen。']
(ROOT/'34_topk成员核查结果.txt').write_text('\n'.join(lines),encoding='utf8')
a=json.loads((ROOT/'ddf_geometry_cache_audit/02_results.json').read_text(encoding='utf8'))
lines=['Material Passport: academic-research-suite / experiment-agent; cache-only output-coordinate derivative audit, no model inference/training.',
       'DDF matched-positive几何方向新核查（探索性）：BASE24两个seed、development00-07、两个aux层，始终用同前向原GPU union。不是网络参数梯度或优化端点，不是AP。',
       '原代码loss_local的DDF正项权重为student detachedIoU；负项为teacher maxsigmoid但无GT几何，不在本表。原T=5与自然对照T=1固定同权重、归一化。闭式KL梯度已对Torch独立合成验证。',
       'seed | 配对层曝光 | active(studentIoU>0) | teacher worse | teacher worse的IoU权重占比 | T1/T5负内积 | teacher worse中T5负内积']
for s in a['pooled']:
    allg=s['groups']['all'];w=s['groups']['teacher_worse']
    lines.append(f"{s['seed']} | {allg['pairs']} | {allg['active_pairs']} | {allg['teacher_worse']} | {w['student_IoU_weight_sum']/allg['student_IoU_weight_sum']:.6%} | {allg['T1_opposing']}/{allg['T5_opposing']} | {w['T5_opposing']}/{w['active_pairs']}")
lines += ['负内积指∇(5L1+2GIoU)·∇DDF<−1e-12，仅在输出corner logits处；直接沿-DDF更新时对几何目标的一阶方向不利。共享网络雅可比、全局裁剪、Adam历史、后层互相影响均未包含，不能据此推网络训练有害。',
          'T5负内积没有多于T1，teacherbetter组T5反而更少。因此不以当前数据优先投入“高温破坏几何”的假设。',
          '最终教师并非每个query/GT都更准，teacherworse组在两个seed约占43%的studentIoU正权重且多呈逆几何方向。频次本身不证明禁用蒸馏能提升最终检测；teacherworse简单门控已有很强先例，不能认作新原理。',
          '最近原文风险：Distilling Object Detectors with Task Adaptive Regularization, arxiv2006.13108，作者摘要已明确根据师生谁表现更合理自适应转移bbox知识；本轮只读摘要，不声称核实全部公式。Localization Distillation CVPR2022/后续TPAMI已有分布KD；D-FINE §4.2已提出GO-LSD。',
          '原始完整分组/每批数据及来源SHA见ddf_geometry_cache_audit/02_results.json；32层批次条目，框重构误差最大8.895e-8；模型前向0、训练步0。']
(ROOT/'35_DDF新缓存核查结果.txt').write_text('\n'.join(lines),encoding='utf8')
print('\n'.join(lines[3:]))
