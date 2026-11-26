"""Persist verified A1/A2 evidence and change the research priority,not the goal."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np

BASE=Path(__file__).resolve().parent
NOW=datetime.now(timezone.utc).isoformat()
A1=BASE/'A1_冻结特征读出诊断_v1'
A2=BASE/'A2_已有预测框误差分解_v2'
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
s=read(A1/'analysis/summary.json'); e=read(A1/'receipts/extraction_complete.json'); g=read(A2/'summary.json')
assert read(A1/'backup/local_verification.json')['verified_files']==51
assert s['linear_readouts_fitted']==32 and s['full_model_forwards_cumulative']==185
assert not all(s['gate_checks'].values()) and not g['gate_passed']
assert e['main_model_state_unchanged'] and max(r['box_max_abs'] for r in e['comparisons'])==0
rows=[json.loads(line) for line in (A2/'per_gt.jsonl').read_text(encoding='utf-8').splitlines()]
unique={r['uid']:r for r in rows if r['stage']=='encoder_topk300' and r['mode']=='geometry'}
assert len(unique)==4491 and sum(r['final_any_iou']<.5 for r in unique.values())==586
per_image=read(A2/'per_image.json')
for stage in ('encoder_topk300','decoder_0','decoder_3'):
    for mode in ('geometry','class_argmax'):
        selected=[r for r in rows if r['primary'] and r['stage']==stage and r['mode']==mode]
        table=next(r for r in g['stage_table'] if r['stage']==stage and r['mode']==mode)
        for key in ('base_rec50','center_rec50','size_rec50','template_rec50'):
            values=[]
            for file in sorted({r['file_name'] for r in selected}):
                values.append(np.mean([r[key] for r in selected if r['file_name']==file]))
            assert abs(float(np.mean(values))-table[key])<1e-12
history=BASE/'history_before_A1_A2_update_v1'
history.mkdir(exist_ok=True)
for name in ('01_候选判别表.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json'):
    if not (history/name).exists():
        (history/name).write_bytes((BASE/name).read_bytes())

# Preserve the frozen original source contract and attach a precise explanatory correction.
correction=dict(utc=NOW,kind='EXPLANATORY_LABEL_ONLY_NO_NUMERICAL_OR_PROBE_CHANGE',
    file='source_contract.json',view='P3_after',
    original_label='final stride8 PBM feature before decoder enc_output',
    corrected_label='encoder output[0]; actual DEIM decoder input_proj[0] is Identity for192->192,then flatten/mask and direct enc_score_head/enc_bbox_head. There is no enc_output module on this active path.',
    basis='Active engine/deim/deim_decoder.py403-426,456-496,525-544',
    numeric_tap_unchanged='model.encoder forward output[0]; all16 outputs matched A0',
    note='Do not substitute the other dfine_decoder.py implementation when describing DEIM execution.')
write(A1/'source_contract_explanatory_correction.json',correction)

board=read(BASE/'01_候选判别表.json')
board.update(latest_update_utc=NOW,status='CENTER_READOUT_PROBLEM_LOCALIZED_ALGORITHM_NOT_VALIDATED',
             mainline_validated=False,current_priority_question='融合特征中可读的微小目标中心线索，为什么未稳定转化为候选框/查询的正确中心位置？如何在固定候选预算内改善这种空间对应？',
             latest_report=str(BASE/'20_A1A2结果与当前候选主线.txt'))
for c in board['candidates']:
    if c['id']=='A_ENCODER_INSTANCE_SEPARABILITY':
        c.update(status='BROAD_DETAIL_ERASURE_EXPLANATION_NOT_SUPPORTED_BY_A1_LINEAR_PROBE',
                 next='Do not add a detail-preserving module from A0 missing boxes. A1 fused features yield better conditional center ranking; this does not prove nonlinear sufficiency or within-cell separability.')
new=dict(id='E_CENTER_TO_BOX_SPATIAL_CORRESPONDENCE',status='PRELIMINARY_DIAGNOSTIC_SUPPORT_NEAREST_METHOD_GATE_OPEN',
         question=board['current_priority_question'],increment_approved=False,causal_evidence=False,
         evidence=['A1 fused-feature conditional center ranking exceeds shallow feature ranking on522 eligible<16 GT.',
                   'A2 on470 persistently missed<16 GT with local selected queries: center oracle recovers more geometry than size oracle;both size bins agree.'],
         nearest_to_clear=['CenterNet/center-based detection','Gaussian or center-guided DETR query selection','QCAR(primary abstract only)',
                           'CLSC DETR(primary abstract only)','DS-Det full decoder method','DAB/Conditional DETR spatial-query designs'],
         possible_increment='A query-center update/readout explicitly tied to local spatial evidence rather than an independent feature-only coordinate regression; this is a design space,not an approved novel operator.',
         stop='Reject generic heatmap supervision,center-query seeding or attention calibration as novelty if closest methods already provide the same mechanism.')
if not any(c['id']==new['id'] for c in board['candidates']): board['candidates'].append(new)
board['new_empirical_evidence']='A0/A0b plus A1:185 full model forwards;main model updates0;A1 CPU diagnostic linear fits32;A2 uses100 cached traces,0 new forwards/fits.'
board['deprioritized_by_A2']='A width/height-specific correction mainline: size oracle is weaker than center oracle and adds no material IoU0.5 recovery beyond a15-template explanatory oracle.'
write(BASE/'01_候选判别表.json',board)

matrix=read(BASE/'14_架构核心文献矩阵.json')
extra=[dict(id='QCAR',title='Query-calibrated attention and visibility-aware anisotropic regression for small object detection',
    url='https://doi.org/10.1016/j.neucom.2026.134465',tier='Neurocomputing primary publisher record;volume700 dated1Nov2026',
    read_scope='Publisher abstract,highlights and introduction/contribution excerpts only;full method not acquired/read',
    author_method='Training-only matched-query Gaussian spatial alignment and progressive anisotropic box regression',
    our_consequence='Generic attention-to-center calibration or staged center/shape loss is already claimed in a close aerial tiny-object work.',
    unresolved='Need full method/source before excluding overlap with a spatially constrained center readout;not reproduced.'),
    dict(id='CLSC_DETR',title='CLSC DETR: Reliable Candidate Ranking via Cross Layer Geometric Support for UAV Small Object Detection',
    url='https://arxiv.org/abs/2608.21457',tier='preprint',read_scope='Primary indexed abstract only;full method/source not read',
    author_method='Cross-layer local geometric support and classification/localization consistency calibration for candidate ranking',
    our_consequence='Generic cross-layer geometry aggregation or confidence/localization consistency is not empty prior art.',
    unresolved='Current problem concerns center coordinates,not just ranking;full method needed before claiming a substantive difference.')]
for row in extra:
    if not any(x['id']==row['id'] for x in matrix['rows']): matrix['rows'].append(row)
matrix['updated_utc']=NOW
write(BASE/'14_架构核心文献矩阵.json',matrix)

state=read(BASE/'15_当前状态与结果校验.json')
state.update(recorded_utc=NOW,goal_status='ACTIVE',previous_goal_turn='progress',
             mainline_validated=False,diagnostics_complete=True,model_forwards=185,cap=200,
             training_updates=0,training_update_scope='Main detection model only;32 diagnostic CPU linear readouts were fitted',
             CPU_diagnostic_readout_fits=32,latest_report=str(BASE/'20_A1A2结果与当前候选主线.txt'))
state['verification']['scope']='Historical A0/A0b integrity and metric snapshot at169 forwards; current phase is below.'
state['latest_authoritative_phase']=dict(utc=NOW,goal_status='active',classification='progress',
    complete_mainline=False,full_model_forwards=185,global_cap=200,main_model_training_updates=0,
    new_architecture_training_runs=0,CPU_diagnostic_linear_readouts_fitted=32,A2_new_fits=0,
    A1_archive_sha256=read(A1/'backup/archive_receipt.json')['sha256'],A1_archive_files_verified=51,
    A2_unique_gt_verified=4491,A2_final_geom50_misses_reproduced=586,
    A2_tables_reaggregated=True,latest_report=str(BASE/'20_A1A2结果与当前候选主线.txt'),
    next='Read the nearest center-query/calibration methods and specify a genuinely distinct,causally testable spatial readout mechanism before architecture training.')
write(BASE/'15_当前状态与结果校验.json',state)
index=read(BASE/'19_A1实际路径与冻结协议索引.json')
index.update(status='COMPLETED_WITH_MODEL_STATE_PRESERVED',completed_update_utc=NOW,
             actual_new_full_model_forwards=16,actual_cumulative_forwards=185,actual_diagnostic_readout_fits=32,
             local_integrity_verification=str(A1/'backup/local_verification.json'),
             explanatory_source_correction=str(A1/'source_contract_explanatory_correction.json'),
             note='Frozen protocol/source bytes preserve the pre-execution record; see the independent explanatory correction and completion receipts.')
write(BASE/'19_A1实际路径与冻结协议索引.json',index)
report='''Material Passport
Origin Skill: academic-research-suite
Origin Mode: bounded frozen-feature diagnostic and cached-prediction analysis
Origin Date: {now}
Verification Status: ANALYZED;integrity and listed metric checks passed;new scientific mainline remains UNVERIFIED
Version Label: A1_A2_center_readout_v1

当前候选主线：融合特征中可读的微小目标中心线索，为什么未稳定转化为候选框/查询的正确中心位置？研究重点转向特征位置与查询/框中心的空间对应。
它不是压缩/恢复，也还不是已确认的新算法。下一步不是立即加模块，而是查清已有中心引导查询、位置校准与显式坐标读出的方法边界。

一、这次实际完成了什么
A1：作者TinyFormer-S-PBM VisDrone EMA保持eval、FP32与原生640输入，固定前16图，14个文件前缀组、4折。提取原生P2、P2经原生stride2卷积后的结果、融合前P3、融合后P3。
16个真实标签线性读出＋16个图内打乱标签对照，共32个CPU拟合；每个193参数，固定ridge0.01，无调参搜索。原检测模型0次参数更新。
每个新前向与保存的A0核对：框最大差0，分数最大差1.1920929e-7，flatTop300类别/query编号完全相同；模型状态摘要不变。累计完整前向185/200，未追加完整训练。
A1归档42,109,977字节、51个文件逐项SHA通过：f2c74884571fff166dce5f279d8bf0a250c3959c9ae69ca585215bb1767b25b78。
A2：只用已有100张A0预测、4491GT做几何分解，0新前向、0新拟合。复核最终几何IoU0.5未覆盖586个，与A0一致。
A2v1在解析预测前因我抄错输入归档哈希退出；v2只按原始回执纠正哈希，分析逻辑AST一致、协议字节相同。失败记录与两个版本均保留。

二、[实测] 不支持“先给编码器补细节”的解释
A1主终点是GT导出的局部候选区域内，中心得分高于邻近非中心位置的比例；按图平均。它不是检测AP，也不是全图寻找未知目标的召回率。
522个合格<16像素GT：P2为0.7305，P2_down为0.7939，融合前P3为0.8118，融合后P3为0.9287。
预定主对比P2减融合后P3：-0.1982，文件前缀分组bootstrap95%区间[-0.3315,-0.1060]；<8及8-16两个尺度方向一致。打乱标签对照分别0.5578和0.4811。
在136个“<16且全部编码候选读出IoU0.5未覆盖”的子集中，融合后P3为0.8333，P2为0.7742，差异区间含0。
[推断] 在这项有限线性读出任务上，融合后的中心线索更容易读取；不能拿A0缺框直接论证融合抹掉小目标细节。
[未证明] 所有定位/边界信息都充足、非线性信息不受损、同格实例可分、任意新编码器无价值。该权重见过原训练图；读出折只隔离了读出拟合，不提供检测泛化结论。

三、[实测] 优先问题是中心位置，而非宽高
A2固定原生选出的stride8查询，只分析GT邻近3x3锚点中的同一批query；分别真值修正中心、修正宽高，并保留可选的原框。固定模板对照是15种尺寸/比例的oracle选择，仅作解释控制。
2538个<16GT中，523个在初始Top300及最终decoder都几何未覆盖；470个附近有选中查询，53个没有。预定主分析为这470个GT，覆盖50图、31个前缀组。
初始候选在该队列的IoU0.5几何覆盖按图平均：原框0；修正中心0.6217；修正宽高0.2498；固定模板oracle0.2589。
宽高修正减中心修正：-0.3719，95%区间[-0.5115,-0.2578]；<8与8-16分别-0.2903和-0.5711，方向一致。
保留预测类别argmax一致时，中心修正0.5132、宽高修正0.1049，宽高减中心-0.4083。最终层也有同方向的几何分解。
[推断] 对这个条件队列，优先研究宽高/分布范围的依据不足；中心位置修正有更大的几何解释空间。
[限制] 真值修正不是可部署算法，不是AP增益或因果证明。局部锚点关联不是真对象身份；一个query仍可能涉及邻近GT。15模板oracle增加了假想候选，仅用于排除简单尺寸解释，不能当公平算法对照。

四、为什么现在可以把问题收窄到“中心读取与空间对齐”
A1与A2结合支持一个值得核查的缺口：局部中心线索可被有限读出利用，同时原生候选的中心误差解释了较大一部分持续失败。
这仍可能来自监督/分配、查询对应、回归参数化、位置编码或数据难度，而不是某个模块必然失效。已有证据不能在这些原因间直接作因果选择。
实际入口已经核对为engine/deim/deim_decoder.py：选择后的内容/框detach进入decoder；encoder候选头只对选中位置有相应辅助监督；同仓库dfine_decoder.py不是相同解码循环。
原生DEIM路径没有enc_output模块。A1冻结说明里P3_after的“before enc_output”措辞已用独立更正文件纠正，未改冻结文件或实际tensor tap，数值结果不受影响。

五、可能的算法空间与最近工作门槛
可继续考察的增量是：让query中心更新或坐标读出显式受其读取的局部空间证据约束，减少特征内容与预测位置各自独立变化产生的错位。这只是设计空间，尚未确定或训练算子。
普通中心热图监督、按中心峰值初始化query、注意力向GT中心对齐、跨层几何聚合都不能直接称新颖。QCAR的主出版方摘要已有Gaussian query attention校准；CLSC DETR摘要已有跨层局部几何支持及分类/定位一致性校准。两篇本次仅摘要/贡献片段，必须继续核查完整方法。
Slim-DETR目前仅有第三方条目被发现，尚未取得可核验原始论文；不把第三方方法描述或标题当作已确认的最近工作证据。
还需与CenterNet、DAB/Conditional DETR、DS-Det完整解码方法逐项对照。简单共享encoder memory、加P2/频域、增加query、换Mamba均不作为已成立创新。

六、数据、算法、资源与下一步
数据仍为VisDrone原图。TinyFormer-S-PBM是当前可核验诊断基线，未锁定最终算法；YOLO26s保留参考，禁止混配方直接排名。
当前总量：185次完整模型前向；检测模型0次新训练更新；32个CPU诊断读出拟合；A2不增加前向/拟合。原200次上限保留，远端本次结束GPU空闲，未启动新训练队列。
先完成中心引导/坐标读出最近工作的全文和源码核查，明确一个具体机制差别，再冻结能区分原因的干预；若只剩已有方法的等价改写，淘汰该增量。单卡24GB训练可行性仍需真实前反向容量和吞吐验证，推理成功不等于训练容量通过。
原七项调研框架见17_架构方向收敛与最近工作对比.txt；本文件更新问题与实验判断，文献范围见14_架构核心文献矩阵.json。

七、可核查位置
A1协议、代码及结果：A1_冻结特征读出诊断_v1/protocol.json、analysis/summary.json；完整特征/拟合系数/回执在backup/A1_readout_probe_bounded.tar.gz。
A2协议与结果：A2_已有预测框误差分解_v2/protocol.json、summary.json、per_gt.jsonl、input_trace_verification.json。
QCAR主来源：https://doi.org/10.1016/j.neucom.2026.134465
CLSC DETR主来源：https://arxiv.org/abs/2608.21457
TinyFormer作者源码：https://github.com/mmpmmpmmpjosh/TinyFormer
研究目标保持active：已把问题收窄到有诊断依据的候选主线，尚未完成新算法的新颖性/机制/效用验证。不追加压缩恢复分支，不重启旧DEIM任务。
AI披露：单助手辅助编码、检索、诊断与整理；未推定用户阅读过文献，未进行独立人工同行复核。
'''.format(now=NOW)
(BASE/'20_A1A2结果与当前候选主线.txt').write_text(report,encoding='utf-8')
write(BASE/'21_A1A2完成与核验回执.json',dict(utc=NOW,status='PASS_EVIDENCE_AND_METRIC_CONSISTENCY_ONLY',
    full_model_forwards=185,main_model_updates=0,diagnostic_readout_fits=32,A1_files_verified=51,
    A2_unique_gt=4491,A2_final_geom50_missing=586,A2_tables_reaggregated=True,mainline_validated=False,
    report_sha256=hashlib.sha256((BASE/'20_A1A2结果与当前候选主线.txt').read_bytes()).hexdigest()))
print(json.dumps(dict(status='A1_A2_DECISION_RECORDED',full_model_forwards=185,readout_fits=32,
                      core_literature_rows=len(matrix['rows']),scientific_mainline_validated=False),ensure_ascii=False))
