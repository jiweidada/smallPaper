"""Reconcile independently verified A8 evidence; no experiments or fitting."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil

R=Path(__file__).resolve().parent
A=R/'A8_固定query的窗口更新诊断_v1'
UTC=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=read(A/'summary.json');v=read(A/'verification.json');receipt=read(A/'backup/receipt.json')
assert v['status'].startswith('PASS_') and v['independent_graph_counts']==90
assert receipt['archive_sha256']==sha(A/'backup/A8_window_memory_results.tar.gz')
assert s['global_full_detector_forwards']==200 and s['main_detector_updates']==0
assert read(A/'completed.json')['summary_sha256']==sha(A/'summary.json')
g=v['aggregate'];base=g['REPLAY_NATIVE'];oracle=g['GT_COVERAGE_SAME_K'];dense=g['ALL_WINDOWS']
assert base['tiny_gt']==base['tiny_center_covered']==69
assert base['lt8_iou75']==oracle['lt8_iou75']==dense['lt8_iou75']==12
assert all(row['coverage']['tiny_mean_area_covered']==1 for row in s['rows'] if row['arm']=='NATIVE')
assert all(row['memory_relative_change']>0 for row in s['rows'] if row['arm'] not in ['NATIVE','REPLAY_NATIVE'])
decision='DEPRIORITIZE_COVERAGE_REPAIR_AS_MAINLINE_NO_MODULE_TRAINING_FROM_A6_A8'
report=R/'28_A8窗口更新负结果与主线重新开放.txt'
files=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
h=R/'history_before_A8_result_v1';h.mkdir(exist_ok=True)
for n in files:
 if not (h/n).exists():shutil.copy2(R/n,h/n)

table=[]
for arm,name in [('REPLAY_NATIVE','原生选窗重放'),('RANDOM_SAME_K','等预算随机'),('GT_COVERAGE_SAME_K','等预算GT覆盖'),('ALL_WINDOWS','全窗口参考')]:
 q=g[arm]
 table.append(f"{name}\t{','.join(str(x) for x in q['windows'])}\t{q['tiny_center_covered']}/69\t{q['lt8_iou50']}/69\t{q['lt8_iou75']}/69\t{q['all_iou50']}/220\t{q['all_iou75']}/220")
body=f'''A8窗口更新诊断：不支持把“补覆盖”直接发展为主线
记录时间：{UTC}

结论
当前没有经验证可投稿的新主线。A8使“微小目标被稀疏编码更新遗漏”的候选进一步降级，不据A6的标签几何缺口训练新门控。
这只约束当前Dome-S、固定3个作者训练案例、固定query初始化下的memory读取路径；不否定整个稀疏编码方向，也不证明一切encoder改进无效。

1. 检索策略与记录
本轮无新增联网文献检索，完成的是固定作者源码、标签/预处理接口与受控诊断。原定向调研记录、34项矩阵和原始检索响应仍分别保留在13/14文件及source_audit_v1。
新增核查来源包括作者VisDrone转换脚本、12通道配置和检查点、原生动态TopQ后处理、MWAS双轴注意力及decoder初始化接口。不能把本轮称为系统综述或全面复现。

2. 核心文献与阅读范围
Focus-DETR说明前景筛选机制；RFLA已研究微小GT正样本缺口；Dome-DETR已有密度引导的窗口编码和query分配；SViT已有任务训练门控；Salience已有层级显著性筛选；UHR-DETR已有面积归一化覆盖；CoRePrune已有集合条件下的删除/恢复效用。
具体阅读范围及源码版本沿用14矩阵。QA-MBG仍仅摘要；本轮没有把摘要补写为全文阅读。Dome本次实际运行了固定官方模型，其他方法不因此变成已复现基线。

3. 最近工作逐项对照和接口修正
A7草案中的Top300口径，在任何Dome预测之前已按实际源码改为动态TopQ：输出12类，原生有效类别编号1–10，旧10类分析ID需加1。未根据结果改阈值或协议。
Dome同一mask同时控制编码更新和可选query筛选。A8固定原生decoder初始化返回元组，等窗口预算臂只替换MWAS更新集合；比较基线为原生mask重放。
每个10×10 P3窗口包含完整位置特征；作者窗口编码器先执行窗口内注意力，再交换轴做选中窗口之间的交互。它不是彼此独立的逐窗口算子。
这与“删除未选token”的实验不同。所有未接受MWAS更新的特征仍保留在原生memory/FPN路径中，GT覆盖臂是已知控制，不是新算法。

4. 我方测量及其证据边界
模型：作者Dome-DETR-S VisDrone转换检查点，固定Git提交2dde3bc1946a3e9fad9abd0612b59fc39bd6b861。1033个state张量严格匹配，参数13,240,007；原生800方形Resize/float32除255，无额外归一化或部署融合。
数据：固定3个不同前缀的训练案例，按标注尺寸/数量预先选择；有效GT共220个，按sqrt(area)<8输入像素划分的微小GT共69个。作者权重已在训练集上训练，这不是独立测试。
原生各图选择92/100、56/100、63/100个窗口，query数386、419、384。69个微小目标中心以及其全部框面积，均已被原生选窗联合覆盖。
三个原生重放的raw框与logit最大差均为0，flat输出索引完全相同。各干预都改变了memory，排除了“干预实际上没有生效”这一解释；这不等于证明其全局效果。

诊断覆盖计数（每个尺寸群单独做有类别约束、同物理query至多匹配一次的最大匹配；不是官方AP/AR，也不是群间联合分区）
方案\t各图更新窗数\t微小GT中心覆盖\t微小IoU50\t微小IoU75\t全部IoU50\t全部IoU75
'''+'\n'.join(table)+f'''

等预算GT选窗的微小IoU50覆盖从51降到50，IoU75仍为12；全窗口参考的微小覆盖也为51/12。全窗口对全部GT的计数有+1/+1，但使用更多窗口，且未改善这里的微小目标计数，不能当成同算力新算法优势。
【我方判断】这些案例没有显示可被“补选更新窗口”修复的微小目标覆盖缺口。因此当前证据不足以批准这一候选的模块设计和训练。
【不能声称】不能把这3个样本的阴性结果推广为Dome/Focus所有图像均无问题，不能据此声称统计显著无效，也不能把A6的Focus目标标签几何直接等同于Dome实际更新或漏检。

5. 研究空白、候选增量与风险
A6的源码几何缺口仍是可核验事实，但它不是新发现的充分证据，更未在这里转化为Dome的实际更新遗漏。当前没有得到通过最近工作去重和因果验证的算法增量。
原有编码器细节、通用坐标精修、更新收益门控等候选保留各自的否证/重合边界，不通过更换名称重新包装。
后续应重新开放具体编码/解码结构问题；必须先有可重复的实际失败机制，并证明已知方法控制不足，再提出与最近工作逐项有区别的算子。当前不锁定下一新模块。

6. 最小验证已完成、资源与下一步
本次完成冻结协议的3图×5臂共15次完整前向，无warmup，无参数更新或新拟合。此前185次加本次15次，架构诊断阶段达到200次上限；本轮不再追加该阶段前向。
冻结模型前后state digest均为{s['state_digest_before']}。本次运行约17.3秒（包含加载/追踪等，非纯推理benchmark）；自己的峰值CUDA分配约0.609GiB，GPU当时有其他进程占用。已证明当前环境能执行这种冻结诊断，尚未证明完整训练的显存、耗时或数天预算。
独立复核：30个归档结果文件SHA、15个原始输出、90项独立网络流匹配计数和矩形覆盖面积全部通过。初版审计把原始query顺序与排序后的输出直接比较，已修正审计索引；没有改实验代码、输出或数值容差。
query初始化由冻结代码钳制并通过原生重放检查，但没有另行序列化每个arm的种子张量摘要；独立核查范围不包含逐张量重放该未存储对象。该限制随报告保留。
下一步为基于本次否证边界重新筛选具体encoder/decoder机制；尚无批准的新训练方案。研究目标继续，旧压缩/恢复分支和旧DEIM任务不重启。

7. 可核验参考及本地证据
Dome-DETR原文：https://arxiv.org/abs/2505.05741
Dome固定源码：https://github.com/RicePasteM/Dome-DETR/tree/2dde3bc1946a3e9fad9abd0612b59fc39bd6b861
SViT：https://arxiv.org/html/2306.07050v3
Salience-DETR：https://arxiv.org/abs/2403.16131
RFLA：https://arxiv.org/abs/2208.08738
CoRePrune：https://arxiv.org/html/2609.26484v1
本地冻结协议：{A/'protocol.json'}
独立核查：{A/'verification.json'}
原始输出：{A/'outputs'}
归档SHA256：{receipt['archive_sha256']}
原始结果归档：{A/'backup/A8_window_memory_results.tar.gz'}
'''
report.write_text(body,encoding='utf-8')

board=read(R/files[0]);board.update(latest_update_utc=UTC,latest_report=str(report),mainline_validated=False,
 status='NO_VALIDATED_MAINLINE_A8_COVERAGE_REPAIR_DEPRIORITIZED',
 current_priority_question='尚无通过机制验证的候选主线；依据A8实际更新覆盖和阴性对照，重新筛选具体编码器/解码器失败机制。')
for c in board['candidates']:
 if c['id']=='F_SPARSE_ENCODER_UPDATE_ALLOCATION':
  c.update(status=decision,increment_approved=False,causal_evidence=False,
   evidence='A6 source target geometry does not establish an actual Dome coverage failure. A8:all69 tiny GT centers/areas already covered;same-K GT control50/12 versus native51/12 atIoU50/75;all-windows51/12. Three exploratory train cases only.',
   next='No coverage-repair module or training from these data. Reopen only with independent evidence of an actual missed-update mechanism and a nontrivial closest-work distinction.')
board['asset_gate']={'strict_Dome_model_load':'PASS1033keys','frozen_inference':'PASS15forwards',
 'identity_replay':'PASS3cases_exact','diagnostic_archive_independent_audit':'PASS','training_memory':'NOT_VERIFIED'}
board['A8_result']={'decision':decision,'path':str(A),'verification':str(A/'verification.json'),
 'new_forwards':15,'total_phase_forwards':200,'main_detector_updates':0}
board['next_decision_protocol']='No new training protocol selected after A8; current200-forward bounded diagnostic phase is complete.'
save(R/files[0],board)

log=read(R/files[1]);log['A8_source_contract_and_measurement']={'utc':UTC,'new_web_queries':0,
 'type':'Local pinned primary-source audit and preregistered within-project bounded diagnostic; not a systematic review.',
 'corrections_before_prediction':['12 output classes with valid native IDs1–10','native flatTopQ rather than draftTop300','window-internal plus cross-window attention'],
 'result':decision,'evidence':str(A/'verification.json'),'archive_sha256':receipt['archive_sha256']}
save(R/files[1],log)
matrix=read(R/files[2]);matrix['updated_utc']=UTC;matrix['latest_report']=str(report)
matrix['latest_empirical_boundary']='A8 on3 fixed author-train cases/69 tiny GT:actual native windows cover all centers/full area;same-K GT and all-window references give no tinyIoU75 count gain. Phase200forwards,0detectorupdates. No new algorithm validated.'
for row in matrix['rows']:
 if row['id']=='Dome_DETR':
  row['read_scope']='Selected primaryv2 density/mask methods and active source;293-file archive verified. OfficialS-VisDrone1033-state-key model strictly loaded,15 FP32 frozen diagnostic forwards at800,3 exact identity controls. No formal AP reproduction or training.'
  row['our_consequence']='Native window coverage must be measured separately from Focus target labels;all69 tiny GT fully covered in A8 fixed cases. Coverage repair not supported as current mainline.'
  row['A8_verification']=str(A/'verification.json')
save(R/files[2],matrix)

source=read(R/files[4]);source['updated_utc']=UTC
source['A8_Dome_active_execution']={'strict_load':str(A/'model_load.json'),'source_commit':'2dde3bc1946a3e9fad9abd0612b59fc39bd6b861',
 'author_numeric_source_edits':0,'compatibility':'Deferred type annotations; omit a statically unused skimage import; model-only package namespaces; author image-conversion class registry decorator omitted.',
 'new_full_forwards':15,'fitted_models':0,'detector_updates':0,'identity':'3cases_exact_raw_boxes_logits_and_flatIDs',
 'state_digest_before_after':s['state_digest_before'],'verification':str(A/'verification.json'),
 'causal_scope':'Controlled memory-read path with native seeds clamped by source wrapper; no separate serialized per-arm seed-tensor digest.'}
source['scope_this_update']='Dome author numeric model executed15times under frozen source-faithful hooks. Prior read-only receipts remain dated historical snapshots.'
source['current_novelty_boundary']='Generic coverage/gating already prior;A8 supplies no positive basis for a new coverage-repair module. Mainline remains unvalidated.'
save(R/files[4],source)

state=read(R/files[3]);state.setdefault('authoritative_phase_history',[]).append(state['latest_authoritative_phase'])
state.update(recorded_utc=UTC,goal_status='ACTIVE',previous_goal_turn='progress_A7_sources_full_assets_and_remote_CPU_load',
 current_goal_turn='progress_A8_strict_load_15frozen_forwards_independent_audit_negative_mechanism_result',
 mainline_validated=False,latest_report=str(report),model_forwards=200,cap=200,training_updates=0,
 current_priority_question=board['current_priority_question'])
state['latest_authoritative_phase']={'utc':UTC,'classification':'progress','goal_status':'active','complete_mainline':False,
 'full_model_forwards':200,'global_cap':200,'bounded_phase_complete':True,'main_model_training_updates':0,
 'CPU_diagnostic_linear_models':64,'scalar_response_heads':96,'new_fits_A8':0,
 'A8_new_full_forwards':15,'A8_decision':decision,'A8_archive':str(A/'backup/A8_window_memory_results.tar.gz'),
 'A8_archive_sha256':receipt['archive_sha256'],'independent_graph_counts':90,'latest_report':str(report),
 'next':'Reopen architecture mechanism screening;no additional forwards in the completed200-forward phase and no new module training supported by A8.',
 'remaining_uncertainty':['A defensible new algorithm mechanism','Independent actual failure evidence','Novelty beyond nearest methods','Generalization','Training time/memory feasibility']}
save(R/files[3],state)
final={'utc':UTC,'status':'A8_VERIFIED_NEGATIVE_COVERAGE_REPAIR_DEPRIORITIZED_GOAL_ACTIVE',
 'report':str(report),'report_sha256':sha(report),'decision':decision,'mainline_validated':False,
 'archive_receipt':receipt,'independent_verification_sha256':sha(A/'verification.json'),
 'phase_full_forwards':200,'new_full_forwards':15,'main_detector_updates':0,'diagnostic_models':64,'scalar_heads':96,
 'files':[{'path':str(R/n),'sha256':sha(R/n)} for n in files]}
save(R/'29_A8完成核验与主线降级回执.json',final)
print(json.dumps({k:v for k,v in final.items() if k not in ['files','archive_receipt']},ensure_ascii=False,indent=2))

if __name__=='__main__':pass
