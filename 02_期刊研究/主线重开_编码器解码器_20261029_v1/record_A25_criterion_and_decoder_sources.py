import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')
B=Path(__file__).resolve().parent
D=B/'A25_完整损失与查询边界核验_v1'
A=B.parent/'文献_相关原始论文_20261004'
REPORT=B/'60_A25新编码器源码与解码器候选筛查.txt'
RECEIPT=B/'61_A25完整损失与新文献核验回执.json'
NAMES=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
assert not REPORT.exists() and not RECEIPT.exists()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

cpu=read(D/'CPU_complete_native_criterion_result.json')
dense=read(D/'CPU_dense_DN_density_result.json')
assert cpu['complete_native_criterion_forward_calls']==3 and cpu['native_matcher_forward_calls']==17
assert dense['complete_criterion_calls']==2 and dense['native_matcher_calls']==10
assert cpu['detector_forwards']==dense['detector_forwards']==cpu['optimizer_updates']==dense['optimizer_updates']==0
assert not cpu['CUDA_initialized'] and not dense['CUDA_initialized']
assert cpu['cases'][0]['nonfinite_raw_loss_keys']==['loss_ddf_aux_0','loss_ddf_aux_1']
assert dense['cases'][0]['nonfinite_raw_keys']==['loss_ddf_dn_2']
assert not dense['cases'][1]['nonfinite_raw_keys'] and dense['cases'][1]['synthetic_gradients']['all_present_gradients_finite']
assert cpu['criterion_losses']==['vfl','boxes','local']
assert cpu['query_boundary_fixture']['native_matching_selected_indices']==[1]

commit=read(D/'DSPE_author_commit.json')
tree=read(D/'DSPE_author_exact_tree.json')
source=read(D/'DSPE_downloaded_source_manifest.json')
assert commit['sha']==source['commit']=='23b92e34198df838c8ed63259e6457e58100edf2'
assert tree['sha']==commit['commit']['tree']['sha']==source['commit_tree']
assert not tree['truncated'] and len(source['files'])==9
source_paths=[]
for row in source['files']:
    path=Path(source['source_root'])/row['path']
    data=path.read_bytes()
    assert sha(path)==row['sha256']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha1']
    source_paths.append(path)

now=datetime.datetime.now(datetime.timezone.utc).isoformat()
paper_specs=[
 ('A25_DSPE_ViT_FrontNeurorobot2026','dbf3ad1ea7a4896bac556561592c5af48ccf57e5842b49f1c5f79d84ae92b54e',
  'DSPE-ViT: a lightweight vision transformer with dynamic sparse positional encoding for dense small object detection in UAV imagery',
  'https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1849093/pdf',
  'Publisher version, 16June2026; complete Method3.1-3.4, implementation4.2 and selected reproducibility discussion; PDF method pages5-6 visually checked. Not all experiments.'),
 ('A25_Group_DETR_ICCV2023','228382d7f501acb7849d3e72fa42776e94ba822e93d4a986fde55fe3e9192f51',
  'Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment',
  'https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Group_DETR_Fast_DETR_Training_with_Group-Wise_One-to-Many_Assignment_ICCV_2023_paper.pdf',
  'Official ICCV2023 version; abstract, selected3.1 algorithm/training/inference/loss passages; Figure2/Algorithm1/Eq5-6 on PDFpage3 visually checked. Not whole3.2, experiments or code.'),
 ('A25_GroupEnsemble_arXiv2603_01847v1','27ea4a62c0c1afb7f6e393f4bdf7d19811aecd57034350934ea4f002a36ba9c6',
  'GroupEnsemble: Efficient Uncertainty Estimation for DETR-based Object Detection',
  'https://arxiv.org/pdf/2603.01847v1',
  'arXivv1, 02March2026; abstract/introduction, selectedIII-A and completeIII-B-D; Eq4 mask and clustering/aggregation checked visually on PDFpage5, with page4 group construction. Not whole experiments or author code.'),
]
paper_meta=[]
for stem,expected,title,url,scope in paper_specs:
    p=A/(stem+'.pdf');t=A/'_text'/(stem+'.txt')
    assert p.read_bytes().startswith(b'%PDF-') and sha(p)==expected and t.exists()
    m={'title':title,'download_url':url,'download_status':'HTTP200_PDF_HEADER_AND_SELECTED_METHOD_VISUAL_CHECKED',
       'bytes':p.stat().st_size,'sha256':sha(p),'PDF':str(p),'text':str(t),'text_sha256':sha(t),'read_scope':scope,'recorded_utc':now}
    sidecar=p.with_suffix('.source.json');assert not sidecar.exists();write(sidecar,m)
    paper_meta.append((m,p,t,sidecar))

crossref=read(D/'VRWKV_DETR_Crossref_metadata.json')['message']
vrwkv={'title':crossref['title'][0],'doi':'10.1016/j.eswa.2026.134086',
       'url':'https://www.sciencedirect.com/science/article/abs/pii/S0957417426029921',
       'PDF_download_status':'NOT_OBTAINED; publisher XML only1858bytes metadata, openaccess0; direct webpage403, indexed publisher preview readable.',
       'read_scope':'Publisher abstract,introduction and section fragments only; not full method/code.',
       'cover_date':'2027-01-15','online_publication_date':'NOT_VERIFIED',
       'record_created_utc':crossref['created']['date-time'],
       'date_boundary':'Crossref created date is not online-publication date. Current preview accessibility is recorded; do not silently treat the2027 issue as an established2026 publication date.',
       'recorded_utc':now}
vr_side=A/'A25_VRWKV_DETR_public_access.source.json';assert not vr_side.exists();write(vr_side,vrwkv)

dspe_comparison={
 'paper_method':{'PE':'Learned PE dimension gates, hard threshold at inference; authors explicitly say tensor/QKV sizes and FLOPs remain unchanged.',
                 'local':'Per-token normalized activation norm, median split, two relative-bias tables for4/8 windows with soft transition; bias is added inside attention logits.',
                 'head':'Conv-stem P2 + SmallObjFPN + ATSS/WIoUv3; paper is not a DETR decoder implementation.'},
 'repository_pin':source['commit'],'verified_files':9,'verified_source_bytes':sum(x['bytes'] for x in source['files']),
 'read_source_scope':'README and complete DSPEModule, LocalPEEnhancer, PERedundancyPruner; selected algorithms.md and reimplementation_guide.md. Neck/loss files downloaded but not fully read.',
 'observed_reference_code':[
   {'file':'dspe_module/local_pe_enhancer.py','lines':[76,122,136,188,208],
    'observation':'One fixed-window bias table, window-mean activation divided by global mean, static pairwise-bias reduction/projection to token embedding, then a density-scaled output vector.'},
   {'file':'dspe_module/dspe_module.py','lines':[118,121,124],
    'observation':'Wrapper adds pruned PE and projected local vector to input tokens; it does not supply dual density bias tables inside a QK attention matrix.'},
   {'file':'dspe_module/redundancy_pruner.py','lines':[52,93,98,103,122],
    'observation':'Default temperature1.0, PE-derived MLP gates; inference hard-threshold code is commented out; target-mean retention penalty.'},
 ],
 'availability':'Nine public files are a reference-component package; no full backbone/training configs/checkpoints in this pinned tree. README itself describes S1 and says a complete release is pending.',
 'repository_metadata_note':'Repository description contains a different manuscript title. Treat the paper-linked repository as a reference snapshot and assess its contents; this is not a research-integrity finding.',
 'consequence':'Generic density-aware PE is prior. The available reference cannot be silently equated to the final paper operator or to a reproduced6M/15.8GFLOPs/accuracy result.',
 'not_proven':'Actual trained full-paper implementation, AP, resource use, LayerNorm input placement or a task failure in this reference.'
}
write(D/'DSPE_paper_vs_reference_operator_audit.json',dspe_comparison)

new_candidate={
 'id':'O_QUERY_BUDGET_EXPANSION_CONTEXT','status':'SOURCE_STAGE_UNTESTED_NO_APPROVED_INCREMENT',
 'question':'同图候选由较小预算扩到较大预算时，新增真实查询是否扰动已有候选，造成未被新增覆盖补偿的小目标检测损失？能否在相同最终候选数与图像编码下保留新增覆盖并稳定检测？',
 'related_existing_candidate':'D_DISTRIBUTIONAL_QUERY_RELATIONS; this subproblem concerns candidate-budget/context changes rather than uncertainty-derived pairwise relations.',
 'source_motivation':['VRWKV-DETR publisher preview describes inference-time expansion300to1500 with the same trained decoder; full details/date priority remain limited.','Inspected native Dome decoder uses inter-query self-attention; dependence is not proof of harmful task effects.'],
 'nearest':['BUTR','DQ-DETR','PaQ-DETR','DETRNN','EASE-DETR','Dual-R-DETR','Group DETR','GroupEnsemble','DN-DETR/native denoising mask'],
 'known_components_not_innovation':['Causal/one-way attention masks','Query groups and independent group attention','Query sorting or ordinary KV caching','Already-shared static image memory'],
 'possible_increment_NOT_CLEARED':'A task-specific staged query-expansion decoder using ordinary trained encoder-selected queries, with a testable old-query dependence contract and optional across-budget query-state reuse. Exact prior difference and AP benefit are unproven.',
 'source_difference_to_check':{'Group_DETR':'Group-wise supervised training, one group retained at inference.',
                              'GroupEnsemble':'Uses multiple groups already learned via Group DETR, symmetric inter-group isolation and BSAS aggregation for uncertainty; not a demonstrated ordinary-checkpoint prefix-expansion method.',
                              'DETRNN':'Confidence/similarity ordering and density grouping with bidirectional RNN/Mamba are existing; no first sorting/Mamba claim.',
                              'DN':'One-way blocking is already present in the native training mask; do not claim the adjacency pattern as invented.'},
 'causal_evidence':False,'increment_approved':False,'real_model_probe_executed':False,
 'distinct_from_padding_fixture':'The CPU padded-row assignment counterexample is an implementation risk, not evidence that adding genuine candidates harms inference.',
 'next':'Inspect exact decoder/cache inputs and closest methods before freezing any real-image test. Ordinary mask/KV reuse alone is insufficient; no new module training.'
}
write(D/'candidate_O_query_budget_context_DRAFT.json',new_candidate)
protocol={
 'status':'DRAFT_NOT_PRESPECIFIED_NOT_EXECUTED','new_execution_allocation':0,
 'purpose':'Separate coverage gain from the effect of additional candidate interactions in a trained detector.',
 'host_candidate':'Existing author VisDrone Dome-S weights, source2dde3bc, frozen eval mode; no new baseline training required for the initial diagnostic if exact interfaces are available.',
 'source_and_data_preconditions':['Verify common initial query IDs/content/reference boxes, same image memory and model weights.','Check available cached tensor positions and actual query lengths before setting budgets.','B1/no padding in the inference assay; no GT-derived query order or deployment mask.'],
 'candidate_budgets':[250,500],
 'arms_to_freeze_before_outcomes':['base-only n queries','native m queries','m queries with symmetric inter-group isolation control','m queries with old rows blocked from new keys but new rows allowed to read old context'],
 'controls':['Compare algorithm arms at exactly the same final m candidates and the same class/postprocess detection cap.','Keep encoder, candidate set, scores/heads, preprocessing and all other decoder paths fixed.','Check the expected old-prefix numerical identity before interpreting a mask intervention.','Do not identify a known grouped-mask control as a faithful reproduction of trained GroupEnsemble.'],
 'metrics':['Primary future task metrics: AP and APsmall on the same valid-GT/ignore policy; final protocol must state the policy and fixed detector output cap.','Common-prefix localization/score changes are diagnostic only; check whether extra candidates recover lost targets.','Duplicate false positives and distinct-query assignment must be checked. Restoring a max-over-prefix IoU bound by construction is not AP improvement.'],
 'timing_boundary':'Any KV-reuse benefit must be measured under an explicit increasing-budget workload; report single-shot m inference separately. Reusing already-shared image features is not a new speedup.',
 'rejection_conditions':['No reproducible uncompensated task harm from expansion.','The proposal reduces to existing masks/grouping without a distinct task-specific algorithm.','No AP benefit at the same final query count, or gains require posthoc NMS/threshold changes.'],
 'not_yet_frozen':['Final image/prefix cohort and independence limits','Exact source interface and numerical tolerances','Paired primary effect thresholds, uncertainty method and final call budget'],
 'no_A22_resume':True,'N_scientific_design_unchanged':True,
}
write(D/'candidate_O_minimum_validation_DRAFT_NO_ALLOCATION.json',protocol)

REPORT.write_text('''本轮没有确认新主线。已完成两类实质推进：把半精度风险推进到完整原生criterion接口的合成复现；新增核查编码器位置建模与推理查询扩容的最近方法，建立独立的O候选问题。N共享状态假设保留未验证，O也尚无真实图像效应或创新通过结论。

1. 检索策略与记录
围绕RoPE/稀疏位置编码/线性高分辨率编码器，以及query数量变化、分组、自注意力、增量解码和KV复用定向追踪。本轮18条检索式、6次网页直访与3个站内find，精确请求和原始结果保存于web_search_records.json。另保存论文链接GitHub仓库的元数据、提交、分支及确切树对象，Crossref和出版社公开XML入口，逐份下载9个源码/说明blob，归档3份公开PDF。
仅依主来源形成方法结论；搜索摘要不视为全文。LA-DETR仅有IEEE摘要，直访418，留在筛查记录，不据此排清算子。VRWKV页面直访403，出版社索引预览可读；XML只有书目且openaccess0，未取得全文。正式卷期为2027-01-15，Crossref创建于2026-08-18不等于在线发表时间；当前预览可访问，准确online日期仍未核验。

2. 核心文献矩阵与阅读范围
矩阵78增至82条：DSPE-ViT、VRWKV-DETR、Group DETR、GroupEnsemble。DSPE读完整Method3.1–3.4、实施4.2和部分复现讨论，核图PDF第5/6页；完整读公开DSPE三个模块与README，其他说明只选读，未复现整模型。VRWKV只读出版社摘要/导言/片段。
Group DETR采用ICCV2023正式PDF，读摘要及3.1算法/训练/推理相关段，核对Figure2/Algorithm1；未读完整分析/实验/代码。GroupEnsemble采用arXiv2603.01847v1，读摘要/导言、部分III-A和完整III-B–D，核对PDF第4/5页的组构造、屏蔽和聚合；未读完整实验或作者代码，不宣称已同行评议。

3. 与最近工作的逐项对比
DSPE-ViT已有密度条件位置建模与PE维度门控，不能把该宽泛方向当作空白。论文也明确说明PE置零不改变Q/K/V维度和FLOPs，不能错误批评它用置零直接省掉密集矩阵运算。当前公开参考代码却采用单个固定窗口表、窗口范数/全局均值缩放、将相对偏置归约投影为token向量再相加；论文最终算法采用中位数划分、两个偏置表和attention-logit内融合。代码推理硬阈值被注释，默认温度也不同。它能作为参考组件阅读，不能与完整论文实现或精度结果等同；没有发现完整骨干/训练配置/权重。9个文件均固定23b92e34198df838c8ed63259e6457e58100edf2并通过Git blob校验。
VRWKV出版社预览已提出线性RWKV高分辨率多尺度建模及同一训练decoder在推理300→1500查询扩容；不能声称首次把RWKV或更多查询用于密集小目标。完整方法与online时间边界保留。
Group DETR已有组内独立自注意力和一组一对一监督，推理保留一组。GroupEnsemble已将训练好的多组查询全部用于推理，组间双向屏蔽并做BSAS/不确定性聚合。因此，“多组查询＋attention mask”在推理阶段同样不是新意。原生DN训练mask也已有单向可见性模式。DETRNN/EASE/Dual-R等继续约束排序、竞争或路由的宽泛创新声明。

4. 文献支持但尚未证实的问题
新增O候选：同一图像的真实候选查询扩容时，新增查询经自注意力改变原候选，是否造成没有被新增覆盖补偿的微小目标检测损失？可否在相同最终候选数下保留覆盖收益并稳定检测？普通自注意力存在依赖，不等于这种依赖有害；本轮没有真实模型效应测量。
这一问题不同于补齐行实现风险，也不同于原D候选的定位不确定性关系问题。它从已有推理扩容方法引出，不能把“没有查到完全同名文章”当创新证明。组隔离、单向mask、排序和KV缓存均已有基础，具体算法差异仍需核查。

5. 候选增量、可行性与风险
O的暂拟方向是适用于普通已训练检测器的分阶段候选扩容，约束原查询受新查询影响的路径，并在明确递增预算场景下考虑查询状态复用。这是算法机制候选，不是已确认创新。现有Dome作者权重与源码可以作为首个冻结诊断宿主；尚未核对完整prefix/cache接口和新协议容量，不能声称新方法已能运行。图像memory早已共享，不能重新计作创新或速度收益。
风险是隔离有益上下文、固定错误旧预测、重复检测增加，或最终仅剩标准mask/KV复用的组合。若在相同最终候选数下没有任务收益，就不靠多给query、重调阈值或追加NMS救结果。DSPE论文报告的16GB可训练不代替本机资源实测；其参考包与论文的差异也不构成我们的论文创新。

6. 本轮验证与最小后续路线
CPU合成验证共5次原生criterion.forward、27次原生matcher.forward，完整检测器前向0、优化器更新0、真实图像0，CUDA未初始化。前3组使用原生接口处理普通/aux/pre/encoder分支：FP16出现具名loss_ddf_aux_0/1=NaN；仅将数学上应为零的DDF求和改用FP32累加后，该组合损失和梯度有限，FP32对照有限。随后2组增加GT[10,902]、DN1804、非相同student/teacher以及DEFE密度/回归分支：原生FP16在loss_ddf_dn_2复现NaN，零项累加修正后全部raw loss与20个合成叶张量梯度有限。原函数返回值会用nan_to_num把这些NaN置零，因此只看最终总loss会漏掉风险。
这些是CPU合成反例和对照，不是原2105批失败状态的重放，也不是GPU AMP全覆盖验证。当前宿主实际criterion类别为VFL/boxes/local，不是MAL。另一个两行候选合成例子证实原matcher可以在batch_queries_num=1时选中补齐的第2行；截断到有效集合后选第1行。尚无真实发生频率、梯度影响或AP证据，不能据此称新架构问题已成立。
原A22冻结源码未改、任务未续跑、没有新训练或新评估。历史累计完整调用2365、更新528（其中526正式、2容量）不变，原分配已关闭，剩余额度不自动转用。
O后续先查接口和素材，拟比较base-only n、普通m、双向组隔离m和单向隔离m；图像编码、查询初值、输出上限与最终候选数相同。公共前缀IoU/分数变化只作机制指标，须检查新增候选有无补偿原检测损失；主要任务证据仍应是同口径AP/APsmall与重复误检。通过硬约束恢复旧输出的max-IoU下界本身不是检测改进。具体样本、门槛、容差和预算尚未冻结，DRAFT明确执行分配0，不是已执行协议。

7. 可核验参考与链接
DSPE-ViT: a lightweight vision transformer with dynamic sparse positional encoding for dense small object detection in UAV imagery. Cai,L.;Li,S. Frontiers in Neurorobotics20(2026),1849093. https://doi.org/10.3389/fnbot.2026.1849093
论文链接参考仓库：https://github.com/LiyaCai-001/DSPE-ViT-1/tree/23b92e34198df838c8ed63259e6457e58100edf2
VRWKV-DETR: Boosting high-resolution small object detection with linear attention and dense queries. Zhang,Y.;Wang,X.;Liu,J.;Zhang,W. ESWA333(2027),134086；本轮只读预览。 https://doi.org/10.1016/j.eswa.2026.134086
Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment. Chen,Q.et al. ICCV2023. https://doi.org/10.1109/ICCV51070.2023.00610
GroupEnsemble: Efficient Uncertainty Estimation for DETR-based Object Detection. Yang,Y.et al. arXiv2603.01847v1(2026). https://arxiv.org/abs/2603.01847v1
DETRNN与BUTR的既有阅读范围继续见核心矩阵：https://arxiv.org/abs/2610.05191 ，https://arxiv.org/abs/2610.33430v1 。不把它们的结论当作本模型实验。

61回执保存源文件哈希、合成数值、阅读范围及版本；3份PDF按项目规则进入中央文献目录，VRWKV仅存访问限制/元数据记录。目标保持ACTIVE。当前有15个候选记录、82条核心文献，仍没有确认可成文的主线。
''',encoding='utf-8')

backup=D/'canonical_before_A25';backup.mkdir(exist_ok=False)
for name in NAMES:shutil.copy2(B/name,backup/name)
old=read(B/'59_A24文献与CPU数值核验回执.json')
for path,digest in old['files_sha256'].items():
    original=Path(path)
    check=backup/original.name if original.parent==B and original.name in NAMES else original
    assert sha(check)==digest,(path,'prior evidence changed before A25')

phase={'phase':'A25','classification':'progress','recorded_utc':now,
 'status':'FULL_CRITERION_SYNTHETIC_FAILURE_REPRODUCED_NEW_DECODER_CANDIDATE_SOURCE_STAGE',
 'goal_status':'active','mainline_validated':False,'complete_mainline':False,'report':str(REPORT),'receipt':str(RECEIPT),
 'new_detector_forwards':0,'new_optimizer_updates':0,'new_real_images':0,'new_training_jobs':0,
 'full_model_forwards':2365,'actual_cumulative_detector_optimizer_updates':528,'formal_detector_training_updates':526,
 'historical_capacity_updates':2,'cumulative_authorized_protocol_cap':39092,'full_forward_cap_closed':True,'counts_terminal':True,
 'native_CPU_criterion_calls':5,'native_CPU_matcher_calls':27,'synthetic_backward_calls':3,
 'actual_A22_failure_root_cause_proven':False,'A22_restarted':False,'N_protocol_changed':False,
 'new_core_papers':4,'core_matrix_rows':82,'candidate_count':15,'new_candidate':'O_QUERY_BUDGET_EXPANSION_CONTEXT',
 'O_real_model_effect_measured':False,'O_algorithm_novelty_cleared':False,'O_execution_allocation':0,
 'DSPE_reference_git_blobs_verified':9,
 'next':'先核查O“真实查询扩容的集合上下文扰动”的精确decoder/cache接口、可用素材和最近方法差异，之后才冻结受控诊断。标准mask、排序与KV复用不作为创新；N保留未验证，原A22终止，不训练新模块。'}

board=read(B/NAMES[0]);assert len(board['candidates'])==14
board['candidates'].append(new_candidate)
board.update(status='NO_CONFIRMED_MAINLINE_QUERY_BUDGET_CONTEXT_CANDIDATE_OPEN',mainline_validated=False,latest_update_utc=now,latest_report=str(REPORT),current_priority_question=phase['next'])
board['A25_evidence']=phase
board['next_exploratory_question']=new_candidate['question']
board['rejected_generic_novelty_claims'].append('A25: Density-aware PE, query grouping, inference group masks, one-way masks, query ordering and ordinary KV reuse have prior mechanisms. O is not approved without a distinct algorithm and matched-final-budget task benefit.')
write(B/NAMES[0],board)
search=read(B/NAMES[1])
search['A25_encoder_and_query_budget_followup']={'recorded_utc':now,'new_search_queries':18,'web_open_calls':6,'web_find_calls':3,'raw_web_records':str(D/'web_search_records.json'),'primary_API_files':[str(D/x) for x in ['DSPE_author_repository_metadata.json','DSPE_author_commit.json','DSPE_author_tree.json','DSPE_author_exact_tree.json','VRWKV_DETR_Crossref_metadata.json','VRWKV_publisher_XML_access.xml']],'source_blob_files':9,'new_PDFs':3,'full_text_limits':vrwkv,'report':str(REPORT)}
write(B/NAMES[1],search)

matrix=read(B/NAMES[2]);assert len(matrix['rows'])==78
rows=[
 {'id':'DSPE_ViT','title':paper_specs[0][2],'authors':['Liya Cai','Shuping Li'],'doi':'10.3389/fnbot.2026.1849093','url':'https://doi.org/10.3389/fnbot.2026.1849093','tier':'Frontiers in Neurorobotics20,1849093(16June2026)','read_scope':paper_meta[0][0]['read_scope']+' Public source scope is limited as documented in DSPE_paper_vs_reference_operator_audit.json.','author_method':'PE dimension gating and density-conditional relative-position bias, SmallObjFPN/ATSS/WIoUv3.','our_consequence':'Generic density-aware PE is already direct UAV prior; available reference uses a different local-vector operator and is not a reproduced full detector.','unresolved':'Complete trained pipeline/checkpoints unavailable in the pinned nine-file reference; no independent AP/resource measurements.','source_commit':source['commit'],'source_operator_audit':str(D/'DSPE_paper_vs_reference_operator_audit.json')},
 {'id':'VRWKV_DETR','title':vrwkv['title'],'authors':['Yu Zhang','Xufan Wang','Jiaxin Liu','Wenyuan Zhang'],'doi':vrwkv['doi'],'url':vrwkv['url'],'tier':'ESWA333(2027),134086; publisher preview accessible, exact online date unverified','read_scope':vrwkv['read_scope'],'author_method':'Preview describes high-resolution multiscale linear RWKV modeling and same-trained-decoder inference query expansion300to1500.','our_consequence':'Generic RWKV transfer or more inference queries is not a new claim; motivates an untested query-context expansion question.','unresolved':'Full method/code and online date not verified; no reproduction; metadata created date is not publication priority.','access_record':str(vr_side)},
 {'id':'Group_DETR','title':paper_specs[1][2],'authors':['Qiang Chen','Xiaokang Chen','Jian Wang','Shan Zhang','Kun Yao','Haocheng Feng','Junyu Han','Errui Ding','Gang Zeng','Jingdong Wang'],'doi':'10.1109/ICCV51070.2023.00610','url':'https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Group_DETR_Fast_DETR_Training_with_Group-Wise_One-to-Many_Assignment_ICCV_2023_paper.pdf','tier':'ICCV2023 formal version','read_scope':paper_meta[1][0]['read_scope'],'author_method':'Independent within-group self-attention and group-wise one-to-one assignment during training; one group retained for inference.','our_consequence':'Query grouping and separated attention are prior; need exact comparison rather than calling a mask new.','unresolved':'Not all analysis/experiments/code read; no our tiny-object effect or inference-prefix improvement follows.'},
 {'id':'GroupEnsemble','title':paper_specs[2][2],'authors':['Yutong Yang','Katarina Popović','Julian Wiederer','Markus Braun','Vasileios Belagiannis','Bin Yang'],'url':'https://arxiv.org/abs/2603.01847v1','tier':'arXivv1,02March2026; not asserted peer reviewed','read_scope':paper_meta[2][0]['read_scope'],'author_method':'Inference with multiple query groups already learned by Group DETR; symmetric group-isolation mask/batch implementation; BSAS clustering and confidence-weighted spatial uncertainty aggregation.','our_consequence':'Inference query-group isolation is direct prior. Does not establish ordinary-checkpoint prefix expansion or our task benefit.','unresolved':'No author-code or experiment reproduction; additional groups must already be learned/high-quality; uncertainty objective differs from new tiny-object AP hypothesis.'},
]
assert not any(row['id'] in {x['id'] for x in rows} for row in matrix['rows'])
matrix['rows'].extend(rows)
matrix.update(updated_utc=now,latest_report=str(REPORT),latest_empirical_boundary='A25:5 CPU criterion calls and27 matcher calls on synthetic tensors only;0detector calls/updates/images. O is source-stage, N untested; A22 remains terminal without AP.')
write(B/NAMES[2],matrix)
state=read(B/NAMES[3])
state.update(recorded_utc=now,goal_status='ACTIVE',mainline_validated=False,diagnostics_complete=False,model_forwards=2365,training_updates=528,formal_detector_training_updates=526,cap=39092,full_forward_cap_closed=True,counts_terminal=True,latest_report=str(REPORT),latest_authoritative_phase=phase,current_priority_question=phase['next'],current_goal_turn='A25_full_native_criterion_and_new_encoder_decoder_source_audit')
state.setdefault('authoritative_phase_history',[]).append(phase)
write(B/NAMES[3],state)
audit=read(B/NAMES[4]);audit.update(updated_utc=now);audit['A25_native_criterion_and_new_architecture_sources']=phase
audit['DSPE_reference_operator_audit']=str(D/'DSPE_paper_vs_reference_operator_audit.json')
audit['O_query_budget_context']=new_candidate
write(B/NAMES[4],audit)

files=[p for p in D.iterdir() if p.is_file()]+source_paths+[REPORT,Path(__file__),vr_side]+[B/name for name in NAMES]
for m,p,t,sidecar in paper_meta:files.extend([p,t,sidecar])
receipt=dict(phase)
receipt.update(previous_goal_turn_classification='progress: A24 verified a numeric operator counterexample and excluded broad novelty using full primary methods.',
               files_sha256={str(path):sha(path) for path in files},
               previous_A24_files_verified_with_canonical_backup=len(old['files_sha256']),
               CPU_normal_cases=cpu,CPU_dense_DN_cases=dense,DSPE_operator_audit=dspe_comparison,
               O_candidate=new_candidate,O_protocol_status='DRAFT_NO_ALLOCATION',paper_metadata=[x[0] for x in paper_meta])
write(RECEIPT,receipt)
for path,digest in receipt['files_sha256'].items():assert sha(path)==digest,path
assert len(read(B/NAMES[2])['rows'])==82 and len(read(B/NAMES[0])['candidates'])==15
assert read(B/NAMES[3])['model_forwards']==2365 and not read(B/NAMES[0])['mainline_validated']
print(json.dumps({'status':'RECORDED_AND_VERIFIED','files_verified':len(receipt['files_sha256']),'matrix_rows':82,'candidates':15,'new_detector_calls':0,'new_training_jobs':0,'mainline_validated':False,'report':str(REPORT)},ensure_ascii=False))
