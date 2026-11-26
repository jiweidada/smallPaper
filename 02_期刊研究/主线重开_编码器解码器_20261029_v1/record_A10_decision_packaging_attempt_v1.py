"""Archive the completed parameter-free screen and reconcile research state."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A10_跨实例负去噪参考框几何审计_v1'
NOW=datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


s=read(OUT/'summary.json')
v=read(OUT/'independent_verification.json')
c=read(OUT/'completed.json')
assert sha(OUT/'summary.json')==c['summary_sha256']
assert sha(OUT/'protocol.json')==c['protocol_sha256']
assert s['decision']=='DEPRIORITIZE_CROSS_INSTANCE_NEGATIVE_DN_FROM_LOW_GEOMETRIC_EXPOSURE'
assert v['target_draw_events_verified']==71856 and v['tiny_exposed_target_draws']==34
assert s['cumulative_full_model_forwards']==200

report=ROOT/'32_A10负去噪近邻对比与主线候选淘汰.txt'
report.write_text('''当前判断：跨实例负去噪冲突不列为主线；微小目标几何定位仍是问题方向，算法增量尚未成立。

1. 本轮实际完成
固定RHINO作者仓库d9baf3bc1149ebef2ef03c89b26b93106b82075a，下载10个相关源码文件并逐个验证Git blob；没有安装或执行RHINO模型。通过Europe PMC公开接口取得CCDN-DETR完整XML，核实DOI10.3390/s24061793；本轮完整阅读去噪方法3.4.1/3.4.2及图4说明，未将整篇标为已读。
对Dome作者的参数无关去噪生成函数进行CPU等价检查，7类固定合成输入通过。共享随机数组、恒等类别嵌入，最大logit差9.54e-7、初始参考框差1.19e-7。首次检查因Tensor注解未绑定而退出，随后仅修复加载命名空间；原始冻结文件、噪声规则、次数、阈值与判据未修改，失败和兼容修复均留存。

2. 最近工作的逐项边界
CCDN-DETR（Sensors2024）：限制各真实框周围正负噪声范围，并约束极小负框尺寸；负样本仍接受背景分类监督。已读方法未出现跨全部其他GT的碰撞检查，但未取得作者噪声生成源码，不能据此声称所有实现都没有该检查。
https://www.mdpi.com/1424-8220/24/6/1793
https://doi.org/10.3390/s24061793

RHINO（WACV2025；本轮读arXiv v6，2026-05-05）：在正去噪预测与普通预测之间加入匹配，减少不合适的正去噪监督；负去噪背景项保留。其公开PHC配置与头部源码也将原负去噪项保留为背景。论文公式对过滤正样本的描述与已查PHC代码处理有所不同，未将二者宣称为完全相同的算子。论文限制节报告COCO水平框迁移下降1.6AP，属作者结果，未复现。
https://arxiv.org/html/2305.07598v6
https://github.com/SIAnalytics/RHINO/tree/d9baf3bc1149ebef2ef03c89b26b93106b82075a

Shadow-DETR（Neural Networks2026）：相似负query的分类冲突及额外回归监督已有直接先例；当前仍为摘要、引言和方法片段阅读，未读完整匹配规则，不宣称去重完成。
https://doi.org/10.1016/j.neunet.2025.108524

STEP-DETR（ICCV2025）：已读3.5及3.6部分，背景负查询的输入嵌入处理也已有专门设计。它属于半监督文本引导方法，不能仅换到VisDrone就称为新思路；未取得并执行源码。
https://openaccess.thecvf.com/content/ICCV2025/papers/Shehzadi_STEP-DETR_Advancing_DETR-based_Semi-Supervised_Object_Detection_with_Super_Teacher_and_ICCV_2025_paper.pdf

3. A10冻结审计方案
只复用既有100张训练图的4491个有效GT，800方形输入尺度下552个GT属于sqrt面积<8像素。每图16次固定抽样；Dome原生噪声尺度1.0、标签噪声比0.5、12类输入编码、角点扰动、绝对尺寸修正与inverse_sigmoid/sigmoid均保留。
主要事件：每个GT在一次抽样中，是否被第一去噪组中“源于另一个GT、实际输入类别与该目标一致”的负参考框以IoU>=0.5覆盖。选择首组以避免真实batch最大GT数量造成组数混杂。先对图内GT及16次抽样取均值，再等权平均图像，并按文件前缀分组bootstrap。
预设优先级门槛：微小GT>=200、前缀组>=15；图均率>=5%且区间下界>2%才进入下一项因果审计；区间上界<2%则降级。此门槛是资源筛选规则，不是统计显著性或论文创新证据。

4. 结果与独立核验
总共生成144896个负参考框（包含原生B1全部组）。微小GT的主要事件：34/8832个目标抽样对，合并率0.38496%；按图平均0.35507%，分组bootstrap区间[0.20670%,0.49671%]，49张含微小GT图、29个前缀组。
不要求输入类别一致的次要图均率为0.59311%；使用B1所有去噪组且要求类别一致为0.44423%。这些是预定敏感性结果，未用于改换主要判据。自身源GT重叠的次要图均率为0.33572%，同样不据此建立新主线。
已独立用标量矩形相交/并集计算重建全部71856个GT抽样对的主要事件，与缓存标志逐项一致；100份参考框NPZ的SHA、逐GT计数和按图均值一致。次要阈值与所有组的事件没有另做独立重算，明确保留该核验范围。

5. 科学结论与限制
【本次测量】在固定有效GT与固定噪声模型下，主要跨实例几何歧义事件较少，未过预设门槛。
【判断】当前不优先投入负去噪冲突的模块设计或训练，不通过降低IoU阈值、挑拥挤子集或增加seed重新寻找正结果。
【未证明】初始参考框重叠不等于最终输出误标，也不证明梯度有害；没有训练增强和真实batch过程，不能推断完整训练中的发生率。忽略/未标注目标不在统计范围。未将A9框定位不足归因于去噪。

6. 对主线的影响
上一目标回合完成A9和近邻证据归档，属于有实际进展；本回合又完成近邻方法/源码核查和A10淘汰，也属于进展。当前没有通过机制与新颖性双重核查的论文主线。
后续回到编码器表示、特征读取和解码器框坐标生成的结构机制；不继续压缩恢复分支，不恢复旧DEIM任务，不把通用非均匀分布回归、跨层教师或负样本忽略改名为创新。下一项结构干预需独立的新协议和可证伪依据，当前200次完整前向阶段已结束，不自动扩大。
本轮新增完整模型前向0、拟合0、主模型更新0；累计仍200次完整前向、64个诊断线性模型/96个标量响应头。CPU参考框审计不能证明单卡24GB的训练可行性。目标继续ACTIVE。
''',encoding='utf-8')

source_audit={
    'utc':NOW, 'RHINO': {'commit':'d9baf3bc1149ebef2ef03c89b26b93106b82075a','download_verification':str(ROOT/'comparison_sources_v1'/'RHINO_A10_v1'/'download_verification.json'),'downloaded_files':10,'read_scope':'Selected config,PHC/PH negative targets,detector initialization and complete bbox-noise generator; primary3.5,Algorithms1/2,limitationG and selected surrounding sections. No full dependency closure or model execution.','negative_DN_observation':'Original negative indices remain background in inspected PHC code; DN assigner receives only positive-DN predictions.','paper_code_boundary':'v6 filtered-positive formula versus pinned PHC background-label treatment differs; not resolved as identical implementations.'},
    'CCDN': {'doi':'10.3390/s24061793','publisher':'https://www.mdpi.com/1424-8220/24/6/1793','fulltext_xml':str(ROOT/'source_audit_v1'/'A10_CCDN_EuropePMC_fulltext.xml'),'xml_sha256':sha(ROOT/'source_audit_v1'/'A10_CCDN_EuropePMC_fulltext.xml'),'reading':'Complete3.4.1/3.4.2 andFigure4 caption; prior selected framework/setup/ablations. Full XML acquired is not equivalent to full article read.','source_not_acquired':True},
    'Shadow':{'reading':'Abstract,introduction and section snippets only; full method/source not obtained.'},
    'A10':{'protocol':str(OUT/'protocol.json'),'result':str(OUT/'summary.json'),'independent_verification':str(OUT/'independent_verification.json'),'decision':s['decision'],'new_full_model_forwards':0,'new_fits':0},
}
write(OUT/'closest_operator_audit.json',source_audit)

# Bundle immutable result evidence and code; source repositories retain their own receipts.
backup=OUT/'backup'
assert not backup.exists()
backup.mkdir()
code_names=['a10_dn_geometry.py','a10_author_operator_equivalence.py','a10_author_operator_equivalence_compat_v1.py','prepare_A10_geometry.py','run_A10_geometry_once.py','record_A10_compatibility.py','verify_A10_geometry.py','record_A10_decision.py']
items=[(p,str(p.relative_to(OUT))) for p in OUT.rglob('*') if p.is_file() and backup not in p.parents]
items += [(ROOT/n,'code/'+n) for n in code_names]
archive=backup/'A10_parameter_free_geometry.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for path,name in items:
        t.add(path,arcname=name)
file_manifest=[{'path':name,'bytes':p.stat().st_size,'sha256':sha(p)} for p,name in items]
with tarfile.open(archive) as t:
    assert len(t.getmembers())==len(file_manifest)
    for r in file_manifest:
        raw=t.extractfile(r['path']).read()
        assert hashlib.sha256(raw).hexdigest()==r['sha256']
write(backup/'file_manifest.json',{'files':file_manifest})
archive_receipt={'status':'PASS_ARCHIVE_MEMBER_HASHES','files':len(file_manifest),'bytes':archive.stat().st_size,'sha256':sha(archive),'archive':str(archive)}
write(backup/'receipt.json',archive_receipt)

names=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
snapshot=ROOT/'history_before_A10_update_v1'
assert not snapshot.exists()
snapshot.mkdir()
for n in names: shutil.copy2(ROOT/n,snapshot/n)
records={n:read(ROOT/n) for n in names}
board=records[names[0]]
board.update({'latest_update_utc':NOW,'latest_report':str(report),'remaining_deliverable':'Identify a specific measured architecture-dependent geometric-precision mechanism and a nontrivial difference from closest operators before choosing an algorithm; validation and training resources remain unresolved.','next_decision_protocol':'A10 cross-instance DN frequency gate failed. No DN module/training or extra forward in the completed200-forward phase. Reopen actual encoder/decoder coordinate-generation mechanism screening.'})
board['source_only_candidate_not_mainline']={'question':'Cross-instance negative-DN ambiguity','status':s['decision'],'result':str(OUT/'summary.json'),'prior_source_scope':str(OUT/'closest_operator_audit.json'),'novelty_not_established':True}
board['A10_result']={'decision':s['decision'],'primary':s['primary'],'tiny_exposed_target_draws':34,'tiny_target_draws':8832,'archive':archive_receipt,'no_algorithm_approved':True}
board['new_empirical_evidence']='200 complete model forwards;64 diagnostic fitted models/96 scalar heads. A9 cached capacity analysis and A10 parameter-free DN references add no fits,full forwards or main-model updates. A10 primary occurrence low;candidate deprioritized.'
board['candidates'].append({'id':'G_CROSS_INSTANCE_NEGATIVE_DN_AMBIGUITY','question':'Do negative DN references coincide with another tiny real target and create a useful distinct decoder-training problem?','status':s['decision'],'evidence':'100 fixed train images4491GT,552tiny;primary34/8832,target-draw exposures;image mean.355%,CI[.207%,.497%].','increment_approved':False,'causal_evidence':False,'nearest':['CCDN-DETR','RHINO','Shadow-DETR','NAN-DETR','STEP-DETR'],'next':'No module or training. Retain as a negative screen;do not tune thresholds/subsets to promote it.'})

log=records[names[1]]
raw=sorted((ROOT/'source_audit_v1').glob('A10_*.json'))
log['supplement_utc']=NOW
log['A10_primary_methods_and_parameter_free_screen']={'raw_web_responses':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in raw],'query_inputs':'Each archived JSON includes the exact request and response. Broad search also returned irrelevant/secondary hits;these were not method evidence.','fulltext_success':'EuropePMC fullTextXML for PMC10974197;DOI verified against XML,PubMed and publisher.','limited_access':['Publisher CCDN HTML429/XML403;official archive XML succeeded.','Shadow full method remains unavailable;no paywall bypass.','Web tool API endpoints inaccessible;public author-code metadata read through CLI.'],'source_audit':source_audit,'A10_archive':archive_receipt,'report':str(report)}
log['query_groups'].append({'name':'A10 exact search requests','requests':[{'archive':str(p.relative_to(ROOT)),'request':read(p)['request']} for p in raw]})

matrix=records[names[2]]
rhino=next(r for r in matrix['rows'] if r['id']=='RHINO')
rhino.update({'url':'https://arxiv.org/html/2305.07598v6','tier':'WACV2025; primary arXiv v6 revised2026-05-05','read_scope':source_audit['RHINO']['read_scope'],'author_method':'Bipartite positive-query filtering;improved variant preserves localization loss. Inspected PHC code keeps original negative-DN background supervision.','our_consequence':'Generic harmful-DN filtering is prior art; source and paper variants must be specified. Authors reportCOCO horizontal transfer-1.6AP,not reproduced.','unresolved':'No full model executed. Filtered-positive formula/source treatment difference unresolved;no exact all-other-GT collision safeguard claim beyond inspected files.','source_commit':source_audit['RHINO']['commit']})
ccdn=next(r for r in matrix['rows'] if r['id']=='CCDN_DETR')
ccdn.update({'doi':'10.3390/s24061793','url':'https://www.mdpi.com/1424-8220/24/6/1793','tier':'Sensors2024,24(6),1793','read_scope':source_audit['CCDN']['reading'],'author_method':'Bounds positive/negative noise around each source GT and constrains minimum negative size;negative samples use background classification.','our_consequence':'Constrained denoising is prior art. A10 source-bound cross-instance exposure is low anddoes not support a new module.','unresolved':'No author generator source acquired or executed;no assertion about uninspected implementations.'})
assert not any(r['id']=='STEP_DETR' for r in matrix['rows'])
matrix['rows'].append({'id':'STEP_DETR','title':'STEP-DETR: Advancing DETR-based Semi-Supervised Object Detection with Super Teacher and Pseudo-Label Guided Text Queries','url':'https://openaccess.thecvf.com/content/ICCV2025/papers/Shehzadi_STEP-DETR_Advancing_DETR-based_Semi-Supervised_Object_Detection_with_Super_Teacher_and_ICCV_2025_paper.pdf','tier':'ICCV2025','read_scope':'Primary indexed complete3.5 and beginning3.6 only;not full methods/paper.','author_method':'Teacher/pseudo-label text guided DN queries;background-negative input embeddings receive different/random text conditioning.','our_consequence':'Generic changes to negative-query input-label/text conditioning have prior art.','unresolved':'No source acquired/executed;semi-supervised setup differs;no cross-GT safeguard established from this reading.'})
matrix.update({'updated_utc':NOW,'latest_report':str(report),'latest_empirical_boundary':'A10 parameter-free geometry: low cross-instance negative-DN exposure under fixed GT/noise assumptions. Does not establish harmful supervision,architecture cause or novelty.'})

status=records[names[3]]
assert status['model_forwards']==200 and status['mainline_validated'] is False
status['authoritative_phase_history'].append(copy.deepcopy(status['latest_authoritative_phase']))
status['latest_authoritative_phase']={'utc':NOW,'classification':'progress','goal_status':'active','complete_mainline':False,'full_model_forwards':200,'global_cap':200,'bounded_phase_complete':True,'main_model_training_updates':0,'CPU_diagnostic_linear_models':64,'scalar_response_heads':96,'A10_new_full_forwards':0,'A10_new_fits':0,'A10_parameter_free_negative_refs':144896,'A10_author_operator_synthetic_cases':7,'A10_primary_target_draws_independently_verified':71856,'A10_decision':s['decision'],'A10_archive':archive_receipt,'latest_report':str(report),'next':'Return to actual architecture-specific localization mechanisms;no DN module or extra forward in the completed phase.','remaining_uncertainty':['Causal geometric-precision mechanism','Novel architecture operator','Independent validation','Training time/memory feasibility']}
status.update({'recorded_utc':NOW,'previous_goal_turn':status['current_goal_turn'],'current_goal_turn':'progress_A10_primary_methods_pinned_source_and_parameter_free_negative_screen','latest_report':str(report)})
audit=records[names[4]]
audit.update({'updated_utc':NOW,'A10_DN_closest_and_operator_audit':source_audit,'current_novelty_boundary':'No approved mainline or operator. A10 cross-instance negative-DN candidate deprioritized by low source-bound geometry occurrence;nearest work scope remains explicit.'})
for n,val in records.items(): write(ROOT/n,val)
receipt={'utc':NOW,'status':'PASS_A10_PARAMETER_FREE_SCREEN_ARCHIVED_AND_RECONCILED','goal_status':'ACTIVE','previous_goal_turn_classification':'progress_A9_completed_capacity_analysis_and_primary_evidence_archive','current_goal_turn_classification':'progress_A10_primary_methods_source_and_negative_geometry_gate','mainline_validated':False,'full_model_forwards':200,'new_full_forwards':0,'new_fits':0,'main_detector_updates':0,'diagnostic_fitted_models':64,'scalar_response_heads':96,'A10':{'decision':s['decision'],'primary':s['primary'],'independent_verification':v,'archive':archive_receipt},'matrix_rows':len(matrix['rows']),'files':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [report]+[ROOT/n for n in names]],'not_verified':['Novelty','Causal localization mechanism','Independent generalization','Training feasibility']}
write(ROOT/'33_A10完成核验与候选淘汰回执.json',receipt)
print(json.dumps({'status':receipt['status'],'goal_status':'ACTIVE','mainline_validated':False,'full_forwards':200,'matrix_rows':len(matrix['rows']),'archive':archive_receipt},ensure_ascii=False))
