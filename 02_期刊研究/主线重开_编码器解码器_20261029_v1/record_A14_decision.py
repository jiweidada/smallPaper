"""Reconcile primary encoder reading and completed algebra controls, not model runs."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
A14 = ROOT / 'A14_新编码器位置保留近邻审计_v1'
NOW = datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

done = read(A14/'completed.json'); result = read(A14/'algebra_summary.json')
for key, name in [('protocol_sha256','algebra_protocol.json'),('summary_sha256','algebra_summary.json'),('cases_sha256','algebra_cases.json')]:
    assert done[key] == sha(A14/name)
assert result['verified_git_blobs']==18 and result['new_full_model_forwards']==0
assert result['new_fits']==result['main_model_updates']==0 and result['cumulative_full_model_forwards']==213
receipts = [read(ROOT/'comparison_sources_v1'/(n+'_A14_operator_source_receipt.json')) for n in ['LPS','SPD','SAPA','LoftUp']]
for rec in receipts:
    for f in rec['files']:
        p=Path(rec['local'])/f['path']; assert sha(p)==f['sha256']

source_audit = {'utc':NOW, 'status':'COMPLETED_LISTED_SOURCE_OPERATOR_COMPARISON_NOT_AUTHOR_MODEL_REPRODUCTION',
    'sources':receipts, 'observations':[
        {'method':'SPD-Conv','paper_scope':'Complete3.1/3.2and4.1;notentirepaper.',
         'code_scope':'common.pyspace_to_depth93-102,Focus212-221;suppliedspace_depth_s.yaml13-50;README. No fulltrainingimport/parser closure.',
         'fact':'space_to_depth concatenates[00,10,01,11]. Focus packs then convolves. Several suppliedsmall-config reductions instead listConv(stride1),space_to_depth,C3.',
         'inference':'Do not label everyauthorstage as the same pack-then-linearconv orclaimwholeSPD-Conv is lossless.'},
        {'method':'LPS','paper_scope':'Complete4.1/4.2/4.3;selectedpreliminaries;notallappendixproofs.',
         'code_scope':'Selectedpolydown/polyup/lps_utils/logitlayer/segmentationfactory plusREADME. No author model execution.',
         'fact':'Downsample returns probability/index information;unpool_multistage passesp_dict[layer] to the corresponding upsampler.',
         'inference':'Learningphasechoice andpassingphaseprovenancetoadecoder are existingmechanisms.'},
        {'method':'Shift-equivariantViT','paper_scope':'arXiv2305.16316v2 section4andAppendixA2.2example;notfullCVPR2024finalversionclosure.',
         'fact':'Adaptive token/window/patchmerging andcircularRPE;theencoder-decoder example passesboth mergingandwindowindices into unroll/unpool.',
         'inference':'Generic encoder-to-decoder position-index transport isalreadyexplicitprior.'},
        {'method':'SAPA_family','paper_scope':'2022complete3/4.1/4.2andselected5/6;2023extensionIII-BandIII-C excerpts,notwholeextension.',
         'code_scope':'sapa_base.py,sapa_deform.py,deform_func.py,README only;CUDAkernelsnotinspected,compiledorexecuted.',
         'fact':'SAPADeform getsqueries fromhighresolutiony,keys/offsets fromlowresolutionx,and reassemblesx asvalues;offsetshapecanvarybyupsampledpoint.',
         'inference':'Fineguidance,content-similarity reassemblyanddynamicpointselection alreadyhave a directprior.'},
        {'method':'LoftUp','paper_scope':'Completecoordinate-basedrepresentation3,4.1masktraining,and4intro;notcomplete4.2orappendix.',
         'code_scope':'LoftUp.forward,ImplicitFeaturizer,CrossAttentionLayer,CATransformer;selectedstage1configandREADME.',
         'fact':'RGB+Fouriercoordinatequeries readlowresolutionkeys/values;queryresidualandpointwiseFFNarepresent;attentionstackhasnoquery-queryattention.',
         'inference':'Given identicalprecomputedqueryrows/fixedkeysvalues,subsettingrowscommuteswiththisdeterministicattentionstack. DenseRGBconvolutions,globalMinMaxstatisticsandBNtrainingpreventanunqualifiedend-to-endsparseexecutionclaim.'}],
    'limits':['18listedGitblobsverifieddoesnotmean18filesallreadorfullrepositoryclosure.',
        'No real-imagealiasingfailureoralgorithmgainmeasured.',
        'No claimthatallphase-awarearchitecturesare equivalent;algebraassumptions areexplicit.']}
write(A14/'source_operator_comparison.json',source_audit)

next_note = {'utc':NOW,'status':'EXPLORATORY_QUESTION_NOT_EXECUTABLE_PROTOCOL_OR_APPROVED_INCREMENT',
    'question':'在相同query和细化预算下，局部特征随图像位置变化的方向是否能为微小框提供更准确的位置更新？',
    'distinctions':['ImagefeatureJacobian dF/dpixelposition differsfromthe interpolationcoefficientranktestedbyA0.',
        'A5 scalarcenterfieldTaylorcontrolwasnegative;thatnegativeisretainedanddoesnotproveaquery-conditionedvectoralignmentworks.',
        'DEQDetlatentqueryrefinementJacobian isdifferentfrom image-coordinatefeatureJacobian;neither genericJacobiannorfixedpointuseisnew.'],
    'known_nearest':{'PixLoc':'PrimaryfeaturemetricLM3.1and3.2read;needsreferenceimage/3Dpointdescriptors ratherthanunknownobjectqueries.',
        'DEQDet':'Primary3.2/3.3read:sharedimplicitdecoder,RAG,RAP;notnewtoaddrefinementawareness.',
        'TAPTRv2':'Knownkey-awareattentionandpositionupdatecontrol.', 'DARK':'KnownscalarheatmapTaylorcontrol;A5negative.'},
    'possible_operator_only':'Aquery-compatibletargetembeddingandlocalfeaturemetriccouldproduce a damped2Dpositionstep;thisis an unverifiedadaptation,notclaimedorimplementedasnew.',
    'requirements_before_any_model_run':['Defineinference-timequerytargetwithoutGTorreference-imageleakage.',
        'ShowoperatoralgebraanddifferencefromPixLoc/TAPTRv2/DEQDetandothergradient-basedboxrefiners.',
        'Specifywhichnativequery/memorytensorsandcoordinateunitsareused;do not mistakea detachedautogradpathforzero spatialinformation.',
        'Definefaithfulknown-methodandmatched-capacityMLPcontrols,plus spatial-derivativepermutationcontrol.',
        'Freezeindependentfinitecapture/trainingbudgetandprediction-blinddata/primarymetricbeforeexecution.'],
    'falsifiers':['No inference-timetargetwithoutGTcanbedefined.','Knowncontrolgivesalltheeffect.',
        'MatchedcapacityMLPorspatiallyshuffledderivativesgiveequivalentresults.','Onlyoracletemplateswork.'],
    'current_empirical_support':'A13showslocalizationresidualandhighconstructivesupportonly;noactualfeature-metricfailurevalidated.',
    'new_forwards_authorized_by_this_note':0,'mainline_validated':False}
write(A14/'next_spatial_update_question.json',next_note)

report=ROOT/'38_新编码器算子对照与A14判断.txt'
report.write_text('''当前结论：尚未确认可立项的新算法主线。新编码器的位置保留方向已完成一轮原始方法与算子核查；“保存相位索引传给decoder”“把高分辨率特征重新打包”“只在query位置上采样”均不能单独作为新的定位机制。本轮没有检测器前向或训练。

1. 检索策略和记录
沿用定向非系统前期调研，围绕polyphase/space-to-depth/shift equivariance、query-dependent feature reassembly、coordinate-based feature upsampling展开；随后查image-feature spatial Jacobian、feature-metric alignment与decoder refinement。精确查询、原始返回和访问范围保存在source_audit_v1/A14_*.json，未将第三方AI摘要当方法证据，也不把检索未返回结果当作无人研究的证明。
CVF部分PDF/HTML返回403，MDPI AeroPinWorld全文请求返回429；保留失败响应，转读公开作者arXiv和源码。公开GitHub元数据的TLS失败有记录；重新访问后取得固定提交和目录，未关闭TLS。没有自动审批拒绝，也没有等待用户批准的步骤。

2. 核心文献与阅读范围
SPD-Conv（Sunkara、Luo，ECML PKDD2022）：完整3.1、3.2和4.1已读。空间交错切片后拼接通道，再用非步长卷积是明确先例。已取作者提交c09f7928b8d2c33034fc3e9889a866f8bf37b44d的4个文件；没有执行作者模型。
LPS（Rojas-Gomez等，NeurIPS2022）：完整4.1—4.3已读。学习全局相位选择，并将概率/索引传入反采样。已取ef28ff29aa058c5f3fdf05b9a096f196ee1d0f30的6个文件；未整篇附录核证或运行作者模型。
Making Vision Transformers Truly Shift-Equivariant（CVPR2024）：读arXiv v2完整第4节及A2.2示例，未完成终稿逐项对照。窗口偏移及patch合并索引向解码头传递，已有明确实现示例。
SAPA（NeurIPS2022）及On Point Affiliation in Feature Upsampling（2023扩展）：会议版完整第3节、4.1和4.2，扩展版III-B/III-C片段及动态算子已读。两者属于同一方法家族，不作两项独立创新证据。已取当前作者tiny-smart/sapa提交a5acd520895c69886ce7175010d50a71d50b5a0e的4个接口文件；未构建CUDA核。
LoftUp（Huang等，ICCV2025）：完整第3节、4.1及第4节开头已读，4.2/附录未通读。用RGB和坐标Fourier特征作query，以低分辨率特征作key/value，已有坐标驱动特征恢复。已取0751a9b3cf7d5d4d1be181abc6f44fe3f43c77d8的4个文件；未下载权重或训练。
AeroPinWorld（Electronics2026）：仅出版商检索返回的摘要/3.3方法片段，全篇访问429，不能称全文已排重；其UAV相位感知stride-2设计是直接相关线索。
PixLoc（CVPR2021）：所读arXiv的3.1特征对齐主体、3.2、部分3.3和5.4；依赖已知参考图像/3D点描述子的特征度量LM定位是先例，不能把通用空间梯度加阻尼当创新。未执行作者实现。
DEQDet（ICCV2023）：完整3.2和3.3及补充B/C索引片段已读，非全篇。共享隐式decoder、细化感知梯度RAG及扰动RAP已有先例；源码仅README/检索片段范围，未下载固定版本或执行。

3. 逐项算子比较
LPS和shift-equivariant ViT已经传递下采样/窗口索引。差别若只在“把索引交给检测query”，仍需解释新增计算和作用，不能靠换任务证明新颖性。
SAPA-D以高分辨率点生成query、低分辨率特征生成key和偏移，按相似性重组低分辨率value。LoftUp则把RGB/坐标query经残差直接带入输出。故“细节只用于选权重”和“细节直接形成value/残差”是两类已有设计，不能只与双线性插值相比。
LoftUp的CATransformer中没有query间注意力，在相同预计算query、固定key/value及确定性计算下，可逐行独立求值。把它改为只计算检测采样点，至少在这个注意力栈范围属于已有算子的子集求值；可以节省实际计算，但不能自动称为新的特征保留机制。其RGB卷积、跨空间MinMax统计及训练BN仍须单独处理，本轮未证明整个LoftUp可等价稀疏执行或更快。
SPD作者代码的small YAML有多处“stride1 Conv→space_to_depth→C3”，首层Focus才是“交错打包→Conv”。没有把论文块示意强行当作全部代码链。后述线性等价只对应指定pack→conv，不声称整套作者网络等价于单层卷积。

4. A14代数核查：能排除什么，不能排除什么
纯NumPy、固定3个随机seed、float64，不导入或执行作者Torch代码。已核对4份固定版本回执、18个Git blob及SHA256。
18个案例核对交错相位打包后k=1/3的线性卷积，与重排权重后的stride2、2k卷积逐项一致。它说明指定线性算子的重参数化关系，不证明原生复杂网络没有性能差别。
3个案例核对按相位重编号feature/key/value/坐标时，全量注意力读出不变；故单纯打包并未提供额外位置信息。故意只重排key而不重排value的负对照不等，说明一致重排这个条件是必要的。
3个案例核对“固定key/value、无query间耦合”的注意力加逐点FFN栈，与query子集求值相等；这是行独立性质的代数控制，不是作者完整MHA/GELU模型复现。
3个零空间案例证明打包后再作降维线性投影并不保证可逆；不能把SPD这一步的可逆性扩展成整个降采样块无损。
共27个控制通过，最大等价算子数值差5.33e-15。它们没有使用真实图像，也不证明本机模型存在相位混叠、特征丢失或可训练改进。基础代数性质不宣称新定理。

5. 当前有依据的空白与候选判断
能确认的研究问题仍是A13所观察到的微小框高精度定位残差；原有坐标范围在自由logit构造下足以表达多数合格框。尚不能确定是编码、读取、回归学习或监督导致，不能据此直接设计“细节保留”模块。
本轮淘汰的是通用相位索引传递/打包/子集求值作为独有新机制的说法，没有否定所有新的编码器，也未证明相关任务适配或工程优化一概没有价值。
下一候选问题暂记为：同一query的局部特征随位置变化方向，能否提供比原生回归更准确的微小框位置更新。其与A0的插值系数秩、A5的标量中心场Taylor控制及DEQDet的latent-state Jacobian必须严格区分。PixLoc/TAPTRv2/DEQDet已覆盖一般原理，当前没有确立独特算法增量。
潜在的query兼容特征度量与小型2D位置求解仅为待定义假设，未实现、未选定为论文主线。推理时如何在没有GT或参考模板的情况下得到可信目标描述子，是首先要解决的条件；若只能依赖oracle模板，应直接淘汰。不能把GT用于训练监督与GT泄漏进推理混为一谈。

6. 最小验证路线与风险
相位通用候选已在文献/算子层面失去独有机制依据，当前不安排其模块训练或AP搜索。27个代数控制仅完成这一阶段的验证。
新空间更新假设需要先给出可执行算子、图像坐标单位及合法query目标定义，再固定原生方法、TAPTR类已知位置更新、相同容量MLP、空间导数置换等必要对照。原生query和memory必须来自预测，不能用GT中心充当输入。新数据选择、预算、主指标和停止条件须在新一轮模型执行前冻结；本轮没有将草案视为可执行协议。
主要风险包括重复已有特征对齐/梯度框细化、query与value不在同一语义空间、局部梯度指向背景纹理、病态2D求解，以及增加容量或oracle输入造成假提升。现有缓存未含完整原生query与空间梯度，不能靠修改缓存分析口径给出因果结论。
单卡约24GB能够承担当前冻结推理，不代表新的端到端训练资源已核实。LoftUp等论文的分类/分割训练配方不直接转用为800输入的检测配方。架构分支累计213次完整前向、主检测器更新0、既有64个诊断拟合/96个标量响应头；本轮新增均为0。有界前向阶段保持结束，旧DEIM和压缩/恢复任务未重启。目标继续ACTIVE。

7. 可核验参考链接
SPD-Conv：https://arxiv.org/html/2208.03641
LPS：https://arxiv.org/html/2210.08001
Shift-equivariant ViT：https://arxiv.org/html/2305.16316
SAPA会议版：https://arxiv.org/html/2209.12866
SAPA扩展：https://arxiv.org/html/2307.08198
LoftUp：https://arxiv.org/html/2504.14032
AeroPinWorld出版商线索（全文未取得）：https://www.mdpi.com/2079-9292/15/7/1364
PixLoc：https://arxiv.org/html/2103.09213
DEQDet：https://arxiv.org/html/2308.09564
作者代码固定版本及逐文件SHA：A14_新编码器位置保留近邻审计_v1/source_operator_comparison.json。
''',encoding='utf-8')

names=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
history=ROOT/'history_before_A14_update_v1'; assert not history.exists();history.mkdir()
for name in names:shutil.copy2(ROOT/name,history/name)
write(history/'manifest.json',{'utc':NOW,'files':[{'file':n,'sha256':sha(history/n)} for n in names]})
records={name:read(ROOT/name) for name in names}

# Save a bounded evidence archive; full-model/standalone replay is not claimed.
members=[(p,str(p.relative_to(ROOT))) for p in A14.iterdir() if p.is_file()]
for name in ['a14_encoder_operator_algebra.py','fetch_A14_encoder_sources.py','fetch_alignment_operator_sources.py']:
    members.append((ROOT/name,name))
raw=sorted((ROOT/'source_audit_v1').glob('A14_*.json'))
members.extend((p,str(p.relative_to(ROOT))) for p in raw)
for rec in receipts:
    p=ROOT/'comparison_sources_v1'/(Path(rec['local']).name.split('_operators_')[0]+'_operator_source_receipt.json')
    members.append((p,str(p.relative_to(ROOT))))
    members.extend((Path(rec['local'])/f['path'],str((Path(rec['local'])/f['path']).relative_to(ROOT))) for f in rec['files'])
backup=A14/'backup'; assert not backup.exists();backup.mkdir()
archive=backup/'A14_encoder_source_and_algebra_evidence.tar.gz'
manifest=[{'file':rel.replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size} for p,rel in members]
assert len({x['file'] for x in manifest})==len(manifest)
with tarfile.open(archive,'w:gz') as tf:
    for (p,_),item in zip(members,manifest):tf.add(p,arcname=item['file'])
with tarfile.open(archive,'r:gz') as tf:
    assert len(tf.getmembers())==len(manifest)
    for item in manifest:assert hashlib.sha256(tf.extractfile(item['file']).read()).hexdigest()==item['sha256']
write(backup/'member_manifest.json',{'utc':NOW,'members':manifest})
archive_receipt={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(members),
    'status':'PASS_ALL_ARCHIVE_MEMBER_HASHES','scope':'Evidencecopies;notfullauthorrepositoryorstandaloneportablemodelreplay.'}
write(backup/'receipt.json',archive_receipt)

board=records[names[0]]
question='微小目标精确定位：查证局部图像特征的空间变化与query位置更新是否匹配；相位索引/打包等通用机制已有先例，具体算法未确定。'
board.update({'latest_update_utc':NOW,'latest_report':str(report),
    'status':'NO_CANDIDATE_PASSED_MECHANISM_AND_NOVELTY_GATES_AFTER_A14',
    'current_priority_question':question,'next_decision_protocol':'Definequery-compatiblefeaturemetricandclosestknowncontrols beforeafrozenboundedprotocol. No genericphase-moduletrainingorautomaticnewmodelpasses.',
    'new_empirical_evidence':'A14primaryreading and18pinnedsourceblobs plus27scopedNumPyalgebracontrols;0newdetectorpasses/fits. Real-imagefailuremechanismunvalidated.'})
board['candidates'].append({'id':'I_GENERIC_PHASE_PROVENANCE_AND_QUERY_SUBSETTING',
    'status':'DEPRIORITIZED_AS_DISTINCT_MECHANISM_BY_PRIMARY_PRIOR_AND_SCOPED_EQUIVALENCE',
    'nearest':['SPD-Conv','LPS','Shift-equivariantViT','SAPA','LoftUp','AeroPinWorld(partial)'],
    'increment_approved':False,'causal_evidence':False,'limits':'Doesnotrejectallphaseoperatorsorengineeringoptimization;onlygenericmechanismclaimsandlistedalgebraassumptions.',
    'evidence':str(A14/'source_operator_comparison.json')})
board['next_exploratory_question']=next_note
board['A14_evidence']={'archive':archive_receipt,'algebra':result,'source_audit':str(A14/'source_operator_comparison.json')}

log=records[names[1]];log['supplement_utc']=NOW
raw_records=[{'file':str(p.relative_to(ROOT)),'sha256':sha(p),'request':read(p)['request']} for p in raw]
log['query_groups'].append({'name':'A14position-preservingencodersandspatial-updatepriors','requests':raw_records})
log['A14_encoder_source_and_algebra']={'source_audit':str(A14/'source_operator_comparison.json'),'archive':archive_receipt,
    'scope':'Targetednonsystematic;rawsearchreturnsmayincludeunrelatedandsecondarypagesbutthese arenotprimarymethodevidence.',
    'read_scope_explicit':True,'report':str(report),'new_model_runs':0}

matrix=records[names[2]]
newrows=[
 {'id':'SPD_Conv','title':'No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects',
  'url':'https://arxiv.org/html/2208.03641','tier':'ECMLPKDD2022;inspectedarXivv1','read_scope':'Complete3.1/3.2/4.1;4pinnedfilesselectedoperators/config/README,noauthormodelrun.',
  'author_method':'Interleavedspace-to-depth andstride1convolution;suppliednativeconfigorderrecordedseparately.',
  'our_consequence':'Genericphasepackingalreadyprior;linearpack-convhasastride2weightreindexing,andpackinglosslessnessdoesnotguaranteecompressioninvertibility.',
  'unresolved':'NoAPorreal-imagealiasingtest;fullauthorimport/parserchainnotclosed.','code_commit':receipts[1]['commit']},
 {'id':'LPS','title':'Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks',
  'url':'https://arxiv.org/html/2210.08001','tier':'NeurIPS2022','read_scope':'Complete4.1/4.2/4.3;6pinnedfiles,selectedoperatorchain;notwholeappendixortraining.',
  'author_method':'Learnedphasechoice,probability/indexprovenance,andpairedunpooling undercircularshiftequivarianceconditions.',
  'our_consequence':'Passingphaseindicestothedecoderisexplicitprior.','unresolved':'Classification/segmentationguaranteesdonotprovetinyboxprecisionorallboundary/tieconditions.','code_commit':receipts[0]['commit']},
 {'id':'ShiftEq_ViT','title':'Making Vision Transformers Truly Shift-Equivariant',
  'url':'https://arxiv.org/html/2305.16316','tier':'CVPR2024;inspectedarXivv2Nov2023',
  'read_scope':'Complete4andA2.2encoder-decoderexample;notfullpublishedfinalversionorallproofs.',
  'author_method':'Adaptive tokenization/windows/patchmerging,circularrelativeposition,andmerging/windowindex-awareunpooling.',
  'our_consequence':'Genericindex-carryingencoder-decoderalreadycovered.','unresolved':'Nodetectionexecution;fullfinalsourceclosureunavailable.'},
 {'id':'SAPA_family','title':'SAPA: Similarity-Aware Point Affiliation for Feature Upsampling',
  'url':'https://arxiv.org/html/2209.12866','extension_url':'https://arxiv.org/html/2307.08198',
  'tier':'NeurIPS2022plus2023arXivextension,samemethodfamily',
  'read_scope':'Conferencecomplete3/4.1/4.2,selectedresults/discussion;extensionIII-B/III-Cexcerpts.4pinnedPythoninterfacefiles;noCUDAbuildorkernelclosure.',
  'author_method':'Highresolutionguide-to-lowresolutionaffinityreassembly;extensionaddsdeformablepointselection.',
  'our_consequence':'Guidedreassembly,dynamickernels,learnedoffsetsandquery-keyguidanceareprior.',
  'unresolved':'Noauthorreproduction;guidingselectiondiffersfromdirectlyinjectingfinevalues.','code_commit':receipts[2]['commit']},
 {'id':'LoftUp','title':'LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models',
  'url':'https://arxiv.org/html/2504.14032','tier':'ICCV2025pp9913-9923',
  'read_scope':'Complete3and4.1,4intro;4pinnedfilesandselectedactiveoperators. Notcomplete4.2/appendix;noweightsortraining.',
  'author_method':'RGB/Fouriercoordinatequeries,crossattentiontolowresolutionfeatures,andmask/self-distilledfeaturetargets.',
  'our_consequence':'Coordinatequeriedliftinganddirectqueryresidualalreadyprior. Querysubsettingcommutesonlywiththefixedinputindependentattentionstack.',
  'unresolved':'Densepreprocessing/globalMinMax/BNprecludeanunqualifiedend-to-endsparse-equivalenceorspeedclaim.','code_commit':receipts[3]['commit']},
 {'id':'AeroPinWorld_partial','title':'AeroPinWorld: Revisiting Stride-2 Downsampling for Zero-Shot Transferable Open-Vocabulary UAV Detection',
  'url':'https://www.mdpi.com/2079-9292/15/7/1364','tier':'Electronicspublisherrecord2026',
  'read_scope':'Publisherindexedabstractand3.3methodexcerptsONLY;fullpagefetch429;notfullmethodcleared.',
  'author_method':'Indexedexcerptdescribesphase-awarestride2PConv atselectedearly/headtransitions.',
  'our_consequence':'DirectUAVphase-downsamplingpriorrequirescomparison,notanunclaimedblankarea.',
  'unresolved':'Fullmethod/code/metricsnotreviewed;authorsnotfullymetadata-verified.'},
 {'id':'PixLoc','title':'Back to the Feature: Learning Robust Camera Localization from Pixels to Pose',
  'url':'https://arxiv.org/html/2103.09213','tier':'CVPR2021',
  'read_scope':'3.1featuremetricLMbody,3.2,selected3.3/5.4;notwholepaperorappendix. CurrentGitHuboptimizerexcerptonly,nopinnedcodeclosure.',
  'author_method':'Referenceimage/3D-pointfeaturealignment withimage-coordinateJacobians,confidenceweightsandper-coordinatelearneddamping.',
  'our_consequence':'Feature-gradient-awarepositionsolvinganddampingareprior;unknownobjectquerytemplatesareunresolved,taskchangealoneisnotnovelty.',
  'unresolved':'Noauthorrun or detector-compatibletemplateestablished.'},
 {'id':'DEQDet','title':'Deep Equilibrium Object Detection','url':'https://arxiv.org/html/2308.09564','tier':'ICCV2023',
  'read_scope':'Complete3.2/3.3;selected3.1/4.1,supplementB/Cexcerpts;officialREADME. Nopinnedsourceclosureortraining.',
  'author_method':'Sharedimplicitquerydecoder,two-steprefinement-awaregradients,andrefinement-awareperturbation.',
  'our_consequence':'Genericrecurrence,fixed-pointrefinementandlatent-Jacobianawarenessalreadydirectdetectionprior.',
  'unresolved':'LatentstateJacobiandiffersfromimage-coordinatefeatureJacobian;neitherprovesnativeDomecausalfailure.'}]
ids={r['id'] for r in matrix['rows']};assert not any(r['id'] in ids for r in newrows)
matrix['rows'].extend(newrows)
matrix.update({'updated_utc':NOW,'latest_report':str(report),
    'latest_empirical_boundary':'A14isprimarysource/algebraonly;nogenericphaseprovenanceincrementapproved. Spatial-updatehypothesisnotmechanismvalidated.'})

status=records[names[3]];assert status['model_forwards']==213 and status['mainline_validated'] is False
status['authoritative_phase_history'].append(copy.deepcopy(status['latest_authoritative_phase']))
status['latest_authoritative_phase']={'utc':NOW,'classification':'progress','goal_status':'active','complete_mainline':False,
    'full_model_forwards':213,'cumulative_authorized_protocol_cap':213,'bounded_phase_complete':True,
    'main_model_training_updates':0,'CPU_diagnostic_linear_models':64,'scalar_response_heads':96,
    'A14_new_forwards':0,'A14_new_fits':0,'A14_algebra_cases':27,'A14_pinned_git_blobs':18,
    'A14_decision':result['decision'],'A14_archive':archive_receipt,'latest_report':str(report),
    'next':'Definespecificquery-to-spatialfeaturecompatibilityoperatorandclosestcontrols;noautomaticadditionalforwardphase.',
    'remaining_uncertainty':['Deployableoperator','Distinctincrement','Real-imagefailuremechanism','Causalefficacy','Generalization','Trainingfeasibility']}
status.update({'recorded_utc':NOW,'previous_goal_turn':status['current_goal_turn'],
    'current_goal_turn':'progress_A14_primary_encoder_sources_and_operator_algebra',
    'latest_report':str(report),'current_priority_question':question})
audit=records[names[4]]
audit.update({'updated_utc':NOW,'A14_encoder_sources_and_algebra':source_audit,
    'A14_algebra_result':result,'next_spatial_update_question':next_note,
    'current_novelty_boundary':'Genericphaseprovenance,losslesspacking,coordinateupsamplingandsubsetqueryevaluationnotnewmechanismsbythemselves;spatial-updateadaptationunvalidatedandmustclearPixLoc/TAPTRv2/DEQDet.'})
for name,value in records.items():write(ROOT/name,value)
receipt={'utc':NOW,'status':'PASS_A14_PRIMARY_SOURCE_AND_SCOPED_ALGEBRA_RECONCILED',
    'goal_status':'ACTIVE','previous_goal_turn_classification':'progress_A13_cachedsupportandtrajectoryreconciliation',
    'current_goal_turn_classification':'progress_A14_primaryencodermethodsandoperatorequivalence',
    'mainline_validated':False,'full_model_forwards':213,'new_full_model_forwards':0,
    'new_fits':0,'main_model_updates':0,'matrix_rows':len(matrix['rows']),
    'raw_search_and_open_records':len(raw),'verified_git_blobs':18,'algebra_cases':27,
    'archive':archive_receipt,'next_question_status':next_note['status'],
    'files':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [report]+[ROOT/n for n in names]]}
write(ROOT/'39_A14源码与算子去重核验回执.json',receipt)
print(json.dumps({'status':receipt['status'],'mainline_validated':False,'goal_status':'ACTIVE',
    'matrix_rows':len(matrix['rows']),'raw_records':len(raw),'source_files':18,
    'algebra_cases':27,'new_model_forwards':0,'full_model_forwards':213,'archive_members':len(members)}))
