from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
ENG=ROOT.parents[1]
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=load(ROOT/"offline_complete.json")
new=load(ROOT/"remote_downloaded_30172/D0_results_v1/complete.json")
nv=load(ROOT/"remote_independent_verification.json")
ov=load(ROOT/"offline_independent_verification_v2.json")
assert old["status"]==new["status"]=="COMPLETE" and nv["status"]==ov["status"]=="PASS"
assert new["counters"]=={"model_forwards":32,"model_backward_calls":0,"optimizer_updates":0}
primary=load(ROOT/"remote_downloaded_30172/D0_results_v1/analysis_summary.json")
lit=load(ENG/"研究历史与文献审计_20261026_v1/literature_table_data.json")
byid={x["id"]:x for x in lit}
methods=[
 {"id":"P1","old_literature_id":"L124","reference":"Wang, J., Xu, C., Yang, W., & Yu, L. (2022). A normalized Gaussian Wasserstein distance for tiny object detection. arXiv:2110.13389v2.",
  "urls":["https://arxiv.org/abs/2110.13389v2"],"read_scope":"本地原PDF题头已核v2/2022-06-14；读取NWD定义、框高斯表征、常数C及AI-TOD设置节选。未全文通读或复现。",
  "original":"以框对应高斯分布的Wasserstein距离构造指数相似度，减轻小框平移敏感性；C与数据集有关，原AI-TOD平均绝对尺寸12.8。",
  "inference":"普通平滑几何距离是必须考虑的已知答案，换GIoU为NWD不能当作本项目新增算法。",
  "difference":"本轮仅将NWD作为Hungarian几何代价对照，保留原分类和L1项，不复现anchor/RKA；固定C移植到640输入，未调参，不是最优NWD效果上界。"},
 {"id":"P2","old_literature_id":"L244","reference":"Liu, S., et al. (2023). Detection transformer with stable matching. arXiv:2304.04742v1; ICCV 2023.",
  "urls":["https://arxiv.org/html/2304.04742v1","https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Detection_Transformer_with_Stable_Matching_ICCV_2023_paper.pdf"],
  "read_scope":"本地文件虽名ICCV2023，题头实为arXiv v1/2023-04-10；本轮核读§2.1–2.4定位监督与位置调制匹配，未通读全文。网页公式与归档正文对照。",
  "original":"指出跨decoder层匹配的不稳定，提出基于定位质量的正例分类监督和位置调制分类匹配代价。",
  "inference":"“稳定匹配/用定位质量约束分类”本身已经有先例，不能因换到航拍或ECDet称新。",
  "difference":"跨层/优化路径稳定性与本轮固定输出下标注边缘扰动不同；原PSL也不等同本项目MAL。未运行Stable-DINO训练对照。"},
 {"id":"P3","old_literature_id":"L234","reference":"De Plaen, H., De Plaen, P.-F., Suykens, J. A. K., Proesmans, M., Tuytelaars, T., & Van Gool, L. (2023). Unbalanced optimal transport: A unified framework for object detection. CVPR.",
  "urls":["https://openaccess.thecvf.com/content/CVPR2023/papers/De_Plaen_Unbalanced_Optimal_Transport_A_Unified_Framework_for_Object_Detection_CVPR_2023_paper.pdf"],
  "read_scope":"本地原PDF摘要与§2.1–2.3、背景质量、熵正则及运输示意的节选；未通读所有证明或复现。",
  "original":"把Hungarian匹配表为带背景质量的OT特例，用熵正则及放松质量约束获得连续/软连接。",
  "inference":"“GT质量守恒/分数式分配/软化Hungarian”属于已知操作族，不能仅凭这些性质宣布新意。",
  "difference":"这不证明标注特定扰动的期望分配与UOT算子完全等价；当前尚未提出超出该族的具体、值得实现的差异。"},
 {"id":"P4","old_literature_id":"L136","reference":"Sun, H., Li, Y., Yang, L., Cao, X., & Zhang, B. (2025). Uncertainty-aware gradient stabilization for small object detection. ICCV.",
  "urls":["https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.pdf"],
  "read_scope":"本地CVF原PDF摘要、§3.1及§3.2方法节选（分布回归、熵约束、特征扰动）；未完整审查数学推导或实验。",
  "original":"采用分类式非均匀定位网格、预测不确定性最小化和由不确定性引导的特征扰动精炼。",
  "inference":"通用不确定性调权、非均匀坐标bins或加扰动本身不足以构成本项目新方法。",
  "difference":"特征扰动/回归梯度问题不等于本轮的标注扰动→实例分配；本项目原生FDR已是分布式坐标回归。"},
 {"id":"P5","old_literature_id":"L113","reference":"Yang, C., Zhang, P., & Zhao, J. (2026). SA-Matching DETR: A lightweight transformer detector with enhanced scale adaptive matching. CVPR Findings.",
  "urls":["https://openaccess.thecvf.com/content/CVPR2026F/papers/Yang_SA-Matching_DETR_A_Lightweight_Transformer_Detector_with_Enhanced_Scale_Adaptive_CVPRF_2026_paper.pdf"],
  "read_scope":"原PDF摘要、§3.2匹配复制函数式(3)及§3.3和相关消融说明；不声明全文通读。",
  "original":"按尺度复制GT以增加监督，原文小目标因子s=1，主要增加较大目标正例。",
  "inference":"尺度自适应正例扩充已有直接先例；不能把改query/GT数作独立创新。",
  "difference":"它没有直接等同于半像素扰动的全局归属边界，本轮也未增加GT/query数；不能拿它当作本假设已被证伪的性能证据。"},
 {"id":"P6","old_literature_id":"L134","reference":"Chen, X., Yu, S., Cheng, C., Wang, H., & Tian, T. (2025). An uncertainty-aware DETR enhancement framework for object detection. arXiv:2507.14855v1.",
  "urls":["https://arxiv.org/html/2507.14855v1"],
  "read_scope":"本地PDF及官方HTML摘要、方法结构、§4.2式(11)匹配代价及相关消融节选；未全文通读。",
  "original":"对框的不确定性建模，以GW距离和Bayes Risk改进输出、分类及匹配代价。",
  "inference":"“不确定性感知匹配代价”已非空白。",
  "difference":"风险加权的单次代价与局部多分配平均未证明等价；当前也没有足以支持后者投入的新证据。"},
 {"id":"P7","old_literature_id":None,"reference":"Zhu, H., Xu, C., Yang, W., Zhang, R., Zhang, Y., & Xia, G.-S. (2024). Robust tiny object detection in aerial images amidst label noise. arXiv:2401.08056v1. Journal version: DN-TOD, Pattern Recognition (2026), 113448.",
  "urls":["https://arxiv.org/html/2401.08056v1","https://doi.org/10.1016/j.patcog.2026.113448","https://github.com/ZhuHaoranEIS/DN-TOD"],
  "read_scope":"arXiv v1摘要、引言和§4.2 TLR/RBR节选；期刊2026版仅摘要与publisher元数据；作者README方法概述。PDF下载因本地TLS认证失败未取得，未逐公式/表格核对，不把v1阅读写成期刊全文阅读。",
  "original":"已研究tiny标签噪声；用类别校正、学习趋势重加权及递归框再生成处理类别/框噪声。",
  "inference":"“小框标注不精确值得处理”不是无人研究的问题。",
  "difference":"不把大幅/真实框噪声与本轮±0.5原图像素敏感性混同。2024 v1与2026摘要的任务/数据差异未逐项核验，未使用其性能数值。"}
]
for m in methods:
    m["PDF_artifacts"]=[]
    if m["old_literature_id"]:
        for f in byid[m["old_literature_id"]]["pdf_files"]:
            p=Path(f)
            if p.exists():m["PDF_artifacts"].append({"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size})
    m["paper_reproduced_in_A47"]=False
save("01_核心文献矩阵与逐项对比.json",{"mode":"targeted non-systematic preliminary research; single assistant","status":"ANALYZED","methods":methods})
queries=[
 "tiny object detection DETR matching stability assignment localization uncertainty 2025 2026 arxiv",
 "UAV small object detection transformer encoder decoder 2026 novel architecture occlusion density",
 "DETR tiny object label perturbation matching instability normalized Wasserstein distance assignment stability",
 "DETR stable matching small objects box jitter 2025 2026",
 "\"small object\" \"annotation\" \"matching\" \"uncertainty\" DETR 2025",
 "\"UAV\" \"perspective\" \"object detection\" \"transformer\" scale",
 "\"aerial\" \"FOVEA\" \"detection\"",
 "\"tiny object detection\" \"normalization\" \"attention\" softmax sink",
 "\"DETR\" \"tiny\" \"aspect ratio\" \"attention\" 2026",
 "\"DETR\" \"MAL\" \"small\" classification IoU target 2025",
 "\"tiny object\" \"matching stability\" transformer",
 "\"DETR\" \"box perturbation\" matching",
 "\"small object detection\" \"quality\" \"classification\" \"noise\" 2026"]
save("02_检索策略与记录.json",{"date":"2026-11-01","mode":"定向前期调研；非系统综述，不报告全数据库命中或PRISMA筛选数。",
 "strategy":"先用73条历史路线与当前A0–A46问题查重；一般新架构/场景候选作定位，随后只深入匹配稳定性与标注小扰动。追溯NWD/Stable-DINO/UOT/UGS等基础先例及2025–2026近邻。",
 "sources":"web search; arXiv官方页；CVF原PDF；publisher摘要；作者GitHub；项目文献总表及归档原PDF。二手页面仅作定位，不支持技术结论。",
 "successful_query_batches":4,"successful_query_strings":queries,
 "supplemental_search_limit":"后续perturbed/soft/probabilistic/ensemble matching等补查多次连接失败；另一搜索服务也返回传输失败。三次arXiv站内索引打开为cache miss。不把失败检索当成没有先例，也不声称穷尽最新文献。",
 "direct_reading":"逐篇URL、阅读范围、原件SHA见01文件。CVF UOT网页直开403但本地原PDF可读；DN-TOD网页可读但PDF未取得。",
 "exclusions":"反无人机本体检测与当前航拍地面目标任务不同；只堆模块、只换宿主不入选。检索中遇到标注2026-11-14的未来卷期页面，未据此作截至11-01的最新性或创新性判断。",
 "initial_question_not_finding":"小框分配对微小标注变化是否过敏？不是已确认缺陷或新颖性结论。"})
save("03_服务器复用与授权事实.json",{
 "user_supplied_new_endpoint":True,"credential_saved":False,"old_original_root_not_found_at_new_endpoint":True,
 "correction":"先前只检查/root固定路径而说旧目录和运行环境都不在，范围过大；已纠正。没有证据证明旧工作整体已丢失。",
 "reused_dataset":new["data_root"],"train_images":6471,"all6471_annotations_checked_before_capture":True,
 "selected128_image_and_annotation_files_locally_SHA_reverified":256,"existing_modern_python":"/opt/conda/envs/mamba/bin/python",
 "environment":"Python3.11.13/torch2.7.1+cu126/torchvision0.22.1+cu126; original environment was torch2.6.0+cu124. Not a bitwise rerun claim.",
 "new_isolated_root":"/root/uav_a47_20261101_v1","uploaded_asset_bundle_bytes":200427520,
 "scope":"复用数据和现代环境，只补字节固定的源码、初始化/EMA checkpoint以及任务目录中的pycocotools。未重装共享环境，未上传训练集，未恢复旧训练。",
 "readiness_test":"one synthetic CUDA tensor sum only; not a detector forward; no optimizer.",
 "earlier_connection_and_download_issues":"30890 banner failed; 30172 persistent channel failures resolved by fresh-per-query connections. A first mirrored wheel fetch was interrupted; official PyPI pinned wheel succeeded and its SHA was verified.",
 "new_detector_forwards":32,"new_optimizer_updates":0})
save("04_统计解释与范围核查.json",{"mode":"single-assistant validate checklist, not an independent reviewer panel",
 "coverage":"11/11 checked","items":[
 {"type":"Simpson","disposition":"同时报告GT总体率与同图同类宏平均，后者差距较小；不以总体差距声称尺寸因果。"},
 {"type":"Ecological","disposition":"GT/draw观测与图像层汇总分开；不把模型间总体比较当单目标或训练作用。"},
 {"type":"Berkson","disposition":"旧val按边界/计数筛选309/548后哈希选128；新train从6030/6471标签合格图选128，外推受限。"},
 {"type":"Collider","disposition":"不按低置信度或坏预测选图；旧缓存源合法性规则预设，实际无128对被排除；仍不作因果主张。"},
 {"type":"BaseRate","disposition":"明确原图tiny定义、GT和图片分母；16draw为重复观测，不当独立样本。"},
 {"type":"RegressionToMean","disposition":"没有按极端失败分选样本，也没有前后训练收益比较。"},
 {"type":"Survivorship","disposition":"记录309/548与6030/6471合格来源；所有选中样本保留；完整退出与失败回执保留。"},
 {"type":"LookElsewhere","disposition":"本项目长期探索，结果不构成校正后的显著性检验。两幅度、两算子与阈值预设，均完整保留。"},
 {"type":"ForkingPaths","disposition":"远端方案先冻结，连接问题后另冻结旧缓存诊断；新入口仅改路径/环境接入；未改选图/幅度/阈值救结果。"},
 {"type":"CorrelationCausation","disposition":"标签扰动会改变该匹配函数输出；不能推成真实标签误差或训练伤害/AP根因。"},
 {"type":"ReverseCausality","disposition":"无训练时间序列及学习因果识别；冻结输出诊断不说明坏定位与匹配变化的自然方向。"}],
 "verifier_correction":"初版把原FP32逐tiny框IoU与FP64结果强行要求差<2e-6而失败（首次4.059e-6）；保留v1，v2独立按原FP32精度重算，IoU差0，全部原始owner/CSV通过。未重跑模型或匹配，未放宽研究阈值。",
 "bootstrap_or_pvalue":None,"verification_status":"ANALYZED; execution, identity and raw-index arithmetic checked; no new method efficacy validated."})
def per(s):
 b=s["cohorts"]["B_0.5"]["tiny"];m=s["cohorts"]["M_0.5"]["tiny"];c=s["common_support"]["B_0.5"]
 return b,m,c
table=[]
for label,s in [("旧BASE24 seed1，val128",old["summaries"]["1"]),("旧BASE24 seed2，同一val128",old["summaries"]["2"]),("ECDet-S seed20261101，train128",primary)]:
 b,m,c=per(s)
 table.append(f'{label}: tinyGT={b["GT"]}，所在图={b["images"]}；B归属变动={b["owner_flip_fraction"]*100:.4f}%，B正例类别身份改变={b["positive_class_identity_change_fraction"]*100:.4f}%；M归属变动={m["owner_flip_fraction"]*100:.4f}%，M正例类别身份改变={m["positive_class_identity_change_fraction"]*100:.4f}%；同图同类tiny−larger宏平均={c["image_class_macro_tiny_minus_larger"]*100:.4f}pp（{c["images"]}图）；M−B干净匹配IoU均值={m["clean_matched_iou"]-b["clean_matched_iou"]:.6f}。')
report=f"""A47　微小框匹配稳定性：下一候选选题与必要验证结果
日期：2026-11-01（America/Los_Angeles）

Material Passport
Origin Skill: academic-research-suite / deep-research + experiment-agent
Origin Mode: targeted preliminary research + run/analysis
Verification Status: ANALYZED；执行、输入身份和原始分配重聚合已核验；没有新算法有效性结论
Version Label: A47_v1

本轮结论
已推进到下一候选并完成必要诊断。当前不把“半像素标注扰动造成tiny匹配/分类监督普遍失稳”推进为优先论文主线，不启动三组12轮新训练。主线仍未找到。
这里的停止不是所有匹配学习或标签噪声研究都无价值，也不是一项新算法训练失败；本轮没有实现或训练新匹配方法。原生匹配有可见敏感性，但没有达到预先固定的投入门槛，且泛化的平滑距离、软分配与不确定性监督已有先例，尚未形成独立增量。

1　自主目标、排重和研究问题
用户授权自主设定下一目标并跑必要验证。目标是先找到与旧路线有实质区别的一个具体问题，或给出及时退出的实证理由，不为用掉预算而搭模块。
选择的问题：在固定检测输出下，原图坐标小幅变化是否使tiny目标的普通query责任归属和实际分类监督比同图同类较大目标更易变；普通NWD平滑距离是否已足够处理这一敏感性？
与旧记录的关系：
- D035处理回归支持范围的匹配代价，训练未达固定终点，不能写成所有新匹配都失败；本轮不使用其范围门或重启其训练。
- D036处理回归支持所有权/GO修复；本轮不改GO或扩大正例、query配额，也不把换owner数量当新增检测。
- D062是量化观测集合框监督的旧受控预检；本轮测离散分配和分类正例身份，区别于框回归读出的集合期望。但仍使用假定的像素区间，不能换名声称解决了真实标注噪声。
- D073是监督容量诊断，不自动推出新分配方法；A42/A43训练停止、A44撤回、A46头类负梯度抑制未推进，全部维持。
因此本轮允许一项诊断，不代表批准新算法。检索与方法矩阵见01、02文件。

2　最近方法及原文/推断边界
NWD [P1]已有平滑几何相似度。常数C与数据有关；本轮固定借用12.8输入像素，保留原focal分类/L1与几何权重。这个M是已知度量的有限适配对照，不是原anchor/RKA复现或最优NWD的上界。
Stable-DINO [P2]已有基于定位质量的正例分类监督和位置调制代价，针对跨层及多优化路径。MAL与其PSL不是同一损失，但“定位质量指导分类”已不新。
UOT [P3]已有背景质量、运输质量分配与熵正则。GT总质量守恒的软分配不因移到新宿主而构成创新；这不证明任何标注特定的扰动期望都与UOT严格等价。
UGS [P4]已有非均匀离散回归、不确定性约束与特征扰动精炼。它与标签边缘扰动不同；通用bins/不确定性加权/扰动均不能直接算本项目增量。
SA-Matching [P5]按面积复制GT，原文small因子s=1，主要补较大目标监督。它不是本轮标注扰动的直接答案，不能拿其题名当成已解决本问题。
Uncertainty-aware DETR [P6]已有风险感知匹配代价；DN-TOD [P7]已研究tiny标签噪声和趋势重加权/框再生成。P7期刊2026版仅摘要，arXiv2024 v1只读方法节选，未把它们混成已读期刊全文。
这些原文支持“问题与通用办法已有研究”，不支持“所有已有办法已经足够”或“新框架都没机会”。本轮没有训练这些论文的完整方法。

3　验证前固定的机制与停止规则
把标注框四条边分别加入原图坐标的均匀扰动：主幅度±0.5像素，预设敏感性±1像素。每图8组独立扰动及其反向，共16次，作用于所有GT，类别/实例身份固定。这是假定边缘区间敏感性，不是已识别的真实标注噪声模型；独立边误差也不等价于真实x/y/w/h标注过程。
B为原生focal-class + L1 + GIoU Hungarian，M仅换几何相似度为NWD。query数均300，不新增GT、query、模型参数或训练。
tiny按原图框area<=256，small为(256,1024]，larger>1024。归属变动是每GT每次扰动的query索引变化；正例类别身份改变指原匹配query被变为未匹配或另一类别，同类GT之间换owner不算这项变化。两者不等于错误、有害梯度、TP或AP损失。
同图同类中计算tiny−larger差，在图内先等权类，再等权图。最低支持为200个tiny、20张tiny图、200个larger及20张共同支持图。
继续需同时达到：tiny归属变动>=10%，同图同类差>=5pp，正例类别身份改变>=5%。普通平滑控制另看归属变动相对减少>=50%且干净匹配IoU均值降幅<=0.01。这些是工程预算筛选线，不是统计显著性或学科价值判定线。
未改阈值、幅度、群体或把±1像素敏感性结果升为主结局。

4　实际数据与算法，以及服务器情况的更正
最初检查原/root路径未见旧目录，就说旧数据和环境都不在，判断范围过大。新入口30172实际有可复用的训练集和现代环境：
数据：{new["data_root"]}
共6,471张训练图；采集前逐一核验6,471个对应标注SHA，选中128图与对应标注在采集前后校验，下载后又与本地256个真实文件逐字节SHA比对通过。
环境：/opt/conda/envs/mamba/bin/python，Python3.11.13、torch2.7.1+cu126、torchvision0.22.1+cu126。它是可用的现有环境，并未证实是原来torch2.6.0的那套venv；不声称跨版本位级复现。
新隔离目录：/root/uav_a47_20261101_v1。只上传约200.4MB的固定源码/权重材料和一个小依赖；数据未重新上传，共享环境未安装新包，pycocotools仅装到任务目录。
宿主：VisDrone + ECDet-S，A43原版B第12轮EMA（不是恢复定位读取或蒸馏）。源代码71文件、初始化、checkpoint及清单SHA与旧资产一致。
此前已另行冻结并完成旧BASE24两seed的val缓存诊断。它们是HGNetv2-B0 + HybridEncoder + D-FINE/DEIM decoder的baseline臂，不是ECDet。两seed共享128图和同一份GT；不能叠加为双倍独立样本。旧val按标签边界/计数筛选有309/548合格图；新train有6030/6471合格图。分别按预设哈希选128图，不按低分或坏预测选图。
不把旧val与新train/不同模型的数值差解释成架构效应、数据集泛化或方法复制成功。

5　结果与分母
以下均为16次扰动下的平均率；不是“这些比例的目标永远受损”，也不是召回/AP：
{chr(10).join(table)}

新ECDet-S cohort：全体7,306 GT，其中tiny1,909、small2,728、larger2,669。新模型的同图同类比较有138个cell、75张图；旧两seed各200个cell、95张图。支持门槛均满足，继续的三项主效应门槛均未达到。
±1像素仅作预设敏感性：旧两seed tiny归属变动6.8112%/6.7770%，ECDet7.9557%；没有据此重新设主幅度。
一个需要保留的相反现象：NWD在ECDet中使归属变化更大（4.4264%→5.9357%），同时正例类别身份变化更小（2.2459%→0.9691%）。这说明换query索引本身不能当作训练有害性的代理。
NWD使干净标注下选中的匹配框平均IoU略低，但框预测本身完全相同；这不是模型定位性能或AP下降，也不是NWD训练无效的证据。当前没有新的训练AP。

6　执行、独立复核与限制
旧缓存分析退出0，耗时约{old["elapsed_seconds"]:.1f}秒，未构建检测器或进行检测器前向；256个图像×模型clean匹配均与独立NumPy/SciPy公式的分配相同。
新远端进程退出0，耗时约{new["elapsed_seconds"]:.1f}秒（含文件I/O、模型与匹配分析）；32次完整模型前向、teacher0、模型反向0、优化器更新0。权重和state哈希前后一致。该轻量诊断不能证明完整训练容量或时延优势。
远端完整归档约1.85MB，共22个文件；ZIP CRC和21项清单SHA通过。任务PID/start身份已消失，终态观察GPU无计算进程，无自动训练队列。
独立复核直接读取原始owner数组及原始预测，覆盖旧16,384组与新8,192组扰动分配，逐GT owner、分类身份、分层及同图同类宏平均均复得；复核未新求解匹配、未新调用模型。
第一版逐框FP64数值复核因与原FP32的微小舍入差超过2e-6而退出；原件保留，v2按原精度独立重算IoU差0。没有重跑实验、改数据或放宽研究门槛。其余限制和11/11统计误读检查见04文件。

本阶段不测：真实标注误差的分布、早期训练责任切换、辅助/DN层、模型梯度干扰、训练因果收益或新算法AP。固定第12/24轮状态不能排除训练初期或更大自然噪声的重要性。

7　有依据的剩余问题、增量与去留
仍可讨论的疑点是标注形成过程和早期训练中的责任不确定性，已有文献表明它们值得研究。但本轮没有自然噪声证据、没有与UOT/Stable-DINO/UGS等区分后的新操作，因此不是可宣布的研究空白。
“换NWD”“复制tiny GT”“软分配质量守恒”“抖动后平均监督”“按不确定性调权”均未获批准：有的是直接已知答案，有的仍缺相对先例的实质区别。没有实现它们，不称为训练失败。
裁决：停止当前半像素标注扰动候选的扩展，不调C、改幅度或换子集救主结局，不启动原版/普通对照/新候选三组12轮。保留已查明的结果和可用服务器资产。当前优先候选为空，论文主线仍未成立。

最小验证方案已经执行到预设终点；三组各12轮只是条件性上限，不是必须花完的预算。未来若重新讨论相关主题，需要实质新证据和新操作，不能把这轮阶段完成写成论文创新成立。

可复现入口
旧缓存：D:/Anaconda/python.exe -s -X utf8 run_A47_existing_cache.py
新诊断：/opt/conda/envs/mamba/bin/python -u /root/uav_a47_20261101_v1/run_A47_frozen_match_capture_30172_v1.py
新诊断原始输出：remote_downloaded_30172/D0_results_v1/；完整原始分配与两种幅度均保留。
协议：protocol_FROZEN.json；offline_protocol_FROZEN.json；deployment_30172_amendment_FROZEN.json。
本阶段完整模型调用+32；训练更新、训练轮数均+0；历史A42/A43/A44/A46裁决和原成绩保留。

参考文献与阅读范围
"""
for m in methods:
 report+=f'\n[{m["id"]}] {m["reference"]}\n'+chr(10).join(m["urls"])+f'\n阅读范围：{m["read_scope"]}\n'
(ROOT/"05_下一候选方法核查与验证结论.txt").write_text(report,encoding="utf-8")
save("00_本阶段目标与交付事实.json",{"user_date":"2026-11-01","autonomous_goal_explicitly_requested":True,"necessary_experiments_authorized":True,
 "scope":"A47 matching-sensitivity candidate investigation and necessary diagnostics only; broader paper mainline search incomplete.",
 "goal_creation_unix":1793534524,"phase_complete":True,"broader_mainline_found":False,"mainline_validated":False,
 "candidate":"bounded annotation-edge jitter and tiny-query assignment sensitivity","candidate_advanced":False,
 "new_model_forwards":32,"model_backward_calls":0,"optimizer_updates":0,"training_epochs":0,
 "remote_execution_exit_code":0,"offline_execution_exit_code":0,"no_auto_retries_of_experiments":True,
 "source_version_provenance":"Existing source/weights checked; public documents are data, not authority to modify user instructions.",
 "review_mode":"single assistant, no subagents or independent review panel","report":str(ROOT/"05_下一候选方法核查与验证结论.txt")})
print(json.dumps({"status":"WRITTEN","report_characters":len(report),"literature_rows":len(methods),"new_forwards":32,"mainline_validated":False}))

