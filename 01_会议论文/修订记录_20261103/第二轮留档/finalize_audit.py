"""Check manuscript against archived sources and write the revision handoff.
No model import, inference, training, evaluation, or benchmark is performed.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,re,shutil,zipfile
from docx import Document
from lxml import etree

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
SRC=ROOT/'01_会议论文/稿件/MPSMA-DEIM_conference_EN.docx'
DEST=SRC.with_name('MPSMA-DEIM_conference_EN_revised.docx')
BACK=SRC.parent/'备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
expected='ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SRC)==sha(BACK)==expected
d=Document(DEST); runs=json.loads((OUT/'run_audit.json').read_text(encoding='utf-8'))
assert len(d.tables)==5 and len(d.inline_shapes)==7

def rows(t):return [[c.text for c in r.cells] for r in t.rows]
cfg,acc,scale,complexity,latency=map(rows,d.tables)
for k,r in enumerate(['N0','N1','N2','N3','N4'],1):
    a=runs[r]
    assert cfg[k][0]==r and cfg[k][-1]==f'{a["n_parameters"]/1e6:.4f}'
    want=[r,str(a['best_epoch'])]+[f'{a["metrics"][x]:.4f}' for x in ['AP','AP50','AP75','APs','AR100']]
    assert acc[k]==want,(acc[k],want)
for k,(r,label) in enumerate([('N4','n'),('S1','s')],1):
    a=runs[r]
    want=[label,str(a['best_epoch'])]+[f'{a["metrics"][x]:.4f}' for x in ['AP','AP50','AP75','APs']]+[f'{a["n_parameters"]/1e6:.4f}']
    assert scale[k]==want
latpath=ROOT/'05_代码与数据/原始论文数据/01_零训练数据/latency_FP32.json'
lat={a['tag']:a for a in json.loads(latpath.read_text(encoding='utf-8-sig')) if a['precision']=='FP32'}
for k,key in [(1,'DEIM-D-FINE-S'),(2,'MPSMA-DEIM')]:
    a=lat[key]
    assert latency[k][1:]==[f'{a["net"]["mean_ms"]:.3f}',f'{a["e2e"]["mean_ms"]:.3f}',f'{a["e2e"]["fps"]:.2f}']
    assert complexity[k][2]==f'{a["params_M"]:.2f}'
assert [r[1] for r in complexity[1:]]==['24.86','29.63']

body=[];refs=[];inrefs=False
for p in d.paragraphs:
    if p.text=='References':inrefs=True;continue
    (refs if inrefs else body).append(p.text)
refnums=[int(re.match(r'^\[(\d+)\]',s).group(1)) for s in refs if re.match(r'^\[(\d+)\]',s)]
assert refnums==list(range(1,12))
cited=set()
for m in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', '\n'.join(body)):
    ids=[int(x.strip()) for x in m[1].split(',')]
    if all(1<=x<=11 for x in ids):cited.update(ids)
assert cited==set(range(1,12)),cited
text='\n'.join(body)
for n in range(1,7):assert f'Fig. {n}.' in text
for n in range(1,6):assert f'Table {n}.' in text
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with zipfile.ZipFile(DEST) as z:
    doc=etree.fromstring(z.read('word/document.xml'))
    styles=etree.fromstring(z.read('word/styles.xml'))
    comments=etree.fromstring(z.read('word/comments.xml'))
    assert len(doc.xpath('//m:oMath',namespaces=ns))==6
    comment_ids=set(comments.xpath('/w:comments/w:comment/@w:id',namespaces=ns))
    anchors=set(doc.xpath('//w:commentRangeStart/@w:id',namespaces=ns))
    assert len(comment_ids)==5 and comment_ids==anchors
    assert not doc.xpath('//w:pBdr',namespaces=ns)
    assert not styles.xpath('//w:pBdr',namespaces=ns)
    assert not doc.xpath('//w:ins|//w:del',namespaces=ns)

manifest=json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))
for item in manifest:assert sha(item['source'])==sha(item['snapshot'])==item['sha256']
archive=OUT/'source_snapshot/archived_benchmark';archive.mkdir(exist_ok=True)
extra=[]
for p in [latpath,latpath.with_name('模块参数量分解.txt'),latpath.with_name('零训练数据汇总.md'),ROOT/'90_历史与工具/旧期刊投稿材料/《计算机应用与软件》投稿/_work/latency.py']:
    target=archive/p.name
    if not target.exists():shutil.copy2(p,target)
    assert sha(target)==sha(p)
    extra.append({'source':str(p),'snapshot':str(target),'sha256':sha(p)})
(OUT/'benchmark_source_manifest.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')

dataset=json.loads((OUT/'dataset_eval_audit/annotation_audit.json').read_text(encoding='utf-8'))
saved_eval=json.loads((OUT/'dataset_eval_audit/saved_eval_metadata.json').read_text(encoding='utf-8'))
assert dataset['train']['annotations']==343205 and dataset['val']['annotations']==38759
assert dataset['val']['original_category_counts']['0']==1378 and dataset['val']['original_category_counts']['11']==32
assert len(saved_eval)==7 and all(r['params']['maxDets']==[1,10,100] for r in saved_eval)
image_sources=json.loads((OUT/'dataset_eval_audit/figure6_source_manifest.json').read_text(encoding='utf-8'))
embedded_hashes={hashlib.sha256(d.part.related_parts[s._inline.graphic.graphicData.pic.blipFill.blip.embed].blob).hexdigest() for s in d.inline_shapes}
for a in image_sources:assert sha(a['source'])==sha(a['snapshot'])==a['sha256'] and a['sha256'] in embedded_hashes
checks={'source_and_backup_sha256':expected,'revised_docx_sha256':sha(DEST),'tables_match_raw_logs_and_archived_latency':True,'references_count':11,'all_references_cited':True,'figures':6,'inline_images':7,'native_equations':6,'comments_with_matching_anchors':5,'source_snapshot_files_verified':len(manifest),'benchmark_snapshot_files_verified':len(extra),'figure6_source_images_verified':2,'saved_eval_metadata_files_inspected':7,'annotation_counts_verified':True,'no_new_experiments':True,'layout_review':'See visual_qa.json for inspection of the final Word export.','submission_ready':False,'note':'These are artifact and source checks, not experimental replication or an ARS integrity PASS.'}
(OUT/'validation_report.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')

runlines=[]
for key,a in runs.items():
    runlines.append(f'{key}: {a["run"]}\n  日志第 {a["best_log_line"]} 行，epoch {a["best_epoch"]}（从0起）；AP={a["metrics"]["AP"]:.12f}，APs={a["metrics"]["APs"]:.12f}；训练参数={a["n_parameters"]:,}；train/val batch={a["train_batch"]}/{a["val_batch"]}。')

report=f'''MPSMA-DEIM 英文会议稿修订说明与未解决问题
更新日期：2026-11-03，第二轮补充；目标月份：2026年12月；会议尚未选择。
当前结论：已完成本轮内容与证据修订。稿件尚不能作为已达到投稿要求的版本提交；关键公平对照和复现证据仍未闭合。未运行新训练、推理、重评、profiling、时延测试或远程任务。

一、继续维护的文件
修订稿：{DEST}
原稿：{SRC}
原稿备份：{BACK}
原稿和备份SHA256相同：{expected}
本轮修订稿SHA256：{sha(DEST)}
本轮只更新英文修订稿，原英文稿、中文稿、历史实验和训练源码均保留。保留原作者、单位与联系信息；盲审及具体模板等选会后处理。
后续继续维护 MPSMA-DEIM_conference_EN_revised.docx，并按日期追加修订记录。revise_manuscript.py 是本轮从原稿重建的脚本；后续有人工编辑时不要直接重跑覆盖，应先备份并合并更改。

二、本轮已落到稿件的主要修改
1. 原消融表改为配置对照：新增N0–N4明确标识、stem、stage aggregation/gates、activation、batch及训练参数；精度单列为Table 2，取各自最高AP同一行的完整指标，不逐指标挑选峰值。
2. 原基线N0使用batch16，其余使用batch4；N1/N3还将代码特有的se双卷积聚合路径改成ESE聚合。已撤回“只换SiLU”“只加MPS”“模块独立收益”和“协同增益”的归因。
3. N0到N3增加1,597,430参数，四阶段分别增加4,094、53,246、557,052、983,038；stem/encoder/decoder张量形状一致。该差异不能由无参数的激活替换解释。
4. n完整配置N4的AP=0.226957240816；相对N0的AP/AP50/AP75/APs/AR100差值按未舍入日志计算为3.75/5.91/3.87/2.66/4.08个百分点。正文将其限定为配置差异。N1的大目标AP高于N4，移除“所有指标都最优”的泛化。
5. n与s分开报告：两者骨干均为B0；s增加stride8输入、宽度128→256、采样点[6,6]→[3,6,3]，学习率和参数组也不同。s精度不能作为“仅扩大骨干”或独立模块效应。
6. 移除外部test-dev结果与本地val结果混排的优越性排名；移除由这些跨协议数值推导的精度/参数优势。表中保留本地单次训练结果。
7. 重画图1–5，并保存可编辑SVG：MPS补齐28→32→16融合及末次stride2；MSCA-Lite没有先GAP，三尺度平均后算空间能量门，再乘原F；Dilated使用d=1/2/4、ratio8及C/8分组产生C通道，其两路门都乘原F。第二轮将图6换成两张可追溯的验证集原图，图注列文件名，像素未经修改，已核对嵌入图像与源文件SHA一致。
8. 六条公式与上述结构统一：Lite/Dilated方差分母HW，stage2 SimAM实现分母HW−1；Dilated有效核大小3/5/9，空间门保留C通道。SiLU只替换可配置骨干块，Dilated内部ReLU仍在。
9. 重写摘要、引言、相关工作、贡献段、实验分析和结论，保留可以说明的结构设计及单运行观察；不再声称新的SiLU/SimAM原理、统计显著性、种子稳定性或机载实时保证。
10. 将训练参数与部署参数分开。44.30 FPS明确为RTX3090/FP32/batch1/640输入的网络加后处理计时，输入已经位于GPU；不包含图像解码、预处理和传输。保留计时加载权重不完整可追溯的限制。
11. 清理参考文献，最终11条均与正文对应；补正D-FINE题名，加入ESE的CenterMask来源，并在第二轮加入官方VisDrone评价源码。删掉未在修订正文使用的条目，不把删掉的原19条全部称作已完成全文核查。
12. 第二轮完成只读数据与评价核查：当前train/val转换标注为343205/38759框，全部iscrowd=0且无ignore字段；与原始score=1、类别1–10标注逐框对应，没有数量/类别差异，最大坐标差约0.00143像素。val原始1378条ignored-region和32条others不在转换文件中。六组latest.pth及S1独立eval.pth保留的评价元数据均为10个IoU阈值、101个recall阈值、10类、maxDets=[1,10,100]、面积区间[0,1024]/[1024,9216]/[9216,1e10]。官方工具会用ignored-region筛除相应检测，AP上限为500，不能仅删除原始忽略框就宣称等同官方。正文4.1和指标定义已据此更新。未重算AP。

三、精度结果的直接来源
原始运行根目录：C:\\Users\\jiwei\\Desktop\\DEIM\\outputs
以下文件均已复制到本记录的source_snapshot/N0…N4、S1；source_manifest.json保留原路径与SHA256。run_audit.json保存完整12项指标、args、checkpoint元信息；configuration_differences.json保存配置差异。
{chr(10).join(runlines)}

表4、表5来源：source_snapshot/archived_benchmark，包括原latency_FP32.json、模块参数量分解.txt、零训练数据汇总.md及原latency.py。该JSON文件内同时有FP32和FP16记录，本稿只取precision=FP32两行。
原始时延记录的e2e标签已改称Combined，以匹配脚本实际范围。22.572/19.011对应约18.7%的延迟增加；不把两次独立均值的差当作直接测出的后处理延迟。

四、方法证据与边界
方法快照在source_snapshot/current_code。核心路径：
engine/backbone/hgnetv2_MSCA_EnhancedStem.py：EnhancedStemBlock及HG块聚合、B0分阶段配置。
engine/extre_module/custom_nn/attention/MSCA.py：MSCA_Dilated（89行起）、MSCA_Lite（157行起）。
engine/extre_module/custom_nn/attention/simam.py：SimAM能量门。
engine/backbone/hgnetv2.py、hgnetv2_silu.py、hgnetv2_MSCA.py：不同运行变体的聚合及激活。
engine/solver/det_solver.py：恢复best_stg1的控制流；不是原训练版本认证。
当前源码、args和checkpoint张量形状相互支持结构描述；无参数行为（dilation、激活、插值、lambda等）不能仅靠checkpoint形状确定。当前代码哈希不得冒充训练时代码哈希。

五、投稿前必须处理的问题
P0 公平主对照：现有六组不能隔离方法收益。仅修改文字无法使batch16/4及聚合结构不同的比较成为单因素消融。至少需要同条件基线与完整方法，或找到可证明同条件的既有记录。
P0 历史实现与数据评价版本：第二轮已核查现存标注哈希、转换对应及存量评价器参数；但尚未把这些当前文件绑定到历史运行时不可变源码/标注快照、评价器发行版本和运行环境清单。已明确本地COCO-style口径与官方ignore/maxDets口径不同。正文已去掉原“统一Windows11/RTX4080”的无充分依据断言。
P0 权重与最优日志对应：N2/N4/S1的best_stg2元信息last_epoch分别296/305/294，最高日志AP索引分别300/306/302。当前solver恢复逻辑说明last_epoch不能直接当作保存时日志索引，但不能据此认证已复现最优权重。需要原保存回执/控制台记录，或另行授权后重评指定SHA权重。N3的配置记录resume到自身last，不能只凭pretrained:false断言完整训练从头未中断。
P1 部署测量链：旧表GFLOPs为24.86/29.63，另一树为25.37/30.14；不能把两个版本混写。旧时延脚本先deploy再strict=False加载，原JSON未保存missing/unexpected keys；baseline resume为null。精度权重与已测时延结构的严格同一性尚未认证。若保留准确率—时延联合结论，应补固定版本、正确加载且可验证的部署测量。
P1 稳定性：末20轮AP均值/标准差只说明单运行后期变化。没有多seed结果，不声称显著性、可重复稳定性或泛化。
已解决 图6来源：第二轮替换为验证集两张原图并列出文件名；复制件和DOCX内嵌文件与原件哈希相同。选取排序后的首张和中间一张，仅作输入样图，不作为效果证据。原稿拼图仍留在原稿和第一轮留档。
P2 会议：尚未指定会议、截稿日、页数、盲审和格式；当前为内容工作版，未宣称符合任一会议模板。

六、最小补实验方案（方案已具体化，尚未授权和运行）
A. 回答“完整方法在公平训练条件下是否优于宿主基线”。最低两组：相同n规模的宿主原始聚合基线和完整MPSMA；固定batch4、seed0、数据/标注哈希、评价器、640输入、320epoch、增强和学习率、优化器/EMA以及预先确定的端点选择。两组均fresh；结构相同的部分采用可核验的共同初始化，新增结构独立但固定初始化规则。完整方法包含聚合结构变化，其收益只能称整个组合的收益，不能归因到MPS或MSCA单项。
评价口径必须在开跑前固定并明确命名；若主张官方VisDrone结果，必须使用含ignore处理及相应maxDets的官方协议，不能把本轮已查明的本地COCO-style数值直接改名。新增评价同样需要授权。
为何不能只补一条基线就默认成立：历史N4缺少冻结源码/数据/评价关联，且训练控制流/保存元信息待核清。只有该链路先补齐并能严格复用时，才可能只新训一个匹配基线；否则两组新跑才可比。
B. 若仍要单模块归因：在统一ESE聚合、标准stem、ReLU的共同控制上，最少五组为控制、仅MPS、仅hierarchical attention、仅SiLU、完整组合。改变某因素时其余因素固定；每组同seed同预算。这个控制不同于原始SE-path宿主，若还要证明对原宿主的收益，需加原宿主对照，或与A共用对应组。五组不足以证明三因素协同；若坚持协同，需要完整2^3设计并定义交互项，当前建议撤掉协同主张。
C. 解决权重与指标链：优先只读找保存回执；不足时，申请对指定SHA checkpoint进行固定评价器验证，并保存加载键检查、预测文件及评价结果。这仍属于新重评，需要明确授权。
D. 解决部署结论：如保留速度结论，另行申请两模型同设备同软件版本测量，先在非deploy结构严格加载已核验权重，再转换并核对输出，预先固定输入和计时范围；此处只列方案，未实施。
不把两组单seed结果等同于显著性证据；多seed是后续增强证据的选择，不能以“最小方案”之名声称完成稳定性验证。

七、修改与验证记录
revision_patch.json：逐段旧文/新文、图表操作、原稿及输出哈希；不宣称ARS机器完整性PASS。
validation_report.json：原稿/备份相同、表格数字与来源、11条引用对应、6幅编号图（7个内嵌图像）6式、5个问题批注、图6源图SHA及22份源码/运行快照核验（具体计数以JSON为准）。
acronym_report.json：摘要和正文独立的辅助缩写核查；部分复合/大小写缩写不被脚本识别，人工核对MSCA/SiLU/NMS/ESE/ReLU/IoU/FLOP/GFLOP的定义。品牌、硬件名、HW数学量及正文已说明的AP50/AP75见allowlist；“no findings”不等于完整自动认证。
visual_qa.json：最终DOCX用Word导出并经Poppler逐页检查；包内render_docx.py因缺少soffice不可用，采用本机Word只读导出作为替代。PDF/PNG仅为内部排版核查材料。
citation_archive.json及引用核查记录.txt：文献链接、版本、下载状态、哈希及实际阅读范围。8篇PDF已在原文库复用/归档，2篇仅核查出版社页面/摘要，明确不称已读全文；第11条为官方源码，其只读快照在dataset_eval_audit/official_toolkit。
dataset_eval_audit：第二轮完整静态核查，包括各split标注统计与SHA、原始逐文件清单、逐框对应结果、七份存量评价元数据、两张源图证明。已有S1独立评价TensorBoard文件未检出simple_value标量，不能从它推定最优权重对应关系。第一轮输出及报告保留在第一轮留档。

本轮验收：主要数值可追到日志；已暴露并纠正比较与归因问题；当前可取得实现对应的图文已统一；贡献表述已收敛；论文结构完整。实验公平性及历史运行复现链仍有重要缺口，因此整体“达到投稿要求”的目标尚未完成。
'''
(OUT/'修订说明与未解决问题.txt').write_text(report,encoding='utf-8-sig')

links={
1:'https://www.mdpi.com/2072-4292/16/1/149',
2:'https://xplorestaging.ieee.org/document/10640832/',
3:'https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html',
4:'https://openreview.net/pdf?id=MFZjrTFE7h',
5:'https://proceedings.neurips.cc/paper_files/paper/2024/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract-Conference.html',
6:'https://openaccess.thecvf.com/content_CVPR_2020/html/Lee_CenterMask_Real-Time_Anchor-Free_Instance_Segmentation_CVPR_2020_paper.html',
7:'https://proceedings.mlr.press/v139/yang21o.html',
8:'https://arxiv.org/abs/1702.03118',
9:'https://openaccess.thecvf.com/content_ICCVW_2019/papers/VISDrone/Du_VisDrone-DET2019_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ICCVW_2019_paper.pdf',
10:'https://doi.org/10.1007/978-3-319-10602-1_48',
11:'https://github.com/VisDrone/VisDrone2018-DET-toolkit'}
scopes={1:'出版社网页摘要和元数据；PDF请求403。只支持无人机检测困难的一般背景，不据此引述具体表格或公式。',2:'IEEE摘要与IGARSS2024官方会议程序；只支持卷积局部上下文与Swin结合的概要。未取得全文。作者、会议、页码和DOI核查；未比较其精度。',3:'CVF出版元数据与本地原文；定向核对dense matching / matchability-aware loss。',4:'ICLR/OpenReview原文；定向核对分布式边框细化与题名，保留正式ICLR2025条目。',5:'NeurIPS正式论文及出版页；定向核对consistent dual assignments与NMS-free训练；页码/DOI另与出版分发页相符。',6:'CVF正式原文；定向核对VoVNetV2和eSE，原文式(6)–(7)与相关段落支持归属。未声称本项目ESE每一实现细节与原文完全一致。',7:'PMLR元数据与本地原文；核对energy门的来源。本文中HW和HW−1的区别以代码为准，不将改写后公式冒称原始SimAM逐字公式。',8:'arXiv v3（2017-11-02）原文；定向核对SiLU=x·sigmoid(x)，不将其列为本工作新激活函数。',9:'CVF正式论文3.1节；核对6471/548和十类。该论文写1580个testing图像，与外部仓库1610 test-dev口径不同；修订稿不再在此引用下给test-dev数量。',10:'Springer正式ECCV元数据，另归档arXiv1405.0312v1原文（八位作者）；不混用后续版本作者表。评价器是否与官方设置完全一致仍待运行资料确认。'}
scopes[11]='官方evalDET.m指定VisDrone2019-DET-test-challenge并报告AP maxDets=500；dropObjectsInIgr.m同时筛除与ignored-region大面积重叠的GT和检测。读取并归档源码，未执行；不能据此声称已获得本项目官方分数。'
archive_data=json.loads((OUT/'citation_archive.json').read_text(encoding='utf-8'))
archive_data.append({'reference':11,'download_status':'Official source-code snapshots archived, never executed'})
c=['引用核查记录（2026-11-03，第二轮）','修订版共11条引用（10篇论文及1个官方源码仓库），均在正文使用；本记录只陈述实际核查范围，不声称逐页通读全部论文。','原19条参考文献中的未使用条目移除，并非把所有原始引用都判为错误。']
for n in range(1,12):
    c.extend(['',refs[n-1],links[n],scopes[n],f'本地归档状态：{archive_data[n-1]["download_status"]}'])
c.extend(['','原文库：'+str(ROOT/'03_文献资料/原始论文_20261004'),'元数据、PDF哈希和正文提取路径见citation_archive.json；目录为原文库迁移后的实际路径。','其他核查链接：','IGARSS2024官方程序：https://2024.ieeeigarss.org/view_paper.php?PaperNum=1464','YOLOv10出版分发页：https://www.proceedings.com/079017-3429.html','原外部比较表所在仓库：https://github.com/z1069614715/objectdetection_script/tree/master/visdrone2019-benchmark','此仓库的test-dev/预训练协议不与本地val/from-scratch配置混排得出优越性结论。'])
(OUT/'引用核查记录.txt').write_text('\n'.join(c),encoding='utf-8-sig')
(ROOT/'01_会议论文/当前修订入口.txt').write_text('当前内容修订稿：'+str(DEST)+'\n修改说明与未解决问题：'+str(OUT/'修订说明与未解决问题.txt')+'\n状态：内容修订完成一轮；尚不能投稿。未运行新实验。会议未选定。\n',encoding='utf-8-sig')
print(json.dumps(checks,ensure_ascii=True,indent=2))
