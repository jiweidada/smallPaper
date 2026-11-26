from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
import json,hashlib,csv

BASE=Path(__file__).resolve().parent
SOURCE=Path(r'C:\Users\jiwei\Desktop\大论文\大论文框架\大论文框架与导师汇报方案_20261102.docx')
OUT=BASE/'大论文框架_导师审阅版_20261102.docx'
doc=Document(SOURCE)
body=doc._element.body
for element in list(body):
    if element.tag!=qn('w:sectPr'): body.remove(element)
for style_name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3'):
    st=doc.styles[style_name]
    st.font.color.rgb=RGBColor(0,0,0)
    st.font.italic=False
doc.styles['Normal'].font.size=Pt(11.5)
doc.styles['Normal'].paragraph_format.line_spacing=1.15
doc.styles['Normal'].paragraph_format.space_after=Pt(4)
doc.styles['Title'].font.size=Pt(20)
doc.styles['Title'].paragraph_format.space_after=Pt(8)
for name,size in [('Heading 1',14),('Heading 2',12.5)]:
    st=doc.styles[name]; st.font.size=Pt(size)
    st.paragraph_format.space_before=Pt(9)
    st.paragraph_format.space_after=Pt(4)
    st.paragraph_format.keep_with_next=True
for sec in doc.sections:
    sec.header.paragraphs[0].text='硕士学位论文框架  ·  2026年11月'

def p(t,style=None):
    q=doc.add_paragraph(t,style)
    q.paragraph_format.widow_control=True
    return q
def h(t,level=1): return doc.add_heading(t,level)
def line(t):
    q=p(t)
    q.paragraph_format.space_after=Pt(3)
    q.paragraph_format.keep_together=True

p('硕士学位论文框架',style='Title')
p('论文题目  面向无人机航拍图像的小目标检测算法改进研究',style='Subtitle')
h('一 研究目标与主要内容')
p('围绕无人机航拍图像中小目标细节不足、尺度变化和低照度干扰等问题，研究小目标特征表达与场景适应方法，并通过检测系统验证其应用效果。')
p('研究内容一：以 DEIM-D-FINE 为基础，研究多分支渐进式细节保留与分层多尺度上下文建模，形成 MPSMA-DEIM 检测方法。')
p('研究内容二：拟研究低照度条件下的小目标检测适应性，分析已有模型的漏检与定位误差，比较常规增强策略，并在此基础上确定改进方法。')
p('研究内容三：实现无人机小目标检测系统，完成图像与视频检测、模型切换、结果展示及导出，并开展功能与性能验证。')
h('二 章节安排')
h('第1章 绪论',2)
for x in ['1.1 研究背景与意义','1.2 国内外研究现状','1.3 无人机小目标检测的关键问题','1.4 主要研究内容与技术路线','1.5 论文组织结构']: line(x)
h('第2章 相关理论与实验基础',2)
for x in ['2.1 小目标特征提取与多尺度建模','2.2 注意力机制与特征增强','2.3 DETR系列检测方法与DEIM训练框架','2.4 低照度成像与目标检测','2.5 数据集 评价指标与实验协议','2.6 本章小结']: line(x)

doc.add_page_break()
h('第3章 融合渐进式细节保留与多尺度上下文的检测方法',2)
for x in ['3.1 问题分析与方法总体框架','3.2 多分支渐进式Stem设计','3.3 分层多尺度上下文建模','3.4 整体模型与训练设置','3.5 消融实验与方法对比','3.6 模型复杂度 推理效率与结果分析','3.7 本章小结']: line(x)
h('第4章 低照度条件下无人机小目标检测的适应性研究',2)
for x in ['4.1 低照度条件下的检测失效分析','4.2 低照度数据与评估协议','4.3 常规适应策略及其局限','4.4 面向检测任务的适应方法','4.5 对比实验 消融实验与适用范围分析','4.6 本章小结']: line(x)
h('第5章 无人机小目标检测系统设计与验证',2)
for x in ['5.1 系统需求与总体架构','5.2 图像与视频检测功能实现','5.3 模型管理 结果展示与导出','5.4 系统功能与性能测试','5.5 本章小结']: line(x)
h('第6章 总结与展望',2)
for x in ['6.1 研究工作与主要结论','6.2 研究局限与后续展望']: line(x)
h('三 当前进展',2)
p('MPSMA-DEIM 小论文初稿、相关实验记录及检测系统原型已形成，当前正在统一稿件中的方法描述与实验口径。第四章为拟开展研究，具体改进方法尚待论证与验证。')

doc.core_properties.title='硕士学位论文框架'
doc.core_properties.subject='面向无人机航拍图像的小目标检测算法改进研究'
doc.core_properties.comments=''
doc.save(OUT)

metrics=Path(r'C:\Users\jiwei\Desktop\小论文投稿\数据\01_零训练数据\run_metrics.csv')
with metrics.open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
selected=[r for r in rows if r['tag']=='best_AP' and r['run'] in ['deim_hgnetv2_n_custom320','deim_hgnetv2_EnhancedStemNO_NOMSCA_Silu_n_visdrone_320','deim_hgnetv2_EnhancedStem_NOMSCA_NOSilu_n_visdrone_320','deim_hgnetv2_EnhancedStemNO_MSCA_NOSilu_n_visdrone_320','deim_hgnetv2_MSCA_EnhancedStem_SiLU_320']]
(BASE/'核对到的五组指标记录.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')

review='''MPSMA-DEIM英文会议稿审查意见
2026年11月2日

总体判断
有继续整理为应用型会议投稿的价值，但当前版本不宜直接提交。论文已有完整方法、五组消融表、规模扩展、效率和可视化材料，英语整体可读。主要问题在实验归因与方法表达的一致性。尚未指定会议，因此本判断不等于某会议的录用意见，也不保证EI收录。
本次为单助手稿件审查，未运行新实验，未认证原始实验可复现性。

一 最优先核查 消融是否真正只改变指定因素
严重程度：Major。证据：table: 英文稿Table 1；dataset: 数据/01_零训练数据/run_metrics.csv中best_AP记录。
该CSV能对应论文五组AP数字，但记录的参数量如下：
基线 deim_hgnetv2_n_custom320：3,737,611；AP 0.1895。
SiLU-only：5,335,041；AP 0.2117。
MPS-only：5,340,366；AP 0.2182。
MSCA-only：7,535,617；AP 0.2078。
完整配置：7,540,942；AP 0.2270。
仅将ReLU换成SiLU不应凭空增加1,597,430个可学习参数。这里可能存在基线结构、配置、参数统计口径或记录对应关系的差异。当前不能把+2.22个AP百分点全部归因于SiLU，也不能只凭论文写了“相同设置”就确认严格受控。
结论置信度：5/5，参数记录差异明确；差异具体原因尚未证实。找到的后期备份配置把SiLU组指向HGNetv2_Silu并采用ESE配置，可作追溯线索，不能代替原始运行快照。
最小修正：对照原始五组配置、实际模型结构、日志和参数统计方式，列出差异。若是统计或索引错误，修正记录；若确有额外结构变化，修正实验标签和贡献归因，必要时另行安排配套对照。本轮不执行训练。

二 必须修正 英文正文与结构图不一致
严重程度：Major。证据：equation: Section 3.3；figure: Figures 4 and 5。
图4把聚合特征送入Global AvgPool再送入SimAM，而英文公式是在空间特征图上计算均值和方差后生成注意力；图中的乘法对象也未清楚对应公式Out=F*A。
图5的空洞率标为1、2、3，英文正文为1、2、4；图中采用Spatial Conv(k×k)，英文采用能量函数形式的空间注意力。两者并非同一计算过程。
置信度：5/5，稿件内部可直接核对。抽查的本地MSCA.py快照在聚合、空洞率及输出上更接近英文公式，但其是否为表1各组训练源码尚待关联。
最小修正：以对应已报告权重的实际实现为依据统一正文、公式、图2至图5与参数表，再重绘错误图示。不得凭任意一份较新代码决定旧实验实际实现。

三 必须修正 跨数据划分的精度比较
严重程度：Major。证据：table: Table 2；text: Section 4.1的比较说明与Section 4.3的优劣分析。
表2文献行使用test-dev与预训练权重，本稿实测行使用validation与从头训练，表题却写成全部在validation上的比较。正文仍使用精度相近、APs提高等推论，加入限制说明并不能使比较成为公平对照。
置信度：5/5，稿件自己说明了两种协议。
最小修正：将文献结果与本地同协议结果分表；撤回跨协议收益和排名推论。s尺度结果目前可作为本方法的规模扩展展示，不能把0.219到0.2876直接视为同条件算法收益。

四 贡献需要收敛到可辩护的具体差异
严重程度：Major。证据：text: Sections 1、2、3.2、3.3；equation: Section 3.3的SimAM风格权重。
当前更接近基于已有检测框架的结构组合与场景改进。多分支、池化、空洞卷积、ESE/SimAM本身不应被整体描述成新机制。SimAM原论文已给出无新增可学习参数的能量函数注意力，因此应突出你的分支构造、阶段配置和融合方式的具体区别，而不是仅使用新的模块名。
置信度：4/5，已对照本稿及SimAM原文相关方法部分，未进行穷尽性新颖性检索。
最小修正：明确每个组件的继承与改动，用最贴近的结构参照说明差别。SiLU继续保留为实验配置因素。五组配置完整方案最好，可支持配置比较；严格的互补或协同因果结论需更强控制或降低措辞。

五 证据范围与表述
严重程度：Minor。证据：text: Abstract、Sections 4.4、4.5、Conclusion。
1. 后20轮AP标准差0.0003仅说明一条训练轨迹后期波动，摘要和结论的“vary by”应明确为标准差，不等于极差或多seed稳定性。
2. latency_FP32.json可对应44.3 FPS，测于RTX 3090；应报告计时范围、FP32与硬件，不直接推断机载实时要求。该记录中基线resume=null，宜补充说明计时模型的权重状态与后处理路径。
3. 英文4.1说“All experiments”在RTX 4080，但时延用RTX 3090，改成训练环境与推理测试环境分别描述。
4. 图7中的检测框增多不能单独证明漏检减少；需与GT及一致置信度阈值对应。
5. main branch中的“concatenates”两层串行卷积宜改为“stacks”或“applies ... sequentially”；统一模型大小写、图中模块名称、引用格式及公式编号。
置信度：5/5，稿件和现有时延记录可直接对应。

对投稿定位的建议
保留这项工作，先完成实验归因与图文一致性修正，再依据具体会议范围和要求决定投稿。无需为了修稿立刻再增加新模块。若最重要的受控比较无法解释清楚，英文润色或换成会议模板都不能解决核心问题。
EI Compendex是索引数据库，会议论文集收录另有遴选过程；EI标签本身不是统一的论文难度等级。官方FAQ明确说明提交评估不保证收录。

核查范围与未核查项
已读英文稿完整正文、四张结果表、七幅图；核对本地run_metrics.csv、latency_FP32.json及MSCA.py后期备份片段。未重评权重，未证明五组原始数据流相同，未逐条核验所有19条参考文献，未指定会议。
审稿建议不代表方法无效，也不替代对原始实验的进一步核对。

资料位置
英文稿：C:/Users/jiwei/Desktop/小论文投稿/小论文/MPSMA-DEIM_conference_EN.docx
指标：C:/Users/jiwei/Desktop/小论文投稿/数据/01_零训练数据/run_metrics.csv
时延：C:/Users/jiwei/Desktop/小论文投稿/数据/01_零训练数据/latency_FP32.json
代码片段：C:/Users/jiwei/Desktop/小论文投稿/outputs/small_object_encoder_decoder12_20261010_v5/terminal_collection/seed1/extracted/native/engine/extre_module/custom_nn/attention/MSCA.py
SimAM原文：https://proceedings.mlr.press/v139/yang21o.html
EI官方FAQ：https://suggestor.ei.engineeringvillage.com/faq/index
'''
(BASE/'英文会议稿审查意见_20261102.txt').write_text(review,encoding='utf-8-sig')
(BASE/'编辑记录.json').write_text(json.dumps({'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'output':str(OUT),'removed':'内部审查记录、核查清单、操作说明、导师汇报话术、来源路径、重复边界说明','retained':'题目、目标、三项研究内容、六章目录、简短当前进度','scope':'第四章仍为拟开展研究；未更改英文论文原文'},ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT)
