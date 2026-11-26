from pathlib import Path
import hashlib
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '大论文框架与导师汇报方案_20261102.docx'
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(.7), Inches(.7)
sec.left_margin, sec.right_margin = Inches(.8), Inches(.8)
sec.header_distance, sec.footer_distance = Inches(.3), Inches(.3)

def font(style, size, east='宋体', bold=False):
    s = doc.styles[style]
    s.font.name = 'Calibri'
    s.font.size = Pt(size)
    s.font.bold = bold
    s.font.color.rgb = RGBColor(0, 0, 0)
    r = s.element.get_or_add_rPr()
    fs = r.find(qn('w:rFonts'))
    if fs is None:
        fs = OxmlElement('w:rFonts'); r.insert(0, fs)
    fs.set(qn('w:eastAsia'), east)
    for a in ('w:asciiTheme','w:hAnsiTheme','w:eastAsiaTheme'):
        fs.attrib.pop(qn(a), None)
    pf = s.paragraph_format
    pf.space_after = Pt(5)
    pf.line_spacing = 1.2
    return s

font('Normal', 11.5)
font('Title', 22, '黑体', True)
font('Subtitle', 11, '宋体')
for name, size in [('Heading 1',16),('Heading 2',13),('Heading 3',11.5)]:
    st=font(name,size,'黑体',True)
    st.paragraph_format.space_before=Pt(10)
    st.paragraph_format.space_after=Pt(5)
    st.paragraph_format.keep_with_next=True
for name in ('Header','Footer'):
    font(name,9)
for style in doc.styles:
    if style.type == 1:
        ppr=style.element.find(qn('w:pPr'))
        if ppr is not None:
            for node in list(ppr):
                if node.tag==qn('w:pBdr'): ppr.remove(node)

hp=sec.header.paragraphs[0]
hp.text='大论文框架整理  ·  导师讨论版  ·  2026年11月2日'
fp=sec.footer.paragraphs[0]; fp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
fp.add_run('第 ')
fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); fp._p.append(fld)
fp.add_run(' 页')

def p(text, bold=False, style=None):
    para=doc.add_paragraph(style=style)
    para.paragraph_format.widow_control=True
    run=para.add_run(text); run.bold=bold
    return para

def h(text, level=1):
    return doc.add_heading(text,level)

def item(title, text):
    para=doc.add_paragraph()
    para.paragraph_format.space_after=Pt(6)
    para.paragraph_format.keep_together=True
    para.add_run(title+'  ').bold=True
    para.add_run(text)
    return para

def page(title):
    doc.add_page_break(); h(title)

def table(headers, rows, widths):
    t=doc.add_table(rows=1, cols=len(headers))
    t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    pr=t._tbl.tblPr
    borders=OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+edge); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'4'); e.set(qn('w:color'),'D9D9D9'); borders.append(e)
    pr.append(borders)
    margins=OxmlElement('w:tblCellMar')
    for edge in ['top','left','bottom','right']:
        e=OxmlElement('w:'+edge); e.set(qn('w:w'),'95'); e.set(qn('w:type'),'dxa'); margins.append(e)
    pr.append(margins)
    for i, w in enumerate(widths): t.columns[i].width=Inches(w)
    for cell, txt in zip(t.rows[0].cells,headers): cell.text=txt
    rh=OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(rh)
    for data in rows:
        cells=t.add_row().cells
        for c, txt in zip(cells,data): c.text=txt
    for ri,row in enumerate(t.rows):
        for ci,cell in enumerate(row.cells):
            cell.width=Inches(widths[ci]); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ri==0:
                sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'EDEDED'); cell._tc.get_or_add_tcPr().append(sh)
            for para in cell.paragraphs:
                para.paragraph_format.space_after=Pt(0)
                para.paragraph_format.line_spacing=1.12
                for run in para.runs:
                    run.font.size=Pt(10.5); run.bold=ri==0
        no=OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(no)
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
    return t

# Page 1
p('大论文框架与导师汇报方案',style='Title')
p('基于现有 MPSMA-DEIM 稿件与硕士论文初稿整理',style='Subtitle')
h('建议采用的研究框架')
p('建议保留题目《面向无人机航拍图像的小目标检测算法改进研究》，以已有 MPSMA-DEIM 工作作为第一项研究，以低照度条件下的检测适应性作为第二项研究的拟定问题，最后通过已有系统进行应用验证。当前可以确定章节关系和写作范围；第二项方法及其有效性仍需后续研究建立。')
p('核心逻辑是先研究检测网络如何利用有限的小目标细节，再研究输入照度降低后检测能力如何变化及如何改善。两章共同围绕无人机小目标检测，分别处理网络表征与场景适应问题。低照度范围来自现有论文初稿，本轮将其收窄为待验证的问题，不预先指定增强器或损失函数。')
table(['章节','建议名称或内容','现有基础与状态'],[
('第1章','绪论','已有文字；需按新框架改写研究内容'),
('第2章','相关理论与实验基础','已有文字；统一模型、数据与指标口径'),
('第3章','融合渐进式细节保留与多尺度上下文的检测方法','已有小论文与实验表；需处理版本差异'),
('第4章','低照度条件下无人机小目标检测的适应性研究','研究方案；尚无已验证的第二项算法'),
('第5章','无人机小目标检测系统设计与验证','已有原型与5项CPU功能测试记录'),
('第6章','总结与展望','研究完成后按实际证据收束')
],[.65,3.45,2.8])
h('对工作量与创新的判断',2)
p('建议按两项相互关联的研究内容组织，但不能把“两项研究”直接说成“两项已完成的算法创新”。MPS 与 MSCA 属于同一项既有工作，不拆成两个独立研究章节；系统章节体现实现与应用工作，其分量由实际功能和验证支撑。学校对学位论文工作量的具体认定，本轮未核查。')

# Page 2
page('第1章与第2章的写作框架')
h('第1章 绪论',2)
item('1.1 研究背景与意义','围绕无人机航拍检测的应用需求，说明小目标像素少、尺度变化及背景干扰带来的识别困难。')
item('1.2 国内外研究现状','按航拍小目标检测、细节保留与多尺度建模、低照度检测三个主题梳理。每组研究最后说明仍需解决的问题，避免仅按模型年代罗列。')
item('1.3 问题分析与研究目标','提出两个问题：网络前端如何保留小目标可用信息；低照度下已有检测模型的失效如何分析和改善。第二个问题在当前阶段属于研究假设。')
item('1.4 主要研究内容与技术路线','分别对应第3章的方法、第4章的适应性研究和第5章的系统验证。将已完成内容与计划内容分开陈述。')
item('1.5 论文组织结构','说明六章之间的递进关系，不在此提前写入未取得的实验结论。')
h('第2章 相关理论与实验基础',2)
item('2.1 小目标特征提取与多尺度建模','介绍卷积、下采样、感受野、特征融合与本文实际使用的注意力机制；删除与后文无关的通识堆砌。')
item('2.2 DETR系列检测方法','区分 DETR 的集合预测、D-FINE 的检测结构与 DEIM 的训练和匹配机制，明确本文采用的具体模型尺度及配置。')
item('2.3 低照度成像与检测问题','说明亮度、对比度与噪声对可用视觉信息的影响，区分图像看起来更清楚与目标更容易被检出这两个评价目标。')
item('2.4 数据集与数据划分','介绍 VisDrone2019 的训练、验证及 test-dev 划分；真实低照度数据另行确定，明确类别、标注和拆分单位。')
item('2.5 评价指标与比较规范','统一 AP、AP50、AP75、APs、参数量和时延定义；写清输入尺寸、训练预算、初始化、种子及推理计时边界。')
item('2.6 本章小结','说明这些基础如何服务于后续两项研究。')
p('写作原则：第2章提供共同定义与协议；第3章、第4章分别补充本章特有的设置，不重复介绍整套基础知识。')

# Page 3
page('第3章 既有小论文对应的研究')
h('融合渐进式细节保留与多尺度上下文的检测方法',2)
p('研究问题：在既定输入分辨率下，如何改善无人机小目标的特征表达。以现有 MPSMA-DEIM 为主体，完整解释动机、结构、实现与结果。章名突出研究问题和方法作用，具体缩写放在正文中定义。')
item('3.1 问题定义与方法总览','解释浅层下采样和上下文利用方面的设计动机，给出与基线的结构差异图。')
item('3.2 多分支渐进式Stem','说明各分支输入、步长、尺寸对齐、融合与输出；高频分支的作用应使用现有证据支持。')
item('3.3 分层多尺度上下文建模','说明各阶段采用何种注意力，给出 MSCA-Lite 与 MSCA-Dilated 的实际公式及参数。')
item('3.4 整体模型与训练配置','单独交代全骨干 SiLU 替换，明确其属于配置因素；区分 n 与 s 两个尺度，说明初始化和训练预算。')
item('3.5 受控消融与结果分析','优先呈现同尺度、同划分、同训练协议的五组消融；分别解释单因素变化与完整配置结果。')
item('3.6 对比实验与效率分析','严格比较与文献参考值分表呈现；参数量、计算量、时延和可视化分别说明，不将跨划分数值直接相减归因。')
item('3.7 局限与本章小结','明确现有证据的数据集与模型尺度范围，并引出低照度条件下的待研究问题。')
h('回填正文前必须统一的三件事',2)
table(['问题','本章处理方式'],[
('旧稿与英文稿实验口径不同','英文稿表1报告 n 尺度 AP 0.1895→0.2270；旧论文采用另一组逐步叠加数值。按原始记录核定，不能混拼为同一实验。'),
('中文与英文方法表达有差异','MSCA-Lite 的聚合和输出形式、MSCA-Dilated 的空洞率及输出形式并不一致。最终以实际代码、配置和权重对应关系为准。'),
('结果和推论需要对应','五组消融支持这些配置下的性能比较；未覆盖全部组合，不能据此严格证明各模块之间的协同机制。')
],[1.55,5.35])
p('以上是稿件一致性核查，不要求立即重跑实验。英文稿已有时延结果和更完整的消融表，应先核对归属再回填，避免继续沿用旧稿的“全部待补”清单。')

# Page 4
page('第4章 第二项研究的拟定范围')
h('低照度条件下无人机小目标检测的适应性研究',2)
p('本章定位为研究方案。具体问题是：照度降低时，第3章方法的小目标检测性能如何变化；在固定检测器和训练预算下，面向检测任务的适应策略能否优于直接检测、常规低光增强及简单数据增强。当前不把任何候选模块写成已成立的创新。')
item('4.1 低照度下的检测失效分析','按照度、目标尺寸和遮挡等条件分析漏检与定位误差。目标是获得可重复的失效证据，判断是否值得进一步设计方法。')
item('4.2 数据与评估协议','合成低照度用于控制变量，先划分原图再退化，固定生成参数；真实低照度数据用于检验外部适用性，数据来源和类别映射确定后再写入正式稿。')
item('4.3 简单策略与改进需求','比较直接检测、训练时低光增强、常规图像增强后检测。先确认这些方案留下的具体不足，再提出方法设计。')
item('4.4 面向检测的适应方法','这一节保留方法位置，但暂不命名新模块、不固定复杂损失。后续只围绕4.1与4.3证实的问题提出可解释的改变，并交代与已有方法的实质差异。')
item('4.5 实验结果与分析','以检测指标为主，报告普通照度与低照度结果、分层表现、失败样例和额外开销；图像质量仅作辅助指标。')
item('4.6 本章小结','依据实际结果说明有效范围及局限，不提前写入性能提升结论。')
h('将来验证时的最小比较关系',2)
table(['比较项','要回答的问题'],[
('基线与第3章方法','原有改进在照度变化下是否仍有作用'),
('直接检测与简单策略','已有低成本手段能解决多少问题'),
('候选方法与最强简单策略','新增设计是否提供额外价值'),
('普通照度与低照度测试','改善是否以常规场景退化或过大开销为代价')
],[2.15,4.75])
p('进入算法章节的条件：问题能够复现，方法具有可说明的差异，收益能够在一致协议下重复得到。若只有合成数据，只能报告合成低照度范围的结论；若简单策略已解决问题，应保留诚实的分析结果并重新评估研究内容，不能靠增加模块补足“创新点”。当前方案不意味着第二项算法已经完成，也不意味着已满足学位要求。')

# Page 5
page('第5章与第6章的写作框架')
h('第5章 无人机小目标检测系统设计与验证',2)
item('5.1 需求与使用场景','明确输入类型、模型切换、检测结果展示、导出和异常提示等需求，说明系统服务于离线图像和视频检测。')
item('5.2 系统架构与运行流程','给出交互界面、推理服务、模型管理、结果保存的关系及数据流，说明代码与模型配置如何对应。')
item('5.3 核心功能实现','介绍图片检测、视频逐帧处理、基线与改进模型切换、日志和结果导出。保留可复现的输入与输出示例。')
item('5.4 功能与异常测试','利用已有测试记录说明有效输入、空图片、无效视频及短视频处理结果；明确每个测试实际覆盖的范围。')
item('5.5 推理性能与应用验证','后续在固定硬件和一致协议下报告延迟、资源占用与功能场景结果。系统测试和算法精度实验分别报告。')
item('5.6 本章小结','总结已实现能力与实际适用范围。第4章方法验证成功后再作为模型选项接入。')
p('现有基础：已归档的5项CPU功能测试均记录通过，但测试文件明确标注 formal_performance_claim=false。界面低光增强仅为确定性预览，不是第4章训练成果；这些记录不能证明机载实时部署或长视频稳定运行。')
h('第6章 总结与展望',2)
item('6.1 研究工作总结','分别总结特征表达研究、已实际完成的第二项研究和系统实现，以真实完成内容为准。')
item('6.2 主要结论与适用范围','将每项结论对应到具体实验和条件；研究内容、技术改动与学术贡献分开表述。')
item('6.3 局限与后续研究','说明数据覆盖、稳定性、效率和实际部署方面仍未解决的问题，避免把计划内容写成成果。')
h('三章之间的衔接',2)
p('第3章提供一个已有依据的改进检测器；第4章以统一协议研究它在低照度条件下的不足及适应策略，同时保留原始基线以检查适用范围；第5章承接实际验证通过的模型。新的英文期刊选题只有与这一对象、问题和证据相容时才并入大论文，不能仅因另有实验就强行放入第四章。')

# Page 6
page('英文小论文现状与近期整理顺序')
h('已有稿件的定位',2)
p('已找到 MPSMA-DEIM_conference_EN.docx，包含摘要、方法、实验、结论及参考文献，是现有工作的英文会议稿。文件存在可以支持“英文稿已形成”，但不足以认定已达到某个期刊或会议的录用标准。本轮核查了正文与表格，未完成原始训练日志、全部引文及目标刊会要求的逐项复核。')
h('投稿前优先修正的内容',2)
item('实验比较口径','英文稿表2把验证集实测值与文献中的 test-dev 值放在一起，表题却统称验证集；正文还使用“精度相近”“小目标增益集中”等推论。应分表或移至相关工作参考，不能只在表下加说明后继续作公平优劣判断。')
item('方法和实验版本','中英文 MSCA 公式存在差异，大论文又使用更早的消融表。先整理代码、配置、日志、权重与图表的对应关系，再统一三份稿件；文件修改时间较新不等于内容必然正确。')
item('稳定性与部署表述','英文稿以最后20轮的波动描述训练后期稳定，这不等同于多随机种子稳定性。RTX 3090上44.3 FPS属于文中报告的平台结果，不能直接推断满足机载应用的实时要求。')
item('贡献表述','将 MPS 与分层 MSCA 的具体设计作为待论证的贡献，说明相对常规结构的区别；SiLU单列为实验因素。五组消融中的完整配置最好，不足以单独证明严格的模块协同。')
h('当前可以先完成的整理',2)
p('第一，统一小论文和大论文第三章的事实版本，建立每张表对应哪组实验的清单。第二，按本框架改写绪论、第三章及系统章节，保留第四章研究方案。第三，按目标刊会的范围与格式修订英文稿，把2026年12月作为拟提交时间。新的期刊创新继续作为待研究事项，不设为现有稿件开始修订的前提。')
p('以上为文稿整理与研究安排。本轮不启动新训练，不预告第二项创新已经找到；12月是投稿计划，不是录用或检索承诺。具体刊会尚未确定，本文件不作级别或录用概率判断。')

# Page 7
page('可向导师汇报的文字')
h('第一项 小论文',2)
p('老师，目前小论文已经在原有 MPSMA-DEIM 工作基础上形成了一份英文会议稿。我计划在12月完成针对目标刊会的修改并投稿，近期先统一方法描述和实验比较口径。同时，面向英文期刊的新创新点还在重新梳理和论证，目前尚未形成可以确定下来的新贡献。')
h('第二项 大论文',2)
p('大论文已有正文初稿和检测系统原型，但研究框架还需要收紧。我拟继续围绕无人机小目标检测展开：第三章以现有 MPSMA-DEIM 工作研究细节保留和多尺度特征表达；第四章拟研究低照度条件下的检测适应性，先明确失效问题和简单方法的局限，再确定改进方案；第五章完成系统实现与应用验证。目前第一项工作和系统已有材料，第二项研究仍处于方案阶段。接下来先统一小论文与大论文的数据和方法版本，并按这个框架推进写作。')
h('若导师追问第二个创新是否确定',2)
p('目前还没有确定。我已经把第二项研究的问题范围收窄了，但不会把设想直接写成创新点。现阶段能够汇报的是已有成果、拟研究的问题以及如何验证它，而不是已经完成了两项算法改进。')
h('本次整理依据',2)
p('材料1  小论文目录中的 MPSMA-DEIM_conference_EN.docx 和 MPSMA-DEIM_修订稿_真实数据版.docx。主要核对英文稿表1至表4、方法部分及对应中文表述。')
p('材料2  大论文／调研成果／大论文初稿归档／大论文初稿_系统闭环回填版_v2.md 及同名Word文件。用于核对六章结构、第四章暂停说明及现有系统内容。')
p('材料3  大论文／调研成果／面向无人机航拍图像的小目标检测算法改进研究_完整大纲_定稿版.md。作为既有研究范围和章节组织的参考。')
p('材料4  大论文／调研成果／第五章系统闭环_v1／system_test_results.json。用于确认已保存的功能测试及其性能声明边界。')
p('说明：本文件是重新整理后的框架与汇报方案，保留原稿。已完成事实、研究建议和未验证内容在文中分别标明；其中第四章仍为拟开展研究。')

doc.core_properties.title='大论文框架与导师汇报方案'
doc.core_properties.subject='无人机小目标检测硕士论文框架与投稿进度汇报'
doc.core_properties.author=''
doc.core_properties.keywords='MPSMA-DEIM 无人机 小目标检测 大论文框架'
doc.save(OUT)

report='''导师汇报文字
2026年11月2日

1. 小论文
老师，目前小论文已经在原有 MPSMA-DEIM 工作基础上形成了一份英文会议稿。我计划在12月完成针对目标刊会的修改并投稿，近期先统一方法描述和实验比较口径。同时，面向英文期刊的新创新点还在重新梳理和论证，目前尚未形成可以确定下来的新贡献。

2. 大论文
大论文已有正文初稿和检测系统原型，但研究框架还需要收紧。我拟继续围绕无人机小目标检测展开：第三章以现有 MPSMA-DEIM 工作研究细节保留和多尺度特征表达；第四章拟研究低照度条件下的检测适应性，先明确失效问题和简单方法的局限，再确定改进方案；第五章完成系统实现与应用验证。目前第一项工作和系统已有材料，第二项研究仍处于方案阶段。接下来先统一小论文与大论文的数据和方法版本，并按这个框架推进写作。

如果导师追问第二个创新是否确定
目前还没有确定。我已经把第二项研究的问题范围收窄了，但不会把设想直接写成创新点。现阶段能够汇报的是已有成果、拟研究的问题以及如何验证它，而不是已经完成了两项算法改进。
'''
(ROOT/'导师汇报文字_20261102.txt').write_text(report,encoding='utf-8-sig')
sources=[
Path(r'C:\Users\jiwei\Desktop\小论文投稿\小论文\MPSMA-DEIM_conference_EN.docx'),
Path(r'C:\Users\jiwei\Desktop\小论文投稿\小论文\MPSMA-DEIM_修订稿_真实数据版.docx'),
Path(r'C:\Users\jiwei\Desktop\大论文\调研成果\大论文初稿归档\大论文初稿_系统闭环回填版_v2.md'),
Path(r'C:\Users\jiwei\Desktop\大论文\调研成果\第五章系统闭环_v1\system_test_results.json')]
(ROOT/'source_manifest.json').write_text(json.dumps([{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in sources],ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT)
