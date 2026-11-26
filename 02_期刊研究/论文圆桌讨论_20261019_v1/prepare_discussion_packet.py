"""Prepare an evidence-bounded discussion packet; no remote, model, or application calls."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

HERE=Path(__file__).resolve().parent
EN=HERE.parent
PAPER=EN/'相位机制论文_20261018'
DATA=PAPER/'data'
sources={
 'E1':DATA/'independent_full12.csv',
 'E2':DATA/'paired_full12_deltas_pp.csv',
 'E3':PAPER/'英文稿/main.tex',
 'E4':PAPER/'英文稿/tables/paired_detection_v5.tex',
 'E5':PAPER/'英文稿/tables/cost_rows_v3.tex',
 'E6':PAPER/'英文稿/tables/toolkit_rows.tex',
 'E7':PAPER/'英文稿/tables/ignore_pair_rows.tex',
 'E8':PAPER/'本轮修订说明_20261018_v5.txt',
 'E9':EN/'data/07_第二数据集_20261017/独立中期评估_20261018_v1/summary/小目标收益判断_第36轮_20261018.txt',
 'E10':EN/'data/07_第二数据集_20261017/GPU重申请恢复_20261019_v2/resume_acceptance.json',
}

def read_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

rows=read_csv(sources['E1']);deltas=read_csv(sources['E2'])
metrics=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
assert len(rows)==40 and len({(r['seed'],r['arm']) for r in rows})==20
lookup={(r['seed'],r['arm'],r['split']):r for r in rows}
for d in deltas:
 left=lookup[(d['seed'],d['arm'],d['split'])];right=lookup[(d['seed'],d['reference'],d['split'])]
 for metric in metrics:
  observed=100*(float(left[metric+'_raw'])-float(right[metric+'_raw']))
  assert abs(observed-float(d[metric]))<1e-9
assert all(math.isfinite(float(r[m+'_raw'])) for r in rows for m in metrics)

topic='''在已有 SPD-Conv、UAV-DETR SOEP 和 HMF-DEIM 结构先例下，我们的轻量 P2→P3 分支 C 及训练期 C-mean、rank16/rank16-mean 对照，究竟形成了什么有发表价值的新认识？请根据共同资料独立判断：哪些主张成立，哪些只是已有算子的实现或“保留更多信息会有收益”；是否足以组织成面向 SIVP 的英文实证论文，以及最小还缺什么证据。

本场只讨论研究贡献与证据，不启动训练、不操作服务器、不修改稿件或自动监测。UAVDT 最终结果尚未完成；第36轮仅为中期诊断，正文预计情景不作证据。不得预设论文能发或必退稿，也不以增加模块、实验数量或四位专家的多数票替代证据。

第1轮：每位独立给出最强可守主张、最强反对理由、一个可能改变判断的证据缺口。
第2轮：至少回应另外两位的具体论点，指出证据编号、隐含假设及同意或反对的理由；不要只附和。
第3轮：各自给出收紧后的贡献句、保留/删除/降级主张表、至多两项必要补证（区分已有数据分析与新训练），以及 UAVDT 最终正向/混合/负向三种情形下的论文处理方案。允许保留分歧。

每个数字注明划分、种子、比较对象、评估协议与单位。每轮先给结论，随后给证据；建议每人约900–1500个中文字，围绕论证，不复述整份材料。'''

perspectives={
 'Codex A｜机制与结构':'''你负责机制与结构解释，但不承担“替作者证明创新成立”的任务。逐项判断 C 是否只是已知算子的轻量参数化，以及训练期消除子格差异后究竟能推断什么。特别检查：pixel_unshuffle+1×1 等价于作用在 U 上的2×2 stride-2卷积；固定相位通道置换可被后续权重吸收；phase不是傅里叶相位或已证明的亚像素坐标编码；C/C-mean的输入子空间和梯度不同；rank16/rank16-mean仅具有相同最大映射秩，不是所有容量与优化过程相同。区分“子格差异具有增量用途”和“空间位置机制已被证明”。从 SOEP/SPD/HMF-DEIM 的已有结构中界定本研究真正新增的问题与答案。允许你的最终判断为贡献不足。引用共同资料 E1–E10；未读原始文件不要声称独立复核。''',
 'Codex B｜实验与统计':'''你负责证据质量、效应量和实验可解释性。主比较是同种子 C−C-mean 与 rank16−rank16-mean，同时必须检查 C−baseline。核对 test-dev 主配对两个约0.16pp增益、三种子均值/SD/中位数、小目标ARs和AP75例外，以及中大目标代价。三个种子只作有限描述；验证/test-dev、类别、IoU、GT数、多个指标都不是额外训练重复。同名seed也不保证中断恢复后的逐步随机流一致。检查共同检出上的条件定位与固定分数阈值的选择效应；平移只有seed0且没有C-mean；UAVDT中期原生AP70与辅助COCO-style终点不能混用。不要给出未经功效分析的“几个种子能可靠识别多少pp”，也不要把样本bootstrap当训练seed不确定性。提出能实际改变判断的最小分析，说明已有数据是否够用。''',
 'DSH｜替代解释与反例':'''你负责构造最有力的替代解释和可证伪反例，不为反对而反对。首先检验：是否仅因为 C-mean 处理损失了信息或自身退化，才放大 C−C-mean；纹理、通道混合和优化差异是否也能解释结果，而无需“位置编码”解释。检查 raw-P2 64通道均值池化+零初始化64→256投影（仅投影约16,384参数）的简单同预算候选：它尚未实测，不能宣称会赢，但是否比再加模块更能区分解释？还要讨论 rank16 更便宜且部分seed优于C，对“最小/最优实现”主张意味着什么。SOEP-lite只是自建简化参照，不能当作完整SOEP被击败。对你提出的每项追加实验说明：排除哪种解释、可能的三类结果、各自如何改变结论、是否值得算力；至多保留两项真正必要者。''',
 'Codex C｜贡献与审稿':'''你负责从目标读者和模拟期刊审稿角度判断论文主线，不宣称代表真实编辑或保证录用。核心问题：读者除“均值化可能丢失有用信息”外，能获得什么可复用的设计认识？判断哪些内容构成一项连贯实证贡献，哪些只是实验或审计清单，不把每个对照拆成算法创新。当前只讨论面向SIVP的主题和论证契合度，不臆造分区、录用率、版面费或期刊强制实验数量。正文v5已有AP50/AP75/ARs、尺度/定位/平移和完整12项/逐类附表，不得再把这些笼统列成完全缺失。针对UAVDT最终支持、混合、不支持三种情况给出诚实叙事；不能把探索性AP75结果后见替换原预设AP70主终点。最终提出最短且可守的研究问题、两句贡献表述和投稿前证据缺口；若增量过小应明确说明原因。'''
}

def delta_rows(arm,reference,split):
 return sorted([r for r in deltas if r['arm']==arm and r['reference']==reference and r['split']==split],key=lambda r:int(r['seed']))

parts=['''共同参考资料｜相位/子格差异信息传递论文｜2026-10-19

使用范围
这是给四位专家相同的一份“研究主线讨论证据包”，不是全文同行评审的完成证明。资料以本机2026-10-18论文v5、统一独立评估和2026-10-19恢复验收为快照。没有访问服务器更新数字，没有新增训练或推理。未读文件的内容不得标为已核验；若无法打开来源文件，只能依据包内资料并标注限制。

本场不执行外部工具写操作：不训练、不修改代码/稿件、不操作服务器、不管理自动监测。检索文献若可用，只能读；引证须给可核实的原文出处，不能捏造作者代码复现。

数字口径
成绩为百分制，差值为绝对百分点(pp)。主表为锁定best_stg2 EMA的统一独立评估；不能混减训练日志、末轮、全程最佳或不同协议。项目COCO、固定VisDrone工具包规则、忽略区COCO敏感性、UAVDT原生AP70和COCO-style辅助诊断各自单独命名。

训练重复与资料规模
完成的VisDrone为20个训练模型、每个模型val与test-dev两份评估，共40份划分级结果；每份12个COCO指标，共480个汇总值。它们不是480次独立实验。同权重的两个划分不是两个训练seed。C/C-mean和rank16/rank16-mean各有seed0/1/2；baseline、lowpass、singlephase、SOEP-lite只有seed0/1，无seed2 baseline。

[E3] 方法和研究问题
检测器为DEIM–D-FINE-S类小模型，HGNetv2-B0，输入640×640。P2是stride4的64×160×160特征；P3是stride8的256×80×80特征。原P3–P5检测层、300 queries保留。
C：U=ReLU(BN(Conv1×1(64→16)(P2)))；Z=pixel_unshuffle(U,2)，得到64×80×80；P3'=P3+Wr Z，Wr是零初始化、无偏置的64→256 1×1投影。位置在主干P3产生后，更深层也消费修改后的P3。新增17,440参数。
C-mean：相同层、名义参数量和初始化规则，把每个压缩通道四个offset均替换为其均值，然后送原Wr；从训练开始就如此，不是只在推理时破坏特征。
分解：Σk Wk zk=(Σk Wk) zbar + Σk Wk(zk−zbar)。这是标准线性分解，不是新理论。C可访问均值及差异项；C-mean去掉差异项。输入信息/子空间与梯度必然改变，不能宣称只隔离了“位置”这一单一因果变量。
rank16：在Z后加无偏置64→16 mixer，再接零初始化16→256 restore。rank16-mean在mixer前复制均值。两者各新增6,176参数，最大线性秩都≤16；mixer正交初始化。它们缩小“更高最大映射秩独自解释优势”的解释，不排除所有函数类、梯度或优化差异。
lowpass/singlephase：先64→16 BN/ReLU，再2×2平均池化/上左单采样，再线性16→60→256（中间无非线性），各新增17,376参数，最大映射秩≤16。C-mean与lowpass在压缩后的均值输入上可表达同类线性映射，但参数化/优化不同，不是等效性统计检验。

术语边界
phase指P2上2×2网格的四个offset，不是傅里叶相位；P2 stride4，因此也不能直接称为已经验证的输入亚像素坐标编码。pixel_unshuffle+1×1可重写成作用在U上的2×2 stride2卷积。固定通道排列能被后续权重吸收，不能把固定排列顺序当新颖性。推理时打乱相位的严重下降只能说明训练模型依赖该结构，不能单独证明机制。

[E3] 结构先例及复现边界
SPD-Conv：已提出space-to-depth+非stride卷积，原文 https://arxiv.org/abs/2208.03641 。
UAV-DETR SOEP：P2经SPD与stride1卷积，在neck与P3拼接，再接CSP-OmniKernel；仍P3–P5，无额外P2头。原文 https://doi.org/10.3390/s25154582 。
HMF-DEIM：P2经含四offset Cut的DRFD下采样，在FocusFeature融合；去P5、解码P3/P4。原文 https://doi.org/10.3390/s26072187 。
这些描述是项目按论文/图核对的记录，作者代码未进行运行核验。本材料不声称做过完整复现或穷尽相关文献；不能主张最早把P2四offset带入P3而不加P2头。
SOEP-lite为本项目自建简化参照：raw P2 unshuffle到256通道，depthwise3×3 stride1 BN/ReLU，压到16；与neck投影P3的256通道拼接，经非线性272→16，再零初始化16→256残差；不含OmniKernel。新增15,424参数，两个seed。不是完整SOEP，不能把它的落后当作打败UAV-DETR或HMF-DEIM。
本包没有提供只改变接入位置、其余完全匹配的独立证据，因此不能仅凭“接入位置不同”赋予它新颖性或因果优势。若引用历史位置消融，应另读取其原始结果；不要从C与SOEP-lite的多变量差异归因接入位置。

[E3] VisDrone实验与评估
VisDrone2019 DET：6471 train、548 val、1610 test-dev，10类别，整图分别沿两轴stretch到640；无切图/TTA/letterbox。96轮，batch4，AMP，AdamW，基本LR2e−4、主干与新分支1e−4、weight decay1e−4；flat-cosine，warmup1000步、flat46、末阶段12；共享公开初始化SHA前缀85497586。
best_stg2由末阶段验证总体AP选择，不按APs或test-dev选模；使用选定EMA统一FP32独立评估。部分历史训练中断后恢复，同名seed不等于逐步同步或比特相同的随机轨迹。
原项目COCO转换遗漏官方ignore矩形/others处理，因此与官方规则不同。正文已保留项目COCO主配对，并补固定官方工具包规则和ignore-aware COCO敏感性。工具包结果是CPU规则复评，不是挑战服务器成绩；工具包总体AP不是COCO小目标APs。
当前v5正文已加入AP50/AP75/ARs、中大目标代价、尺寸/定位/平移；附表有完整12项与逐类总体AP。不要把这些笼统说成尚未做。逐类总体AP不是逐类小目标AP。

[E1/E2] 关键配对（未舍入原值计算，以下显示6位小数）
split,comparison,seed,delta_AP,delta_APs,delta_ARs,delta_AP75_overall,delta_APm,delta_APl
''']
for split in ['val','testdev']:
 for arm,reference in [('C','baseline'),('C','C-mean'),('rank16','rank16-mean'),('C','SOEP-lite')]:
  for r in delta_rows(arm,reference,split):
   parts.append(','.join([split,arm+' - '+reference,r['seed']]+[f'{float(r[k]):+.6f}' for k in ['AP','APs','ARs','AP75','APm','APl']])+'\n')

parts.append('\n[E4] 固定小目标面积组的AP50/AP75与ARs，来自已生成论文表；不是总体AP50/AP75，以下显示到3位小数。\nsplit,comparison,seed,delta_APs,delta_AP50_small,delta_AP75_small,delta_ARs\n')
split=None
for line in sources['E4'].read_text(encoding='utf-8').splitlines():
 if '\\textit{Validation}' in line:split='val'
 elif '\\textit{Test-dev}' in line:split='testdev'
 elif ' & ' in line and '$-$' in line:
  values=[v.strip().replace('\\','') for v in line.split('&')]
  assert len(values)==6 and split
  parts.append(','.join([split,values[0].replace('$-$',' - '),*values[1:]])+'\n')

primary=delta_rows('C','C-mean','testdev');vals=[float(r['APs']) for r in primary]
parts.append(f'''
主配对效应量：C−C-mean test-dev APs均值{statistics.mean(vals):.6f}pp，样本SD{statistics.stdev(vals):.6f}pp，中位数{statistics.median(vals):.6f}pp（n=3）。方向一致不等于统计显著，两个约0.16pp差值必须与seed0较大差值同时讨论。不能给“两个seed可靠识别0.5pp”这种未经功效分析的通用规则，也不能从GT多推导训练重复多。

[E5] 成本审计（同口径640、FP32、未deploy融合）
模型,总参数,新增参数,GFLOPs,同会话前向时延中位数ms
Baseline,10231997,0,25.551,23.406
C,10249437,17440,25.814,23.292
rank16,10238173,6176,25.670,23.451
SOEP-lite,10247421,15424,25.747,23.786
C新增约0.170%参数、1.031% FLOPs；rank16约0.060%参数、0.466% FLOPs。时延在同一RTX3090会话、30 warmup/100 CUDA-event计时，TF32 off；当时无并发训练计算，但后台分配、主机与时钟未完全隔离，微小差值不支持速度排名。rank16成本更小且test-dev seed2 APs高于C（原值见完整表），C不是已证明的普遍最优。

[E6/E7] 评估规则敏感性
固定VisDrone工具包规则test-dev总体AP：C−baseline seed0/1=+0.803/+0.688pp；C−C-mean seed0/1/2=+0.649/+0.269/+0.246pp（正文显示到3位，存在舍入）。
ignore-aware COCO下 C−C-mean APs：val +0.730/+0.840/+0.569pp；test-dev +0.849/+0.194/+0.151pp。它是官方ignore过滤后算COCO指标的敏感性分析，不是工具包总体AP，也不是新训练重复。

[E3] 尺寸、定位与平移（已做，不能再次当作全部缺失）
640坐标sqrt(wh)分档[0,4)、[4,8)、[8,16)、[16,32)，与原图面积COCO APs分开。
val四档GT=693/6163/14007/12418；test-dev=6493/15773/23501/19881。
C−C-mean 8–16px组AP640：val +0.675/+0.863/+0.392pp；test-dev +0.690/+0.379/+0.426pp（三seed）。其他档并非都同向；rank16的8–16档val seed2也为负。
匹配诊断使用score≥0.3、同图同类、按IoU降序一对一贪心，IoU0.5/0.75；这是诊断匹配，不等同COCO AP分数排序匹配。分别统计共同、新增、丢失、均未检出。
test-dev IoU0.5，C−C-mean新增2658/2482/2444、丢失1856/1889/1913；共同检出目标的平均归一化中心误差由0.09030/0.08975/0.08999降至0.08770/0.08786/0.08801。它是条件误差，不是所有GT无条件定位改善；seed1的平均宽度误差在两划分两IoU均有反例。小目标AP75 seed1 test-dev也下降，不能宣称框几何全面更准。
平移只测seed0的baseline/C/lowpass，没有C-mean；输入偏移1/2/4/8/32px，反向对齐框与统一有效边界后统计。C的绝对ROI AP更高，但翻转率和相对零位移的AP损失无一致优势。4px翻转率baseline/C/lowpass=9.228/9.236/9.081%；8px=4.450/4.091/4.072%；32px=2.272/2.164/2.060%。绝对位置编码等因素也使32px不是严格等变证明。

[E9/E10] UAVDT：必须区分中期诊断和最终结论
六个96轮既有任务：baseline/C/C-mean×seed0/1。按序列24 train(18990帧)、6 val(5153帧)、20官方test(16592帧)；640整图，batch4，共同数据集专用初始化。最终预设主终点是原生类别无关车辆AP70，以val AP70选择阶段2 best_stg2，再独立评估val/test。不是VisDrone权重直接零样本跨域迁移。
本包最新人工验收仅确认续训：LA2026-10-19 18:39，A baseline/C第47轮、C-mean第46轮，B baseline/C第42轮、C-mean第41轮。它们不是完成的96轮结果。本场不更新这些状态；自动监测由另一个对话负责。
已有六份固定第36轮EMA统一CPU FP32独立评估，仅val 5153帧、87719GT，非best_stg2/官方test。C−同seed baseline：
指标,seed0,seed1
原生总体AP70,-0.334,-0.438
总体COCO-style AP(0.50:0.95),+0.556,+0.146
总体COCO-style AP50,-0.364,-0.266
总体COCO-style AP75,+0.821,-0.194
原图COCO-style APs,-0.421,+0.809
640输入<32px辅助AP,+0.047,+0.535
640输入<32px辅助AP50,-1.057,-0.234
640输入<32px辅助AP75,+1.361,+0.961
640输入<32px原生先匹配分档辅助AP70,-0.709,-0.111
原图COCO-style APs的C−C-mean=+1.109/+2.431pp；640<32px辅助AP75的C−C-mean=+2.757/+5.049pp。这可能同时包含C-mean退化，不能替代C−baseline的净收益判断。
这里没有输入<8px的GT，不能验证这两档。辅助COCO-style/maxDets100、原生先匹配分档指标是诊断，不是官方主分数；cap300对<32px AP75正向敏感性保持。两个IoU口径的方向不能被挑选来替换预设AP70主终点。
CPU独立原生AP70较同轮GPU日志高0.061–0.279pp；源代码/配置/GT已核对，具体原因尚未锁定。本包配对均在同一次CPU协议内，不混减日志数字。
稿件v5曾保留明确标记的“UAVDT预计正向情景”，只是内部写作占位，不是实测证据。讨论不能引用它作支持；投稿前必须以完整最终结果替换或删除。若最终AP70不支持，不能后见地把探索性AP75结果改包装成原本预设的验证成功。

尚未完成的竞争性证据及待讨论问题
raw-P2的64通道2×2平均池化+零初始化64→256投影，投影约16,384参数，是一个尚未实测的简单近预算候选。它不经过先64→16 BN/ReLU后均值的相同路径，可能回答更直接的实用预算问题，但不是纯单变量对照。不能假设它优或劣。
完整SOEP/HMF-DEIM、额外P2检测头、第二检测器尚未复现。不是要求它们全部必做；专家需说明哪项能改变哪条判断，区分科学解释与工程性能主张。现阶段不执行新增训练。
区分“本分支中的子格差异有用”“低预算设计原则”“新检测算法/新位置机制”“同预算最优”。不要用实验清单数量当创新点数量，也不要只靠措辞变窄制造新颖性。

[E1] 全部40份统一独立评估的完整12指标
以下显示为百分制、到3位小数，原精度另存同目录CSV。不是官方工具包分数。best_stg2所对应轮次为1起。
seed,arm,split,best_round,{','.join(metrics)}
''')
for r in rows:
 parts.append(','.join([r['seed'],r['arm'],r['split'],r['selected_round_one_based']]+[f'{float(r[m+"_percent"]):.3f}' for m in metrics])+'\n')

parts.append('\n证据来源索引（相对“英文小论文”目录；路径不等于已经读过文件）\n')
for eid,path in sources.items():parts.append(eid+' '+str(path.relative_to(EN)).replace('\\','/')+'\n')
parts.append('''
本次讨论必须交付的判断
1. 一条最强可守的研究问题与答案，至少一个使该答案变弱的反例。
2. 主张分级：可写/需降级/无证据，分别注明源编号、划分、seed、协议、效应量。
3. 至多两项必要补证，每项注明识别目标、可能结果及改变结论的条件；把可由已有数据完成的分析与新训练分开。
4. UAVDT最终正向、混合、负向三种结果下如何保留、改写或放弃主张。
5. 最终明确记录未解决分歧；不以多数同意等同验证，不声称真实期刊录用概率。
''')
materials=''.join(parts)
assert len(materials)<60000

settings='''页面填写和讨论顺序

01_议题：完整粘贴到“想一起推敲什么”。
02_共同参考资料：用“导入文本”导入，四位专家共享同一包，不要每位选不同事实。
03_四位专家视角：按席位分别粘贴到“独立研究视角”。席位可按截图顺序为Codex A、Codex B、DSH、Codex C；专家名称可附职责。

设置建议：3个Codex+1个DSH；保持你想用的已验证模型及推理档位。选“独立互评”，固定3轮。每人每轮一次回应，合计12次主要回应，连接测试另算。先不要选“不限·手动停止”，以免观点复述和上下文不断累积。
第1轮各自独立初判；第2轮回应具体异议；第3轮各自收敛成贡献句和证据缺口。没有必须达成共识的任务；人类决定下一步。

从页面实际显示观察：DSH当前模型标签也可能是GPT-6.1-Sol（codex/gpt-6.1-sol）。DSH是Agent/调用通道名称，不自动等于DeepSeek或不同基础模型。若目的是跨模型复核，应另选确实可用且已验证的不同模型；不能把改名字当模型多样性。即使不同模型，也不是四份独立实验证据。
“程序已找到”不等于已登录或模型可用。若要测试真实连接，应由你在决定开始时执行；本包没有调用任何专家，没有上传材料、发送提示或消耗专家调用额度。

本场只做研究主线讨论，不让专家改代码、改稿、跑训练或管理自动监测。若后续需要全文复审，应另给真实全文及附表；不能把这次证据包讨论冒充已完成全文审稿。

判断讨论是否有用：
— 四位是否明确提出了不同的、可核查的质疑？
— 是否解释了相同数据为何支持有限主张、又不足以支持更强主张？
— 是否把已有v5分析误说成未做，或把UAVDT预计段当作事实？
— 是否给出比“再加模块/再做一堆实验”更具体的最小补证？
— 是否保留了负结果、例外和未解决分歧？

推荐的首场目标是判定贡献边界和最小补证，不是让四位轮流保证能发，也不是强迫它们轮流否定。
'''

outputs={
 '01_议题_粘贴到议题框.txt':topic,
 '02_共同参考资料_导入TXT.txt':materials,
 '03_四位专家视角_分别粘贴.txt':'\n\n'.join(name+'\n'+text for name,text in perspectives.items()),
 '04_设置与三轮流程.txt':settings,
}
for name,text in outputs.items():(HERE/name).write_text(text,encoding='utf-8')
precision_columns=['seed','arm','split','selected_round_one_based']+[m+'_percent' for m in metrics]
with (HERE/'核查用_40份完整12指标_原精度.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=precision_columns);writer.writeheader();writer.writerows({k:r[k] for k in precision_columns} for r in rows)
with (HERE/'核查用_全部同种子配对差值_原精度.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(deltas[0]));writer.writeheader();writer.writerows(deltas)
manifest={'scope':'prepare only; no experts invoked, no app settings changed, no remote operations',
 'materials_characters':len(materials),'UI_materials_limit_characters':60000,
 'VisDrone_models':20,'split_evaluations':40,'full_metric_values':480,
 'paired_C_Cmean_testdev_APs_mean_pp':statistics.mean(vals),'paired_C_Cmean_testdev_APs_sample_SD_pp':statistics.stdev(vals),
 'source_files':[{'id':eid,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for eid,path in sources.items()],
 'outputs':[{'name':name,'characters':len(text),'sha256':hashlib.sha256((HERE/name).read_bytes()).hexdigest()} for name,text in outputs.items()]}
(HERE/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'materials_characters':len(materials),'files':list(outputs),'models':20,'evaluations':40,'full12_cells':480,
 'all_exported_paired_deltas_verified_against_raw_0_to_1_values':True,'experts_invoked':False},ensure_ascii=False,indent=2))
