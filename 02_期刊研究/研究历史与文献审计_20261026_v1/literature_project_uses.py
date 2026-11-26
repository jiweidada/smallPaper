"""Explicitly documented project uses, separate from thematic similarity."""
import direction_decisions as D

CASES=[]
def use(pdf_ids, titles, direction_ids, text, source, anchor, reading_scope=''):
    CASES.append(dict(pdf_ids=pdf_ids,titles=titles,direction_ids=direction_ids,
                      text=text,source=source,anchor=anchor,reading_scope=reading_scope))

chapter=D.MAIN+'360_线程A_补核_三篇章节号_20261026_v1.md'
use([146],[],['D007'],
    'SPD及非步长卷积已有直接原文；用于撤回pixel-unshuffle/相位重排本身为新算子的主张。不能据此抹掉C-C-mean的受控增量证据。',
    chapter,'## 1. SPD-Conv','历史局部方法核读：§3.1–3.2。')
use([113],[],['D007'],
    'SOEP的SPDConv与浅层增强是直接近邻；限定首次使用/首次浅层接入等主张，不判整个C桥逐式相同。另一篇同名UAV-DETR不能替代此篇。',
    chapter,'## 2. UAV-DETR','历史局部方法核读：§3.3、§3.3.1–3.3.2。')
use([34],[],['D007'],
    'HMF-DEIM作为同DEIM家族浅层增强/融合先例；具体接入路径和单变量增量仍须区分。',
    chapter,'## 3.','历史局部方法核读：§3.3.3、式12。')
use([6],[],['D031','D032','D037'],
    'Top-k分布值估计定位质量已有先例；不能把用峰/分布质量本身命名为新方法。并不否定所有分布读取。',
    D.MAIN+'174_单边次峰必要上界_完整阴性回交.txt','GFLv2')
use([128,116,153,244,119],[],['D009','D056','D058','D068'],
    '曾被用于给尺寸、覆盖和各向异性问题设先例闸门；这些已知方法没有在本地同协议比较。独立审查明确：不能从有先例推成问题已解决，也不能把尺寸后处理失败当成其训练方法失败。',
    D.MAIN+'360_v7_问题定义交付_独立核对_20261026.txt','7. 已有方法未在本协议测试')
use([30,32,33],[],['D014','D015'],
    '借鉴学习特征上的LK/GN/LM几何迭代；用于限制“给query加GN/LM就新”的主张。跟踪、单应性、相机位姿与检测框任务不同，不能作本项目效果证据。',
    D.LIB+'持续讨论新增原文阅读索引_20261005.txt','119 Deep-LK')
background=D.DATA+'新文献驱动结构研究_20261005_v1/严格自审_候选降级与否决理由_20261005.txt'
use([157,160,163],[],['D015'],
    '协方差白化、Mahalanobis注意力、辅助样本估计干扰的工具已有先例；实际借鉴是统计读取动机。其优化检测/信噪比的依据不能直接变成IoU最优或外围P2为纯背景的证明。',
    background,'七、新颖性审查')
use([164,165,156],[],['D014','D015','D023'],
    '作为特征驱动框精炼、浅层信息和查询选择的近邻；仅支持构件/问题先例，不等于本地候选与这些方法完全同构或已完成公平比较。',
    background,'3. RefineBox')
use([236,237],[],['D042'],
    'DCR原文与两版作者推理代码已核对：独立分类支路、固定回归框、早期特征、额外CE与梯度隔离已有。v2概率相乘不等于本项目保持分数预算的干预。未复现作者训练效果。',
    D.DATA+'分类读出与分数预算核验_20261007_v2/04_本轮机制结果与目标裁决.txt','DCR原论文与作者两版本源码')
use([233],[],['D043'],
    '实际借用sigmoid近似rank，构造固定encoder预算的惩罚；不是直接照用Smooth-AP检索AP目标。原记录声明局部方法及作者117行代码核对；新预算组合尚无独立贡献验收。',
    D.LIB+'_text/全池排名近邻_Smooth_AP_ECCV2020_20261007_实际方法阅读与借鉴边界.txt','实际阅读范围：',
    '历史局部方法核读：§4/4.1式6–7及§5.3；记录另有作者源码7–16、79–116行对照，未读完28页。')
use([228],[],['D062'],
    '已存TOLF原文/审查用于确认小目标标注噪声监督和分布建模已有直接近邻。量化逆胞是否有独立增量需另证；不能只因TOLF存在就否掉全部确定量化模型。',
    '英文小论文/code/annotation_precision_novelty_review_20261026_v1/367c_集合监督候选_独立创新性审查.txt','TOLF')
haar='英文小论文/code/haar_detail_review_20261025_v1/355_primary_literature_check.json'
use([],['CFBA-FPN / CFCI'],['D007','D008'],
    '已存出版社§3.1–3.2核对记录：P2引导高频残差注入；作为频率补偿直接近邻。尚未完成逐式与作者代码等价审查，不能声称完全相同。',
    haar,'CFBA-FPN / CFCI','历史局部方法核读记录：出版社§3.1–3.2；未完成逐式/代码等价审查。')
use([],['ESE-Net'],['D008'],
    '已存出版社PDF p3 §2.1记录：红外分割中丢低频及对角、保留水平垂直Haar。删除LL不是新想法；任务与融合路径不同，非本检测桥的效果或同构证明。',
    haar,'ESE-Net','历史局部方法核读记录：PDF第3页§2.1；红外分割任务。')
use([],['SARES-DEIM v2'],['D007','D008'],
    '已存v2 III-C核对：SDEP的P2 space-to-depth注入，与MoE专家内Haar WTConv分别描述。不能把线性space-to-depth全频换基称新算子。',
    haar,'SARES-DEIM v2','历史局部方法核读记录：v2 §III-C；未完成整套结构等价核查。')
temporal='英文小论文/code/temporal_acquisition_precheck_20261025_v1/358_literature_triage.json'
use([],['Minimum Delay Object Detection From Video'],['D064'],
    '首次检测延迟与弱证据累积已有直接先例；本地记录定位Algorithm2。不能将首次达标或通用证据记忆单独包装为新设定。',
    temporal,'Minimum Delay Object Detection From Video','历史局部方法核读记录：作者PDF Algorithm2相关文本；非全文阅读声明。')
use([],['A Delay Metric for Video Object Detection: What Average Precision Fails to Tell'],['D064'],
    '支持AP不能完整反映首检延迟、AD及误警约束已有；不是新时序方法。',
    temporal,'A Delay Metric for Video Object Detection')
use([],['Seq-NMS for Video Object Detection','LSTFE-Net: Long Short-Term Feature Enhancement Network for Video Small Object Detection',
        'XS-VID: A Large-Scale Benchmark for Small Object Detection and Tracking in Videos'],['D063','D064'],
    '通用时间分数传播、长短时特征增强及视频小目标基准已有。阅读层级不一：Seq-NMS为摘要，LSTFE页面403仅索引，不能声称已核全部方法。',
    temporal,'Seq-NMS for Video Object Detection')
use([],['Convolutional Neural Network Reference for Track-Before-Detect Applications'],['D064'],
    '已有track-before-detect沿轨迹累积弱信号；仅用于限制泛称证据记忆的首创主张，不是本地视频检测效果。',
    temporal,'Convolutional Neural Network Reference')
novel='英文小论文/code/annotation_precision_novelty_review_20261026_v1/367c_集合监督候选_独立创新性审查.txt'
use([],['Learning from Imprecise and Fuzzy Observations'],['D062'],
    '不精确标签集合上的最小损失已有理论；用于限制“区间内不罚”目标的首创主张，不证明具体检测训练目标已被穷尽。',
    novel,'1305.0698')
use([],['Can We Trust Bounding Box Annotations for Object Detection?','Noise-Aware Evaluation of Object Detectors'],['D062'],
    '提供框标注/噪声评估先例；噪声评估与训练损失是不同任务，不能以评估论文名称直接否定新训练设定。',
    novel,'WACV')

def match(pdfs,title):
    nums={r['n'] for r in pdfs}
    return [c for c in CASES if nums.intersection(c['pdf_ids']) or title in c['titles']]
