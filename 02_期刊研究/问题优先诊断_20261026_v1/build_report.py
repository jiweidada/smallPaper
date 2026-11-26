"""Assemble requested problem map/cards/plans from verified local measurements."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,csv
from pathlib import Path
from collections import Counter
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=read(OUT/'results.json');S=read(OUT/'supplementary_checks.json');V=read(OUT/'verification.json')
groups=R['groups'];N=R['scope']['validation_GT']
def g(seed,kind='all',label='all'):return next(x for x in groups if x['seed']==seed and x['group_type']==kind and x['group']==label)
def q(n,d):return f'{n:,}/{d:,}（{100*n/d:.2f}%）'
def table(headers,rows):
    return '\n'.join([' | '.join(headers),' | '.join(['---']*len(headers))]+[' | '.join(map(str,r)) for r in rows])
def put(n,s):
    with open(OUT/n,'x',encoding='utf-8') as f:f.write(s)
labels={'G':'全部原始框均无合格几何覆盖','S':'有几何覆盖，真类分数均≤0.001','K':'有合格真类分数，但真类候选未入top300','C':'真类候选入top300，后被每类100上限清空','M':'保留合格同类候选，实际一对一评价未命中','T':'实际评价命中'}
map50=table(['IoU≥0.50的顺序状态','seed1：计数/全部GT','seed2：计数/全部GT'],[[labels[k],q(g(1)[k+'50'],N),q(g(2)[k+'50'],N)] for k in 'GSKCMT'])
map75=table(['IoU≥0.75的顺序状态','seed1：计数/全部GT','seed2：计数/全部GT'],[[labels[k],q(g(1)[k+'75'],N),q(g(2)[k+'75'],N)] for k in 'GSKCMT'])
size=table(['640平面短边','共同GT分母','seed1无IoU.5几何框','seed2无IoU.5几何框','seed1实际TP.5','seed2实际TP.5'],
 [[a,g(1,'short_side640',a)['n_gt'],g(1,'short_side640',a)['G50'],g(2,'short_side640',a)['G50'],g(1,'short_side640',a)['T50'],g(2,'short_side640',a)['T50']] for a in ['[0,4)','[4,8)','[8,16)','[16,32)','[32,inf)']])
visible=table(['原始遮挡标记','共同GT分母','seed1实际TP.5','seed2实际TP.5'],[[a,g(1,'occlusion',a)['n_gt'],g(1,'occlusion',a)['T50'],g(2,'occlusion',a)['T50']] for a in ['0','1','2']])
density=table(['32px内其它GT中心数','共同GT分母','seed1实际TP.5','seed2实际TP.5'],[[a,g(1,'density32',a)['n_gt'],g(1,'density32',a)['T50'],g(2,'density32',a)['T50']] for a in ['0','1-4','5-14','15+']])
cnames=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
classes=table(['类别','共同GT分母','seed1 G.5','seed2 G.5','seed1 K.5','seed2 K.5','seed1 TP.5','seed2 TP.5'],
 [[c,g(1,'class',str(i))['n_gt'],g(1,'class',str(i))['G50'],g(2,'class',str(i))['G50'],g(1,'class',str(i))['K50'],g(2,'class',str(i))['K50'],g(1,'class',str(i))['T50'],g(2,'class',str(i))['T50']] for i,c in enumerate(cnames)])
metric=table(['对象（同一数据/评价口径）','seed1 AP / APs，%','seed2 AP / APs，%'],[
 ['BASE24 model：本次原口径复现',f"{R['metrics']['1']['stored_id_native300']['AP']*100:.6f} / {R['metrics']['1']['stored_id_native300']['APs']*100:.6f}",f"{R['metrics']['2']['stored_id_native300']['AP']*100:.6f} / {R['metrics']['2']['stored_id_native300']['APs']*100:.6f}"],
 ['同checkpoint EMA24：历史现场捕获',f"{S['EMA24_reference']['1']['metrics_percent']['AP']:.6f} / {S['EMA24_reference']['1']['metrics_percent']['APs']:.6f}",f"{S['EMA24_reference']['2']['metrics_percent']['AP']:.6f} / {S['EMA24_reference']['2']['metrics_percent']['APs']:.6f}"]])
stage=table(['阶段/单位','seed1','seed2'],[[k,R['stage_counts']['1'][k],R['stage_counts']['2'][k]] for k in ['raw_queries','raw_invalid_queries','all_query_class_pairs','native_post_score_pairs','native_unique_queries','native_extra_pair_slots_same_query','cap100_pairs','cap100_dropped_pairs']])
precheck=table(['固定框/分数的K300→K600检验','seed1','seed2'],[
 ['实际TP.5',f"{R['precheck']['1']['TP50_native']}→{R['precheck']['1']['TP50_K600']}",f"{R['precheck']['2']['TP50_native']}→{R['precheck']['2']['TP50_K600']}"],
 ['新增/丢失GT（IoU.5）',f"{R['precheck']['1']['new_hit50']}/{R['precheck']['1']['lost_hit50']}",f"{R['precheck']['2']['new_hit50']}/{R['precheck']['2']['lost_hit50']}"],
 ['实际TP.75',f"{R['precheck']['1']['TP75_native']}→{R['precheck']['1']['TP75_K600']}",f"{R['precheck']['2']['TP75_native']}→{R['precheck']['2']['TP75_K600']}"],
 ['ΔAP，百分点',f"{R['precheck']['1']['delta_pp']['AP']:+.6f}",f"{R['precheck']['2']['delta_pp']['AP']:+.6f}"],
 ['ΔAPs，百分点',f"{R['precheck']['1']['delta_pp']['APs']:+.6f}",f"{R['precheck']['2']['delta_pp']['APs']:+.6f}"],
 ['评价保留检测行数',f"{R['precheck']['1']['evaluated_rows_native']}→{R['precheck']['1']['evaluated_rows_K600']}",f"{R['precheck']['2']['evaluated_rows_native']}→{R['precheck']['2']['evaluated_rows_K600']}"],
 ['IoU.5未匹配检测行数',f"{S['derived_counts']['1']['native_FP50']}→{S['derived_counts']['1']['K600_FP50']}",f"{S['derived_counts']['2']['native_FP50']}→{S['derived_counts']['2']['K600_FP50']}"]])
onepage=f'''问题地图｜2026-10-26

当前结论：保留2张待继续区分原因的问题卡。最终候选的几何覆盖缺口规模较大；后续筛选也有可重复的固定输出损失。当前尚未定位到足以支持新网络或损失设计的机制，不声明创新成立。

研究目标：为轻量航拍10类目标检测寻找可验证的性能/资源缺口，关注APs、总体AP和严格IoU质量。问题是否曾被研究不决定新方法能否创新。
诊断对象：现有BASE24、seed1/2、last_epoch=23的model原始权重，HGNetv2-B0 + 三层D-FINE decoder，300 queries；132是训练配方总轮数，不是本缓存已训练轮数。不能用本图代表完整96/132轮EMA。
数据：train 6471图/343205 GT仅核数量；val全部548图/38759有效GT用于失败地图。原始标注40169行，其中1378类0区域、32类11条目不在当前COCO GT中；原始有效38759与COCO逐类几何一一对应，最大坐标差0.001280px。未读test/test-dev。
评价：项目绑定custom pycocoeval，bbox IoU .50:.05:.95、101点召回、10类宏平均、maxDets=[1,10,100]。COCO small按原图面积<1024；tiny短边<4在拉伸后640平面。两者不能互换。非VisDrone官方挑战协议。
可观测：原始末端300框/300×10 logits→3000类别候选→top300→score>0.001→每图每类cap100→一对一评价。编码器选前、decoder中间态、特征/注意力与完整训练监督未缓存；这些阶段在地图中标为无法识别。

{map50}

表是顺序操作分解，不是独立病因分解。每个seed的六格之和均38759。为避免GT id=0和“未匹配=0”哨兵混淆，GT级主图在内存使用id+1副本；原ID指标另行原样复现，源标注未改。

{size}

小尺寸、遮挡和密度与失败关联，但没有因此确定宽度、位置、特征或分配为主因。两seed共同在IoU.5漏检12156/38759个GT，其中共同无原始几何覆盖9442/38759。

已做零训练筛选检验：框、logits、GT、score阈值和cap100固定，只将K300改为K600。新增TP.5为1443/1483，ΔAP为+0.187428/+0.219084pp，ΔAPs为+0.171318/+0.197975pp；评价检测行数增加到286557/294563，未匹配行数也明显增加。只支持当前固定输出存在筛选效应，未识别额外类别假设与新query各占多少。

最值得先做：冻结A原top300、B只在A已有query身份内取top600、C全query取top600，对照B-A与C-B。B尚未执行，不能预报结果。详细协议见followup_plan.json。
证据边界：同checkpoint EMA24已保存AP为16.603378/16.408690%，高于raw的13.924192/14.823708%；raw/EMA不能混写。没有完整终点模型的同口径末端/内部失败地图，尚不能把本轮现象升格为最终投稿基线的已确定瓶颈。
执行：本地CPU；模型构建/新模型前向/训练/GPU/远端/test读取均0。2766个输入文件收口SHA无变化，另核对1096次源图SHA。原预测314618行逐项复现差0，历史12指标两seed差0。
'''
put('01_一页问题地图.txt',onepage)
cards='''问题卡｜仅2张，均不含解决方案

排序理由：卡1覆盖真实GT更多，直接影响检测召回与严格IoU质量，但目前只有末端观察；卡2影响规模较小，却已得到固定输入的受控筛选效应，下一步区分成本更低。研究价值顺序为卡1→卡2；即时零训练检验优先卡2。

问题卡1：最终300个query对大量真实目标没有形成足够的几何覆盖

问题是什么：在BASE24 raw的全部验证GT中，即使忽略类别，仍有12486/38759（seed1）与11826/38759（seed2）没有任何IoU≥0.5的框；IoU≥0.75时分别为24848/38759与24018/38759。这里测的是末端输出缺口，尚未确定它在网络哪一步出现。
影响多大：短边<4组G.5为3715/4952、3727/4952；4—8组为4163/10294、4255/10294；>=32组为491/3430、245/3430。所有尺寸分层都已报告，没有仅选择Geo0。两个seed共同G.5为9442/38759。其数量不是可部署TP或AP收益上限。
证据到哪一步：已观察全部GT与300 raw框；核过类别候选/阈值/评价上限及实际一对一匹配。尚无对应的编码器选前或decoder逐层全val证据，无训练后机制验证。原始框“没有覆盖”是问题定义本身，不能用同一事实证明某个回归病因。

竞争解释及所需证据：
1. 输入缩放、可见信息或特征没有形成可区分信号。需要受控的信息/输入对照及特征证据；仅看最终框中心不能证明采样位置或信息缺失。
2. 编码器query选择/初始参考没有覆盖该目标。需要真实入选身份、参考框、选择前score/anchor。当前源码先选topK再只对入选memory回归bbox；未入选位置没有自然生成的回归框，不能虚构“被丢掉的好编码器框”。
3. 初始或中间几何较好，后续decoder细化变差。需要沿同一query身份逐层追踪并报告全部GT的获得/失去覆盖，而不是每层更换配对后串成单一轨迹。
4. 当前raw24状态和较成熟/EMA状态差异。已有EMA终端TP和AP参考支持先检查版本可迁移性，但没有EMA raw300池；不能凭AP更高推断具体G组已被解决。
5. 遮挡、密度、类别与尺寸共同作用。原始遮挡0/1/2的TP.5及密度组均已列，粗分层标准化后密度差仍存在，但没有随机干预，不能指定主因。

历史关系：保留旧宽度门槛、保持宽高比、ROI/四槽补全、DC替换等各自的阴性；不将它们当作全部几何方法无效。旧x区间/中心/宽高替换oracle不进入本卡效能证据。
研究问句：同一目标在真实入选参考、pre与decoder各层是否曾被覆盖；该轨迹在EMA分支是否还出现？
最小零训练检验（计划，未执行）：固定hash选择32张val图，两seed相同；只在下一阶段对EMA做64次CPU前向，导出真实topK query身份、入选参考、pre及3层框/分类分数。现有raw末端与EMA历史GT_hits为参照。按全部GT计数“从有到无/从无到有/始终有/始终无”，同时报告固定入选参考最佳query的真实轨迹与各层全池覆盖，二者不混用。只定位可观测阶段；参考框不是已检测框，未入选anchor也不是漏检候选。
数值/成本：不改模型数学或权重，hash绑定输入；记录hook前后末端输出一致性，CPU backend差异显式保留；IoU临界±1e-5另列，不用容差改判。历史32次CPU前向含加载为9.99/11.54秒，64次只可据此作约22秒的预算线索，hook与导出实际成本未测；规划5分钟，硬上限900秒。导出上限20MB，不导出特征全图，不训练、不用GPU或test。
支持与反对：若初态/中间态有而末端无，支持该固定阶段存在几何退化；若始终无，仅说明缺口在所测更早阶段已存在，不能据此确认特征根因。EMA若不复现某现象，只停止将raw24现象外推到EMA的假说。局部32图不能代替全val或完整96/132轮确认。

问题卡2：已有合格几何框的真类候选，在输出选择中损失了可测检出

问题是什么：在已有IoU≥0.5几何框且真类sigmoid>0.001的GT中，2359/26273（seed1）、2392/26933（seed2）没有真类候选进入原top300；随后cap100又使291/23914、242/24541个GT失去全部合格真类候选。分类阈值、全图排序和每类上限是不同阶段。
影响多大：绝对分母均可回到全部38759 GT；K损失约6.09%/6.17%全部GT。K600的实际新增TP.5为1443/38759、1483/38759，没有原TP.5丢失；ΔAP仅+0.187428/+0.219084pp，ΔAPs仅+0.171318/+0.197975pp。评价未匹配行数从132725→261638、134235→268898；没有时延/FLOPs测量，不能只报新增TP。
合适参照：同框、同logits、同GT、同阈值/评价cap的原top300；同时报告类别、尺寸组以及已有几何覆盖的条件分母。tiny只占其中145/1237、145/1225个K损失GT；这一问题不限tiny，也不替代卡1。
证据到哪一步：本轮protocol先冻结再算，两seed的K600 AP/APs与TP.5点估计均过原数值门槛；这是当前输出读出干预的实测效应。图像配对bootstrap只计算TP率差，未计算AP置信区间，不称总体显著收益。尚无上游机制、等预算收益、训练或最终模型可迁移性证据。

竞争解释及所需证据：
1. 原top300已包含合格框身份，但同一query的真类项排名较后。需要限制原query身份集合、仅增加其类别候选的控制B。
2. 合格框的整个query在原top300中没有身份。需要比较相同600类别候选预算下，是否允许这些新query身份进入的控制C-B。
3. 分类判别/校准或类别频率造成低相对排名。需要按真实类别、真实IoU与负例共同检查score排序；score>0.001并不代表分类已正确，整体重标分可能同时抬高FP。旧按尺寸重标分阴性保持。
4. 评价每类100上限与一对一冲突。已单列C、M。cap后最大基数匹配TP.5为23523/24222，实际为23476/24182，固定图结构差47/40；既非147/117全部可救，更非训练matching的收益上限。

历史关系：20261007的硬单类First/Collapse/Refill在epoch53 TRAIN16/32双视图+NMS上损失6个原正确覆盖GT，AP/APs均为负。该阴性保留；它不能直接代表本轮BASE24单图全val，但本轮也不恢复硬单类主线。新的B保留原query身份、增添类别假设，属于区分现象的控制，不是新方法。
研究问句：固定预测池中，新增检出依赖额外类别假设、原先未入选的query身份，还是两者同时作用？
最小零训练检验：followup_plan.json已冻结、B尚未执行。A=原top300；U=A出现过的query身份；B=仅从U的未改分数中取top600；C=全query top600。B-A检查增加原框类别假设；C-B在600行同预算下检查允许原先缺失query身份的读出效果。score>0.001、maxDets100、GT与框不变，GT不参与选择。若合法pair不足600或出现非法框，同预算对照记N/A，不补齐、不滤框、不换K。
支持与反对：B-A或C-B须各自在两seed同时有AP/APs正差>1e-6原值、TP.5净增，才支持对应的固定读出效应；允许两者同时成立。混合/阴性按原样保留，不由单个阳性指定主因。预检即使通过，仍需目标终点模型确认，随后才查现有方法的适用条件与剩余缺口。
停止范围：只停止未获支持的那条解释/控制；继续探索卡1与其它未排除解释，不降低门槛或换群体救回本轮。
'''
put('02_两张问题卡.txt',cards)
fp=table(['评价保留行在IoU.5的描述类型','seed1 / 156201行','seed2 / 158417行'],[[k,R['FP_map']['1']['0']['counts'].get(k,0),R['FP_map']['2']['0']['counts'].get(k,0)] for k in ['TP','same_class_overlap50_unmatched','other_class_overlap50','same_class_overlap10to50','other_overlap10to50','all_overlap_below10']])
std=[]
for s in ('1','2'):
    x=S['derived_counts'][s]['density_standardization'];std.append(f"seed{s}：共同91层、低密6296个GT与高密12725个GT；合并GT权重标准化TP.5率为{x['standardized_TP50_rate'][0]*100:.3f}% / {x['standardized_TP50_rate'][1]*100:.3f}%。")
ema_size=table(['640短边','共同GT分母','EMA24 seed1 TP.5 / TP.75','EMA24 seed2 TP.5 / TP.75'],
 [[a,g(1,'short_side640',a)['n_gt'],str(next(z for z in S['EMA24_reference']['1']['groups'] if z['group_type']=='short_side640' and z['group']==a)['TP50_EMA'])+' / '+str(next(z for z in S['EMA24_reference']['1']['groups'] if z['group_type']=='short_side640' and z['group']==a)['TP75_EMA']),str(next(z for z in S['EMA24_reference']['2']['groups'] if z['group_type']=='short_side640' and z['group']==a)['TP50_EMA'])+' / '+str(next(z for z in S['EMA24_reference']['2']['groups'] if z['group_type']=='short_side640' and z['group']==a)['TP75_EMA'])] for a in ['[0,4)','[4,8)','[8,16)','[16,32)','[32,inf)']])
report=f'''Material Passport
Origin Skill: academic-research-suite / experiment-agent
Origin Mode: validate（单助手inline，非独立盲审）
Origin Date: 2026-10-26（客户端日期）
Verification Status: ANALYZED；缓存预测/历史评价数值复现通过，不是机制或新方法验证
Version Label: problem_first_v1

{onepage}

完整事实与证据边界

1. 模型/数据/协议
{metric}
两checkpoint的model与EMA各有690/692个浮点state张量不同。当前缓存由model分支产生，不能因为同一个last.pth就标为EMA。
EMA参考只重建既有matching与GT_hits一致性，没有新前向，也没有从该文件取得全部未入选query。EMA的TP.5=25077/24924、TP.75=14698/14484（分母均38759），沿用历史ID0哨兵口径；不是其G/S/K/C/M机制图。
{ema_size}
这个同轮参照显示版本差异实质存在，也显示小尺寸严格命中依然少；但无法由末端命中数推出原因相同。完整96/132轮不能用上述两种24轮模型替代。

原始TXT的40169记录包含38759条score1有效10类GT、1378条类0区域与32条类11；COCO仅保留有效GT，ignore字段缺失、iscrowd全0。当前检测FP是此转换数据及评价器的操作结果，不能未经官方忽略规则复核就等同挑战官方FP。
val.json的坐标保留YOLO归一化转换的小数，不能写成所有COCO宽高都严格为整数。与原始TXT的类/框双射验证覆盖全部38759 GT，最大差0.001280px；本轮没有把该微小转换差夸成噪声主因。

2. 末端预测到评价的完整可观测链路
{stage}
每seed有164400个query行、1644000个(query,class)候选对；164400个top300项不是164400个独立query身份。每图最终top300身份数范围为96—247 / 83—269。72562/71492个额外类别槽不是已证明无用的重复检测。
score>0.001实际丢弃0/164400个top300项；全部有IoU.5几何框的GT也至少有一个真类候选超过此阈值（S=0/26273、0/26933）。这只排除了该特定阈值在当前末端读出的直接损失，不能排除相对分数排名、分类质量或其它阶段。
无NMS接入这份native PostProcessor路径。每类100截断可在评价器中完成；349归档导出预先做同样截断，本次原生输出复现它的156201/158417行，框与分数最大差均0，12指标差均0。
只有3/548张图超过300个GT（每图1—317）；不能凭“300个query”直接说所有拥挤图受固定数量上限控制。一个query可输出多个类别项，几何与类别预算必须分开检查。

3. 全GT失败地图
{map50}

{map75}
这六格按同一GT依次检查，所以可加和；其顺序不是潜在原因的独立可加贡献。其它重叠现象（遮挡/密度/类别）不得与六格相加。
G.5→G.75新增的原始几何区间GT为12362/12192；实际TP.5但非TP.75为10331/10264，没有TP.75但非TP.5的GT。本数据上集合已实核，未只凭两项总数相减假定嵌套。
raw任意类maxIoU中位数为0.654290/0.671516；连续分布的0/10/25/50/75/90/100分位及逐GT值见results.json与per_GT.csv，不把二元阈值作为唯一描述。
两seed共同漏检（IoU.5）12156/38759，共同命中21055/38759，seed1单独命中2421/38759、seed2单独命中3127/38759。seed复现指相同数据上的两个已训练权重，不代表两次独立抽样总体。
同类argmax框的IoU.5覆盖只有21233/38759、22166/38759，而实际多类别候选评价TP为23476/24182。argmax类不符不能当该query的真类项不存在。例：awning-tricycle的argmax覆盖44/532、108/532，但实际TP为204/532、222/532；car作参照的argmax覆盖11480/14064、11635/14064，实际TP11340/14064、11551/14064。

4. 合适参照与选择偏差
{size}

{classes}

{visible}

{density}

低密度参考不是随机对照，常与类别、尺寸、遮挡相关。按类别×5个短边组×3个遮挡标记建立共同层后，仅用有两组的91层：
{chr(10).join(std)}
这些是加权率，不能伪装为整数命中数；共同层以外325个低密GT和769个密集GT未进入该标准化比较。原始全GT分组保留，不删掉它们。粗分层与图像依赖仍使解释停留在关联，不能认定密度或stride碰撞为根因。
COCO面积small 26586个GT、medium11105个、large1068个，面积定义在原图；对应逐阶段表均在failure_map_by_group.csv。tiny4952个是另一坐标/分组口径。

5. 预测行的失败地图（与GT分母分开）
{fp}
TP之外均为操作性未匹配行。“与异类GT重叠”不是已确定的分类病因，“与同类GT重叠但未匹配”不等于每一行都可无损删除，“全部重叠<0.1”不等于已确认纯背景。原始忽略区域处理缺失，也限制这些解释。
提高显示分数会改变这些组成：score>0.5组保留11311/10729行，其中TP9312/9094；score>0.3组保留35270/32737行，其中TP17299/17126。完整0/.05/.1/.3/.5分数参照均保留在results.json；不是调阈值寻找赢家。

6. 已完成的最小缓存干预
{precheck}
原protocol.json在运行前冻结（SHA {R['protocol_sha256']}）。两seed的原始框和logits不变，仅把全图类别候选K=300改为600；每图600个导出行，score阈值与每类100上限不变，不用GT选择输出。模型仍只生成原来的300个query，没有新增解码；扩大的只是这些已生成query的类别候选读出。
两seed通过事先写明的AP/APs原值正差>1e-6及TP.5净增门槛。这是工程数值判据，不是统计显著性或新颖性判据。输出pair增多可以带来新的query身份被保留，不能与模型新生成query混写。
图像配对1000次bootstrap给TP.5率差95%描述区间：seed1 [3.479407,3.968280]pp；seed2 [3.593971,4.103854]pp。仅以图像为重采样单位，不把GT当iid；未做AP bootstrap，不据此宣称AP显著性，也不证明新数据总体效应。
这项检查同时改变输出行预算与query/class组成，因此还没有将原因拆开。下一步冻结B正是为区分这些解释；本轮不提出方法候选。

{cards}

7. 下一项冻结计划与阶段出口
唯一即时优先项：followup_plan.json（SHA {S['followup_plan_sha256']}）。A/C已算，B未算；增加2次B臂CPU评价，规划≤5分钟，硬上限900秒，新模型前向0。全548图与全部GT固定。主指标/对照/支持与反对条件/非法输入处理/抽样单位/成本都已冻结，不能在看到B后改变。
卡1的32图EMA轨迹只作为之后确需新增前向时的计划，当前不执行。若最终研究目标固定在完整96/132轮模型，先补其相同数据、权重分支、评价协议的地图，再把任何机制结论迁移过去。
已定位：当前末端几何覆盖缺口；具体top300/cap100损失；K600固定输出效应；raw/EMA与标注/评价口径差异。
待区分：额外类别假设与原未保留query身份；几何缺口在哪个真实阶段出现；EMA或完整训练终点是否保留相同失效。
当前无法识别：输入信息、内部特征、采样位置、训练监督/assignment的因果作用，以及可实现新方法与现有方法的残余缺口。
方法查新与创新：本阶段未新增检索或宣称文献缺口。问题即使已有方法研究，也不因此出局；需在机制证据后比较现有方法的条件、简单强对照与真实剩余缺口。

8. 复核、历史与限制
主脚本读完2766个输入文件后逐项重哈希，变化0；另做1096次源图SHA比对，错误0。两seed各548对raw/full缓存的预测、类别候选与GT均逐项复现。历史349原生预测共314618行逐项最大分数/框差0；历史12指标两seed差0。
第一次scalar检查混用FP32边界算术，最大差7.5822e-8；补充显式FP64独立标量检查1096对，差≤1e-12。原图native坐标与缓存640坐标在IoU.5没有跨阈值pair，在.75有1个临界pair：seed1 image107/query34/GT8144，0.749999942587 vs0.750002604863。本报告以原图原生评价为准；不通过放宽阈值消除差别。
GT id=0敏感性：内存id+1令AP变化仅+0.000143627/+0.000048257pp，APs+0.000444207/+0.000163443pp。原口径AP保留，未覆盖标注。该微小差别不能解释主要失败地图。
所有复核由同一助手与同一会话内的脚本完成，不是独立盲审。Windows评价二进制与历史Linux .so不同；同数据归档12指标精确复现已通过，不把它写成所有跨平台二进制逐位等价。
历史读取范围、源文件SHA及未重跑内容列在supplementary_checks.json。旧360A的“碰撞不是主因”“tiny低分是判别力不足”等历史解释没有被当作本轮定论；旧阴性结果只约束原假说。未重新阅读该清单所引全部论文，不声称完成260篇查新。

统计谬误扫描（11/11检查）：
1 Simpson：全体、尺寸/类别/遮挡/密度分层及共同层比较并列，未仅凭聚合率归因。
2 Ecological：预测行、query身份、GT、图像与AP单位分开，不由行数直接推GT收益。
3 Berkson：先全部GT，Geo0仅作后分组；val反复开发属性显式保留。
4 Collider：没有把按IoU筛出的失败子集再当位置原因证据；分层不是因果校正。
5 Base-rate：每个主要现象给计数/分母，并同时保留未匹配输出成本。
6 Regression-to-mean：同一固定框/分数对照，无重训练前后增益外推。
7 Survivorship：两seed全部548图、全部GT与7条未入选非法raw框都保留。
8 Look-elsewhere：本轮只做冻结K300/K600对照，不选择新K；承认历史反复探索。
9 Forking-paths：协议先冻结，原ID与正ID、原图与640平面差别分列，不事后改阈值。
10 Correlation/causation：尺寸/密度/遮挡为关联；只有固定输出读出规则是本轮受控改变。
11 Reverse-causality：GT只用于评价，不参与K300/K600输出选择；无训练方向性归因。

核心文件（均在本目录）：
01_一页问题地图.txt；02_两张问题卡.txt；followup_plan.json及.sha256；protocol.json及.sha256；
failure_map_by_group.csv（全尺寸/类别/遮挡/密度/面积/图像GT数）；per_GT.csv（77518行）；per_detection.csv（314618行）；
results.json；scope_audit.json；input_manifest.json；verification.json；supplementary_checks.json；input_domain_validation.json；
audit.py、validate_input_domain.py、supplement.py及run.log（可追溯实现与执行）。

本轮停在问题与机制验证计划，不生成训练命令或方法候选。某项假说失败，只关闭该假说；创新探索继续。
'''
put('00_诊断报告.txt',report)
# Deterministic image list for the optional later geometry trace; chosen without outcomes.
ds=read(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
sample=sorted(ds['images'],key=lambda im:hashlib.sha256(('problem_first_20261026_trace32|'+im['file_name']).encode()).hexdigest())[:32]
plan={'status':'PLAN_ONLY_NOT_EXECUTED','question':'EMA branch persistence and selected-reference/pre/decoder geometry transitions',
 'images':[{'id':i['id'],'file_name':i['file_name']} for i in sample],'sample_rule':'32 smallest SHA256(problem_first_20261026_trace32|filename), no outcome selection',
 'seeds':[1,2],'state':'same last.pth checkpoint ema.module; raw final cache is reference',
 'maximum_new_global_forwards':64,'new_training_steps':0,'GPU':False,'output_limit_bytes':20000000,
 'source_contract':'Record true topk indices, selected references, pre predictions and all3 decoder layers; do not fabricate unselected bbox-head predictions as natural encoder boxes.',
 'outputs':'AllGT per-stage geometry and real class-score records; fixed selected-reference best-query trajectories separately; subset-only inference.',
 'current_stage_action':'Do not run. Freeze a concrete instrumentation/parity contract before execution in a later stage.',
 'budget':'Historical32-forward receipts9.985/11.540s only give a rough64-forward22s starting point; hooks/export unmeasured. Plan5min; hard timeout900s.'}
with open(OUT/'geometry_trace_plan.json','x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
(OUT/'geometry_trace_plan.sha256').write_text(sha(OUT/'geometry_trace_plan.json'),encoding='utf-8')
# Scientific static overview, exported for review without modifying source data.
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,ax=plt.subplots(1,2,figsize=(13.5,5.4),gridspec_kw={'width_ratios':[1.15,1]})
 colors={'G':'#40699a','S':'#8a8a8a','K':'#df9e3b','C':'#9670ad','M':'#d66f6f','T':'#60a28c'}
 names={'G':'No geometric coverage','S':'Class-score floor','K':'Top300 selection','C':'Per-class cap100','M':'Unmatched with coverage','T':'Matched'}
 ys=np.arange(4);left=np.zeros(4)
 for k in 'GSKCMT':
  vals=np.array([g(s)[k+str(t)]/N*100 for s,t in [(1,50),(2,50),(1,75),(2,75)]])
  ax[0].barh(ys,vals,left=left,label=names[k],color=colors[k],height=.57)
  for y,v,l in zip(ys,vals,left):
   if v>5:ax[0].text(l+v/2,y,f'{v:.1f}%',ha='center',va='center',color='white',fontsize=9)
  left+=vals
 ax[0].set_yticks(ys,['Seed1 / IoU.50','Seed2 / IoU.50','Seed1 / IoU.75','Seed2 / IoU.75']);ax[0].invert_yaxis();ax[0].set_xlim(0,100)
 ax[0].set_xlabel('% of all 38,759 GT');ax[0].set_title('Observed final-output failure map',loc='left',fontweight='bold')
 ax[0].legend(loc='lower left',bbox_to_anchor=(-.02,-.49),ncol=2,frameon=False,fontsize=8.5)
 bins=['[0,4)','[4,8)','[8,16)','[16,32)','[32,inf)'];x=np.arange(5)
 for seed,color in [(1,'#40699a'),(2,'#d38d2f')]:
  gg=[g(seed,'short_side640',a) for a in bins]
  ax[1].plot(x,[(a['n_gt']-a['G50'])/a['n_gt']*100 for a in gg],marker='o',linestyle='--',color=color,label=f'Seed{seed}: raw geometry')
  ax[1].plot(x,[a['T50']/a['n_gt']*100 for a in gg],marker='s',color=color,label=f'Seed{seed}: actual TP')
 ax[1].set_xticks(x,bins);ax[1].set_ylim(0,100);ax[1].set_ylabel('GT coverage / hit rate, %');ax[1].set_xlabel('GT short side in stretched640 plane, px')
 ax[1].set_title('All size groups / IoU.50',loc='left',fontweight='bold');ax[1].grid(axis='y',alpha=.2);ax[1].legend(loc='lower right',frameon=False,fontsize=8.5)
 for a in ax:
  a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
 fig.suptitle('BASE24 raw model states | Validation548 | Operational partitions, not causal attribution',fontsize=11,x=.5,y=.99)
 fig.subplots_adjust(bottom=.31,top=.87,left=.13,right=.98,wspace=.34)
 fig.savefig(OUT/'failure_map.png',dpi=150,facecolor='white');plt.close(fig)
except ImportError:
 put('plot_unavailable.txt','matplotlib unavailable; all tables and raw data delivered. No dependency installation attempted.')
manifest={str(p.name):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.name!='delivery_manifest.json'}
with open(OUT/'delivery_manifest.json','x',encoding='utf-8') as f:json.dump({'files':manifest,'scope':'Problem map,2 cards,plans; no method candidates','models_or_training_executed':False},f,ensure_ascii=False,indent=2)
print('Reports and frozen plans created. Files:',len(manifest))
