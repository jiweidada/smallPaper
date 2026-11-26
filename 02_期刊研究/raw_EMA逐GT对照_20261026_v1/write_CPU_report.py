"""Human-readable report of verified counts; excludes unexecuted causal claims."""
import csv,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent;D=P/'同后端CPU重放_20261026_v1/analysis_v1'
def read(p):return json.loads(Path(p).read_text('utf-8'))
R=read(D/'results.json');C=read(D/'independent_counts.json');V=read(D/'verification.json');assert V['status']==C['status']=='PASS'
groups=list(csv.DictReader(open(D/'grouped_failure_and_geometry.csv',encoding='utf-8-sig')))
def group(s,b,typ,g):return next(r for r in groups if r['seed']==str(s) and r['branch']==b and r['group_type']==typ and r['group']==g)
def pct(n,d):return f'{n}/{d} ({100*n/d:.3f}%)' if d else f'{n}/{d}'
text=['同后端CPU raw/EMA逐GT诊断与问题卡更新｜2026-10-26','',
'本轮结论：两条问题继续保留，但仍没有确定上游病因或新方法。受控CPU对照确认raw/EMA保存状态有稳定的端点差异；末端几何缺口与候选保留缺口在EMA中仍存在。pre至final会丢失一部分几何覆盖，同时新增更多覆盖，不能把它直接升级成“decoder普遍破坏定位”。','',
'一、任务与数据口径',
'VisDrone2019-DET转换验证集548图、38759个有效GT，10类；不等同官方VisDrone挑战评价。两seed同epoch24（last_epoch23）的last.pth，分别model与ema.module。原训练配方写132轮不代表本checkpoint已完成132轮。模型为原baseline；原损失、结构和权重均未改。',
'相同本地CPU/FP32/2线程/batch1/eval/no_grad；同PIL RGB拉伸640和输入文件。sigmoid300×10→flatten top300→score>0.001→COCO每图每类maxDets100，无NMS。AP采用原GT ID；逐GT匹配以gtMatches中的正检测ID识别GT0，不能把native dtMatches=0哨兵当GT0未匹配。','',
'二、同后端逐GT结果（恢复=raw漏/EMA中；反向=raw中/EMA漏；每行四格和为38759）',
'seed IoU  raw命中 EMA命中 恢复 持续漏检 反向丢失 共同命中']
for s in (1,2):
 for th in (50,75):
  q=R['same_backend_transitions'][f'{s}_{th}'];text.append(f'{s}  {th/100:.2f}  {q["raw_hit"]}  {q["EMA_hit"]}  {q["recovered"]}  {q["persistent_miss"]}  {q["reverse_loss"]}  {q["both_hit"]}')
 q=R['same_backend_transitions'][f'{s}_50'];miss=38759-q['raw_hit']
 text.append(f'  seed{s} .50：恢复率{pct(q["recovered"],miss)}；持续率{pct(q["persistent_miss"],miss)}；反向率{pct(q["reverse_loss"],q["raw_hit"])}。')
text+=['','同CPU四快照持续漏检：.50为'+str(C['same_CPU_four_snapshot_persistent']['50'])+'/38759，.75为'+str(C['same_CPU_four_snapshot_persistent']['75'])+'/38759。四个快照均为G50：'+str(C['same_CPU_all_four_G50'])+'/38759。旧“10504”来自raw CPU/原生EMA口径，保留原报告；新CPU口径为10499，不混用。','',
'三、执行上下文参照与AP（下列AP均以百分数展示；差为百分点）']
for s in (1,2):
 c=R['archived_vs_CPU_EMA'][str(s)];old=c['old_EMA_hit_all10'][0];gain=c['old_miss_CPU_hit_all10'][0];loss=c['old_hit_CPU_miss_all10'][0]
 a=R['summaries'][f'{s}_raw']['metrics'];b=R['summaries'][f'{s}_EMA']['metrics']
 text.append(f'seed{s} 原生EMA→CPU EMA .50新增{pct(gain,38759-old)}、丢失{pct(loss,old)}，命中状态改变{gain+loss}/38759；AP差{c["AP_metrics_CPU_minus_archived_pp"]["AP"]:+.6f}pp。')
 text.append(f'  同CPU raw→EMA：AP {a["AP"]*100:.6f}→{b["AP"]*100:.6f}，AP50 {a["AP50"]*100:.6f}→{b["AP50"]*100:.6f}，AP75 {a["AP75"]*100:.6f}→{b["AP75"]*100:.6f}，APs {a["APs"]*100:.6f}→{b["APs"]*100:.6f}。')
text+=['同CPU下改变的是整体state_dict（参数及buffer）；可以识别保存状态在该流程下的作用，尚不能把作用归到某一层、仅归到权重平均机制、推到完整训练阶段或普遍泛化。旧EMA与CPU EMA仍有少量真实个体差异，不能说所有张量相同。','',
'四、全GT末端失败地图（每行分母38759；S=0也明确保留）',
'G=无任意类别合格末端框；S=有几何但真类分数均不达阈值；K=真类合格对未进top300；C=每类cap100去除；M=保留但实际未匹配；T=实际匹配。它们是顺序操作状态，不是独立病因贡献。',
'seed 状态 IoU G S K C M T']
for s in (1,2):
 for b in ('raw','EMA'):
  q=R['summaries'][f'{s}_{b}']['allGT']
  for th in (50,75):text.append(f'{s} {b} {th/100:.2f} '+' '.join(str(q.get(f'state{th}_'+k,0)) for k in 'GSKCMT'))
text+=['','五、三个自然观察点：initial参考框/pre框/final框',
'覆盖为“池中至少一个框IoU达标”，不带类别，不是TP；初始参考框也不是实际输出检测。pre发生在第一层decoder block之后、第一层分布式框细化之前；pre之后至final还包含若干计算，不能直接指定某层。',
'三位bit依次表示initial/pre/final几何覆盖：']
for s in (1,2):
 for b in ('raw','EMA'):
  q=R['summaries'][f'{s}_{b}']['allGT'];text.append(f'seed{s} {b} .50（各格分母38759）：'+', '.join(k+'='+str(q.get('geometry_bits50_'+k,0)) for k in ('000','001','010','011','100','101','110','111')))
  d=C['pre_to_final'][f'{s}_{b}_50']
  text.append(f'  pre有/final无：{pct(d["pre_to_final_loss"],d["pre_covered"])}；pre无/final有：{pct(d["pre_to_final_gain"],d["pre_uncovered"])}。参照分母不同，不能只比较两率。')
for s in (1,2):
 q=R['summaries'][f'{s}_EMA']['allGT'];text.append(f'seed{s} EMA末端G中000：{pct(q["geometry_bits50_000"],q["state50_G"])}。')
 flow=R['state_flows_and_G_recovery_geometry'][str(s)];bits=flow['rawG_recovered_EMA_bits50'];n=sum(bits.values());initial=bits.get('111',0)+bits.get('101',0)
 text.append(f'  rawG→EMA实际命中：{n}；其中EMA initial已有覆盖{pct(initial,n)}，其余到后面观察点才出现覆盖。因此恢复不能全归到后续框细化。')
text+=['同branch内槽位顺序保持，已记录initial/pre最佳槽位在final的IoU；不同模型状态的同一query序号不当同一对象。两状态共有encoder空间token的平均数为192.779/300和191.770/300（548图各自均值）；这仅说明成员变化，不能证明选择是病因。',
'探索性稳健性补计：EMA pre→final的.50丢失中，pre至少.55且final低于.45的GT为19/421、17/404；不是所有丢失都只是阈值贴边。该.05边界在阶段结果之后做描述性检查，不伪称预注册。','',
'六、合适参照组（EMA .50；短边按640输入尺度；完整类别/密度/遮挡/截断计数见grouped_failure_and_geometry.csv）']
for s in (1,2):
 for g in ('[0,4)','[4,8)','[8,16)','[16,32)','[32,inf)'):
  q=group(s,'EMA','size',g);n=int(q['GT']);text.append(f'seed{s} 短边{g}：G {pct(int(q["state50_G"]),n)}，K {pct(int(q["state50_K"]),n)}，T {pct(int(q["state50_T"]),n)}。')
text+=['这些尺寸组也混合了类别、密度和可见性差异，不能把组间差异直接归因于尺寸，更不能预设宽度、位置或训练分配是主因。','',
'七、问题卡（暂保留2张）',
'卡1：末端候选池缺少合格几何。影响：raw G50=12486/38759、11826/38759；EMA G50=10931/38759、11068/38759。EMA G里000占10162/10931、10320/11068。证据已到：全GT池覆盖、同后端状态对照及三个自然观察点的区间定位。',
'竞争解释：选中空间token缺少可用对象信息；选中token有信息但参考框回归未产生合格框；分辨率/可见性/拥挤与标注几何共同作用；训练监督与优化历史使相关状态不足。现有资料还不能拆开这些解释。',
'还缺：未入选token在同一状态下的可用几何、特征/注意力采样与训练历史的直接证据；完整训练checkpoint迁移性和该族对AP的明定反事实代价。',
'最小零训练计划：先用现有selected-token IDs将raw/EMA initial覆盖按共有/新增token拆开，在全GT及恢复/持续/反向/共同命中组分别计数，判断“恢复是否必须依靠新增成员”这条具体解释。该检验仅约束恢复子群，不能解释000全部病因；未入选框仍未知。须单独冻结后运行，当前未执行。pre丢失作为子现象保留，暂不升第三卡。','',
'卡2：已有合格真类候选在输出选择中未保留。影响：raw K50=2359/38759、2392/38759；EMA K50=2355/38759、2361/38759。固定raw框/分数将K300放至600已恢复1443/1483个TP50，同时显著增加输出及FP；AP仅+0.187428/+0.219084pp（原正ID敏感性分支），不是全部候选缺口的可训练收益。',
'注意：raw与EMA都处于K的同一GT仅1188/2359、1215/2392（分母为raw K）。两端K总数接近，不代表同一批GT持续卡在K。',
'竞争解释：同query多类假设竞争；原未输出query的好真类对被挤出；类别间/质量分数排序；每类cap和匹配的后续作用。没有证据可以把K等同单一分类根因或训练assignment失败。',
'最小零训练计划：已有冻结A/B/C——A原top300，U为A入选query ID；B仅在U内取top600对；C全部query取top600（C已有）。框、分数、阈值、cap、匹配固定；每图须有600个有效对，否则B记N/A，不能补对/改阈值。B-A约束多类假设扩展，C-B约束未输出query纳入；不作可加因果分解。B尚未执行，不能报告B结果。','',
'八、文献讨论与停止的假说',
'TIDE及Hoiem诊断文献区分错误频数与AP代价。TIDE的missed-GT oracle改变召回分母，其ΔAP不是未来模型能找回多少GT的承诺；多个独立oracle收益也不能直接相加。我们本轮只借用诊断原则，未把G/K强行映射为TIDE的某个错误类，也未直接运行其工具。',
'当前匹配规则使高IoU阈值的命中GT集属于低阈值命中集，故“.5漏、提高阈值反而中”为0不具鉴别力；此假说已停止，其他探索继续。证明与解释修订见边界与匹配机制_20261026_v1/解释修订_匹配单调性.txt。',
'来源原文：https://dbolya.github.io/tide/paper.pdf ；https://dhoiem.cs.illinois.edu/publications/eccv2012_detanalysis_derek.pdf ；https://dhoiem.cs.illinois.edu/projects/detectionAnalysis/detectionAnalysis_v2.pdf 。精确版本、SHA和阅读范围见文献库_sources_问题优先诊断讨论_20261026_v1.json。尚未建立具体解决方法的缺口，不进行新模块/损失筛选。','',
'九、核验与Material Passport',
f'2192个被动观察文件共{read(D.parent/"complete.json")["output_bytes"]}字节（{read(D.parent/"complete.json")["output_bytes"]/1048576:.2f}MiB），总2192次CPU前向含16次复用预检；后续分析0次新forward、0次训练。两个raw分支所有final boxes/logits完全一致，原12项评价指标均重现，2789个输入SHA不变。seed2 raw有7个非正宽高末端框，全部未导出；其他状态没有，均保留原值，不修复或删框。',
'预注册分析protocol SHA256：'+R['analysis_protocol_sha256'],
'独立核算77518行：四格、阶段bit、G/K/C/M/T分区和跨seed集合通过。有限验证数据计数为主，不作iid-GT显著性推断；仅两个seed，不能声称跨模型普遍机制。状态对照、阶段观察、具体读出干预、oracle敏感性与训练因果效应分别标明。',
'原权重/源数据/旧报告不覆盖；先前原生EMA对照的权威数值仍在attempt02，新同后端结果在同后端CPU重放_20261026_v1/analysis_v1。原父目录results.json为序列化失败遗留，不使用。','',
'真实讨论继续记录在“两个检测问题的研究讨论”（thread 01a17b5a-0a45-7921-9591-3ac370e081b5），不是一次性审查。讨论者的下一轮回应将单独保存，不提前称其同意本报告。']
dest=P/'04_同后端结果与两张更新问题卡.txt'
with open(dest,'x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
print(dest)
