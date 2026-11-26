"""Publish the current diagnostic checkpoint, preserving earlier snapshots."""
import json,hashlib,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent;Z=P/'后续零训练预检_20261026_v1';D=P/'同后端CPU重放_20261026_v1/analysis_v1';DISC=ROOT/'讨论伙伴_错误诊断文献_20261026_v1'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):
 with open(p,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
R=read(D/'results.json');G=read(Z/'shared_token_results.json');B=read(Z/'B_results.json');assert B['status']=='COMPLETE'
assert read(Z/'independent_followup_counts.json')['status']=='PASS'
last=list(DISC.glob('04*.txt'));assert last,'Wait for the other chat\'s final B discussion before publishing this checkpoint.'
text=['本轮最终诊断进展：机制预检与两张问题卡｜2026-10-26','',
'当前不需要用户介入。本轮完成原始全GT失败地图、raw/EMA逐GT配对、同后端CPU控制、共有空间token见证和原冻结B对照，并与“两个检测问题的研究讨论”进行了实际往返。没有训练、GPU/远程/test访问，也没有提出新网络模块或损失函数。','',
'采用全部548验证图/38759 GT、两个epoch24 baseline seed；不是完整132轮模型，也不是官方VisDrone挑战成绩。以下每项均分别报告seed，详细全部尺寸/类别/密度/遮挡分层在CSV。','',
'同CPU逐GT（IoU.50；每行四格分母38759）',
'seed 恢复 持续漏检 反向丢失 共同命中']
for s in (1,2):
 q=R['same_backend_transitions'][f'{s}_50'];text.append(f'{s} {q["recovered"]} {q["persistent_miss"]} {q["reverse_loss"]} {q["both_hit"]}')
text+=['恢复率分母是raw漏检：3137/15283、2382/14577；反向丢失率分母是raw命中：1528/23476、1643/24182。与原生EMA对照口径稍有不同；原生EMA→同CPU EMA只有33/38759、22/38759个GT的.50状态改变。旧报告保留，最新CPU结果见04报告及analysis_v1。','',
'问题卡1｜末端候选池缺少合格几何',
'影响：EMA末端G50为10931/38759、11068/38759，其中initial/pre/final三个自然观察点均无覆盖的000为10162/10931、10320/11068。原raw G为12486/38759、11826/38759。pre有而final无的局部现象保留，暂不单立第三卡。',
'已证实：大批GT在可观测候选池没有IoU达标框；同CPU改变保存整体state会改变部分GT的几何及最终结果。不能用原CPU/GPU上下文差异解释掉全部状态效应。',
'新增零训练检验：以全部raw000为基准，将EMA initial正见证按相同空间token ID的共有/新增成员拆开（不按query序号、不假定对象身份）。',
'seed  raw000总数  EMA初始无见证  仅共有  仅新增  两类都有']
for s in (1,2):
 q=next(r for r in G['primary_counts'] if r['seed']==s and r['IoU']==.5 and r['population']=='raw000')
 text.append(f'{s} {q["GT"]} {q["no_witness"]} {q["common_only"]} {q["added_only"]} {q["both"]}')
text+=['在EMA initial已覆盖的1778/951例中，共有ID提供正见证595/1778、324/951；仅新增ID见证1183/1778、627/951。故“初始覆盖改观一律必须引入原缺席空间入口”被反驳。raw000的共有ID正见证还证明：至少一个相同空间入口的reference已在decoder交互前跨过IoU阈值。',
'若共有池raw/EMA的初始最大IoU为r_C/e_C，取EMA共有池最佳ID i*，同ID的IoU变化≥e_C−r_C>0。两边最佳ID不必相同。这里只给成立的下界关系，尚未计算其数值分布。',
'限制：595/324不要求EMA最终命中，不能称TP恢复数；无initial见证也不等于final漏检。约2/3新增only不是“选择错误的病因占比”。同ID表示同网格入口，特征、encoder变换、框回归和buffer状态仍都可能改变。对持续000，上游原因仍未识别。',
'竞争解释与缺失证据：有用的未入选入口可能被得分排序排除；也可能现有表示/回归状态尚未在任何入口形成合格几何；尺寸、可见性、拥挤和训练历史可能共同作用。当前没有未入选框/特征或真实attention采样记录，不能把框中心当作采样证据。',
'最小下一零训练计划（未执行、需另行冻结）：在同CPU、两个seed EMA固定状态下，对全部548验证图记录topK之前的valid空间位置及得分，用同一个已有enc_bbox_head计算其初始reference；只增加已有头的诊断执行，不新增头或训练。对全部GT比较“未入选位置在此诊断反事实中能否计算出合格初始reference”，并以实际命中、尺寸/类别组为参照。额外头调用的输出不是原生路径已生成后丢弃的框，几何合格也不保证decoder实际检出。若存在，仅定位固定表示/现有头反事实全集中的候选可用性；若不存在，仅反对“现有头可给出合格初始框、只是入口被topK排除”这一窄解释，不能证明没有目标信息或后续decoder无法恢复。该计划需要新增CPU前向，当前缓存本身不提供答案；不虚构未入选框。','',
'问题卡2｜合格真类候选在输出读出中未保留',
'影响：raw K50为2359/38759、2392/38759；EMA K50为2355/38759、2361/38759。但raw/EMA都处于K的同一GT只有1188/2359、1215/2392（分母raw K），不能把总量稳定理解为固定个体持续受损。',
'已补齐冻结A/B/C对照：A=原top300；U=A所用query身份；B=只在U内取top600类假设；C=全query取top600。框/分数/GT/score>0.001/每类cap100与匹配固定。全部图可行性门槛通过，U中有效对最少960/830，没有填充、修框或改阈值。',
'AP/APs差以下为百分点，采用原正GT-ID敏感性分支，与同CPU表的原ID AP绝对值分别标明；逐GT计数均包括GT0。',
'seed 对照  ΔAP  ΔAPs  TP50新命中  TP50丢失  TP50净变化']
for s in ('1','2'):
 q=B['seeds'][s];z=q['sets']['50'];text.append(f'{s} B-A {q["B_minus_A_pp"]["AP"]:+.6f} {q["B_minus_A_pp"]["APs"]:+.6f} {len(z["B_new_vs_A"])} {len(z["B_lost_vs_A"])} {len(z["B_new_vs_A"])-len(z["B_lost_vs_A"])}')
 text.append(f'{s} C-B {q["C_minus_B_pp"]["AP"]:+.6f} {q["C_minus_B_pp"]["APs"]:+.6f} {len(z["C_new_vs_B"])} {len(z["C_lost_vs_B"])} {len(z["C_new_vs_B"])-len(z["C_lost_vs_B"])}')
text+=['B-A新增的分母raw漏检为1182/15283、1223/14577；B-A丢失为0/23476、0/24182。C-B新增为457/14101、468/13354（分母B漏检），C-B反向丢失196/24658、208/25405（分母B命中）。',
'两项原定双seed“AP/APs>1e-6原单位且TP50净正”门槛均通过，支持两条具体固定读出规则在这两个raw模型上的操作效应。B新TP与原C-A新增1443/1483的交集为986/1015；B也有196/208个C没有的新增GT。',
'代价：每seed A导出164400对，B/C均328800对。cap后A=156201/158417，B=304181/309288，C=286557/294563；FP50 A=132725/134235，B=279523/283883，C=261638/268898。候选扩展的收益伴随大量额外错误输出，不能只报恢复数量。',
'停止的窄假说：新增query身份不是取得任何读出收益的必要条件，因为B-A已阳性；仅原query附加类假设不能复现C找回的全部对象，因为C仍有457/468个B未命中的GT；C也不逐GT支配B。',
'分层参照：C-B在class4/6/7/8的TP50两个seed分别为−4/−6、−12/−14、−6/−2、−2/−2，对应各类全GT分母1975/1045/532/251。总体阳性不代表每类都改善；各组完整A/B/C计数见B_by_class_area.csv。',
'仍保留：同query类别假设竞争、跨query排序、每类cap及匹配竞争都可能参与。C-B改变成员替换与cap后输出规模，不能把净261/260或其AP增量当新增query的独立病因贡献。B-A无反向丢失也不能单独包装成保护机制；固定顺序下追加低分检测可有结构性单调性。',
'最小下一零训练计划（未执行）：用既有raw缓存重建B/C的真类几何可用、top600、cap100和实际匹配状态，逐GT对齐C新增457/468及C丢失196/208，区分读出成员替换、cap去除/释放、剩余匹配竞争。使用完整GT及共同命中/共同漏检参照，不只挑成功例；可复用现有hit表，不新增forward。它定位操作差异，仍不等同训练分类/assignment根因。','',
'下一证据的优先级',
'G规模更大，K的固定干预区分力更具体；研究优先级不能只按其中一个排序。另一个对话主张优先检验同CPU EMA上的同语义A/B/C是否迁移，本对话原倾向先厘清B/C成员/cap/匹配差异。收束安排：raw B/C逐GT操作地图作低成本配套核查；另冻同CPU EMA同语义A/B/C迁移作下一主检验，仍0新forward；G未入选初始几何诊断因需新CPU前向，单独列计划、排在缓存检验之后。本轮没有追加上述检验。涉及原因的结论继续限定到短训练模型，EMA迁移也不能替代完整训练终点证据。失败只停止相应迁移主张，不停止问题和创新探索。',
'TIDE/Hoiem仅支持本轮诊断原则：频数、AP敏感性与可训练收益分开；oracle的权限逐项说明，不能用理想修正承诺真实收益。未宣布研究问题或方法无人做过，未作系统文献综述。机制支持后再查解决该具体问题的现有方法及缺口，当前不提新模块/损失。','',
'证据质量与归档',
'主诊断/两个后续检查使用全GT与两个seed，77518行分别独立重计；两个B评价全局完成后再分组。图配对1000次bootstrap仅给TP率区间，不给未计算的AP区间或显著性。',
'同CPU重放总2192次forward（含复用预检），之后共有ID和B检查均0新增forward；模型状态SHA前后不变，0训练。原raw完全复现，原指标完全重现；同CPU分析2789输入、共有ID分析2196输入、B分析1105输入及补充4项原评价依赖分别有SHA核验。',
'B耗时193.330秒，仅2次新增评价；原A/C评价复用，A预测逐项比对、C导出计数/来源配方复核，没有宣称再次评价C或比对不存在的C预测文件hash。',
'原数据/权重/旧报告/原冻结协议保留。04报告是执行后续检查前的快照，其中“B未执行/身份检验待执行”的计划状态现由本05报告更新；原父目录results.json仍为失败序列化遗留，不能当权威数值。',
'最新同CPU逐GT文件：'+str(D/'per_GT_stages_and_transitions.csv'),
'后续原始结果：'+str(Z/'shared_token_results.json')+'；'+str(Z/'B_results.json'),
'对话者第十一轮回应：'+str(last[0]),
'本轮仍交付问题地图、2张问题卡和最小后续检验；某个假说停止不意味着创新探索停止。']
with open(P/'05_机制预检结果与最新问题卡.txt','x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
state=read(P/'work_state.json');state.update({'version':2,'latest_completed_stage':'Paired CPU replay, common-token initial witnesses and original frozen B, followed by multi-round two-chat discussion.',
 'last_wait_cursor':sys.argv[1] if len(sys.argv)>1 else None,'discussion_status':'Latest bounded discussion completed; no background automation promised.',
 'completed_current_counts':{'same_CPU_seed1_50':R['same_backend_transitions']['1_50'],'same_CPU_seed2_50':R['same_backend_transitions']['2_50']},
 'current_reports':['05_机制预检结果与最新问题卡.txt','04_同后端结果与两张更新问题卡.txt'],
 'not_executed':['New training','Modules/loss proposals','Dense unselected encoder-reference diagnostic','B/C perGT full operational transition attribution','Method/novelty gap search'],
 'next_independent_work':['Freeze same-CPU EMA A/B/C transfer as next main zero-forward check; full raw B/C operational maps are a small companion check.','Only afterwards consider separately freezing a same-CPU existing-head dense reference availability probe, no training; selected caches cannot answer unselected availability.'],
 'important_limits':['All effects are two BASE24 seeds; no full-training generalization.','Spatial-token identity is not object/feature identity.','G/K state maps are not causal contributions.','B/C cap and membership differ; AP improvements do not establish trainable gains.'],
 'previous_state_snapshot':'work_state.json preserved; historical completed/not-executed lists therein describe the earlier raw/native-EMA pairing, not current state.'})
# Replace old pending lists, rather than leaving contradictory current work items.
state['completed']=['All-GT original scope/failure audit','Raw/native-EMA pairing','Lower-bound arithmetic and matching-monotonicity interpretation correction','Read-only same-CPU replay and allGT staged analysis','Shared-token initial witness analysis','Original frozen B after every-image feasibility gate','Independent row counting and two-chat interpretation discussion']
write(P/'work_state_v2.json',state)
with open(P/'README_最新结果入口.txt','x',encoding='utf-8') as f:f.write('当前先读05_机制预检结果与最新问题卡.txt。\n04为后续零训练检验前的完整CPU结果快照，执行状态由05更新。\n原raw/原生EMA对照的权威数字在attempt02；新的同CPU结果在同后端CPU重放_20261026_v1/analysis_v1；共有ID与B在后续零训练预检_20261026_v1。\n父目录results.json是旧序列化失败遗留，不使用。\n当前工作状态=work_state_v2.json；work_state.json与delivery_manifest.json保留为早期快照。最新交付清单=delivery_manifest_v2.json。\n')
files={str(p.relative_to(P)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in P.rglob('*') if p.is_file() and p.name not in ('delivery_manifest_v2.json','work_state_v2.json')}
external={str(p):sha(p) for p in DISC.glob('*.txt')}
write(P/'delivery_manifest_v2.json',{'authority':'05 report for current interpretation; explicit numeric authority directories as README','files':files,'external_public_discussion_files':external,
 'mutable_excluded':['work_state_v2.json'],'original_snapshots_preserved':True,'operations':'CPU observations2192 forwards, no training; follow-up common-token0 forward and B0 forward/2 native evals; no GPU/remote/test.'})
print('Final diagnostic checkpoint written',len(files),'files')
