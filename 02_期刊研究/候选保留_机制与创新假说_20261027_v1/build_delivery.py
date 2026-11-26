"""Build a traceable research handoff from completed frozen probes."""
import json,csv,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PAIR=ROOT/'raw_EMA逐GT对照_20261026_v1';REP=PAIR/'同后端CPU重放_20261026_v1';TRANS=PAIR/'rawBC操作地图与EMA迁移_20261026_v1';PARTNER=ROOT/'讨论伙伴_候选保留主线取证_20261027_v1'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,s):
 with open(HERE/n,'x',encoding='utf-8') as f:f.write(s)
def jout(n,j):write(n,json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cout(n,rr):
 with open(HERE/n,'x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def table(fields,rr):return '\n'.join([' | '.join(fields)]+[' | '.join(str(r[k]) for k in fields) for r in rr])
def f6(x):return f'{x:.6f}'
def main():
 dirs=['cap_dependency','fixed_budget_capacity','class_prefix_bound','matching_and_witness','pre_witness_information','class_history_information','single_swap_readout'];evidence={};results={}
 for d in dirs:
  p=HERE/d;z=read(p/'attempt01/results.json');v=read(p/'attempt01/verification.json');h=sha(p/'protocol.json');assert z['status']=='COMPLETE' and v['status']=='PASS' and z['protocol_sha256']==h==(p/'protocol.sha256').read_text('utf-8').strip()
  assert not (p/'attempt01/failure.json').exists()
  evidence[d]={'protocol_sha256':h,'protocol':str(p/'protocol.json'),'results':str(p/'attempt01/results.json'),'verification':str(p/'attempt01/verification.json'),'verification_sha256':sha(p/'attempt01/verification.json'),'results_sha256':sha(p/'attempt01/results.json'),'elapsed_seconds':z['elapsed_seconds'],'operations':z.get('operations',{})};results[d]=z
 support=read(HERE/'class_history_information/support_denominators_v1/results.json');assert support['status']=='PASS'
 replay=read(REP/'analysis_v1/results.json');match=results['matching_and_witness'];hist=results['class_history_information'];swap=results['single_swap_readout'];free=results['fixed_budget_capacity'];cap=results['cap_dependency'];pre=results['pre_witness_information'];prefix=results['class_prefix_bound']
 old={'raw':read(TRANS/'raw_operations_v1/results.json'),'EMA':read(TRANS/'EMA_transfer_v1/results.json')}
 order=[('raw',1),('raw',2),('EMA',1),('EMA',2)]
 maps=[];contrasts=[];bases=[];cs=[];ss=[];bounds=[];transition=[];critical=[]
 for state,seed in order:
  key=f'{state}_s{seed}';r=replay['summaries'][f'{seed}_{state}']['allGT'];m={'条件':key,'图像':548,'GT':38759,**{k:r.get(f'state50_{k}',0) for k in ('T','G','S','K','C','M')}};assert sum(m[k] for k in ('T','G','S','K','C','M'))==38759;maps.append(m)
  ar=swap['conditions'][key]['arms']['A'];br=old[state]['seeds'][str(seed)]['B'] if state=='raw' else old[state]['seeds'][str(seed)]['arms']['B'];cr=old[state]['seeds'][str(seed)]['C'] if state=='raw' else old[state]['seeds'][str(seed)]['arms']['C']
  bs={'条件':key,'GT':38759,'BA_AP_pp':f6(100*(br['metrics']['AP']-ar['metrics']['AP'])),'BA_APs_pp':f6(100*(br['metrics']['APs']-ar['metrics']['APs'])),'BA_TP50净增':br['TP50']-ar['TP_all10'][0],'CB_AP_pp':f6(100*(cr['metrics']['AP']-br['metrics']['AP'])),'CB_APs_pp':f6(100*(cr['metrics']['APs']-br['metrics']['APs'])),'CB_TP50净增':cr['TP50']-br['TP50']};bases.append(bs)
  for th,t in ((50,0),(75,5)):
   for left,right in (('B','A'),('C','B')):
    l=match['conditions'][f'{state}_s{seed}_{th}_{left}'];q=match['conditions'][f'{state}_s{seed}_{th}_{right}'];contrasts.append({'state':state,'seed':seed,'IoU':th/100,'contrast':left+'-'+right,'GT':38759,'images':548,'actualTP_delta':l['actual_TP']-q['actual_TP'],'same_cap_rank_delta':l['same_cap_pool_max']-q['same_cap_pool_max'],'greedy_gap_delta':l['greedy_cardinality_gap']-q['greedy_cardinality_gap'],'left_greedy_gap':l['greedy_cardinality_gap'],'right_greedy_gap':q['greedy_cardinality_gap']})
  z=hist['conditions'][key]['domains']['B_minus_A'];c=z['counts'];M=z['M2'];ci=M['image_bootstrap95'];qm=hist['conditions'][key]['domains']['GT_qualified']['M2'];pc=pre['conditions'][key]['counts'];sw=swap['conditions'][key];ha=sw['contrasts']['H_minus_A'];hs=sw['contrasts']['H_minus_S']
  cs.append({'条件':key,'B减A_pair分母':c['candidate_pairs'],'pre赢家pair':c['pre_argmax_candidate'],'pre赢家合格50':c['pre_argmax_and_qualified50'],'其余pair':c['candidate_pairs']-c['pre_argmax_candidate'],'其余合格50':c['GT_qualified50']-c['pre_argmax_and_qualified50'],'M2可比pair':M['overlap_candidates'],'M2可比图':support['conditions'][key]['images_with_at_least_one_supported_candidate'],'M2可比分档':M['overlap_strata'],'M2合格率差_pp':f6(100*M['risk_difference']),'图像bootstrap95_pp':f'[{100*ci[0]:.6f},{100*ci[1]:.6f}]','合格者关键性M2差_pp':f6(100*qm['risk_difference']),'关键性95_pp':str([100*x for x in qm['image_bootstrap95']]),'代理饱和图':pc['proxy_global_saturated'],'代理单删关键pair':pc['proxy_critical_full']})
  ss.append({'条件':key,'图像分母':548,'实际交换图':sw['images_swapped'],'H和S补入不同图':sw['different_H_S_images'],'GT分母':38759,'H_A恢复50':len(ha['gained50_IDs']),'H_A丢失50':len(ha['lost50_IDs']),'H_A恢复75':len(ha['gained75_IDs']),'H_A丢失75':len(ha['lost75_IDs']),'H_A_AP_pp':f6(ha['delta_pp']['AP']),'H_A_APs_pp':f6(ha['delta_pp']['APs']),'H_S_AP_pp':f6(hs['delta_pp']['AP']),'H_S_APs_pp':f6(hs['delta_pp']['APs']),'H_S恢复50':len(hs['gained50_IDs']),'H_S丢失50':len(hs['lost50_IDs'])})
  b=free['conditions'][state][str(seed)]['50'];a=match['conditions'][f'{state}_s{seed}_50_A'];bounds.append({'条件':key,'GT':38759,'A实际TP50':b['A_actual'],'A同cap池最大值':a['same_cap_pool_max'],'满池同300预算图上界':b['full300_oracle'],'A漏关键pair_导出':a['critical_missing_export'],'A漏关键pair_cap后':a['critical_missing_cap'],'满池全部关键pair':a['full_critical_pair_nodes'],'剩余非单点关键容量差':a['capacity_loss_not_individual_critical'],'prefix最大闭包':prefix['conditions'][f'{state}_s{seed}_50']['largest_initial_closure']})
  if state=='raw':
   for th in (50,75):
    w=match['raw_actual_pair_witnesses'][f's{seed}_{th}'];critical.append({'seed':seed,'IoU':th/100,'C新增关键':w['C_new_critical_matched_pair'],'C新增分母':w['C_new_GT'],'C丢失旧B关键':w['C_lost_critical_matched_pair'],'C丢失分母':w['C_lost_GT'],'共同命中C关键':w['common_C_critical_matched_pair'],'共同命中分母':w['common_C_GT']})
 for seed in (1,2):transition.append({'seed':seed,**replay['same_backend_transitions'][f'{seed}_50']})
 cout('全部GT失败地图.csv',maps);cout('raw_EMA逐GT转移_原口径.csv',transition);cout('BA_CB同协议实际效应.csv',bases);cout('同池匹配与池容量差.csv',contrasts);cout('固定300图空间与关键遗漏.csv',bounds);cout('raw真实匹配关键性见证.csv',critical);cout('类别历史条件信息_完整分母.csv',cs);cout('固定单交换结果_完整分母.csv',ss)
 assert not pre['strict_information_advance_gate_pass'] and hist['strict_information_advance_gate_pass'] and not swap['strict_readout_advance_gate_pass']
 ledger=[
 {'id':'cap100_necessity','status':'CLOSED_REFUTED','claim':'cap100 is necessary for measured C-B joint benefit','evidence':'cap600 benefit persists in allfour states/seeds; other truncation effects remain'},
 {'id':'physical600_requirement','status':'CLOSED_REFUTED_IN_FIXED_GRAPH','claim':'More matching coverage physically requires600 outputs','evidence':'full matching bound fits300 in4384cases; not actual AP attainability'},
 {'id':'class_prefix_hard_budget','status':'CLOSED_REFUTED_IN_RELAXATION','claim':'within-query class prefix cost makes matching bound unreachable at300','evidence':'4384 equal constructive certificates; largest closure215'},
 {'id':'greedy_matching_main_C_B','status':'LIMITED_REAL_NOT_MAIN_CARDINALITY_EXPLANATION','claim':'same-pool greedy cardinality gap drives C-B netTP','evidence':'C-B gap changes+1,-1,-2,0 at.50; all0 at.75; underlying A gaps40-59 remain'},
 {'id':'selection_capacity','status':'SUPPORTED_FIXED_POOL_MECHANISM','claim':'output selection can discard GT-graph-irreplaceable candidate pairs','evidence':'actual-gain and actual-loss witnesses both include full-graph critical nodes; no training-cause or deployable-label claim'},
 {'id':'pre_witness_information','status':'CLOSED_DEGENERATE','claim':'frozen pre units/capacity single-deletion proxy discriminates critical pairs','evidence':'proxy truncated rank300 all2192images, critical0, no conditional overlap or CI'},
 {'id':'class_history_qualification','status':'SUPPORTED_LIMITED_INFORMATION','claim':'pre winning-class identity adds qualified-box information in comparable B-A coarse strata','evidence':'allfour primary M2 CIs positive; supports60115/49210/39716/40316 of164400; not whole-domain causal effects or criticality'},
 {'id':'class_history_criticality','status':'NOT_SUPPORTED_FOR_PROTECTION','claim':'pre winning class identifies irreplaceable qualified candidates','evidence':'allfour conditional differences negative; EMA1 interval fully negative'},
 {'id':'single_swap_readout','status':'CLOSED_RULE_FAILED','claim':'one frozen pre-win insertion/pre-disagree deletion safely reads out information','evidence':'AP/APs H-A negative in allfour; score-addition comparison also fails gate'},
 {'id':'Hlineage','status':'FROZEN_PLAN_NOT_EXECUTED','claim':'class-conditioned supporting-neighbor correspondence adds bidirectional capacity information beyond fusion/marginal geometry','evidence':'concrete sets, marginal-preserving nulls, geometric reference and two-sided labels defined; no data result or new-method claim'}]
 jout('假说状态账本.json',ledger);jout('证据入口与协议绑定.json',evidence)
 card={'number':1,'problem':'候选保留有真实的覆盖收益，也会反向删除不可替代覆盖；单框资格/置信无法直接替代安全交换风险。','population':'全部548val图/38759GT，BASE24两seed，raw与EMA分别报告；原C-B主问题保留，后续B-A pair分析明确命名。','impact':{'BA_CB':bases,'actual_critical_witnesses_raw':critical,'capacity_bounds':bounds,'one_swap_failure':ss},'observations_vs_causes':'保留池容量变化得到冻结缓存反事实支持；为何训练产生这些分数/框尚未确认。cap必要、类序硬预算和当前pre容量代理关闭；同池贪心不是当前C-B基数主解释；资格信息不等于关键性。','evidence_stage':'问题存在+固定预测读出机制+一条有限资格信息前提；未到可部署方法/新意/训练泛化验证。','missing':['能同时识别补入收益和删除损伤的GT-free信息','排除普通分数融合、静态空间/类别多样性、几何对应等竞争解释','相同真实预算下的正向作用验证','完整近邻的公平比较、独立数据/完整训练模型上的确认'],'minimal_next_zero_training_test':str(HERE/'Hlineage_下一最小零训练检验_冻结计划.json'),'innovation_hypothesis':'Hlineage仅为候选；只有真实类条件支持血缘优于强置换和几何对应、且补入和删除两侧都过关后，才进入方法候选。'}
 jout('问题卡_01_候选保留.json',card)
 text=f'''候选保留：问题、机制、已有方法边界与可证伪主线候选
2026-10-27；与现有对话“两个检测问题的研究讨论”持续往返形成。对话ID：01a17b5a-0a45-7921-9591-3ac370e081b5。

当前结论
保留一张问题卡：如何在保留有效类别/位置候选时，识别并控制反向覆盖损失。固定预测中的问题和池容量损伤已有证据，但没有已验证的新方法。下一条可证伪候选是Hlineage：类条件支持是否持续来自同一query血缘，对补入收益与删除风险是否有超出普通分数融合、边际邻域和几何对应的信息。
本轮没有以现象直接指定网络模块/损失；也没有把一条阴性当成停止整个探索的理由。已经依次关闭多条具体解释，保留其全部协议和失败结果。已有问题也可由新方法形成创新；本轮不以“问题无人研究过”为门槛，也不以给经典图/排序换名称为创新。

一、任务和数据口径
任务为VisDrone2019-DET-val转换后的COCO式bbox检测评价，548图、38759GT、10类；不是VisDrone挑战官方评价。原TXT忽略区/其他类别与转换注释的区别沿用原审计记录。模型是BASE24两seed同一保存checkpoint的raw和ema.module，last_epoch23，即24轮；不能外推为132轮结果。
300query，native HGNetv2-B0/HybridEncoder/DFINETransformer。final分数已经含LQE。native FP32坐标按cxcywh→edges→原图缩放→xywh；sigmoid300×10，展开top300，严格score>.001，无NMS，每图每类最多100。pre来自首decoder层之后、该层LQE之前，不是中间encoder层；与final共享300query血缘。
本轮新评价全部用内存GT id+1克隆消除原GT0与未匹配哨兵混淆，输出恢复原ID。原注释不改。单交换A原生TP各10个IoU计数及.50/.75精确GT集均重现；A新AP与历史original-ID0 AP的微小差异单独保存在其results中，不混入同协议对照。

二、完整失败地图，不预设宽度/位置/分配是主因
每行是全部38759GT的互斥分区（IoU.50）。T=实际命中；G=当前300框无合格几何；S=有几何但真类floor资格无合格项；K=有floor合格真类候选但top300不含；C=导出有合格真类候选但class100后无；M=cap后有边却未实际命中。
{table(['条件','图像','GT','T','G','S','K','C','M'],maps)}
M是逐GT状态，并不等于可净恢复的同池匹配基数差。G只表明该当前几何池无边，不能归因为宽度、特征或query分配。G仍是平行开放问题；当前读出研究不会宣称能恢复不存在的框。
同后端raw→EMA逐GT（各seed全部GT，IoU.50）仍保留：
{table(['seed','GT','recovered','persistent_miss','reverse_loss','both_hit'],transition)}
raw/EMA是同GT、同seed的两个保存状态，不是四个独立训练seed。queryID只在同一状态内追踪；跨状态不把同号query当同对象。

三、原保留效应和竞争解释
A=原生300pair；U=A导出后的不同query集合；B=只在U内取600pair；C=全部300query中取600pair。B-A增加已有query上的类别假说；C-B还改变query身份、类别构成、cap和匹配，不能当作纯单变量病因贡献。
下面AP/APs均是同positive-ID评价口径的绝对百分点：
{table(['条件','GT','BA_AP_pp','BA_APs_pp','BA_TP50净增','CB_AP_pp','CB_APs_pp','CB_TP50净增'],bases)}
原B-A .50恢复raw1182/1223、EMA1173/1225，反向丢失均0；C-B新增/丢失raw457/196、468/208，EMA515/186、472/224。分母均38759GT，不把新增组分母当全部GT，也不把600导出数当cap后预算。C-B收益很小，不能由图上界承诺论文效果。

已排除或收窄的命题
1) “每类100截断是C-B收益必要原因”关闭：解除为600后四条件仍同时有正AP/APs及净TP50；cap600的新/失raw586/184、596/191，EMA670/174、608/223。原ARs负向对cap敏感，不代表所有反向丢失都由cap造成。
2) “更大覆盖物理上必须600输出”在当前匹配图关闭：固定300预算满池上界可实现匹配；总300在4384图条件不绑定。它是有GT、有利匹配/保留的图上界，不是实际score贪心或AP可达到的保证。
3) “类序前缀占满300使更好覆盖硬不可达”在本固定eligible宇宙松界关闭：4384个条件全有相等构造证书，最大前缀闭包215pair。不能据此说实际质量校准已最优，或整个训练方法受同样表达约束。
4) 同池贪心差真实但有限：.50 A/B/C缺口raw1=47/45/46、raw2=40/47/46、EMA1=52/54/52、EMA2=59/59/59；.75分别全0/全2/全1/全2。C-B净TP50与同池容量差依次为261/262、260/259、329/327、248/248，差由缺口变化+1/−1/−2/0精确补齐。这限制的是基数解释，不排除AP排名问题。

四、正机制证据：确实删掉了不可替代候选
“关键pair”严格指：删除该query-class节点会降低固定满池GT图的class100/总300匹配上界；不是某次最大匹配恰好用了它，也不保证固定对应某一个GT身份。
{table(['条件','GT','A实际TP50','A同cap池最大值','满池同300预算图上界','A漏关键pair_导出','A漏关键pair_cap后','满池全部关键pair'],bounds)}
导出/cap遗漏关键数量是图容量证书，不是独立病因比例。非单节点关键的共同删除仍可损伤容量，不能由单点结果否定组瓶颈。
raw实际匹配见证（分子关键pair，分母该新增/丢失/共同命中GT）：
{table(['seed','IoU','C新增关键','C新增分母','C丢失旧B关键','C丢失分母','共同命中C关键','共同命中分母'],critical)}
EMA此处没有同等历史实际pair映射，本轮未补造对应分子。新旧保留池容量变化是可核查读出机制；为何模型给出这些分数/几何、是否由特征/训练/标注造成，尚未确认。

五、实际推进过的信息/作用检验
第一条：固定pre证据单位的容量代理，关闭。
保持原floor、一次类内.5去重、final→pre同类IoU≥.5、class100及总300，全部2192/2192图的代理总容量饱和为300，单删关键量恒0。四条件B∪C域385462/386198/388543/389322pair均无代理关键项；对应真实关键10753/10997/10263/10420。GT-qualified且容量未截断机制层为空，不能计算CI，不能写“统计显著无效”。关闭的是这个截断删点代理，未否定pre所有信息；没有取消预算/调门槛救假说。

第二条：pre赢家类别身份，有有限的资格信息，不能当作不可替代性。
只看明确命名的后续B-A新增pair域，各164400pair。唯一predictor为pre_argmax(q)==c。M2固定final分数、类rank、max、margin、entropy与final邻居、pre几何支持粗分档；有两种predictor的可比格才参与条件差，不合并挑有利格。
{table(['条件','B减A_pair分母','pre赢家pair','pre赢家合格50','其余pair','其余合格50','M2可比pair','M2可比图','M2可比分档','M2合格率差_pp','图像bootstrap95_pp'],cs)}
“合格”仅指final框对真类GT有IoU≥.5边，不是实际TP/AP。该条件差只适用于表中可比格；图像bootstrap500次描述同val图的不确定性，非四独立seed证明，也未排除完整方法/细粒度未控变量。
raw实际B-A新增TP中，仅59/1182、63/1223的匹配pair是pre赢家；共同命中参照19190/23476、20238/24182。绝大多数新增不能用“恢复早层赢家”解释。
更关键，GT-qualified B∪C内的critical50条件差四条件全负：
{table(['条件','合格者关键性M2差_pp','关键性95_pp'],cs)}
EMA1区间整体负；其余不能说确定反向，但没有支持将pre赢家当稀缺性保护标记。

第三条：固定每图≤1次、同类同分数档交换，作用检验失败并关闭。
H仅从B-A补pre赢家，删A中pre非赢家；类原数≤100，确保实际cap可见；300总数、各类数、cap后数完全相同，框/分数不改。S用相同删除项、类、分档和交换数，按final最高分补入作参照，它只隔离该条件下补入依据，不是完整不使用pre的算法。
{table(['条件','图像分母','实际交换图','H和S补入不同图','GT分母','H_A恢复50','H_A丢失50','H_A_AP_pp','H_A_APs_pp','H_S_AP_pp','H_S_APs_pp'],ss)}
四条件H-A AP/APs均下降，联合门槛失败；H-S也不稳定。raw seed2虽净TP50+6，AP仍下降，不能只挑TP或另一个IoU宣称成功。停止此固定交换规则，不临时改交换数/分档/观察点，也不撤销资格信息的独立结论。

六、已有方法与尚未证实的缺口
LQE已在当前代码中；Rank-DETR已有质量与排名对齐；Learning NMS/MDS已有候选上下文/末段重复与分类处理；DDQ逐精炼选择不同query；LDDP已有质量、空间/标签语义、多类全局子集及输出数上限；经典matching/transversal matroid秩不是新数学。因此“质量排序”“跨层支持”“类别+集合+预算”均不能直接作创新名目。
CLSC原p2–3 Eq1–5：跨encoder局部TopK几何/质量支持与final max、margin、entropy可靠度。其逐框汇总与本次类条件支持血缘检验有操作差别，但整套训练可以改变表示，不能宣称它无法利用相应信息。
QR-DETR已有query跨decoder可能向正确或错误对象/类别演化的观察与pass/bypass；Align-DETR讨论层间监督匹配；DETRDistill已有逐decoder分类语义传递。Cascade R-CNN已有多stage classifier概率平均。“类别历史”两个字和多一次类别预测都不是独特新意，pre_argmax是完整pre向量的确定函数，普通融合是必须面对的竞争解释。
当前文献支持的是一个未决比较：在各层质量、类别、空间边际和简单分数融合相当时，“支持是否持续来自同一血缘”是否额外分辨补入/删除真实容量；没有证据说所有现有方法都不能解决，也未完成完整方法复现或穷尽检索。
主要原文链接及实际阅读范围：
CLSC https://arxiv.org/abs/2608.21457v1 （本对话实读提取p3及伙伴页锚；不声称根对话完整通读）。
LDDP https://openaccess.thecvf.com/content_cvpr_2017/papers/Azadi_Learning_Detection_With_CVPR_2017_paper.pdf （伙伴实读p3–5）。
Learning NMS https://openaccess.thecvf.com/content_cvpr_2017/papers/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.pdf （本对话提取p3–4；伙伴页锚更多）。
Rank-DETR https://proceedings.neurips.cc/paper_files/paper/2023/file/34074479ee2186a9f236b8fd03635372-Paper-Conference.pdf （本对话提取p3，伙伴p3–5）。
Cascade R-CNN https://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf （伙伴官方索引Table2/§5.4；未取到本地全文，明确无PDF SHA）。
DETRDistill https://openaccess.thecvf.com/content/ICCV2023/papers/Chang_DETRDistill_A_Universal_Knowledge_Distillation_Framework_for_DETR-families_ICCV_2023_paper.pdf （伙伴官方索引§4.1/§6；整篇403，不称完整阅读）。
其他原件、版本、下载限制、SHA和页锚见“讨论伙伴_候选保留主线取证_20261027_v1”的sources各账本及claim_to_source_v3.csv。伙伴原文审读与本对话实际阅读分开记录。根对话没有复现这些整套检测方法。

七、问题卡01（本轮只保留一张，不凑满三张）
问题是什么：有效候选保留有覆盖增益，但同一次替换会破坏已有不可替代覆盖。逐候选“合格性”不足以决定安全交换。
影响多大：实际C-B四条件净TP50+261/+260/+329/+248，伴随丢失196/208/186/224；AP增量约.021–.037pp。图上界差更大但不是预期效果。全GT几何G仍大，候选读出不能替代几何问题。
证据到哪一步：问题存在、固定预测池/筛选机制及有限资格信息；一条预几何代理和一条具体交换已失败。未到已验证创新/部署收益/训练因果。
还缺什么：同时预测补入真实容量增量和删除真实容量损失的GT-free证据；普通层融合/几何对应/既有集合方法参照；相同真实预算正向作用；独立数据与完整训练模型确认。
最小零训练检验：Hlineage，当前为已冻结计划、未实现/未运行。先信息后作用，未通过前不提网络或损失。

八、Hlineage：下一条具体、可证伪的创新假说
对final(q,c)，Npre取其他query中pre赢家为c且pre框与final_q IoU≥.5者；Nfinal取其他query中final赢家为c且final框与final_q IoU≥.5者；J为两个ID集合交集数。支持集合先冻，query血缘不当作对象身份。
主标签分开：同一实际A cap后池P的删除容量损失r(P)-r(P-d)，与B-A候选的增点图容量r(P+f)-r(P)。共同限原A该类数≤99让增点在class100可见；后者明确是多加一节点松界，不是固定300的安全交换。两侧GT-qualified层是机制主分析，避免只识别FP/TP资格。
负对照只置换身份连线，不打乱数组后重算邻居。弱全局置换仅诊断；主强置换保持final类别×4x4空间×置信档，且固定目标q；保留各层全部框/分数/类别与邻居数量。再与固定mean/max类别融合和class+IoU几何对应比较。若强置换无可动项就记不可辨识，不放宽网格找阳性。
门槛：真实对应必须在两个seed、raw/EMA的补入和删除两侧均超出强置换及几何对应的预定信息判据；它仍只是有限控制的信息关。空邻域、重复背景簇高J、孤立关键候选J0、正确晚层修正低J等均是直接反例。只一侧有效、只预测资格、或信息被普通融合/边际/几何解释，就关闭本版，不改定义维护答案。
原文近邻若已有等效证据，也关闭这一个新意版本；问题本身仍保留。只有信息关和近邻检验支持后，再冻结真正同300预算联合交换及native AP验证，随后才进入方法设计。
完整冻结定义、置换、标签/比较/资源见Hlineage_下一最小零训练检验_冻结计划.json；SHA {sha(HERE/'Hlineage_下一最小零训练检验_冻结计划.json')}。

九、交付与边界
本轮7项缓存探查均完成且核验PASS，另补全条件支持分母；8次cap评价+12次单交换评价=20次CPU COCO评价，0新模型/头前向、0权重加载、0训练、0GPU/远程/test读取。旧数据/权重/代码和失败材料保留。
假说状态账本.json区分关闭/受限支持/未执行；证据入口与协议绑定.json索引各协议SHA和结果核验；CSV含全部实际计数和分母，逐GT/精确ID/证书在各attempt01。既有raw-EMA与C-B人群未被后续pair域覆盖。
本轮成果是问题地图、1张问题卡、原因反证链、原文方法边界及下一条可证伪候选主线；不是“创新已经成立”的结论。没有需要用户补批的动作，也没有因某个失败停止其他探索。
'''
 write('00_当前主线候选_问题卡与最小验证.txt',text)
 jout('delivery_checks.json',{'status':'PASS','completed_probes':len(evidence),'completed_probe_verifications_all_pass':True,'Hpre_stopped':True,'Hclass_qualification_gate_only':True,'Hswap_stopped':True,'next_Hlineage_executed':False,'cards':1,'COCO_evaluator_calls':20,'new_model_forwards':0,'training':0,'inputs_unchanged_per_probe':True,'old_cohorts_preserved':True})
 print('Delivery built',len(text),'Chinese/text characters;',len(evidence),'completed probes,1card; Hlineage NOT EXECUTED')
if __name__=='__main__':main()
