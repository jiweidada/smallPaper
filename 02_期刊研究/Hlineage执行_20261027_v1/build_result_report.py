"""Honest negative-result report and current-state update."""
import json,csv,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OLD=ROOT/'候选保留_机制与创新假说_20261027_v1';PAIR=ROOT/'raw_EMA逐GT对照_20261026_v1'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,s):
 with open(p,'x',encoding='utf-8') as f:f.write(s)
def js(p,j):write(p,json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 z=read(HERE/'attempt01/results.json');audit=read(HERE/'audit_v1/results.json');parity=read(HERE/'argmax_parity_v1/results.json');verify=read(HERE/'attempt01/verification.json')
 assert z['status']=='COMPLETE' and verify['status']==audit['status']==parity['status']=='PASS' and not z['strict_information_advance_gate_pass']
 rows=[]
 for k,v in z['conditions'].items():
  p=v['primary_qualified'];a=audit['conditions'][k];c=v['full_counts']
  rows.append({'条件':k,'原图分母':548,'原GT分母':38759,'候选pair全域':c['candidate_pairs'],'合格pair分母':p['qualified_pairs'],'合格图像数':p['images_with_qualified_pairs'],'容量增损1':p['label1'],'容量增损0':p['label0'],'可辨识pair':p['informative_pairs'],'可辨识图':p['informative_images'],'可辨识正例':p['positive_labels'],'可辨识反例':p['negative_labels'],'真实CMI_bits':p['real_CMI_bits'],'几何CMI_bits':p['geometry_CMI_bits'],'强置换95_bits':p['strong_null95_bits'],'弱置换95_bits':p['weak_null95_bits'],'强置换全部199均高于真实':not a['real_exceeds_strong_null_min'],'可行性过关':p['information_feasible'],'推进过关':p['advance_gate_pass'],'真实和几何J不同pair':a['real_geo_J_difference_pairs'],'真实和几何J相同比例':a['real_geo_J_same_fraction']})
 with open(HERE/'Hlineage_全部八域计数与比较.csv','x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 simple='\n'.join(' | '.join(str(r[k]) for k in ['条件','候选pair全域','合格pair分母','容量增损1','容量增损0','可辨识pair','可辨识图']) for r in rows)
 stat='\n'.join(f"{r['条件']} | {r['真实CMI_bits']:.8f} | {r['几何CMI_bits']:.8f} | {r['强置换95_bits']:.8f} | {r['弱置换95_bits']:.8f}" for r in rows)
 geometry='\n'.join(f"{r['条件']}: {r['真实和几何J不同pair']}/{r['合格pair分母']} pair不同；{100*r['真实和几何J相同比例']:.3f}%相同。" for r in rows)
 text=f'''本轮结论：没有找到可保留的创新主线
2026-10-27。此文件更新上一轮“下一条仅有计划”的状态；旧封存报告/计划/失败结果不改写。

先回答停止理由
上一轮在Hlineage计划冻结后结束，是停早了；当时没有权限、资源或用户要求上的阻碍，也没有找到已验证主线。用户要求实事求是后，本次实际完成Hlineage及两项实现/数值审计，再与“两个检测问题的研究讨论”继续讨论。
现在可以成立的结论是：本轮没有找到同时接通机制、实际作用和现有方法差异证据的创新主线。问题真实存在，但当前具体候选没有足够证据推进。结束本轮的理由是这一实际研究结果；不能把“统计做完”或“计划写好”叫找到主线，也不能把没找到说成项目不可能创新。

一、Hlineage已执行，全部八域未过预定关口
原计划SHA62f105a7fd24394a1034798cff261fc5905d0aded4da349b03f660fbc35a6e04；执行规格SHA{z['protocol_sha256']}。执行前已固定分档、邻域、399项中的199强/199弱加1几何对应、标签、同一CMI公式、1e−12bit数值容差及可行性门槛；没有看到结果后调阈值。
548图/38759GT，BASE24两seed分别raw/EMA，2192缓存。2种方向的分母是候选pair，不能当作GT数或把两个保存状态充作四seed。真实J为pre/final类条件邻域支持ID交集，query血缘不等于对象身份。
删除标签为同一原生A实际class100池P的r(P)−r(P-d)；补入为r(P+f)−r(P)，共同限定原A该类导出数≤99。后者是增一节点的图松界，不是固定300安全交换/AP。两侧主分析仅在GT-qualified pair内，GT只作离线机制标签，不进入可见特征/候选资格。

条件 | pair全域 | 合格pair分母 | 容量增损1 | 容量增损0 | 可辨识pair | 可辨识图
{simple}
每一原始条件的母体仍为548图/38759GT；表内合格图数、正反分母、全部分档及覆盖量在CSV/results中保留。“可辨识”要求同一固定控制格内真标签和真实J均变化。执行前可行性门槛：至少100个此类pair、20图、正反各10；这是保守工程门槛，不是功效证明。

CMI比较（bits，同一控制编码、同一分母；单例/常数格贡献0，未筛格挑正）：
条件 | 真实对应 | 几何对应 | 强置换95分位 | 弱全局95分位
{stat}
八域均未过强参照；真实值甚至低于各自全部199个强置换样本。四删除域和EMA seed1补入域具备预定可行支持，所以不能将所有失败归为样本不足。raw seed1/2补入只有48/20可辨识pair，正例9/7；raw seed2仅19图，EMA seed2补入81pair，也未过可行性关。
真实对应在八域均超过弱全局95分位，但原主关口是保留final类别、粗空间和置信的强置换，不能用弱阳性维持假说。raw seed1/2补入及EMA seed2删除还未胜几何对应。

二、哪些是观察，哪些不能推断
观察：本固定J/控制格/参照定义没有通过双向信息推进门槛；三个补入域还缺可辨识支持。两seed结果未给出可推进本版的依据。
不能推断：随机对应更好；所有血缘信息为零；queryID恒等于对象；强置换95分位就是校准P值/置信区间；既有训练方法不能表达该信息。
插件CMI受细格占用和J分档变化影响，强置换并非实验随机分配。完整pre/final向量虽然保持不变，CMI只条件于预定粗档，未排除更细分类信息。结论限定为该假说不推进，不能包装成普遍无效定理。

三、实现与不可辨识性审计已经结清
1. 原A实际cap后图容量2192/2192逐图精确重现；1,180,939个候选精确强置换J上下界校验，136,967次真实边际匹配检查；全部增/损标签为0或1。
2. 代码用sigmoid后argmax而计划写logit argmax，已单独对2192缓存×pre/final=1,315,200个query-stage核验：差异0、最高值并列0，因此本批实现与计划等价。未静默改代码补结果。
3. 八域全部(C,J,Y)整数计数重算，CMI/条件熵与原结果误差≤1e−12。没有任何域满足几何标签全纯的熵上界恒等证书，也没有real/geo全联合表相等。不能谎称这个检验数学上处处不可能；这是实测比较未过。
4. 真实J与几何对应J大部分相同，但非完全相同，具体计数如下：
{geometry}
上述只说明本信号与几何参考的区分量有限，不证明完整方法等效。精确强置换可动项、单例格、标签混合格、熵上界和占用量均在原始结果/审计保留，没有仅打印一个FAIL。

四、放回完整证据链，为什么目前没有主线
已证实问题：原候选保留既补回也删除真实GT匹配容量；原C-B和raw-EMA全GT表保持，未用本次pair域替换。既有质量/排序/集合选择研究过相关问题，本身不妨碍新方法创新。
原因已收窄：cap100必要性、固定图300预算硬不可达/类序硬不可达未成立；同池贪心基数变化不足以解释C-B主增量；筛选池容量变化是真实读出机制，训练根因未确定。
已失败的具体候选：pre几何容量代理在2192图容量饱和为300、单删关键量恒0；pre赢家只有有限资格信息，关键性未获支持；固定同类单交换四条件AP/APs均降；本次血缘交集的双向信息关也未过。
现有方法差异尚未接通：CLSC、Rank-DETR、Learning NMS/MDS、DDQ、LDDP，以及普通stage分类融合/QR/层间知识传递，已覆盖宽泛质量、上下文、集合、层间类别说法。本次没有显示足以将当前候选从这些原则中区分出来并产生实际作用的证据；没有复现全部方法，也不宣称文献穷尽或没有其他方法。
我目前没有下一条由这些结果支撑、操作与参照都明确的具体创新假说。继续给同一失败信号换名字、放宽条件或先想模块，会偏离问题优先的要求。因此本轮如实结束为“未找到可保留主线”。这不关闭已观察的候选保留问题、全GT几何G问题或未来其他证据驱动探索。

五、当前问题卡与状态
仍保留1张“候选保留收益与反向覆盖损伤”问题卡，状态更新为：问题有证据，创新主线未找到。Hlineage状态已从“冻结未执行”更新为“已执行，未过关；域稀疏与实测失败分开报告”。没有新方法候选进入训练阶段。
全GT地图/旧问题卡/文献账本与上一轮结果见：{OLD/'00_当前主线候选_问题卡与最小验证.txt'}。其中关于Hlineage尚未执行的文字是旧快照，当前状态以本文件及attempt01为准。
本次运行196.25秒，全部源核验PASS、2206输入未变；0模型/头前向，0新评价、0训练、0GPU/远端/test。分母CSV、每候选特征、真实/几何/398次置换数组与两项审计均保存。没有把当前研究结论归因于权限或资源限制。
'''
 write(HERE/'00_当前结论_未找到可保留主线.txt',text)
 decision={'client_date':'2026-10-27','current_round_result':'NO_SUPPORTED_INNOVATION_LINE_FOUND','research_problem_retained':True,'new_method_established':False,'why_current_round_ends':'The concrete candidates tested did not connect mechanism, useful readout and evidence beyond nearest-method principles; no further specific hypothesis is currently supported. This is not proof that the project has no innovation opportunity.','Hlineage_status':'EXECUTED_DID_NOT_ADVANCE','all8_comparisons_failed_gate':True,'sample_infeasibility_domains':['raw_s1_insert','raw_s2_insert','EMA_s2_insert'],'feasible_but_no_advance_domains':['raw_s1_delete','raw_s2_delete','EMA_s1_delete','EMA_s2_delete','EMA_s1_insert'],'argmax_parity_PASS':True,'entropy_and_count_audit_PASS':True,'training':0,'parent_snapshot_preserved':str(OLD)}
 js(HERE/'当前研究状态.json',decision)
 card=read(OLD/'问题卡_01_候选保留.json');card['current_status']='Problem supported; no supported innovation line found this round';card['previous_minimal_test_completed']=str(HERE/'attempt01/results.json');card['current_minimal_next_test']=None;card['previous_minimal_zero_training_test']=card.pop('minimal_next_zero_training_test');card['innovation_hypothesis']='None retained this round. The specific Hlineage definition has been executed and did not pass the frozen bidirectional information gate; sparse insertion domains are reported separately.';card['evidence_stage']='Problem and fixed-pool mechanism supported; qualification information limited; concrete readout and Hlineage information gates not passed; no verified new method.';card['reason_no_further_proposed_test']='No next concrete innovation premise currently grounded in these results; do not invent a module or rename a failed proxy.';card['evidence_update']=decision;js(HERE/'问题卡_01_状态更新.json',card)
 statepath=PAIR/'work_state_v2.json';previous=read(statepath);js(HERE/'work_state_before_this_execution.json',previous)
 current=dict(previous);current.update({'version':4,'client_date':'2026-10-27','latest_user_instruction':'找不到主线就用找不到主线的理由停止；实事求是。','current_round_status':'CONCLUDED_NO_SUPPORTED_INNOVATION_LINE_FOUND','current_objective':'Report the actual completed negative evidence chain honestly; retain supported problems without asserting a method/innovation.','current_authority':str(HERE/'00_当前结论_未找到可保留主线.txt'),'current_problem_card':str(HERE/'问题卡_01_状态更新.json'),'current_supported_innovation_line':None,'completed_current_stage':'Hlineage actually executed all2192 caches/eight domains; exact rank, argmax and entropy audits PASS; no domain passed predefined information gate.','root_execution_this_stage':{'CPU_evaluator_calls':0,'new_model_forwards':0,'head_calls':0,'training':0,'GPU':False,'remote':False,'test_reads':0,'analysis_seconds':196.25},'next_hypothesis':{'id':'Hlineage','status':'EXECUTED_DID_NOT_ADVANCE','result':str(HERE/'attempt01/results.json'),'verification':str(HERE/'attempt01/verification.json'),'no_new_hypothesis_advanced':True},'next_independent_work':[],'previous_state_snapshot':str(HERE/'work_state_before_this_execution.json')})
 current['discussion_status']='Actual results delivered to existing research chat; finishing scientific-limits discussion and archive. No new experiment pending.'
 statepath.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('Actual negative-result report built; current authority and status updated. No supported main line claimed.')
if __name__=='__main__':main()
