"""Synthesize completed external discussion plus separately verified primary sources."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import json,hashlib,html

OUT=Path(__file__).resolve().parent
EN=OUT.parents[1]
d=json.loads((OUT/'讨论原始导出_三轮完整.json').read_text(encoding='utf-8'))
assert d['id']=='99accfb2-292b-458c-bf58-96641b4b5d34'
assert d['status']=='completed' and d['error'] is None and d['rounds']==3
messages=sorted(d['messages'],key=lambda m:(m['round'],m['expertId']))
assert len(messages)==12
contexts=[]
byid={m['id']:m for m in messages}
for r in (1,2,3):
 rr=[m for m in messages if m['round']==r]
 assert len(rr)==4 and len({m['expertId'] for m in rr})==4
 for m in rr:
  seen=m.get('seenMessageIds',[])
  assert len(seen)==4*(r-1)
  assert all(byid[mid]['round']<r for mid in seen)
  contexts.append({'round':r,'expert':m['expertName'],'seen_previous_answers':len(seen),'same_round_visible':False,'served_model':m.get('model'),'served_provider':m.get('provider'),'served_effort':m.get('reasoningEffort'),'requested_settings':m.get('requestedSettings')})
(OUT/'十二份完整回答_按轮与席位.txt').write_text('\n\n'.join(f"第{m['round']}轮｜{m['expertName']}\n{m['text']}" for m in messages),encoding='utf-8')

rows=[
 {'id':'R1','question':'容量内联合匹配是否产生有实际代价的GT直接几何监督覆盖流失？','asset':'项目冻结_get_go_indices按(query,GT)频次排序、每query只留一个GT；boxes/FGL复用统一索引。既有DN元数据和阶段诊断可复用，但不是实际训练匹配历史。','closest':'D-FINE §4.2/PDF6页及作者_get_go_indices；Stable-DINO §2.1–2.3；Align-DETR §3.2–3.3/PDF4–6页。','known':'联合跨层监督、跨层目标不一致、位置质量监督和多正样本均已有先例；按query合并的规则来自作者实现。','possible_increment':'仅为待检验假设：N≤Q且每分支全覆盖时，聚合后的逐GT覆盖缺口是否真实出现；在控制尺寸、匹配质量和稳定性后，是否仍能预测后续失败。不是发现跨层不稳定，也不是宣称新聚合算法。','strong_simple_baseline':'各层自身Hungarian、末层Hungarian、当前频次规则；覆盖量和对应质量同时比较，不能只看覆盖个数。','no_training_first_check':'本轮已补原文和作者函数AST对照，并盘点了指定归档。下一步若开展，只能先做隔离的冻结权重分配审计（不更新参数）；它生成新诊断，不能伪装成历史训练重放。本轮未执行。','stop':'仅有组合反例；真实输入不发生或仅在N>Q时发生；效应完全由低匹配质量/稳定性解释；近邻等价覆盖且无额外预测；没有时间对应记录却要推断历史损害。任一项成立则停止升级。','decision':'唯一保留的问题核查候选；没有通过可训练立项或创新成立门槛。','sources':['https://arxiv.org/html/2410.13842v1#S4.SS2','https://raw.githubusercontent.com/Peterande/D-FINE/master/src/zoo/dfine/dfine_criterion.py','https://arxiv.org/html/2304.07527v2']},
 {'id':'R2','question':'独立标注分歧能否预测小幅模型差异的排名不稳？','asset':'少量漏标/车型边界疑点及小幅配对差；疑点由模型高分筛选，不能估计全体频率；没有独立双人复标。','closest':'Object Detection Benchmarks are Incomplete，arXiv2609.21822v1（2026-09，公开预印本）；COCO-ReM ECCV2024。','known':'多标注者、软标签、重新评价和排名变化均已有近邻。2026预印本还报告分数显著改变而排名大体稳定；COCO-ReM主要是mask评价，不与本项目bbox直接混同。','possible_increment':'只有超过原始分差/图像重采样/随机复标的留出预测价值才可能有增量；“改标注会改AP”不够。','strong_simple_baseline':'固定官方协议下随机盲复标+直接重评，原始分差与图像重采样。','no_training_first_check':'先核独立复标人力和同协议多模型逐框输出；当前没有这些材料，本轮未复标。','stop':'只有修标签后AP变化；不能预测留出模型配对；没有超出已有近邻；缺独立复标。','decision':'Codex C首轮提出，第二轮后主动撤出优先序；主助手补查原文后仍不建议立项。','sources':['https://arxiv.org/html/2609.21822v1','https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06053.pdf']},
 {'id':'R3','question':'按目标数量或尺寸限制Mosaic/MixUp能否成为新主线？','asset':'旧132日程只完成28份增强输入，3份>300GT；7批3860标签出现全部有正DN槽。不是当前96日程完整分布。','closest':'DEIM v3补充Table12/PDF12页；Scale-aware AutoAug §3.2–3.3/PDF3–5页；SNIP与Scale Match；DEIMv2 Copy-Blend。','known':'DEIM已报告平均约10/25/50目标对应AP51.7/52.5/52.2；密度非单调、尺度适配和尺度收益代价平衡均已有。约50时掉点不证明超过300query。','possible_increment':'若仅调整开关/数量/缩放，当前无法形成实质增量。更具体的监督覆盖问题已归R1，不另拆一条题目。','strong_simple_baseline':'现成增强日程与简单数量/尺寸控制，不应先写新模块。','no_training_first_check':'不重启失败的128输入回放，不修改正在运行的配方。','stop':'只复现增强强度非单调、尺寸变化或额外监督收益，即停止新颖性主张。','decision':'淘汰为当前新主线。','sources':['https://arxiv.org/html/2412.04234v3','https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Scale-Aware_Automatic_Augmentation_for_Object_Detection_CVPR_2021_paper.pdf','https://arxiv.org/html/2509.20787v4']},
 {'id':'R4','question':'用定位质量、框尺寸或分布统计重新评分？','asset':'第36轮同召回下C额外FP729/475；两seed总体负项不同；原生greedy匹配改为oracle的AP70空间极小。已有MAL与LQE。','closest':'GFLv2 §3.2；Rank-DETR §3.2–3.4；Multivariate Confidence Calibration §3；Cal-DETR §3.3。','known':'分布统计→IoU质量→重排、DETR排序损失、位置/尺度条件化校准、不确定性调制均已有。','possible_increment':'当前没有证明超出这些方法的新问题或可部署信号；不能用GT oracle加分。','strong_simple_baseline':'已有IoU质量评分/条件校准；若所有排序与选择均不变，全局单调变换不改变AP。','no_training_first_check':'已完成的覆盖/评分诊断不重新作为待执行实验。','stop':'只是常规重排、只降ECE不改目标性能、或仅GT可用时有效，就淘汰。','decision':'淘汰为当前新主线。','sources':['https://arxiv.org/abs/2011.12885','https://arxiv.org/abs/2310.08854','https://arxiv.org/abs/2004.13546','https://arxiv.org/abs/2311.03570']},
 {'id':'R5','question':'密度自适应query或选择性高分辨率读取？','asset':'有大图、阶段特征工具与UAVDT视频，但val最多30GT/图、300输出不支持硬数量短缺；新高分辨率路径没有独立瓶颈证据。','closest':'DQ-DETR；LiG-DETR arXiv2610.09511v1 §3.2–3.4；PaQ-DETR；DQSA-DETR/MR-DETR仅摘要核实。','known':'计数引导query预算、共享模式/质量分配、全局局部重组、选择性高分辨率编码已有。LiG仅确认公开预印本，PDF未来会议页眉不作录用证明。','possible_increment':'目前只有一般计算预算问题，尚无超出先例的明确新知识。','strong_simple_baseline':'固定query预算、普通裁图/缩放与已有动态query方法；总延迟必须公平计入。','no_training_first_check':'不新增切图或动态query训练；摘要未核细节的风险保留。','stop':'只复现增大分辨率/计算量收益或按密度增query，就停止新颖性主张。','decision':'淘汰为当前新主线。','sources':['https://arxiv.org/abs/2404.03507','https://arxiv.org/html/2610.09511v1','https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PaQ-DETR_Learning_Pattern_and_Quality-Aware_Dynamic_Queries_for_Object_Detection_CVPR_2026_paper.html']},
]
(OUT/'新主线候选对照表.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

report='''四专家三轮查新与问题筛选：结果与下一步边界

结论：只留下1项值得做不训练问题核查的候选；0项已具备开训立项条件，0项已证实创新。它针对基线本身的训练分配规则，不以挽救C为目标。没有修改现有UAVDT计划或任何运行中的任务。

本轮确实使用了用户的本地圆桌软件：会议99accfb2-292b-458c-bf58-96641b4b5d34，三位Codex、一位DSH，独立互评3轮，每轮4份，共12份完整回答，最终completed、error=null。每轮只看到前轮回答，已按导出seenMessageIds核对为0/4/8份，同轮互不可见。请求配置是同一GPT-6.1-Sol家族和xhigh；DSH是适配器名称，不代表DeepSeek。实际served model/provider/effort字段为空，不能把请求设置写成服务端已核实型号，也不能将四席意见当成四模型独立证据。

查新分工必须说清：专家收到共同包中的原文核查摘录，均声明未另读完整原文。资料包由主助手逐段核对。本轮最终表还纳入了讨论启动后主助手补查的D-FINE§4.2、作者源码、DEIMv2、PaQ、Align-DETR、Stable-DINO、EASE及标注近邻，不能声称四位已分别复核这些追加材料。

优先候选是：目标数未超过query容量时，跨分支联合匹配按query择一，是否导致部分GT失去boxes/FGL直接几何监督；这种现象是否在真实输入中发生，并在控制尺寸、匹配质量和稳定性后仍有失败预测价值？

原文和代码定位已经补齐。D-FINE出版社PDF第6页§4.2明确提出跨层Hungarian结果合为联合监督集、分类仍走O2O；作者公开criterion的_get_go_indices把(query,GT)按跨分支出现次数排序，每个query只保留一个GT。主助手保存web原始行223–243并与项目冻结源码第500–519行做AST对照，忽略docstring后的逻辑相同。因此不能把这项规则说成本项目新发现的算法，也不能先叫它bug。GitHub commit API及直接源码下载遇SSL问题；最终依据是已保存的web工具快照与提取函数，未伪称取得固定Git commit或完整源码下载SHA。

同样重要：Stable-DINO已研究跨层不稳定；Align-DETR（BMVC2024版本）§3.2–3.3/PDF4–6页明确讨论跨层目标错配及mixed matching；RT-DETRv3、PaQ-DETR已有密集/均衡监督。单说“匹配不稳定”“小目标监督不足”“加密集头”都不构成新主线。候选仅剩更窄的GT覆盖与质量取舍问题，近邻差异尚不足以证明可发表的新方法。

专家B给出了不依赖并票的纸面反例：6个query、4个GT(a,b,c,d)，三分支分别为
 [a,b,d,–,–,c]
 [a,d,c,–,b,–]
 [d,b,c,a,–,–]
各分支均覆盖四GT；按每query最高频次聚合，结果为[a,b,c,a,b,c]，d消失。它证明该规则在容量内不保证逐GT覆盖，不是训练实测、更不是伤害证明。实际参与分支和配置需另核；当前框/分类/去噪及共享特征梯度仍可能对该GT有作用。尤其DDF对未选query仍可能有蒸馏，不能把boxes/FGL直接GT目标缺口扩大为“全网完全没学”。

现有资产能做什么，不能做什么：本轮检查的旧selection recorder明确只允许训练后的eval诊断；其档案是位置TopK、锚点/分数、阶段最大IoU和query行号。第36轮UAVDT预测只有框、分数、image_id。这些都不能还原优化器时点的逐层Hungarian身份、普通/DN分配和时间对应。检查范围见已核资产字段与适用范围.json；这是对指定资产的核验，不宣称全磁盘/服务器绝无其他归档。

最强反对意见必须保留：聚合减少覆盖，可能是在过滤低质量、不稳定的对应。覆盖更多不等于监督更好。首轮“系统偏向微小GT”的说法在第二轮被撤回；此前3.76px统计属于另一群体，不能拼接成因果证据。旧132配方和当前96配方、训练匹配与评估匹配都必须分开。四位最终都仅支持前置审计，未支持加训练。

唯一合理的后续门槛（本轮没有执行）：先盘点是否另有训练匹配归档；若没有，可另行设计固定现有checkpoint、固定输入且不更新参数的分配审计，比较聚合前后逐GT覆盖与匹配质量，并保留自身/末层Hungarian参照。它是新采集的冻结权重诊断，不能叫原训练历史重放；也无法独自证明因果损害或后续训练收益。不能直接重启旧失败128输入脚本、改当前训练或为了这项假设扩新队列。

如果真实输入不出现、仅N>Q超限出现、差异被质量/稳定性解释、或者只剩几行组合反例，就停止将其升级为论文主线。若真实有独立模式，也要先提出超过现成逐层/末层匹配的明确新增认识和方案，才讨论另行授权的小规模训练。当前没有新训练授权。

另一个首轮候选是“标注分歧预测排名不稳”。Codex C在第二轮主动撤出优先序，原因是缺独立盲复标、同协议多模型逐框材料和最近原文。主助手补查发现，2026-09预印本Object Detection Benchmarks are Incomplete已做四数据集多标注者/软标签/重评，并报告分数变化明显但排名大体稳定；COCO-ReM（主要是mask）也已有重标后排名改变。因此不建议仅凭少量漏标疑点另开此线，更不能把改GT后的AP变化写成模型收益。

本表不是系统综述或正式专利查新，不证明世界上没有同类方法。它的用途是把最接近的已知工作、可复用资产、潜在增量、简单竞争方案和放弃条件摆在一起，避免下一步继续盲目消耗训练预算。R1是问题审计候选；其余项是被筛掉或撤出优先序的路线。

逐项候选对照：
'''
for r in rows:
 report+='\n'+r['id']+' '+r['question']+'\n'
 for k,label in [('decision','去留'),('asset','资产依据'),('closest','原文定位'),('known','原文已做'),('possible_increment','尚待证实的增量'),('strong_simple_baseline','最强简单对照'),('no_training_first_check','不训练步骤与范围'),('stop','停止条件')]:report+=label+'：'+r[k]+'\n'
 report+='原文链接：'+'；'.join(r['sources'])+'\n'
report+='''
交付：讨论原始导出_三轮完整.json/.md、十二份完整回答_按轮与席位.txt保留专家原话；新主线候选对照表.json/.html为主助手综合并加入追加查新；primary_sources*和候选标注方向_补充原文.json保留原件、阅读/下载状态与SHA。第一次LiG下载失败后同一官方URL成功，失败保留；DQSA/MR及一篇2026增强重加权仅摘要/索引、COCO-ReM PDF本地下载证书失败而web原文可读，均未伪称本地全文已取得。没有关闭TLS验证、绕过证书警告或修改系统信任。

本轮新增训练0、模型前向0、优化器更新0、SSH0、GPU操作0、监测管理0、稿件修改0。发生的是本地文档/字段/源码读取、原文下载、浏览器讨论及结果归档。现有实验是否仍在运行，本轮没有远端实时检查，不据旧快照保证。
'''
(OUT/'新主线候选对照与去留结论.txt').write_text(report,encoding='utf-8')

page='''<!doctype html><meta charset="utf-8"><title>新主线候选对照与去留</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1320px;margin:36px auto;padding:0 24px;color:#243631;background:#fafbf8}h1{font-size:26px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d6dfd7;padding:15px;vertical-align:top;text-align:left}th{background:#edf3ed}td:first-child{width:16%}a{color:#216a58}details{margin:10px 0}.lead{padding:18px;background:#eaf1ec;border-radius:8px}</style><h1>四专家三轮：新主线候选对照</h1><p class="lead">完成12/12份真实回答。只保留1项问题核查候选；0项可训练立项，0项已证实创新。本轮未新增训练、模型前向或服务器操作。</p><p>查新由主助手核原文，专家基于共同摘录互评；表中已注明讨论后的追加查新。详细边界和原话见同目录TXT、JSON、MD。</p><table><tr><th>问题与去留</th><th>现有资产</th><th>原文已做到什么</th><th>仍可能新增什么 / 最小步骤 / 停止条件</th></tr>'''
for r in rows:
 links=' '.join(f'<a href="{html.escape(u,quote=True)}">原文{i+1}</a>' for i,u in enumerate(r['sources']))
 page+='<tr><td><b>'+html.escape(r['id']+' '+r['question'])+'</b><p>'+html.escape(r['decision'])+'</p></td><td>'+html.escape(r['asset'])+'</td><td>'+html.escape(r['closest'])+'<p>'+html.escape(r['known'])+'</p>'+links+'</td><td>'+html.escape(r['possible_increment'])+'<details><summary>最小步骤及简单对照</summary>'+html.escape(r['no_training_first_check'])+'<p>'+html.escape(r['strong_simple_baseline'])+'</p></details><p><b>停止：</b>'+html.escape(r['stop'])+'</p></td></tr>'
page+='</table><p>共同意见不替代原始证据。所有候选均未被当作已成立机制；当前不为C追加训练。</p>'
(OUT/'新主线候选对照表.html').write_text(page,encoding='utf-8')

now=datetime.now(timezone.utc)
asset_manifest=json.loads((OUT/'输入核验.json').read_text(encoding='utf-8'))
for asset in asset_manifest['source_assets']:
 assert hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest()==asset['sha256']
verification={'archived_UTC':now.isoformat(),'archived_LA':now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),'meeting_id':d['id'],'status':d['status'],'error':d['error'],'complete_answers_per_round':{str(r):sum(m['round']==r for m in messages) for r in (1,2,3)},'contexts_checked':contexts,'project_input_assets_unchanged':True,'candidates_for_problem_audit':1,'training_ready_candidates':0,'established_innovations':0,'new_training':0,'model_forwards':0,'ssh_calls':0,'automation_changes':0,'primary_source_scope':'Targeted novelty-risk screen, not exhaustive systematic review. Experts cite packet; supplementary primary checks performed by root afterward.','files':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='本轮交付与核验.json'}}
(OUT/'本轮交付与核验.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'meeting':d['id'],'answers':12,'contexts_checked':len(contexts),'result':'one audit candidate, zero training-ready, zero established novelty','report':str(OUT/'新主线候选对照与去留结论.txt')},ensure_ascii=False))
