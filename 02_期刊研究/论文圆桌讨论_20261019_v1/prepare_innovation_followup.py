"""Prepare discussion inputs only; no agent calls, training, or network actions."""
from pathlib import Path
import hashlib
import json

BASE=Path(__file__).resolve().parent
OUT=BASE/'创新点再讨论_20261019_v2'
OUT.mkdir(exist_ok=True)
DATA=BASE.parent/'data/07_第二数据集_20261017'
ANALYSIS=DATA/'同轮趋势核查_20261019_v2/20261020T043654Z/analysis.json'
EARLY=DATA/'独立中期评估_20261018_v1/summary/小目标收益判断_第36轮_20261018.txt'
initial=BASE/'02_共同参考资料_导入TXT.txt'
d=json.loads(ANALYSIS.read_text(encoding='utf-8'))

topic='''重新讨论研究创新点与下一步：依据既有VisDrone全部结果，以及最新UAVDT中期同轮AP70，判断现有C还有什么实质的新知识；若不足，从已有数据出发提出更值得研究、能被低成本证伪的问题。不要预设保住C，也不要把中期AP70负向直接判成最终失败。我们需要选择值得验证的研究问题，不需要再列一批注意力/融合模块。

讨论三轮，三位Codex＋一位DSH。只讨论；不训练、不运行推理实验、不操作服务器、不改代码/稿件/监测。允许只读查文献；不能访问时明确“近邻未核实”。已知SPD/SOEP/HMF-DEIM先例不得忽略，不能用“四区可能能发”代替新颖性判断。

第1轮独立：先判现有主线可保留的最强发现及其局限；每人至多提出2个候选研究问题。每个候选须列：数据中哪项观察触发它、可证伪假设、最强竞争解释、最近方法及与之实质区别（未核实就标未核实）、无需新训练的第一项鉴别分析、足以放弃该方向的结果。可以结论为目前没有可立项的新点。
第2轮：至少回应两位的具体候选，优先淘汰“已有算子改名”“一般信息保留”“靠损坏对照制造收益”“挑有利IoU/尺寸”“只靠复杂模块涨点”的提案。允许修正自己。区分数据支持、合理推测、待查近邻。
第3轮：各自只推荐1个候选（或明确推荐不立项），给一句研究问题、最可能的新知识、最近方法、最低成本鉴别步骤、何种结果停止、若最终UAVDT正/混合/负会怎样影响它。另说明它是在挽救C还是独立的新研究问题。保留分歧，不以多数票认定创新成立。

每轮每人约1000–1600中文字，直接回答问题，引用资料编号和具体比较；不要重复整份资料。主张新颖性需要原文依据。候选建议不等于已授权执行。'''
roles=[
('Codex A｜机制与可证伪问题','Codex','你负责把观察变成可证伪的研究问题。优先检查是否能分开信息访问、最大映射秩与优化；不把代数分解和已有采样算子当新理论。现有C不一定值得保留。候选必须指向一个可测量的失败过程，说明现有数据是否能鉴别；没有证据不要直接把低分归因于分辨率、查询、匹配或增强。最多2个候选，第三轮只留1个或建议不立项。'),
('Codex B｜数据与实验判别','Codex','你负责证据和最小实验设计。核对最新UAVDT AP70同种子同轮差值，与第36轮CPU独评及VisDrone终评严格分开。第36轮小目标AP75正向不等于预设AP70验证成功；反过来AP70负向也不证明所有小目标指标为负。检查时间/训练阶段、阈值及共同检出选择、指标多重选择与种子不确定性。对每个候选提出无需新训练的鉴别分析和停止条件，拒绝没有可判别结果的试验清单。'),
('DSH｜近邻与最强反例','DSH','你负责对创新候选提出最强反例与近邻挑战。SPD、SOEP、HMF-DEIM已经占据广义空间重排和P2融合；常见动态融合、查询筛选、多尺度、切图、IoU加权等不能只换名字就算新。能查原文就给支持内容及链接，不能查就明确待核实，不捏造首创。优先考虑最简单竞争方案是否能解释效果，辨别mean对照退化与C净收益。只保留能改变判断的最小验证，不要求立即增加训练。'),
('Codex C｜贡献价值与去留','Codex','你负责判断一个候选到底能让读者学到什么。区分研究问题、实现方法和实验清单，防止把“收窄措辞”当新颖性。可以否定所有候选。对值得验证的方向说明相对已有方法具体新增的认识、实际意义、可失败条件及现有资产可复用程度；不要许诺期刊录用。第三轮明确推荐继续研究哪个问题或先不立项，不因已经投入很多而保住C。')]

append=['''[E11] 最新UAVDT同轮日志快照，覆盖旧包的进度记录，不覆盖其最终未完成状态
检查：洛杉矶2026-10-19 21:36:54，UTC2026-10-20 04:36:54。
此为训练日志的验证集原生总体车辆AP70，不是独立CPU评估、best_stg2或官方test，也不是小目标AP。
六个进程均存活、身份与各自输出目录一致，冻结文件校验无变动，loss有限，未见近期错误。仅查询既有日志，未做新训练/推理。
最新完整轮数：seed0 baseline/C为51，C-mean为50；seed1 baseline/C为46，C-mean为45。每seed以三臂共同轮次对比；轮次1起。
''']
for account,s in d['accounts'].items():
    append += [f"seed{s['seed']}（账号{account}）共同完整第{s['latest_common_epoch']}轮："]
    r=s['selected_epochs'][-1]
    append += [f"baseline={r['Baseline']:.6f}，C={r['C']:.6f}，C-mean={r['C-mean']:.6f}（百分制）。"]
    for c in s['comparisons']:
        w=c['windows']['10']
        append += [f"{c['comparison']}：共同最新轮{c['latest']['dAP70_pp']:+.6f}pp；最近10个同轮配对（{w['from_epoch']}–{w['to_epoch']}）均值{w['mean_delta_AP70']:+.6f}pp，正{w['positive_epochs']}轮/负{w['negative_epochs']}轮。连续轮不是独立训练重复。"]
    append += ['近期各轮（epoch: baseline/C/C-mean）:']
    append += [f"{r['epoch']}: {r['Baseline']:.4f}/{r['C']:.4f}/{r['C-mean']:.4f}" for r in s['selected_epochs'] if r['epoch']>=35]
append += ['''两端同看第45轮：seed0 baseline84.218427/C83.441437/C-mean82.520599；seed1 baseline84.420134/C83.722653/C-mean81.452429。不能跨种子直接相减。
增强日程在零起epoch46（第47轮）切换；seed0共同轮次已过节点，seed1共同轮次尚未过。相同epoch内三臂日程一致，阶段差异不能抹去已观察到的同轮差值。
当前AP70没有持续净收益支持，但训练尚未完成；最终仍按预设val AP70选择末阶段best_stg2并独评。不能以探索性AP75替换预设验证终点，也不能根据当前AP70证明所有指标永久无效。

[E12] 已有第36轮小目标独立诊断全文（仅这一固定轮、val、共同CPU协议）
''',EARLY.read_text(encoding='utf-8'),'''
[E13] 历史问题诊断的可用性与限制
2026-10-02的单seed0 val冻结预测错误分解记录：GT38759；共同匹配27792；新增1953、丢失1482。它不是当前20模型的完整多seed分析。
极小目标子组GT5096，C未匹配3089，其中2945在当时“按类别过滤的可用预测池”中没有合格几何候选（95.34%）。这不是所有解码query，不是重标分类可恢复召回的严格上界，预测记录缺query ID，不能直接归因于query数量、encoder选择或匹配失败。
640尺寸审计显示VisDrone test-dev更小，但“分辨率/网格是主因”是历史推测，不是已完成因果检验。训练Mosaic/ZoomOut/IoUCrop可能改变尺寸与监督，逐变换生存审计是否完整尚未确认；不能说已证明增强有害。
当前v5已完成逐类、完整12指标、尺寸、定位、平移和ignore敏感性，不应重复建议这些“全部缺失”；若提更细的分析，明确新增的判别点。

本轮产出要求
应区分：可直接写入的已有实测事实、值得验证但未证实的新假设、近邻未核实的候选、没有发表增量的改名或复述。
不要把先前助手“这条主线不行”或“可以发表”当成事实或必须同意的结论。作者焦虑和沉没成本都不是证据。允许讨论得出需换问题、需先查近邻，或者仍有一条有条件可守的路径。
''']
materials='本轮新增事实[E11–E13]放在旧包之后；涉及最新进度以E11为准。旧包E1–E10的完成VisDrone证据仍保留。\n\n'+initial.read_text(encoding='utf-8')+'\n\n'+'\n'.join(append)
assert len(materials)<60000
(OUT/'01_新议题.txt').write_text(topic,encoding='utf-8')
(OUT/'02_共同资料_含最新UAVDT.txt').write_text(materials,encoding='utf-8')
(OUT/'03_专家角色.json').write_text(json.dumps([{'name':n,'adapter':a,'perspective':p} for n,a,p in roles],ensure_ascii=False,indent=2),encoding='utf-8')
manifest={'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ANALYSIS,EARLY,initial)},'topic_chars':len(topic),'materials_chars':len(materials),'planned_rounds':3,'expert_count':4,'new_training_authorized':False,'new_training_started':False}
(OUT/'输入核验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'directory':str(OUT),'topic_chars':len(topic),'materials_chars':len(materials),'roles':len(roles)},ensure_ascii=False))
