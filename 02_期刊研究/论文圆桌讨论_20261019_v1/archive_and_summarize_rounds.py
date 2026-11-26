from pathlib import Path
import csv
import hashlib
import json
import shutil
import statistics
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
OUT = BASE / '三轮讨论整理_20261019_v1'
OUT.mkdir(exist_ok=True)
SOURCE = Path(r'C:\Users\jiwei\Downloads\paper-roundtable-d94a3093-b96a-474a-862c-ec82712b087e.json')
d = json.loads(SOURCE.read_text(encoding='utf-8'))
assert d['id'] == 'd94a3093-b96a-474a-862c-ec82712b087e'
assert d['kind'] == 'live' and d['mode'] == 'independent'
experts = {e['id']: e for e in d['experts']}
assert len(experts) == 4
assert sorted(e['adapter'] for e in experts.values()) == ['codex', 'codex', 'codex', 'dsh']
msgs = sorted((m for m in d['messages'] if m['round'] <= 3), key=lambda m: (m['round'], list(experts).index(m['expertId'])))
assert len(msgs) == 12
msg_by_id = {m['id']: m for m in d['messages']}
context_checks = []
for rnd in (1, 2, 3):
    r = [m for m in msgs if m['round'] == rnd]
    assert len(r) == 4 and {m['expertId'] for m in r} == set(experts)
    for m in r:
        assert m['text'].strip()
        seen = m.get('seenMessageIds', [])
        expected = {n['id'] for n in msgs if n['round'] < rnd}
        assert set(seen) == expected, (rnd, m['expertId'], seen, expected)
        assert all(msg_by_id[mid]['round'] < rnd for mid in seen)
        context_checks.append({'round': rnd, 'expert': m['expertName'], 'characters': len(m['text']), 'seen_previous_answers': len(seen), 'same_round_answers_in_context': False})

for ext in ('json', 'md'):
    src = SOURCE.with_suffix('.' + ext)
    shutil.copy2(src, OUT / ('原始完整讨论_保留第六轮失败状态.' + ext))

extracted = {k: d[k] for k in ('id', 'topic', 'materials', 'kind', 'mode', 'createdAt')}
extracted.update({'source_status': d['status'], 'source_planned_rounds': d['rounds'], 'source_recorded_round': d['round'], 'extracted_rounds': [1, 2, 3], 'experts': d['experts'], 'messages': msgs, 'note': '这是既有真实讨论的前三轮摘录，不是本助手重新发起的三轮；原记录第六轮失败未影响前三轮12份完整回答。'})
(OUT / '前三轮_12份完整回答.json').write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding='utf-8')
transcript = ['四位专家前三轮完整原文', '', '来源：应用真实讨论导出；保留发言原文，不将后续轮次混入三轮结论。', '原记录设置8轮，第6轮Codex A调用失败；前三轮均4/4完整。', '']
for m in msgs:
    transcript += [f"{'=' * 24}\n第{m['round']}轮｜{m['expertName']}\n发言ID：{m['id']}\n时间UTC：{m['createdAt']}", m['text'], '']
(OUT / '前三轮_12份完整回答.txt').write_text('\n'.join(transcript), encoding='utf-8')

with (BASE / '核查用_全部同种子配对差值_原精度.csv').open(encoding='utf-8-sig', newline='') as f:
    deltas = list(csv.DictReader(f))
with (BASE / '核查用_40份完整12指标_原精度.csv').open(encoding='utf-8-sig', newline='') as f:
    metrics = list(csv.DictReader(f))
metric_index = {(r['arm'], r['seed'], r['split']): r for r in metrics}
for r in deltas:
    a = metric_index[(r['arm'], r['seed'], r['split'])]
    b = metric_index[(r['reference'], r['seed'], r['split'])]
    for key in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl'):
        assert abs(float(r[key]) - (float(a[key+'_percent']) - float(b[key+'_percent']))) < 1e-9

def rows(arm, ref):
    return sorted((r for r in deltas if r['arm'] == arm and r['reference'] == ref and r['split'] == 'testdev'), key=lambda r: int(r['seed']))

checks = {}
for a, b in [('C', 'baseline'), ('C', 'C-mean'), ('rank16', 'rank16-mean')]:
    rr = rows(a, b)
    values = [float(r['APs']) for r in rr]
    checks[a + ' - ' + b] = {'split': 'testdev', 'units': 'percentage_points', 'seeds': [int(r['seed']) for r in rr], 'APs_deltas': values, 'mean': statistics.mean(values), 'sample_SD': statistics.stdev(values), 'median': statistics.median(values), 'full12_deltas': rr}
(OUT / '关键配对数字复核.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')

report = '''四专家三轮讨论整理｜当前小论文贡献与最小补证

结论先行
现有结果支持一项有边界的实证贡献：在当前紧凑检测器、压缩路径和训练配方内，训练期允许P2→P3支路访问2×2子格差异，相对仅传均值观察到小目标收益；在最大映射秩同为16的配对中，这一现象仍出现。它尚不构成新算子、新位置编码机制或通用最优配置。四位均保留“这些局部经验是否超出信息保留常识、足以形成期刊增量”的疑问，未给出投稿充分或录用保证。

一、实际执行与来源
本次接手页面时，同一议题已存在真实讨论，原设置为8轮：第1–5轮每轮4份完整回答，第6轮Codex A调用失败，其余3份完整，共23份。没有把这场历史记录写成新启动的3轮，也没有为凑轮数重复调用。按用户要求，仅取前三轮12份完整发言作本报告主要来源；第4–6轮保留在原始导出中，不混算为三轮依据。
应用记录ID：d94a3093-b96a-474a-862c-ec82712b087e。
创建UTC 2026-10-20 04:19:16.655（洛杉矶2026-10-19 21:19:16.655）；更新UTC 04:27:39.218（洛杉矶21:27:39.218）。
核对导出seenMessageIds：首轮各自未见其他回答；第二轮均见此前4份、第三轮均见此前8份，无同轮发言进入上下文。这只核对了应用记录的上下文安排，不证明判断误差独立。
角色为3 Codex＋1 DSH，显式请求均为GPT-6.1-Sol系列/xhigh；DSH请求codex/gpt-6.1-sol、magpie-responses。消息的实际model/provider/effort字段为空，故仅能报告请求配置，不能说服务端模型身份已确认，更不能当作跨模型家族验证。
专家依据共同证据包摘录，未独立读原始论文/全部实验文件；本助手对关键配对数字再与包内原精度CSV核算。此为四角色讨论整理，不是全稿正式审稿或新增证据。
本轮未新增训练、推理实验、服务器操作、稿件修改或自动监测管理。

二、四位认为能保留的内容
1）分支内部的子格差异具有条件性用途。
所有下列数字为VisDrone test-dev、val总体AP选择的best_stg2 EMA、统一FP32项目COCO评估，同种子差值，单位pp。它们不是官方工具包AP，也不是训练日志结果。
比较                         seed0       seed1       seed2        均值 / 样本SD / 中位数
C−C-mean APs                 +0.837      +0.164      +0.163       0.388 / 0.389 / 0.164
rank16−rank16-mean APs       +0.205      +0.267      +0.571       0.348 / 0.196 / 0.267
C−baseline APs              +0.836      +0.669      未训练该基线
C−baseline 总体AP           +0.628      +0.549      未训练该基线
方向一致是描述性结果，不能把小样本下的正值称为显著、等效或普遍稳定。两组配对不能拼成同一处理的6个seed。

2）仅提高“最大映射秩”不是出现该现象所必需的条件。
rank16两臂秩上限均≤16，仍有上述正向配对。能增加的认识是：可访问信息子空间与秩上限应分别检查。不能改写为“容量混淆全部排除”：均值化仍改变函数类、有效自由度与梯度，且端到端特征也随训练改变。

3）实现确实便宜，但轻量是属性，不能独立算强创新。
C新增17,440参数，约0.170%；FLOPs约增1.031%。rank16新增6,176参数且部分seed高于C，故C可称轻量实现，不能称最小或最优。细小时延差不支持速度排名。SOEP-lite只支持与已测简化候选的比较，不等于胜过完整SOEP/HMF-DEIM。

建议的一句主线：
“我们在轻量P2→P3融合中，测量训练期均值化对子格信息利用的检测代价，并通过同最大映射秩配对区分信息访问约束与秩上限约束的经验后果。”
这句话准确陈述研究对象；它是否足够新仍需从文献及可改变设计选择的发现来判断，不能靠改措辞自动获得新颖性。

三、不能继续挂在论文上的主张
— 首创P2四相位传入P3且不加P2检测头：已有结构先例，删除。
— 新pixel_unshuffle算子、固定相位排列顺序有优越性：无依据；unshuffle＋1×1可改写为U上的2×2 stride2卷积。
— 已证明亚像素位置编码/空间位置因果机制：删除。phase仅指特征网格offset，输入信息、纹理及优化解释未分开。
— 全面提升定位、平移稳定性与所有尺寸：删除这一概括，保留具体指标和反例。
  同上述test-dev协议，C−C-mean seed1的小目标AP75为−0.182pp；seed1 APl约−4.562pp、seed2 APm约−0.459pp。两主配对ARs均正，召回改善比几何全面改善更一致。既有平移仅seed0、无C-mean，也没有一致稳定性优势。
— 同预算最优/最小、打败完整SOEP或HMF-DEIM：删除或改成已测候选内的描述。
— 数据集、类别、GT、指标和IoU数量等于更多训练重复：不能这样统计。现有主要机制配对各3个seed，baseline配对2个seed。

四、主要分歧与三轮中的修正
（1）“对照清楚”是否本身足够新？
Codex C更重视“低秩瓶颈不等于固定均值子空间”的设计认识；Codex A与DSH持续追问它是否超出“保留更多信息会有用”。Codex B认为局部描述可成立，但效应波动限制推广。四位第三轮均未确认SIVP贡献充分，分歧没有被多数票消除。

（2）raw-P2均值参照是不是必须做？
DSH第1–2轮倾向把两次新训练列为必要；Codex B认为只在保留预算推荐时必要。到第3轮，DSH明确接受这一限定，A/C也采用条件性要求。因此最终并没有“无论如何都要再训”的一致建议。
它若胜C，会削弱实用优选；它若落后，只增强相对该候选的工程证据。它同时改变通道数、压缩顺序和非线性路径，不能独自确认优化或压缩机制。

（3）已有预测分析优先做什么？
Codex A首轮提出逐图配对bootstrap，之后降低必要性；Codex B更重视阈值扫描与等召回比较，检验共同检出误差的选择效应。若论文主动删除一般定位改善主张，这些分析可作为补充，不应为保住不必要的机制解释无限加实验。图像重采样须重算整组AP，不能平均逐图AP，也不能代替训练seed不确定性。

五、最需要补的证据——本次只整理，不启动
优先1：完成既有UAVDT六次训练及最终独立评估。
沿用既定val原生车辆AP70选择best_stg2，统一独评val/test，逐seed同时报告C−baseline与C−C-mean，明确CPU独评与训练日志口径差异。不能用第36轮或辅助AP75替换最终预设AP70。
注意：UAVDT总体车辆AP70正向仅增加第二数据集检测效用证据；要讨论小目标还需单独报告预定尺寸指标。它没有rank16配对，因此也不能复现“同秩”结论。

优先2，条件性：仅若要推荐C的预算配置，再考虑raw-P2均值近预算参照。
方案为64通道P2直接2×2均值池化，再零初始化64→256投影，投影16,384参数；沿原VisDrone配方、共同初始化、seed0/1和相同选模/独评。具体总成本仍须实测。
— 两seed优于C：撤回C优选，保留内部配对的局部观察。
— 接近或异向：排序未定，不作等效结论。
— 两seed落后：只支持C相对该候选的工程收益，仍不证明位置机制。
若论文不保留预算优选，这两次训练不是局部实证结论成立的前提，当前也未获本次新增训练授权。

可用现有预测完成、非新增训练：若保留定位解释，扫描分数阈值、比较等召回条件、分别报告共同/新增/丢失检出与误差；数据需逐图框、分数、GT和ignore，汇总表不够。若定位优势消失就撤回概括，不为保住它继续加任务。
不建议因为“厚度不够”立即追加完整SOEP、P2头、第二检测器、更多相位顺序或等价双线性对照；每项必须先对应一个会改变结论的具体问题。

六、UAVDT最终三种结果对应的写法
正向：预设AP70下两seed均胜baseline及C-mean，增加第二数据集车辆检测效用的条件支持；不自动证明新机制或小目标APs复现，更不补足同秩外部验证。
混合：种子、比较对象或指标分歧，逐项报告；仅胜C-mean时警惕对照退化，不能写普遍增益。
负向：撤回跨数据集净收益主张，保留VisDrone局部现象及外部未复现。前三轮部分回答称“失败边界”，本整理将其收紧为“未复现”：没有进一步辨识原因时，负向结果本身不解释为什么失效。

七、给作者的具体判断
现阶段可以写成“信息访问约束与秩约束下的小目标检测实证研究”，不宜包装成新检测器或新位置机制论文。最有价值的潜在认识是：即使采用低秩轻量映射，也不应未经验证就把子格差异预先均值化；本研究量化了当前设置的代价及例外。
真正仍悬着的是这条认识的增量和适用范围，而非缺少更多AP指标。现有12项、逐类、尺度、定位与平移足以显示多面结果；继续堆表不会自动解决新颖性。先完成UAVDT既有验证，并决定是否保留预算推荐，再决定是否值得那两次竞争参照训练。四位意见与现有证据都不支持现在承诺“够投必能中”，也没有据此证明“必然退稿”。

可核查文件
前三轮_12份完整回答.txt / .json：本报告的12份一手发言。
原始完整讨论_保留第六轮失败状态.md / .json：包含额外轮次与失败状态的未改写导出。
关键配对数字复核.json：从原精度CSV重算的数值。
讨论归档核验.json：来源哈希、轮次完整性、上下文可见性与模型字段限制。
'''
(OUT / '四专家三轮结论_创新点分歧与最小补证.txt').write_text(report, encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

now = datetime.now(timezone.utc)
manifest = {
    'archived_at_UTC': now.isoformat(),
    'archived_at_America_Los_Angeles': now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
    'source_meeting_id': d['id'],
    'source_status': d['status'],
    'source_planned_rounds': d['rounds'],
    'source_recorded_round': d['round'],
    'source_total_complete_answers': len(d['messages']),
    'summary_scope_rounds': [1, 2, 3],
    'summary_complete_answers': len(msgs),
    'new_expert_calls_in_this_turn': 0,
    'new_training_or_inference_in_this_turn': False,
    'context_checks': context_checks,
    'requested_model_configs': [{'expert':e['name'], 'adapter':e['adapter'], 'modelSettings':e['modelSettings']} for e in experts.values()],
    'served_model_identity': 'Unverified: model/provider/reasoningEffort fields are null in exported messages; configurations are requests only.',
    'raw_source_sha256': sha(SOURCE),
    'evidence_inputs': {p.name:sha(p) for p in [BASE/'核查用_40份完整12指标_原精度.csv', BASE/'核查用_全部同种子配对差值_原精度.csv', BASE/'02_共同参考资料_导入TXT.txt']},
    'artifacts': {p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name != '讨论归档核验.json'},
}
(OUT / '讨论归档核验.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'output_dir':str(OUT), 'rounds':3, 'answers':len(msgs), 'raw_answers':len(d['messages']), 'context_checks_passed':True, 'metric_checks_passed':True, 'report_characters':len(report)}, ensure_ascii=False))
