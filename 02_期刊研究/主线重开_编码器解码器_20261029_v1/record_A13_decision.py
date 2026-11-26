"""Record completed cached A13 evidence; never launch or re-run a model."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
A13 = ROOT / 'A13_既有轨迹与坐标支持范围审计_v1'
NOW = datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

result = read(A13 / 'summary.json')
done = read(A13 / 'completed.json')
protocol = read(A13 / 'protocol.json')
assert done['status'] == 'COMPLETED'
assert done['summary_sha256'] == sha(A13 / 'summary.json')
assert done['protocol_sha256'] == sha(A13 / 'protocol.json')
assert done['source_code_sha256'] == sha(ROOT / 'a13_cached_trajectory_support.py')
for item in protocol['source_files']:
    assert sha(Path(item['path'])) == item['sha256']
assert result['new_full_model_forwards'] == result['new_fits'] == result['main_model_updates'] == 0
assert result['cumulative_full_model_forwards'] == 213
assert result['independent_graph_count_checks'] == 72
tiny = next(r for r in result['aggregate'] if r['cohort'] == 'lt8')
assert tiny['GT'] == 263 and tiny['projected_assigned75'] == 252
assert tiny['stage_iou75_counts'][-1] == 40

names = ['01_候选判别表.json', '13_架构检索增补记录.json',
         '14_架构核心文献矩阵.json', '15_当前状态与结果校验.json',
         '16_最近工作源码核查.json']
history = ROOT / 'history_before_A13_update_v1'
assert not history.exists(), 'One-time reconciliation already started; inspect instead of rerunning.'
history.mkdir()
for name in names:
    shutil.copy2(ROOT / name, history / name)
write(history / 'manifest.json', {'utc': NOW, 'files': [
    {'file': name, 'sha256': sha(history / name)} for name in names]})
records = {name: read(ROOT / name) for name in names}

archive_files = sorted(p for p in A13.iterdir() if p.is_file())
archive_files.append(ROOT / 'a13_cached_trajectory_support.py')
backup = A13 / 'backup'
assert not backup.exists()
backup.mkdir()
manifest = [{'file': ('code/' + p.name) if p.parent == ROOT else p.name,
             'sha256': sha(p), 'bytes': p.stat().st_size} for p in archive_files]
write(backup / 'member_manifest.json', {'utc': NOW, 'files': manifest})
archive = backup / 'A13_cached_support_and_trajectory.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    for p, item in zip(archive_files, manifest):
        tf.add(p, arcname=item['file'])
with tarfile.open(archive, 'r:gz') as tf:
    assert len(tf.getmembers()) == len(manifest)
    for item in manifest:
        assert hashlib.sha256(tf.extractfile(item['file']).read()).hexdigest() == item['sha256']
receipt13 = {'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
             'files': len(manifest), 'status': 'PASS_ALL_ARCHIVE_MEMBER_HASHES'}
write(backup / 'receipt.json', receipt13)

report = ROOT / '36_A13定位证据与当前主线判断.txt'
report.write_text('''当前判断：尚未找到通过机制、新颖性和有效性核查的论文主线。当前研究问题是无人机航拍微小目标的精确定位，重点查证编码器到解码器的位置保留与读取机制，不能把这一问题范围称为已经确认的新算法或研究空白。

1. 数据、算法和当前边界
主数据集仍为VisDrone。现有架构诊断使用作者TinyFormer-S-PBM和Dome-DETR-S权重；本轮A13复用Dome的12张官方验证图、263个微小GT，输入为800方形，微小定义为变换后sqrt面积<8像素。不能外推为全部VisDrone或所有检测器。
Dome-DETR是当前诊断载体，最终论文骨架尚未锁定；YOLO26s已有历史基线，但尚无与新架构的公平比较协议。DeepSeek的新编码器/解码器只提供机制启发，尚没有经验证可直接迁移的检测算法。压缩和图像恢复不再作为当前主线推进。

2. 定向检索和最近工作比较
检索是定向非系统调研。A13逐层退化查询、SQR原文访问、方法阅读、编码器像素相位初查的精确请求和原始返回已保存在source_audit_v1，见13检索记录。
SQR（CVPR2023）已明确研究中间层正确而最终层错误，并用训练时选择性回收前两层query、增加后段监督处理级联问题，推理仍用基本路径。已读原文摘要、引言、完整4.2和4.3（含公式7—9、Fibonacci监督数和起始层选择）；未通读附录、未执行作者代码。作者为Fangyi Chen、Han Zhang、Kai Hu、Yu-Kai Huang、Chenchen Zhu、Marios Savvides。不能仅凭发现逐层退化宣称新问题。
TPRD已有利用更早层教师预测纠正后层教师的蒸馏方案；此前完整III-A—III-D的阅读范围不变，不能把通用挑选较好中间层作创新。
TAPTRv2固定提交119d1d011305e2b1397cecfd01225247c1d0a1fb的主动导入链已有注意力更新位置、再让后续框回归感知位置更新的算子；视频给定点跟踪与未知目标检测任务不同，但换任务不能单独成立算法创新。
DAPE固定提交b472392f4df4892d264870917c0d1e4d8dfc8568中的SQS已有在proposal中心采样多尺度内容并融合可学习query的机制；逐层位置编码从更新后的reference计算。不能把笼统的“内容与位置结合”或“中心再采样”作独有创新。
新视觉编码器的像素位置/多相采样方向仅开始文献发现，SPD-Conv、Learnable Polyphase Sampling及Making Vision Transformers Truly Shift-Equivariant等均需先读原始方法。当前没有据此选择新候选、提出新颖性结论或追加前向。这里的输入像素相位不同于A11/A12的坐标bin插值相位，不把改名当新问题。

3. A13方法与阅读范围
A13只复用A12缓存，探索性且事后设计；A12的相位校正负结果和29条边越界已知。协议在计算A13结果前冻结，但不称独立确认或盲测。A12的query配对固定，比较初始pre-box和3层FDR坐标；前两层由保存的原生分布均值按源码重建，最终层以实际raw输出为准。
按真实reg_scale=4、bin端点±4、固定初始框，推导四条边可表达区间。将每条真实边界裁剪到对应区间，构造自由logit的几何参照。相邻bin插值可以表达连续均值，但这不证明共享的学习head能根据图像预测出这些logit。构造并非训练出的预测，也不是任意几何问题的全局最优求解。
当前解码器用固定pre-box作为回归坐标系，并用最新框作下一层采样reference，这是原生设计，不能直接称为坐标系错误。缓存没有实际MSDA采样点、注意力或特征，不能判断实际图像读取是否覆盖目标。

4. 实际结果
固定query配对下，263个微小GT达到IoU>=0.75的数量：pre-box为38，三层FDR分别42、39、40。自由logit构造可使252/263达到该几何标准；预设主统计为逐图覆盖率均值0.96775，按前缀bootstrap区间[0.93846,0.99083]，通过用于降低“范围不足”优先级的资源筛选规则。
取消固定GT-query配对、允许所有原始query参与类别无关的一对一几何匹配时，实际最终框49/263达到IoU>=0.75，构造参照为255/263。不是检测AP、分类召回或可实现提升。
固定配对最终未达IoU>=0.75的223个微小GT中，有14个在前两层FDR中曾达标；pre-box曾达标而最终未达标的为10个。未记录中间层分类分数，因此这不是SQR原文定义的TP-fading指标；也不能据此给SQR训练收益设上界。
72项匹配图计数经增广路径与最大流两种算法交叉核对；FP32实际输出与FP64重建在IoU75判定上无分歧。输入12份缓存、协议、源码、结果及归档成员SHA已核验。本轮新增模型前向0、拟合0、主模型更新0，架构分支累计完整前向仍为213。

5. 文献结论、推断与待验证假设
原文/源码结论：跨层退化、query回收、位置更新感知、中心内容采样均有直接先例；Dome的固定回归框架与更新采样框架是实际代码行为。
本次测量：样本中的精确几何定位仍有大量未达标；原有坐标范围在构造参照下可表达多数目标的合格框。
我们的推断：单纯扩大坐标范围或笼统保留最佳中间层，当前作为主增量的依据较弱。该结果不确定编码器、特征读取、回归学习、监督哪一项导致了误差。
待验证假设：新的视觉编码器能否更好地保留细粒度位置，并使解码器读取这些信息。目前没有证据证明位置在编码时丢失、或只在读取时丢失，不能据此直接堆叠新模块。

6. 候选增量、资源与下一步
当前不批准某个算法增量为论文主线，不进行坐标范围/相位修正模块训练。下一步先完成位置保留/采样算子的最近工作对照，再提出一个能区分编码不足与读取不足的具体干预及有限协议；只有出现可重复的机制证据且排除最近工作重合，才选择新结构短测。
单卡约24GB已能完成当前冻结推理，尚不能据此证明新结构训练显存、训练时间或期刊价值。原有有界前向阶段均已结束，不据此自动延长；不重启旧DEIM、压缩或恢复任务。
本轮交接：目标保持ACTIVE，阶段已完成且无新实验在跑。负结果用于淘汰解释，不将报告数量、缓存诊断或构造上界当作已找到主线。

7. 可核验来源
SQR（CVPR2023）：https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Enhanced_Training_of_Query-Based_Object_Detection_via_Selective_Query_Recollection_CVPR_2023_paper.html
所读arXiv v3：https://arxiv.org/html/2212.07593
TPRD：https://arxiv.org/html/2609.19964v1
TAPTRv2：https://proceedings.neurips.cc/paper_files/paper/2024/hash/b7216f4a324864e1f592c18de4d83d10-Abstract-Conference.html
DAPE：https://doi.org/10.1609/aaai.v40i6.42480
Dome源码：https://github.com/RicePasteM/Dome-DETR/tree/2dde3bc1946a3e9fad9abd0612b59fc39bd6b861
A13结果：A13_既有轨迹与坐标支持范围审计_v1/summary.json；协议及限制：同目录protocol.json。
''', encoding='utf-8')

question = '微小目标精确定位不足：查证新的视觉编码器及编码器到解码器的位置保留/读取机制；尚未确认因果环节或算法增量。'
board = records[names[0]]
board.update({'latest_update_utc': NOW, 'latest_report': str(report),
    'status': 'NO_CANDIDATE_PASSED_MECHANISM_AND_NOVELTY_GATES_AFTER_A13',
    'current_priority_question': question,
    'new_empirical_evidence': 'A13 cached12official-val images263tinyGT:40 fixed-assigned actualIoU75 versus252 constructive-free-logit support;14/223 earlier-FDR-pass/final-fail. No newforwards/fits;causal mechanism unresolved.',
    'next_decision_protocol': 'Compare actual position-preserving encoder/readout operators and define a discriminating bounded intervention before any further model pass or architecture training.'})
board['A13_evidence'] = {'summary': str(A13 / 'summary.json'), 'archive': receipt13,
    'decision': result['decision'], 'mainline_validated': False,
    'oracle_limit': 'Free per-query logits using GT;not learned prediction,AP,feature availability or causal evidence.'}
board['rejected_generic_novelty_claims'].append('Generic intermediate-query recollection,attention-based position update,or proposal-center content sampling already have SQR/TAPTRv2/DAPE prior art.')

raw_names = ['A13_layer_trajectory_nearest_search.json', 'A13_SQR_primary_open.json',
             'A13_SQR_method_read.json', 'encoder_pixel_phase_primary_discovery.json']
raw_records = []
for name in raw_names:
    p = ROOT / 'source_audit_v1' / name
    value = read(p)
    raw_records.append({'file': str(p.relative_to(ROOT)), 'sha256': sha(p), 'request': value['request']})
log = records[names[1]]
log['supplement_utc'] = NOW
log['query_groups'].append({'name': 'A13 cached trajectory and encoder pixel-position prior discovery', 'requests': raw_records})
log['A13_cached_geometry_and_primary_reading'] = {'raw_primary_requests': raw_records,
    'source_reading_scope': 'SQRabstract,intro,complete4.2/4.3;TAPTRv2/DAPE fixed-source operator reinspection. Pixel-phase encoder family discoveryonly.',
    'result': str(A13 / 'summary.json'), 'design': protocol['design'],
    'no_new_model_runs': True, 'no_novelty_or_causal_claim': True, 'report': str(report)}

matrix = records[names[2]]
assert not any(r['id'] == 'SQR' for r in matrix['rows'])
matrix['rows'].append({'id': 'SQR',
    'title': 'Enhanced Training of Query-Based Object Detection via Selective Query Recollection',
    'authors': ['Fangyi Chen', 'Han Zhang', 'Kai Hu', 'Yu-Kai Huang', 'Chenchen Zhu', 'Marios Savvides'],
    'url': 'https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Enhanced_Training_of_Query-Based_Object_Detection_via_Selective_Query_Recollection_CVPR_2023_paper.html',
    'inspected_url': 'https://arxiv.org/html/2212.07593', 'tier': 'CVPR2023pp23756-23765;inspectedarXivv3',
    'read_scope': 'Abstract,introduction,complete4.2and4.3includingselectionrecurrence/Fibonacci/startingstage. Notwholepaperorappendix;authorcodenotexecuted.',
    'author_method': 'Training-only selective intermediatequery recollection fromtwo neareststages;inferencebasicpathunchanged.',
    'our_consequence': 'Generic stagewise prediction degradation or retaining earlierqueries ispriorart. A13measuresgeometryonly,not sourceTP-fading.',
    'unresolved': 'No SQR implementation/trainingor APcomparisoninoursetting;cannotboundits trainingbenefitfromfixedoutputreplay.'})
matrix.update({'updated_utc': NOW, 'latest_report': str(report),
    'latest_empirical_boundary': 'A13 lowerspriorityofsupport-rangealone;notproof of anencoder/decoderfault. Pixel-position encoder candidates unselected.'})

status = records[names[3]]
assert status['model_forwards'] == 213 and status['mainline_validated'] is False
status['authoritative_phase_history'].append(copy.deepcopy(status['latest_authoritative_phase']))
status['latest_authoritative_phase'] = {'utc': NOW, 'classification': 'progress',
    'goal_status': 'active', 'complete_mainline': False, 'full_model_forwards': 213,
    'cumulative_authorized_protocol_cap': 213, 'bounded_phase_complete': True,
    'main_model_training_updates': 0, 'CPU_diagnostic_linear_models': 64,
    'scalar_response_heads': 96, 'A13_new_forwards': 0, 'A13_new_fits': 0,
    'A13_decision': result['decision'], 'A13_archive': receipt13,
    'A13_independent_graph_counts': 72, 'latest_report': str(report),
    'next': 'Clearposition-preserving encoder andquery-read operators;require specificmechanism andboundedprotocol before newmodelruns.',
    'remaining_uncertainty': ['Causalmechanism', 'Distinctalgorithmoperator', 'Independentgeneralization', 'Trainingfeasibility']}
status.update({'recorded_utc': NOW, 'previous_goal_turn': status['current_goal_turn'],
    'current_goal_turn': 'progress_A13_cached_support_and_trajectory_primary_prior_clearance',
    'latest_report': str(report), 'current_priority_question': question})

audit = records[names[4]]
audit.update({'updated_utc': NOW, 'A13_cached_support_and_trajectory': {
    'source_commit': '2dde3bc1946a3e9fad9abd0612b59fc39bd6b861',
    'protocol_sha256': sha(A13 / 'protocol.json'), 'summary_sha256': sha(A13 / 'summary.json'),
    'decision': result['decision'], 'archive': receipt13,
    'source_interpretation': 'Fixedinitialregressionframe andupdatedsamplingreference are intentionaldifferentroles;not automatically mismatch.',
    'unobserved': ['ActualMSDAsamplinglocations', 'Attention/features', 'Intermediateclassification', 'Causalposition-informationloss']},
    'current_novelty_boundary': 'Noapprovedalgorithm. SQR/TPRD covergenericstagewisereuse;DAPE/TAPTRv2 covercentercontent/position-awareupdates. Newencoderfamilystillonlydiscovery.'})
for name, value in records.items():
    write(ROOT / name, value)
receipt = {'utc': NOW, 'status': 'PASS_COMPLETED_A13_RECONCILED', 'goal_status': 'ACTIVE',
    'current_goal_turn_classification': 'progress', 'mainline_validated': False,
    'full_model_forwards': 213, 'new_full_model_forwards': 0, 'new_fits': 0,
    'main_model_updates': 0, 'matrix_rows': len(matrix['rows']),
    'A13_decision': result['decision'], 'A13_archive': receipt13,
    'input_npz_sha256_checked': len(protocol['source_files']), 'independent_graph_counts': 72,
    'files': [{'file': str(p.relative_to(ROOT)), 'sha256': sha(p)}
              for p in [report] + [ROOT / n for n in names]],
    'not_proven': protocol['not_claimed']}
write(ROOT / '37_A13归档核验与当前主线状态回执.json', receipt)
print(json.dumps({'status': receipt['status'], 'goal_status': 'ACTIVE',
    'mainline_validated': False, 'matrix_rows': receipt['matrix_rows'],
    'A13_archive_files_verified': receipt13['files'], 'full_model_forwards': 213,
    'new_forwards': 0, 'report': str(report)}, ensure_ascii=False))
