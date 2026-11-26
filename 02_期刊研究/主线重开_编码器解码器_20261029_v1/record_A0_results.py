"""Verify the archived observations and update the mutable research board honestly."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'A0_冻结模型诊断结果_v1'
now = datetime.now(timezone.utc).isoformat()
summary = json.loads((RESULTS / 'analysis/stage_summary.json').read_text())
sampling = json.loads((RESULTS / 'sampling_observation_v1/basis_summary.json').read_text())
budget = json.loads((RESULTS / 'receipts/cumulative_forward_budget.json').read_text())
assert budget['attempted_full_model_forwards'] == 169 <= budget['cap'] == 200
assert Counter(e['phase'] for e in budget['events'])['study100'] == 100
assert summary['images'] == 100 and summary['positive_gt'] == 4491
preflight = json.loads((RESULTS / 'receipts/preflight_pass.json').read_text())
assert all(r['boxes_max_abs'] == r['sigmoid_scores_max_abs'] == 0 for r in preflight['comparisons'])
gt = list(csv.DictReader((RESULTS / 'analysis/coverage_by_gt.csv').open()))
assert len(gt) == 4491 and len(set(r['uid'] for r in gt)) == 4491
mapping = dict(zip(['encoder_all_valid','encoder_topk300','decoder_pre','decoder_0','decoder_1','decoder_2','decoder_3','native_post300'],
                   ['all_enc','topk','pre','d0','d1','d2','d3','native']))
for row in summary['coverage']:
    mask = [r for r in gt if row['stratum'] == 'all' or
            (row['stratum'] == 'lt8' and float(r['sqrt_area_640_stretch']) < 8) or
            (row['stratum'] == '8_16' and 8 <= float(r['sqrt_area_640_stretch']) < 16) or
            (row['stratum'] == '16_32' and 16 <= float(r['sqrt_area_640_stretch']) < 32) or
            (row['stratum'] == 'ge32' and float(r['sqrt_area_640_stretch']) >= 32) or
            (row['stratum'] == 'small_lt32' and float(r['sqrt_area_640_stretch']) < 32) or
            (row['stratum'] == 'small_crowded16' and int(r['small_crowded16']) == 1) or
            (row['stratum'] == 'small_isolated16' and float(r['sqrt_area_640_stretch']) < 32 and int(r['small_crowded16']) == 0)]
    assert len(mask) == row['n']
    for metric in ['geom_any_50','geom_one_50','geom_any_75','geom_one_75','class_any_50','class_one_50','class_any_75','class_one_75']:
        assert sum(int(r[mapping[row['stage']] + '_' + metric]) for r in mask) == row[metric]

backup = RESULTS / 'backup'
receipt = json.loads((backup / 'archive_receipt.json').read_text())
archive = backup / 'A0_A0b_bounded_diagnostics.tar.gz'
assert archive.stat().st_size == receipt['archive_bytes']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['archive_sha256']
file_manifest = json.loads((backup / 'file_manifest.json').read_text())
expected = {r['path']: r for r in file_manifest}
verified = []
with tarfile.open(archive) as tf:
    members = tf.getmembers()
    assert len(members) == len(expected) + 1
    assert {m.name for m in members} == set(expected) | {'backup/file_manifest.json'}
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk()
        data = tf.extractfile(member).read()
        if member.name == 'backup/file_manifest.json':
            assert data == (backup / 'file_manifest.json').read_bytes()
        else:
            exp = expected[member.name]
            assert len(data) == exp['bytes'] and hashlib.sha256(data).hexdigest() == exp['sha256']
            verified.append(member.name)
assert len(verified) == receipt['files'] == 167
verification = {'recorded_utc': now, 'archive_sha256': receipt['archive_sha256'],
                'archive_files_sha256_verified': len(verified), 'GT_rows_reaggregated': 4491,
                'coverage_table_rows_checked': len(summary['coverage']), 'forward_budget': 169,
                'training_updates': 0, 'status': 'PASS_WITHIN_DIAGNOSTIC_SCOPE_ONLY',
                'not_verified': ['novelty', 'causal mechanism', 'generalization', 'new algorithm improvement', 'training memory feasibility']}
(RESULTS / 'local_result_verification.json').write_text(json.dumps(verification, indent=2))

counts = []
for kind in ['geom', 'class']:
    for threshold in ['50', '75']:
        miss = [r for r in gt if int(r['d3_' + kind + '_any_' + threshold]) == 0]
        counts.append({'kind': kind, 'iou': threshold, 'final_missing': len(miss),
                       'already_missing_in_all_encoder_readout': sum(int(r['all_enc_' + kind + '_any_' + threshold]) == 0 for r in miss),
                       'all_encoder_covered_topk_missing_and_final_missing': sum(int(r['all_enc_' + kind + '_any_' + threshold]) == 1 and int(r['topk_' + kind + '_any_' + threshold]) == 0 for r in miss),
                       'first_decoder_layer_covered_but_final_missing': sum(int(r['d0_' + kind + '_any_' + threshold]) == 1 for r in miss)})
(RESULTS / 'persistent_failure_counts.json').write_text(json.dumps(counts, indent=2))

history = ROOT / 'history_before_A0_update_v1'
history.mkdir(exist_ok=True)
board_path = ROOT / '01_候选判别表.json'
if not (history / board_path.name).exists():
    shutil.copy2(board_path, history / board_path.name)
board = json.loads(board_path.read_text(encoding='utf-8-sig'))
board.update(status='FROZEN_DIAGNOSTICS_COMPLETE_MAINLINE_NOT_VALIDATED', mainline_validated=False,
             latest_update_utc=now, latest_report=str(ROOT / '11_A0阶段诊断_结果与淘汰.txt'),
             current_priority_question='Which encoder representation/readout mechanism explains the residual sub-stride localization failures, beyond existing SSA/PBM/frequency fusion?',
             new_empirical_evidence='A0:100 images/4491 GT; A0b:32 images/1395 initially localized GT; 169 total model forwards, zero training; outputs preserved',
             asset_gate={'full_source_verified':True,'official_source_files':187,'author_weight_downloaded':True,
                         'tinyformer_commit':'9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1',
                         'author_weight_sha256':'0ba0e997e4bdfae7800f6f3e55fa714b0302efd4aa5327663290c9705eb8a199',
                         'strict_EMA_state_keys_loaded':673,'first8_final_outputs_maxabs':0,
                         'historical_browser_text_reconstruction':'Failed hashes; superseded by full Git-verified checkout'})
board['constraints']['new_experiments_this_turn'] = 'Completed frozen observation only; zero new training/architectures'
for c in board['candidates']:
    if c['id'] == 'A_ENCODER_INSTANCE_SEPARABILITY':
        c.update(status='STAGE_TRACE_COMPLETE_SEPARABILITY_CAUSE_UNPROVEN',
                 next='Do not infer representation collapse from same-cell GT or missing regressor readout; compare representation versus readout with nearest-work controls')
if not any(c['id']=='C_INTERPOLATION_BASIS_COLLAPSE' for c in board['candidates']):
    board['candidates'].append({'id':'C_INTERPOLATION_BASIS_COLLAPSE','status':'DEPRIORITIZED_SCALE_CONFOUND',
        'question':'Does repeated bilinear grid-basis reading explain tiny-object residual failures?',
        'evidence':'Pooled failure-success normalized-rank gap -0.06385; within <8 pixel +0.00148 and 8-16 pixel -0.00400, with small/inconsistent within-initial-quality gaps',
        'nearest':['Better Sampling/SPR','Look-and-Confirm (abstract+intro only)'],
        'increment_approved':False,'next':'No anti-collapse module or training based on this evidence'})
if not any(c['id']=='D_DISTRIBUTIONAL_QUERY_RELATIONS' for c in board['candidates']):
    board['candidates'].append({'id':'D_DISTRIBUTIONAL_QUERY_RELATIONS','status':'LITERATURE_AND_SOURCE_GATE_PENDING',
        'question':'Do uncertain point-box relations create harmful query competition for tiny objects; could distribution-aware relations avoid it?',
        'nearest':['DETRNN','Relation DETR','EASE-DETR','DS-Det','An Uncertainty-aware DETR Enhancement Framework'],
        'increment_approved':False,'causal_evidence':False,
        'next':'Read closest complete mechanisms and source first; do not use A0 as proof of this different decoder hypothesis'})
board['remaining_deliverable']='A specific falsifiable architecture mechanism, defensible nontrivial nearest-work difference and feasible validation; these remain unverified'
board_path.write_text(json.dumps(board,ensure_ascii=False,indent=2),encoding='utf-8')

coverage = {(r['stratum'],r['stage']):r for r in summary['coverage']}
lines = ['Material Passport', 'Origin Skill: academic-research-suite / experiment-agent',
         'Origin Mode: run (bounded inference diagnostic and CPU analysis)', 'Origin Date: '+now,
         'Verification Status: UNVERIFIED for scientific conclusions; file/metric checks described below',
         'Version Label: A0_A0b_result_v1', '', '当前结论：已经取得可核查的架构诊断，尚未找到通过验证的新算法主线。',
         '上一轮属于 progress：完成官方资产取得、严格权重加载、阶段追踪和采样观察；这些证据改变了候选优先级。', '',
         '一、执行与范围',
         '模型：TinyFormer-S-PBM，作者 VisDrone EMA；官方 commit 9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1。',
         '完整源码187文件通过Git blob校验；权重175310334字节，本地/远端SHA一致。作者未另发独立校验和，不能称作者校验和核验。',
         '673状态项strict加载成功；10类，无随机分类头。Python3.9注解延迟及移除未使用DropPath导入只作用于加载AST，原始数值代码文件未变。',
         '640x640作者原生方形拉伸、ImageNet归一化、未融合FP32 eval、TF32关闭、batch1、300查询、4层decoder。',
         '100张来自既有VisDrone训练集diagnostic100，共4491有效GT；作者权重训练时见过原训练集。这是样本内探索，不是独立验证成绩。',
         'A0预检查29次＋正式100次；A0b固定前32图补充40次。累计169/200次完整模型前向、0训练更新、0新架构、0恢复或压缩干预。',
         '首8张原生与追踪最终框/分数最大差0，类别/query编号一致；A0b首8张及全32张旧端点对照也通过。模型状态摘要不变。',
         '推理原始命令见ssh_operation_receipts.jsonl；进程退出均0。主程序有180秒硬超时，没有实验自动重试。', '',
         '二、[本轮实测] 无分数阈值、类别无关的几何覆盖；这不是AP，也不是性能上界',
         '阶段 | 全部GT IoU>=0.5 | 全部GT IoU>=0.75 | <8像素GT IoU>=0.5 | <8像素GT IoU>=0.75']
for stage in mapping:
    a,b=coverage[('all',stage)],coverage[('lt8',stage)]
    lines.append(f"{stage} | {a['geom_any_50']}/4491 | {a['geom_any_75']}/4491 | {b['geom_any_50']}/901 | {b['geom_any_75']}/901")
lines += ['', '全部编码候选框是对未选token追加现有回归头的诊断读出，未选位置没有同等监督，候选数也不同；不能解释成“编码器潜力上界”。',
          '原生post300按类别与query展平取分数Top300，可能重复同一query；其类别覆盖不能直接与每query argmax类别混为同一种指标。',
          '任意框覆盖与全图一对一匹配分别保留；后者最大化合格匹配数量，再最大化IoU总和。CSV4491行重聚合与64行汇总全部一致。', '',
          '三、[本轮实测] 最终残余失败的阶段联系',
          'IoU0.5几何未覆盖586个，其中475个在全部编码候选诊断读出中也未覆盖（81.1%）；76个是编码全候选覆盖、TopK未覆盖且最终仍未覆盖。',
          'IoU0.75几何未覆盖1761个，其中1583个在编码全候选中也未覆盖（89.9%）；47个为TopK丢失后持续失败。',
          '第1层decoder到第4层几何覆盖从3853增至3905，IoU0.75从2622增至2730；总体在改善，不能称解码器整体在破坏小目标。',
          '同一过程也有局部损失：第1层覆盖而最终丢失30个(IoU0.5)／115个(IoU0.75)。112个近邻GT转换事件只描述query最近GT变化，不是真实例身份切换的因果证据。',
          '[推断] 当前优先检查候选形成阶段的表示、融合或回归读出；这组数据不支持先把增大query数或“防decoder漂移”当成主改进。',
          '[尚未证明] 编码器不可逆丢失信息、同格实例融合、P2直接读取有效、任何新模块能提升AP。', '',
          '四、[本轮实测] 采样重复解释未通过初步区分',
          'A0b记录原生采样位置/权重，并计算双线性插值网格基底Gram矩阵参与秩；它不是特征信息秩。',
          '1395个初始TopK最大IoU>=0.25的GT进入关联描述。全尺度合并，失败/成功组标准化秩差-0.06385。',
          '但<8像素组差+0.00148（失败77、成功255）；8-16像素组差-0.00400（失败46、成功444）。',
          '进一步分初始IoU0.25-0.5、0.5-0.75后差值很小且方向不一致；大目标失败数仅4或0，不能夸大。',
          '[推断] 合并差异主要受尺度分布影响，目前缺少“重复采样导致失败”的特异证据。此候选降级，不追加防塌缩模块或训练。',
          '没有以别的层或新阈值替换预定的第4层观察终点来寻找阳性结论。', '',
          '五、已有方法边界（原文结论与本轮推断分开）',
          '[原文] TinyFormer已有SSA/PBM浅层细节通路：https://arxiv.org/html/2605.25046v1 ；阅读§3.1-3.2、§4.3-4.4及结论，另核查完整对应源码。',
          '[原文] FSDC-DETR研究DEIMv2的CNN/ViT频域差异与融合：https://arxiv.org/html/2607.05176v1 ；阅读摘要、§2.2和§3.1-3.2部分，未读全方法/实验。',
          '[原文] UHR-DETR有覆盖导向稀疏编码及全局/局部解码：https://arxiv.org/html/2604.21435v1 ；阅读摘要、引言和§III-A至III-D，未完整读实验。',
          '[原文] Better Sampling的SPR约束框外采样位置/权重：https://arxiv.org/pdf/2407.06127 ；实际阅读摘要、引言及§4.1-4.3；不把它简化成增加采样点。',
          '[原文] Look-and-Confirm针对聚集采样提出分层树状采样和残差门控：https://doi.org/10.1109/ICIGP68997.2026.11619773 ；仅摘要＋引言，完整方法尚未取得。它不同于CVPR2023同名近似的Hybrid-Matching H-DETR。',
          '[原文] Anchor DETR已处理同位置多对象预测：https://arxiv.org/pdf/2109.07107 ；本次阅读摘要、引言，不能宣称多slot本身新颖。',
          '[原文] DETRNN已有置信度/相似度排序、密度自适应分组和Bi-RNN：https://arxiv.org/pdf/2610.05191 ；阅读摘要、引言、§3.1-3.4及部分结果，未读全附录。',
          '[推断] 共享encoder value、加浅层支路、加频域模块、单纯增加query、简单序列排序/分层采样都不是可直接申报的空白。', '',
          '六、资源与可复核性',
          '当前3090/24GB已成功完成上述FP32推理。8次原生CUDA event计时约25.3-27.3ms（仅单图、未融合、无IO/后处理），不能外推训练显存或与其他论文FPS直接排名。',
          '167个归档文件逐项SHA校验通过；完整归档SHA：'+receipt['archive_sha256'],
          '文件：A0_冻结模型诊断结果_v1/backup/A0_A0b_bounded_diagnostics.tar.gz；逐GT/逐图指标、原始采样与阶段读出、源码/协议摘要均保留。',
          '限制：单检查点、训练集探索、固定首32子样本、可能相关帧、尺寸按拉伸后输入计算；覆盖无分数筛选，不等同AP/泛化/因果改善。', '',
          '七、下一步与主线状态',
          '当前主线尚未完成验证。继续研究编码器、特征交互、decoder机制；不回到JPEG/恢复，不启动旧DEIM任务，不因资产现成就锁死TinyFormer。',
          '新的“定位分布参与query关系”只进入文献/源码核查：须与Relation DETR、EASE-DETR、DETRNN、DS-Det及已有不确定性DETR逐项比较，再决定是否值得诊断。',
          '保留待证伪问题：缺陷来自表示本身还是现有回归读出？当前需要能区分它们的证据，不以“换最新backbone”代替研究问题。', '',
          '披露：本报告由AI辅助、单助手完成检索、源码核查和脚本分析；阅读范围逐项标注，未推定用户已读文献。']
(ROOT / '11_A0阶段诊断_结果与淘汰.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(ROOT / '15_当前状态与结果校验.json').write_text(json.dumps({'recorded_utc':now,'goal_status':'ACTIVE',
    'previous_goal_turn':'progress','mainline_validated':False,'diagnostics_complete':True,
    'model_forwards':169,'cap':200,'training_updates':0,'verification':verification,
    'candidate_board':str(board_path),'latest_report':str(ROOT/'11_A0阶段诊断_结果与淘汰.txt')},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'archive_files_verified':167,'GT_rows_checked':4491,'coverage_rows_checked':64,
                  'model_forwards':169,'training_updates':0,'mainline_validated':False}))
