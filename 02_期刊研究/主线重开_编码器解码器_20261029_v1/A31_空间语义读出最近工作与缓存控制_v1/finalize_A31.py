import hashlib
import json
import pathlib
import shutil
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
D = pathlib.Path(__file__).resolve().parent
B = D.parent
LIB = B.parent / '文献_相关原始论文_20261004'
NOW = datetime.now(timezone.utc).isoformat()
CANONICAL = ['01_候选判别表.json', '13_架构检索增补记录.json',
             '14_架构核心文献矩阵.json', '15_当前状态与结果校验.json',
             '16_最近工作源码核查.json']
REPORT = B / '72_A31语义读出最近工作与否定性验证.txt'
RECEIPT = B / '73_A31缓存控制与独立复算回执.json'


def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    pathlib.Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


assert not REPORT.exists() and not RECEIPT.exists(), 'Already published; do not rerun.'
prior = read(B / '71_A30双头任务与原始支持核验回执.json')
for path, digest in prior['files_sha256'].items():
    assert sha(path) == digest, ('Prior A30 evidence changed', path)
assert len(prior['files_sha256']) == 172
result = read(D / 'cache_result.json')
verification = read(D / 'independent_verification.json')
assert verification['status'] == 'PASS_SOURCE_UNION_SCALAR_READS_ALL9_AP_AND_FIXED_FP'
assert result['new_model_calls'] == result['new_optimizer_updates'] == 0
assert len(result['gates']) == 7 and not any(result['gates'].values())
assert read(D / 'status.json')['status'] == 'COMPLETED'
source_rows = read(D / 'selected_source_manifest.json')['rows']
for row in source_rows:
    data = pathlib.Path(row['local']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    blob = b'blob ' + str(len(data)).encode() + b'\0' + data
    assert hashlib.sha1(blob).hexdigest() == row['git_blob_sha1']
assert len(source_rows) == 8
backup = D / 'canonical_before_A31'
backup.mkdir(exist_ok=False)
for name in CANONICAL:
    shutil.copy2(B / name, backup / name)
prior_remapped = {str(backup / pathlib.Path(p).name) if pathlib.Path(p).parent == B
                  and pathlib.Path(p).name in CANONICAL else p: h
                  for p, h in prior['files_sha256'].items()}
for p, h in prior_remapped.items():
    assert sha(p) == h

tood_scope = ('Selected complete Sec3.1 and Sec3.2.1-3.2.2; method figures/equations visually '
              'inspected in saved PDF renders pages3-5. Author-code selected head/assigner/config '
              'statically inspected at pinned commit. Not whole-paper/experiments/supplement/code execution.')
defcn_scope = ('Existing local arXiv2012.03544v3,26Mar2021; selected Sec3.2.1-3.2.3 and4.2.1 '
               'text; PDFpage5 Fig3/Eq5-6 visually inspected. Selected .3dmf.aux author code '
               'statically inspected; not whole paper/experiments or author implementation run. '
               'Published CVPR2021 metadata verified separately; local PDF bytes are arXivv3.')
ct_scope = ('Selected full Sec3.3 from extracted primary text,includingEq5-6 and following '
            'cross-page interpolation explanation. Saved render ConsistentTeacher_page3 '
            'shows introduction andSec3.1,notEq5-6; no visualEq5-6 verification claimed. '
            'Not full method/experiments/supplement or author-code audit.')
specs = [
    dict(id='TOOD_ICCV2021_A31', title='TOOD: Task-aligned One-stage Object Detection',
         authors=['Chengjian Feng', 'Yujie Zhong', 'Yu Gao', 'Matthew R. Scott', 'Weilin Huang'],
         url='https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html',
         tier='ICCV2021; official CVF PDF', pdf='A31_TOOD_ICCV2021.pdf',
         sidecar='A31_TOOD_ICCV2021.source.json', read_scope=tood_scope,
         author_method='Task-interactive features and layer-attention task decomposition; aligned classification sqrt(P*M); learned offsets resample four decoded box-coordinate maps; task-aligned assignment/soft targets.',
         our_consequence='Generic task alignment,shared/adaptive task features and quality-aware assignment are prior art. Eq5 does not move classification scores to predicted box centers.',
         unresolved='No equivalence between TOOD learned alignment and the A31 fixed geometric score transport; no new operator validated.'),
    dict(id='DeFCN_POTO_CVPR2021_A31', title='End-to-End Object Detection with Fully Convolutional Network',
         authors=['Jianfeng Wang', 'Lin Song', 'Zeming Li', 'Hongbin Sun', 'Jian Sun', 'Nanning Zheng'],
         url='https://arxiv.org/abs/2012.03544v3', tier='CVPR2021; inspected local arXivv3 bytes',
         pdf='早期分类复核_20261006_POTO_DeFCN_CVPR2021.pdf',
         sidecar='A31_DeFCN_readscope.source.json', read_scope=defcn_scope,
         author_method='Global one-to-one Hungarian assignment with spatial prior and class/IoU quality; 3D Max Filtering over adjacent FPN levels; auxiliary one-to-many supervision.',
         our_consequence='Global assignment and local cross-scale competitive filtering are prior art. Inspected .3dmf.aux source applies MaxFiltering to bbox-subnet features and multiplies class probability by learned filter probability.',
         unresolved='A31 does not reproduce POTO/3DMF or test a learned competition operator.'),
    dict(id='Consistent_Teacher_CVPR2023_A31', title='Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection',
         authors=['Xinjiang Wang', 'Xingyi Yang', 'Shilong Zhang', 'Yijiang Li', 'Litong Feng', 'Shijie Fang', 'Chengqi Lyu', 'Kai Chen', 'Wayne Zhang'],
         url='https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Consistent-Teacher_Towards_Reducing_Inconsistent_Pseudo-Targets_in_Semi-Supervised_Object_Detection_CVPR_2023_paper.html',
         tier='CVPR2023; official CVF PDF', pdf='A31_Consistent_Teacher_CVPR2023.pdf',
         sidecar='A31_Consistent_Teacher_CVPR2023.source.json', read_scope=ct_scope,
         author_method='For inconsistent SSOD pseudo-targets: learned 3D spatial/scale offsets resample regression features to align classification features; spatial bilinear and cross-scale interpolation.',
         our_consequence='Generic learned spatial-scale classification/regression alignment is prior art; this paper concerns semi-supervised pseudo-label consistency,not direct proof about UAV small-object failures.',
         unresolved='OnlySec3.3 inspected; no whole-method or code-reproduction claim; does not validate A31 score transport.')
]
records = []
for s in specs:
    s = dict(s)
    pdf = LIB / s.pop('pdf')
    sidecar = LIB / s.pop('sidecar')
    if sidecar.exists():
        meta = read(sidecar)
        assert meta['sha256'] == sha(pdf)
    else:
        meta = dict(kind='existing-paper-reading-scope', file=str(pdf), sha256=sha(pdf),
                    bytes=pdf.stat().st_size, url=s['url'], version='arXiv2012.03544v3,26Mar2021',
                    provenance='Reused existing local PDF; no claim of fresh download from official CVF.',
                    publication_metadata_url='https://openaccess.thecvf.com/content/CVPR2021/html/Wang_End-to-End_Object_Detection_With_Fully_Convolutional_Network_CVPR_2021_paper.html',
                    text_file=str(LIB / '_text' / 'A31_DeFCN_existing_CVPR2021.txt'))
        meta['text_sha256'] = sha(meta['text_file'])
    meta.update(read_scope=s['read_scope'], read_scope_updated_utc=NOW,
                read_by='assistant', human_read_attested=False)
    write(sidecar, meta)
    s.update(local_pdf=str(pdf), pdf_sha256=sha(pdf), access_record=str(sidecar),
             read_by='assistant', human_read_attested=False)
    records.append(s)
write(D / 'new_core_records.json', records)
comparison = {
    'phase': 'A31', 'recorded_utc': NOW,
    'scope': 'Targeted preliminary review; not systematic/exhaustive novelty clearance.',
    'closest_work': records,
    'author_source_manifest': str(D / 'selected_source_manifest.json'),
    'pinned_repositories': {
        'fcjian/TOOD': '1bda27c6f290c5b828c566eb95ea50fed75a53cb',
        'Megvii-BaseDetection/DeFCN': 'bd7e24d7408d63edc878d175d9cf798974095049'},
    'inspected_source_blobs': 8, 'author_code_imports_or_execution': 0,
    'our_intervention': 'Frozen YOLO26 one-to-one probability field,read at anchor+alpha*(predicted center-anchor),same level. Fixed analytic alpha,not learned features. Fixed-pool controls preserve anchor/class identities and boxes.',
    'original_source_conclusion': 'Task alignment,spatial-scale feature resampling and local competitive prediction mechanisms already exist; specific author methods differ as recorded.',
    'observed_evidence': 'All7predeclared A31 promotion gates fail; center transport damages AP even with fixed candidate pool.',
    'inference': 'Predicted geometric center is not a justified drop-in readout location for a frozen classification field.',
    'not_established': ['Encoder information-loss cause', 'Classification/regression misalignment as the cause of UAV misses', 'Failure of learned feature alignment', 'A new algorithm or publishable research gap'],
    'decision': 'Do not advance direct geometric center score transport; reopen architecture screening only when a concrete defect and operator difference are supported.'
}
write(D / 'closest_operator_comparison.json', comparison)

question = ('保留“已有定位覆盖却低分漏检”这一观察，重新筛查编码器到检测头的目标表征与实例判别问题；'
            '先证明具体缺陷并与已有对齐、质量评分、分配及竞争算子区分，再提出算法。直接框中心分数迁移已淘汰。')
phase = dict(
    phase='A31', classification='progress_closest_prior_clearance_and_negative_cached_control',
    recorded_utc=NOW, status='CACHE_CONTROL_AND_INDEPENDENT_REAGGREGATION_COMPLETE_NO_VALIDATED_MAINLINE',
    goal_status='active', mainline_validated=False, complete_mainline=False,
    report=str(REPORT), receipt=str(RECEIPT), new_whole_detector_forwards=0,
    new_GPU_model_calls=0, new_complete_decoder_calls=0, new_optimizer_updates=0,
    new_training_jobs=0, cached_images=100, score_control_groups=9, bilinear_level_calls=1500,
    bootstrap_draws=1000, all_seven_promotion_gates_failed=True,
    research_gap_confirmed=False, new_algorithm_approved=None,
    full_model_forwards=2466, actual_cumulative_detector_optimizer_updates=528,
    formal_detector_training_updates=526, historical_capacity_updates=2,
    cumulative_complete_decoder_only_calls=48, historical_last_layer_suffix_calls=72,
    prior_A30_files_verified=172, selected_pinned_source_blobs=8,
    new_core_papers=3, core_matrix_rows=101, candidate_count=16,
    new_PDF_downloads=2, existing_PDF_reused=1, fresh_heldout_confirmation=False,
    automatic_followups=False, next_model_call_allocation=0, A22_restarted=False,
    counter_scope='Architecture-probe counters; no model calls added. Historical D0/DA training separate.',
    next=question)

lines = [
    'A31：语义读出最近工作与否定性验证', '记录时间（UTC）：' + NOW, '',
    '当前判断：尚未找到可正式立项的小论文算法主线。保留“已有定位覆盖却低置信漏检”的观察；'
    '淘汰“把分类概率读取位置直接移到预测框中心”这一具体方案。不得把该观察直接解释为编码器丢信息、'
    '分类定位错位或分配错误，也不据此宣称新模块。', '',
    '1. 检索策略与记录',
    '采用定向前期调研，检索分类/定位任务对齐、预测框中心采样、端到端一对一检测及局部竞争。'
    '6条查询、3批浏览记录归档于web_discovery_raw.json；优先CVF/arXiv原文和作者仓库。'
    '原始下载记录primary_download_receipt.json；8个作者源码blob经Git/SHA验证，静态阅读且未导入执行。'
    '仅为此次算子查重，不是穷尽文献或新颖性证明。', '',
    '2. 核心文献矩阵与阅读范围',
]
for r in records:
    lines.extend([r['title'], '原文方法：' + r['author_method'], '已读范围：' + r['read_scope'],
                  '版本与链接：' + r['tier'] + '；' + r['url'], ''])
lines.extend([
    '3. 与最接近工作的逐项对比',
    'TOOD：已有任务交互特征与层注意力，分类为sqrt(P*M)，偏移重采样四个框坐标图；'
    '并不是将分类概率读取点固定移到预测框中心。',
    'DeFCN/POTO：已有全局一对一分配及跨尺度3D最大过滤。此次选定作者配置在框分支特征上学习过滤，'
    '再与分类概率相乘；不能简化为直接在分类概率上做局部最大值，也不能把“局部竞争”本身列为创新。',
    'Consistent-Teacher：已有学习的空间与尺度偏移，用于回归特征对齐；其半监督伪标签语境不能直接证明本任务缺陷。',
    '本次控制只改冻结模型分类场的读取点，既不是上述方法复现，也不是新的可训练对齐算子。', '',
    '4. 文献支持的边界与尚未证明的空白',
    '文献足以排除把一般任务对齐、空间尺度重采样、质量评分或局部竞争作为独立创新。'
    '已有框覆盖但低分的现象有缓存证据，却尚无证据确定原因或证明现有方法没有解决。'
    '目前没有经确认的、可支持投稿的研究空白；不能从“我们还没有方法”推导“领域没有方法”。', '',
    '5. 候选增量、可行性与风险',
    '直接预测中心分数迁移：不继续；全部7个预设门槛失败。固定候选池仍明显下降，'
    '排除了仅以新增候选或候选上限解释全部下降。',
    '学习式特征对齐、实例竞争、编码器替换：尚未批准新方案；须先找到可复现的具体失效及区别于最近工作的算子。'
    '本地CPU缓存诊断可运行；这不能证明新架构单卡24GB训练可行。100张开发图已反复使用，存在选择偏差，'
    '任何后续方法均需独立数据确认。', '',
    '6. 最小验证：已执行的冻结缓存控制',
    'VisDrone官方训练集留出的100张开发图，D0固定YOLO26s检查点；未用官方验证集作为本次确认集。'
    '固定模型和框，比较9组读取/池控制，1000次配对图像bootstrap；无新模型前向、无训练。',
    '忽略区域采用固定官方toolkit源码规则：类0区域并集覆盖>=0.5时过滤GT/预测；类11不计分类，'
    '预测先截断500再过滤且不补足。过滤后GT4489，小目标4127。',
    'AP为COCO101@500，配合官方源码对齐的忽略区预处理；不称官方VisDrone完整评测复现。'
    '主小目标按网络输入面积<1024，另报原图面积小目标。不得与A30旧内部crowd口径直接作差。',
    '主同误报预算926，取自原生one-to-one在置信度0.25时的FP；全局阈值按同分整块处理。', '',
    '组名 | 总体AP(%) | 输入小目标AP(%) | 原图小目标AP(%) | FP=926时小目标TP | 重复FP',
])
for name, m in result['metrics'].items():
    op = next(p for p in m['same_FP'] if p['FP_budget'] == 926)
    lines.append(f"{name} | {100*m['AP_original_all']:.6f} | {100*m['AP_input_small']:.6f} | {100*m['AP_original_small']:.6f} | {op['small_TP']} | {op['duplicate_FP']}")
for contrast, ci in result['bootstrap95'].items():
    lines.append(f'{contrast}：输入小目标AP差的配对95%区间 [{100*ci[0]:.6f}, {100*ci[1]:.6f}] 个百分点。')
lines.extend([
    '原生one-to-one与零位移对照AP完全相同。中心迁移相对零位移下降18.267326个百分点；'
    '固定候选池中心迁移相对原生下降7.509280个百分点。7个事先固定的推进门槛全部不通过。',
    '独立复算通过：4491个GT过滤检查、345227条保留预测的区域检查、76726条原生预过滤预测、'
    '1800个标量读取、全部9组AP和固定FP结果、1000个bootstrap抽样索引及3个抽样的独立重聚合；AP最大误差0。',
    '【实测】上述缓存分数操作有害。'
    '【推断】几何框中心不能直接当作已训练分类场的正确语义读取位置。'
    '【未验证假设】学习式特征对齐、实例判别或编码器传递机制可能仍可改善检测；本次没有验证这些假设。',
    '本阶段不自动安排训练、不调节alpha挽救失败方案；无新增模型调用预算。下一项工作是重新筛查具体架构缺陷及算子差异。', '',
    '7. 可核验参考文献、原始证据与状态',
    '三篇原文链接和阅读范围见第2项；完整条目见new_core_records.json。'
    'TOOD作者源码：https://github.com/fcjian/TOOD/tree/1bda27c6f290c5b828c566eb95ea50fed75a53cb',
    'DeFCN作者源码：https://github.com/Megvii-BaseDetection/DeFCN/tree/bd7e24d7408d63edc878d175d9cf798974095049',
    'VisDrone忽略区源码：https://github.com/VisDrone/VisDrone2018-DET-toolkit/tree/005445782213e20cb91bc50a597db3dd949e749a',
    '冻结协议：' + str(D / 'cache_protocol_FROZEN.json'),
    '结果：' + str(D / 'cache_result.json'),
    '独立核验：' + str(D / 'independent_verification.json'),
    '源码与方法对比：' + str(D / 'closest_operator_comparison.json'),
    '当前目标保持active，未找到可验证主线；累计架构分支完整前向2466、优化器更新528、完整DETR解码器重放48，均未增加。',
])
REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')

board = read(B / CANONICAL[0])
assert len(board['candidates']) == 16
board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A31', mainline_validated=False,
             latest_update_utc=NOW, latest_report=str(REPORT), current_priority_candidate=None,
             current_priority_question=question, next_exploratory_question=question,
             A31_evidence=phase,
             A31_selection_note='直接框中心概率读取及半位移控制均不推进。保留低分漏检观察，但原因与新算法均未证实；重新筛查架构缺陷。')
write(B / CANONICAL[0], board)
search = read(B / CANONICAL[1])
search.update(supplement_utc=NOW, A31_spatial_readout_closest_and_cache_control={
    'phase': phase, 'raw_queries': str(D / 'web_discovery_raw.json'), 'query_count': 6,
    'query_batches': 3, 'new_core_papers': records, 'selected_source_blobs': 8,
    'scope': 'Targeted review; selected method sections and author code; no exhaustive novelty claim.',
    'operator_comparison': str(D / 'closest_operator_comparison.json')})
write(B / CANONICAL[1], search)
matrix = read(B / CANONICAL[2])
assert len(matrix['rows']) == 98
assert not any(r.get('id') in {s['id'] for s in records} for r in matrix['rows'])
matrix['rows'].extend(records)
matrix.update(updated_utc=NOW, latest_report=str(REPORT),
              latest_empirical_boundary='A31 cached center-score transport fails all seven gates; no validated architecture cause, gap, or new algorithm. Reading scopes are selected,not full-paper claims.')
write(B / CANONICAL[2], matrix)
state = read(B / CANONICAL[3])
assert state['model_forwards'] == 2466 and state['training_updates'] == 528
state.update(recorded_utc=NOW, goal_status='active', mainline_validated=False,
             previous_goal_turn=state['current_goal_turn'], current_goal_turn=phase,
             latest_authoritative_phase=phase, latest_report=str(REPORT),
             current_priority_question=question, A31_evidence=phase)
write(B / CANONICAL[3], state)
audit = read(B / CANONICAL[4])
audit.update(updated_utc=NOW, A31_readout_closest_and_negative_controls={
    'comparison': str(D / 'closest_operator_comparison.json'), 'phase': phase,
    'read_scope': [dict(id=r['id'], scope=r['read_scope']) for r in records],
    'source_code_executed': False, 'new_algorithm_approved': None})
write(B / CANONICAL[4], audit)

evidence = [p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
evidence += [REPORT] + [B / name for name in CANONICAL]
for r in records:
    evidence.extend([pathlib.Path(r['local_pdf']), pathlib.Path(r['access_record'])])
evidence.extend([LIB / '_text' / 'A31_TOOD_ICCV2021.txt',
                 LIB / '_text' / 'A31_Consistent_Teacher_CVPR2023.txt',
                 LIB / '_text' / 'A31_DeFCN_existing_CVPR2021.txt'])
receipt = dict(phase)
receipt.update(independent_verification=verification,
               result=str(D / 'cache_result.json'),
               previous_A30_evidence_remapped_sha256=prior_remapped,
               files_sha256={str(p): sha(p) for p in sorted(set(evidence))})
write(RECEIPT, receipt)
for path, digest in receipt['files_sha256'].items():
    assert sha(path) == digest
print(json.dumps(dict(status='A31_PUBLISHED_NO_VALIDATED_MAINLINE',
                      prior_files_verified=len(prior_remapped),
                      evidence_files=len(receipt['files_sha256']), core_rows=len(matrix['rows']),
                      all_gates_failed=True, new_model_calls=0,
                      report=str(REPORT)), ensure_ascii=False))
