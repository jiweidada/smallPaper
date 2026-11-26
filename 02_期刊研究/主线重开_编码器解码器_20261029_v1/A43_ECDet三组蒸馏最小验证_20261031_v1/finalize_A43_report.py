"""Write the final A43 report and scoped canonical state, never run a model."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

D = Path(__file__).resolve().parent
BASE = D.parent
V = D / 'result_verification_v1'
N = D / 'next_candidate_A44_v1'
REPORT = BASE / '104_A43三组终点去留与下一候选.txt'
RECEIPT = BASE / '105_A43终点与目标完成审计回执.json'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    gate = read(V / 'final_practical_gate_verification.json')
    controls = read(V / 'B_M_Q_controls_verified.json')
    stats = read(V / 'statistical_interpretation_11_check.json')
    proposal = read(N / 'A44_candidate_proposal.json')
    assert gate['status'] == controls['status'] == 'PASS'
    assert not gate['pilot_pass'] and controls['formal_M_Q_teacher_pairing_verified']
    assert len(stats['checks']) == 11 and stats['coverage'] == '11/11 checked'
    assert proposal['training_allocation_now'] == 0 and not proposal['mainline_validated']
    old_receipt = BASE / '103_A43原版终点独立核验回执.json'
    old = read(old_receipt)
    snap = D / 'state_before_A43_terminal'
    snap.mkdir(exist_ok=True)
    remap_file = D / 'B_terminal_state_historical_remap.json'
    if not remap_file.exists():
        # Verify old signed files BEFORE changing the two canonical states.
        for name, digest in old['files_sha256'].items():
            assert sha(Path(name)) == digest, name
        mapping = {}
        for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
            src = BASE / name
            target = snap / name
            content = src.read_bytes()
            if target.exists():
                assert target.read_bytes() == content
            else:
                target.write_bytes(content)
            mapping[str(src)] = {'preserved_as': str(target), 'sha256': sha(target)}
        write(remap_file, {'status': 'PASS', 'source_receipt': str(old_receipt),
                          'source_receipt_sha256': sha(old_receipt), 'mapping': mapping,
                          'scope': 'Only mutable01/15 moved to verified historical snapshots; receipt103 unmodified.'})
    else:
        saved = read(remap_file)
        assert saved['source_receipt_sha256'] == sha(old_receipt)
        for value in saved['mapping'].values():
            assert sha(Path(value['preserved_as'])) == value['sha256']

    metrics = gate['metrics']
    rows = {a: read(D / 'completed_results' / a / ('arm_' + a) / 'result.json')
            for a in ['B', 'M', 'Q']}
    capacity = read(D / 'startup_evidence/capacity_preflight_result.json')
    prior = read(D / 'canonical_before_A43/15_当前状态与结果校验.json')
    formal_student = sum(r['full_model_forwards'] for r in rows.values())
    formal_teacher = sum(r['teacher_forwards'] for r in rows.values())
    formal_updates = sum(r['optimizer_updates'] for r in rows.values())
    current_forwards = formal_student + formal_teacher + capacity['new_student_forwards'] + capacity['new_teacher_forwards']
    totals = {'A43_formal_student_forwards': formal_student,
              'A43_formal_teacher_forwards': formal_teacher,
              'A43_formal_updates': formal_updates,
              'A43_capacity_student_forwards': capacity['new_student_forwards'],
              'A43_capacity_teacher_forwards': capacity['new_teacher_forwards'],
              'A43_capacity_optimizer_updates': capacity['optimizer_updates'],
              'A43_full_forwards_with_capacity': current_forwards,
              'A43_updates_with_capacity': formal_updates + capacity['optimizer_updates'],
              'architecture_cumulative_full_forwards': prior['model_forwards'] + current_forwards,
              'architecture_cumulative_updates_with_capacity': prior['training_updates'] + formal_updates + capacity['optimizer_updates'],
              'architecture_cumulative_formal_updates': prior['formal_detector_training_updates'] + formal_updates,
              'architecture_cumulative_capacity_updates': prior['capacity_only_optimizer_updates'] + capacity['optimizer_updates'],
              'architecture_cumulative_epochs_sum_across_arms': prior['formal_training_epochs'] + sum(r['epochs'] for r in rows.values())}
    assert totals['A43_full_forwards_with_capacity'] == 49905
    assert totals['architecture_cumulative_full_forwards'] == 83915
    assert totals['architecture_cumulative_updates_with_capacity'] == 58793
    assert totals['architecture_cumulative_formal_updates'] == 58774
    assert totals['architecture_cumulative_capacity_updates'] == 19
    now = datetime.now(timezone.utc).isoformat()
    table = '\n'.join(f"{a}\t{metrics[a]['AP']:.10f}\t{metrics[a]['APs']:.10f}\t{metrics[a]['APs75']:.10f}\t{gate['latency_median_ms'][a]:.4f}"
                      for a in ['B', 'M', 'Q'])
    qb = gate['deltas_vs_B_pp']['Q']
    qm = gate['Q_minus_M_pp']
    mb = gate['deltas_vs_B_pp']['M']
    report = f"""## Material Passport
类型：ARS experiment-agent / run与终点validate；附下一候选的非系统定向前期调研。
生成UTC：{now}
本轮三组最小验证已完成，结论为停止A43-B；整体研究主线仍未验证。A44仅为下一候选草案，训练分配0。
证据：冻结协议/源码、原始标注、保存的权重和预测、独立NumPy矩阵复算、CPU严格加载、真实/proc身份退出记录。未重新跑网络生成预测。
正式审计回执：{RECEIPT}

1. 项目结果与去留
候选问题：同一冻结教师下，以最终匹配query监督早期空间特征，是否比普通区域特征蒸馏更有效提升小目标APs？
三组各12轮，固定第12轮EMA。B是fresh原版；M普通区域蒸馏；Q直接实例query监督。
本项目COCO-style口径，非VisDrone官方榜单。AP单位为百分数，差值为百分点。
组别\t总AP\t小目标APs\t小目标AP75\t3090原生推理中位ms
{table}

M-B：总AP {mb['AP']:+.10f}，APs {mb['APs']:+.10f}，小目标AP75 {mb['APs75']:+.10f}pp。
Q-B：总AP {qb['AP']:+.10f}，APs {qb['APs']:+.10f}，小目标AP75 {qb['APs75']:+.10f}pp。
Q-M：总AP {qm['AP']:+.10f}，APs {qm['APs']:+.10f}，小目标AP75 {qm['APs75']:+.10f}pp。

预先冻结的三项门槛：
Q-B APs≥+0.5pp：未通过；实际{qb['APs']:+.10f}。
Q-M APs≥+0.3pp：未通过；实际{qm['APs']:+.10f}。
Q-B总AP≥-0.2pp：通过；实际{qb['AP']:+.10f}。
三项必须同时通过。因此停止直接query目标监督这一候选的当前投入，不追加教师/层数/权重/阈值搜索。
M在本次协议中也未提高APs。某一控制组或个别类别的正差值不被另行包装成创新。

2. 实际方法和公平比较
VisDrone6471训练图、548验证图；10类；640stretch、ImageNet归一化、确定性翻转。单卡RTX3090，batch8，seed20261101。
共同COCO初始化转10类；都是fresh学生，不是容量权重或A42学生续训。
固定教师始终为旧A42原版B第12轮EMA，SHA bbd5712fe42bc7b11e3b01d46e930f7baa3992dd76832a913b367a58626a6ad6。没有替换为新A43基线。
学生stride4空间特征在匹配GT区域3x3 ROIAlign均值后，经同一48→192投影并L2归一化。
M目标是教师hybrid encoder第2个输出的对应区域均值；Q目标是教师最终分类Linear输入的匹配实例query。
两组都计算两种目标，仅选择用于loss的目标不同；固定平均1-cosine，权重1。teacher eval/no_grad、先于学生前向，保持冻结。
匹配由教师原生最终输出Hungarian分配给实际GT；没有按置信度、类别或尺寸另选子集。每图超过300query容量的GT不匹配并单独计数。匹配可用性不等于正确检测率。
保留原生检测loss。共同AdamW、500步warmup、12轮cosine、BF16前向/FP32 loss与ROI、clip0.1；相同原生EMA和4/8/12轮验证。
详细超参和来源以startup_evidence/execution_protocol_FROZEN.json及下载源码为准。协议SHA cb82214e481387ecf3510f6f7ff76d2161e5828136ba1571b81d5b4dc1365280。
这不是作者完整训练配方/COCO榜单复现；也没有新encoder/decoder架构成立结论。

3. 核验覆盖与资源闭环
一次固定密集batch8容量检查，GT数902/512/462/431/370/351/348/346，峰值约15.65GiB。B1、M2、Q2次容量更新；容量权重全部弃用。辅助分支零预测影响、梯度到达stem、教师冻结及原生导出检查通过。
正式每组9708更新、10159学生完整前向；教师B0、M9708、Q9708。各组epoch1..12完整，退出0，无自动重试或额外轮次。
M/Q12轮全部教师输出、实例分配和两种目标的逐batch滚动摘要一致；B/M/Q12轮采样顺序及每轮前8实际输入batch摘要一致。未逐像素比较每个batch。
每组216条保存的采样进度标量有限，并结合源码逐步loss/全局梯度非有限失败检查；未保存每一步完整梯度张量。
raw与EMA均严格CPU加载为9803112参数原生推理模型；M/Q另存9408个辅助参数，有限且已更新，辅助优化器step9708。推理图中移除辅助模块和hook。
各组归档128个成员逐SHA通过：B166932405、M167041240、Q167114793字节；终点摘要9成员也通过。全部权重、预测、GT、矩阵、日志、协议和源码已本地留存。
独立NumPy计分未导入训练评价器/torch/pycocotools；每组40400精度单元及400召回值逐值一致。标量仅可有约1e-14pp浮点求和舍入。
548份原始标注逐文件校验、独立矩形并集算法重建38602评价GT与157忽略GT；检查保留预测B153766/M153729/Q153867均符合ignore规则。此检查不重新生成被排除/原始网络预测。
APs按裁剪后原图GT面积≤1024，maxDet300；class0并集IoA≥0.5处理，11类不计分。不能跨口径直接对比文献榜单数值。
batch1 BF16原生eager推理warm10+测30；上表实测中位数用于环境记录，不是机载/TensorRT速度或额外分支成本的证明。
终点采集时，四个冻结作业身份均已从/proc消失，GPU compute列表为空；没有排队的其他方向训练。

A43正式学生前向{formal_student}、教师前向{formal_teacher}、更新{formal_updates}；含容量总前向{current_forwards}、更新{totals['A43_updates_with_capacity']}。
架构分支累计完整前向83915，更新58793（正式58774、容量19），训练epoch合计72是不同组相加，每组仍最多12轮。旧D0/DA不在该累计口径内。
CPU/NumPy核验与文献/源码阅读新增网络前向和更新均0。A42阴性保留，旧A22/DEIM/压缩分支未重启。

4. 解释边界：事实、推断、假设
事实：本次Q未超过B/M，冻结投入门槛失败；完整输入/教师目标配对及独立计分支持这一项目结论。
资源判断：停止当前直接query目标主张。不是统计零假设检验，也不能断言更长训练、其他教师或所有query蒸馏永远无效。
尚未证明：浅层语义不足是漏检原因、最终query不适合监督、某类encoder/decoder信息已丢失、或新架构创新成立。
11/11统计与方法谬误检查完成，解释置信度CAUTION。单seed12轮，公开验证集此前多次参与选题，不给p值/CI/显著性或泛化结论。
类别仅作描述：Q-B APs有3/10类正差，Q-M有2/10；没有用其替换主终点、挑子集挽救或追加检验。

5. 下一候选与本轮边界
A44：同一12个逻辑采样位置预算下，按预测框细网格占据为query分配一次细/中尺度读取。
已有ECDet固定[3,6,3]归属，原生offset/attention已经query自适应。草案仅把slot3按min(ref_w*W/8,ref_h*H/8)<2从stride16改读stride8。
最小三组：N0原版；N1所有query静态[4,5,3]；N2预测几何条件化切换。沿用三组同预算12轮，若N2不超过N1和原版便停止，不展开新的探针或阈值搜索。
先做一次有界接入/梯度与容量检查，再冻结运行协议。新实现/容量/速度/收益均未验证，本轮未启动。
RT-DETRv2、AdaMixer和Better Sampling说明静态不等点数、动态尺度采样、采样监督均有直接先例。潜在增量只在明确同读取预算的离散归属及其收益；新颖性风险高，没有“无人做过”结论。
该候选是独立待验证假设，不能从A43阴性推导其必然有效。最小方案通过后仍需已知连续3D sampler对照、多seed/长训练/独立数据，才谈期刊投入。
完整问题、具体算子、最近工作逐项对比、主要风险、可承担验证路线与7条检索记录：
{N / 'A44_候选问题与最小验证路线.txt'}
{N / 'A44_candidate_proposal.json'}
新增阅读矩阵单列addendum，原117行主矩阵保持不变；这不是系统综述或全领域新颖性排查。

6. 可核验来源与阅读范围
RT-DETRv2, arXiv2407.17140v1（完整2.1、选读3.4）：
https://arxiv.org/html/2407.17140v1
AdaMixer, CVPR2022 / arXiv2203.16507v1（完整3.2-3.3、3.4开头，未阅读全文/复现）：
https://arxiv.org/html/2203.16507v1
Better Sampling, arXiv2407.06127v1（本次重读完整4.1及选读4.2/4.3）：
https://arxiv.org/html/2407.06127v1
AdaMixer作者采样函数网页已读；本地GitHub API pin因TLS EOF失败，main网页未声称不可变commit固定：
https://github.com/MCG-NJU/AdaMixer/blob/main/mmdet/models/roi_heads/bbox_heads/sampling_3d_operator.py
当前宿主原文/已有近邻读法与范围见原14_架构核心文献矩阵.json；本次A43的可复算项目证据见result_verification_v1。
"""
    REPORT.write_text(report, encoding='utf-8')
    event = {'phase': 'A43', 'event': 'ALL_ENDPOINTS_VERIFIED_CANDIDATE_RETIRED_NEXT_A44_PROPOSED',
             'recorded_utc': now, 'status': 'BOUNDED_TASK_FULFILLED_COMPLETION_AUDIT_REQUIRED',
             'mainline_validated': False, 'pilot_pass': False, 'verified_endpoint_arms': ['B', 'M', 'Q'],
             'completed_epochs_by_arm': {a: 12 for a in rows}, 'active_arm': None,
             'endpoint_metrics': metrics, 'deltas_vs_B_pp': gate['deltas_vs_B_pp'],
             'Q_minus_M_pp': qm, 'practical_gates': gate['practical_gates'],
             'latency_median_ms': gate['latency_median_ms'], 'counts': totals,
             'formal_M_Q_teacher_pairing_verified': True, 'all_job_identities_absent': True,
             'other_direction_training_queued': False, 'rescue_sweep_started': False,
             'A42_restarted': False, 'A22_restarted': False, 'old_DEIM_restarted': False,
             'compression_branch_reopened': False,
             'new_model_forwards_by_postprocessing': 0, 'new_optimizer_updates_by_postprocessing': 0,
             'next_candidate': proposal['candidate_id'], 'next_candidate_training_allocation': 0,
             'next_candidate_status': proposal['status'], 'report': str(REPORT), 'receipt': str(RECEIPT),
             'next': 'A43 stopped. A44 is an untrained proposal; no automatic next-direction training.',
             'goal_completion_scope': 'Three-arm12epoch pilot, independent verification, frozen decision and bounded next-candidate proposal; NOT validated mainline/novelty.'}
    write(D / 'final_report_evidence.json', event)
    board_path = BASE / '01_候选判别表.json'
    board = read(board_path)
    for row in board['candidates']:
        if row['id'] == 'A41-B':
            row.update(status='RETIRED_AFTER_A43_FIXED12_PRACTICAL_GATES_FAILED',
                       pilot_recommended=False, efficacy_validated=False, pilot_gate_pass=False,
                       evidence=str(REPORT), next='No rescue sweep; preserve negative A43 evidence.')
    next_row = {'id': proposal['candidate_id'], 'status': proposal['status'],
                'question': proposal['question'], 'nearest': ['RT-DETRv2', 'AdaMixer', 'Better Sampling/SPR'],
                'increment_approved': None, 'pilot_recommended': True,
                'pilot_recommendation_scope': 'Only the proposed bounded test; no training allocated or novelty established.',
                'efficacy_validated': False, 'novelty_cleared': False, 'training_allocation': 0,
                'evidence': str(N / 'A44_candidate_proposal.json')}
    old_next = [r for r in board['candidates'] if r['id'] == proposal['candidate_id']]
    if old_next:
        assert len(old_next) == 1
        old_next[0].update(next_row)
    else:
        board['candidates'].append(next_row)
    board.update(status='A43_RETIRED_A44_CONCRETE_PROPOSAL_NOT_RUNNING',
                 mainline_validated=False, mainline_selected_for_pilot=False,
                 goal_status='complete', latest_update_utc=now, latest_report=str(REPORT),
                 current_priority_candidate=proposal['candidate_id'],
                 current_priority_question=proposal['question'], current_host_candidate='ECDet-S',
                 selection_stage_complete=True, pilot_validation_status='A43_FAILED_FIXED_GATES_STOPPED',
                 remaining_deliverable='Current bounded task fulfilled; overall research mainline not validated. A44 remains proposed/untrained.',
                 next_decision_is_not_algorithm_approval=True)
    board['goal_status_scope'] = 'Local bounded-task fulfilment; final app goal result recorded separately after completion audit.'
    board['A43_terminal_result'] = event
    board['A43_execution_scope'] = 'Historical startup snapshot; superseded by A43_terminal_result.'
    board['A44_proposal'] = next_row
    board['new_empirical_evidence'] = 'A43 fixed12EMA independent verification: Q APs14.234834, Q-B-.122546pp, Q-M-.072709pp; both gain gates failed, candidate retired.'
    board['next_exploratory_question'] = proposal['question']
    board['next_decision_protocol'] = {'status': 'A44_DRAFT_NOT_RUNNING',
                                       'proposal': str(N / 'A44_candidate_proposal.json'),
                                       'training_allocation': 0,
                                       'scope': 'Future bounded proposal, not current A43 protocol or novelty approval.'}
    write(board_path, board)

    state_path = BASE / '15_当前状态与结果校验.json'
    state = read(state_path)
    state['previous_goal_turn'] = {'classification': 'verified_wait', 'phase': 'A43 Q epoch9',
                                  'evidence': str(D / 'goal_progress_11_Q_epoch9_verified_wait.json')}
    state.update(recorded_utc=now, goal_status='complete',
                 goal_status_scope=board['goal_status_scope'], mainline_validated=False,
                 diagnostics_complete=True, mainline_selected_for_pilot=False,
                 selection_stage_complete=True, model_forwards=totals['architecture_cumulative_full_forwards'],
                 training_updates=totals['architecture_cumulative_updates_with_capacity'],
                 detector_optimizer_updates_including_capacity=totals['architecture_cumulative_updates_with_capacity'],
                 formal_detector_training_updates=totals['architecture_cumulative_formal_updates'],
                 capacity_only_optimizer_updates=totals['architecture_cumulative_capacity_updates'],
                 formal_training_epochs=totals['architecture_cumulative_epochs_sum_across_arms'],
                 formal_training_epochs_scope='Sum across earlier and current arms, not any single model. Current B/M/Q each exactly12.',
                 training_counts_terminal=True, counts_terminal=True,
                 counters_observed_lower_bounds_while_job_live=False, full_forward_cap_closed=True,
                 current_native_job_terminal=True, cap=49905,
                 cap_scope='A43 new full student+teacher calls including capacity, not cumulative model_forwards.',
                 full_model_forwards_scope='Architecture cumulative whole-detector calls including A43 frozen-teacher calls; excludes old D0/DA branch.',
                 teacher_forwards_A43=19420, latest_report=str(REPORT),
                 current_priority_question=proposal['question'],
                 current_user_task_status='A43_VERIFIED_RETIRED_A44_PROPOSAL_ONLY',
                 latest_authoritative_phase=event, current_goal_turn=event, current_user_task_phase=event)
    state.setdefault('authoritative_phase_history', []).append(event)
    state['A43_terminal_verification'] = event
    state['A43_execution_scope'] = board['A43_execution_scope']
    state['A44_proposal'] = next_row
    state.setdefault('verification', {})['A43_terminal'] = str(V / 'final_practical_gate_verification.json')
    state['goal_completion_tool_receipt'] = str(D / 'goal_completion_tool_receipt.json')
    write(state_path, state)
    print(json.dumps({'report': str(REPORT), 'decision': gate['decision'], 'counts': totals,
                      'candidate_count': len(board['candidates']),
                      'app_goal_update_called': False, 'next': 'Run full completion audit before goal update.'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

