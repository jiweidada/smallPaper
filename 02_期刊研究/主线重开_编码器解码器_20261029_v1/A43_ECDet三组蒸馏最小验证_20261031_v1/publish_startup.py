import datetime
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
B = D.parent
LOG = B.parent / '前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v):p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    verification = read(D / 'startup_download_verification.json')
    assert verification['status'] == 'PASS'
    previous_path = B / '99_A42终点独立核验与去留回执.json'
    previous = read(previous_path)
    for name, digest in previous['files_sha256'].items():assert sha(Path(name)) == digest, name
    snapshot = D / 'canonical_before_A43';snapshot.mkdir(exist_ok=False)
    remap = {}
    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
        source = B / name;dest = snapshot / name;dest.write_bytes(source.read_bytes())
        remap[str(source)] = {'snapshot': str(dest), 'sha256': sha(dest)}
    historical = {'status': 'PASS', 'source_receipt': str(previous_path), 'source_receipt_sha256': sha(previous_path),
                  'all_previous_receipt_files_verified': len(previous['files_sha256']), 'remap': remap}
    write(D / 'historical_A42_canonical_remap.json', historical)
    records = []
    with LOG.open(encoding='utf-8') as f:
        for line in f:
            try:r = json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('id') == 'A43_startup_live_02' and r.get('action') == 'exec':records.append(r)
    assert len(records) == 1 and records[0]['exit_code'] == 0
    obs = json.loads(records[0]['stdout'])
    assert obs['processes']['supervisor']['identity_matches'] and obs['processes']['B']['identity_matches']
    progress = obs['arms']['B'];assert progress['optimizer_updates'] > 0 and progress['status'] == 'TRAINING'
    assert progress['teacher_forwards'] == 0
    write(D / 'startup_live_observation.json', obs)
    cap = read(D / 'startup_evidence/capacity_preflight_result.json')
    plan = read(D / 'startup_evidence/execution_protocol_FROZEN.json')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = B / '100_A43蒸馏三组已启动与固定判据.txt'
    receipt_path = B / '101_A43启动容量与冻结核验回执.json'
    question = '相同冻结教师下，以最终匹配query监督早期空间特征，是否比普通区域特征蒸馏更有效提升小目标APs？'
    full = 34010 + 8 + 4 + progress['full_model_forwards']
    updates = 29664 + 5 + progress['optimizer_updates']
    event = {'phase': 'A43', 'event': 'B_M_Q_PILOT_LAUNCHED_ORIGINAL_RUNNING', 'recorded_utc': now,
             'goal_status': 'active', 'status': 'ORIGINAL_RUNNING_M_Q_QUEUED',
             'mainline_selected_for_pilot': True, 'mainline_validated': False,
             'candidate_id': 'A41-B', 'phase_alias': 'A43-B', 'question': question,
             'active_arm': 'B', 'arms_queued': ['M', 'Q'], 'epochs_max_each': 12,
             'seed': 20261101, 'microbatch': 8, 'gradient_accumulation': 1,
             'source_pin': plan['source_pin'], 'protocol_sha256': verification['protocol_sha256'],
             'capacity_status': 'PASS', 'capacity_student_forwards': 8, 'capacity_teacher_forwards': 4,
             'capacity_optimizer_updates': 5, 'capacity_auxiliary_gradient_probes': 2,
             'capacity_peak_allocated_bytes': {a: row['peak_allocated_bytes'] for a, row in cap['arms'].items()},
             'zero_output_effect_bitwise_equal': True, 'capacity_teacher_streams_identical': True,
             'capacity_teacher_frozen_unchanged': True, 'native_export_strict_load_verified': True,
             'formal_M_Q_teacher_pairing_verified': False, 'comparative_AP_available': False,
             'live_observation': obs, 'cumulative_full_model_forwards_lower_bound': full,
             'cumulative_optimizer_updates_with_capacity_lower_bound': updates,
             'new_teacher_forwards_A43_lower_bound': 4,
             'new_formal_student_updates_lower_bound': progress['optimizer_updates'],
             'next': 'Finish existing sequential B/M/Q to fixed12EMA, independently verify endpoints and paired teacher stream; retire if practical gates fail. No rescue sweep or automatic next-direction training.',
             'report': str(report), 'receipt': str(receipt_path),
             'execution_protocol': str(D / 'startup_evidence/execution_protocol_FROZEN.json'),
             'remote_root': '/root/c10_region_restore_20261029_v1/architecture_probe_v1/a43_ecdet_distill_three_arm_20261031_v1',
             'A42_restarted': False, 'A22_restarted': False, 'old_DEIM_restarted': False,
             'compression_branch_reopened': False, 'other_direction_training_queued': False}
    lines = ['Material Passport', '类型：ARS experiment-agent / run，单助手。',
        '状态：目标active，正式原版已启动，区域蒸馏M和实例query监督Q顺序排队。尚无本候选AP结论。', '',
        '当前问题：' + question,
        '三组均为VisDrone + ECDet-S，每组最多12轮，统一固定第12轮EMA；B为fresh原版，不续训A42。',
        '教师固定A42原版B的第12轮EMA，sha256 ' + plan['teacher_checkpoint_sha256'] + '。教师eval/no_grad，全参数冻结。',
        'M/Q同一教师、同一增强输入、同一最终Hungarian匹配实例、同一48→192学生投影和实例平均1-cosine（固定权重1）。',
        'M目标为教师hybrid encoder的stride16区域特征，Q目标为匹配实例的最终query。学生均读已有stride4 stem的GT3×3 ROIAlign均值。GT仅训练期使用。',
        '推理时物理移除辅助模块及采集hook；模型预测架构保持原版。原生decoder源码逐字节不变，原生检测loss与LQE/GO-LSD保留。', '',
        '已完成接口/容量验收：',
        '- CPU严格装载及投影器初始化一致；参数B9803112，M/Q9812520，额外9408为训练期参数。',
        '- 附加/删除投影器后原生预测逐位一致；辅助梯度可到达stem；教师参数/状态不变；M/Q实际匹配和两类目标摘要一致。',
        '- 同一个固定dense batch8，GT数902/512/462/431/370/351/348/346；B1次更新、M/Q各2次。',
        '- 峰值allocated GiB：B%.4f/M%.4f/Q%.4f，低于22GiB；未搜索batch或回退。' % tuple(cap['arms'][a]['peak_allocated_bytes'] / 2**30 for a in ['B', 'M', 'Q']),
        '- 容量合计8次学生前向、4次教师前向、5次更新、2次辅助梯度探针；这些权重均丢弃，正式训练重新装载公共初始化。',
        '- 密集batch教师最多匹配2400个实例，1322个GT受300query上限未匹配；M/Q使用相同集合。该覆盖限制保留并按轮计数，不偷偷增加query。', '',
        '正式冻结条件：seed20261101，单卡RTX3090 24GiB，640stretch，batch8，不累计；确定性翻转；AdamW backbone2.5e-5/其他及投影5e-4，clip0.1，500warmup及12轮cosine。BF16模型前向，FP32损失，无GradScaler。',
        '各组6471训练图、548验证图；预定第4/8/12轮验证，评价与延时均去辅助模块；终点checkpoint的model/EMA为原版键，训练投影状态另存。',
        '评价COCO-style：原图裁剪GT面积small≤1024、maxDet300、class0矩形并集IoA≥0.5排除GT/预测；others11不计类。不是VisDrone官方榜单分数。',
        '去留门槛：Q-B APs≥0.5pp、Q-M APs≥0.3pp、Q-B总AP≥-0.2pp，三项同时满足；APs75与实际原生延时也报告。门槛是资源判断，非显著性检验。',
        '若Q未超过普通区域蒸馏，停止直接query目标主张；不扫层数、权重、教师或阈值。三组结束后只准备下一候选，不自动另开训练。', '',
        '实际启动核查：B第%d轮%d/809步，累计%d次更新，loss%.6f；supervisor21467/start_ticks105744454，B21562/start_ticks105744796与/proc一致。' % (progress['epoch'], progress['microbatch'], progress['optimizer_updates'], progress['loss']),
        '启动证据9个归档成员以及冻结91个文件与本地来源SHA核验通过。完整正式M/Q配对与指标核验尚待运行完成，不能把容量检查当效果证据。',
        '所有进度计数为保存时的观测下界；总完整前向同时包含学生和新加入的冻结教师调用，教师调用单独记账。', '',
        'A42阴性结果已保留，旧草案仍在原路径作为历史；本次执行协议在：' + event['execution_protocol'],
        '当前目标：' + str(D / 'explicit_goal_creation_receipt.json'),
        '研究边界：当前没有证明语义瓶颈、算法有效、新颖性或期刊价值。单seed12轮即使过门槛也仅表示值得下一阶段验证。']
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    board_path = B / '01_候选判别表.json';state_path = B / '15_当前状态与结果校验.json'
    board, state = read(board_path), read(state_path)
    for row in board['candidates']:
        if row['id'] == 'A41-B':
            row.update(status='PILOT_RUNNING_B_M_Q_FROZEN', question=question, pilot_recommended=True,
                       efficacy_validated=False, increment_approved=None, evidence=str(report), next=event['next'])
    board.update(status='A43_ACTIVE_B_RUNNING_M_Q_QUEUED', latest_update_utc=now, goal_status='active',
                 mainline_selected_for_pilot=True, mainline_validated=False, current_priority_candidate='A41-B',
                 current_priority_question=question, latest_report=str(report), A43_execution=event,
                 pilot_validation_status='A43_B_RUNNING_M_Q_QUEUED', remaining_deliverable=event['next'])
    state.update(recorded_utc=now, goal_status='active', goal_status_scope='New user-authorized A43 execution goal active; A42 goal remains historical complete.',
                 mainline_selected_for_pilot=True, mainline_validated=False, model_forwards=full,
                 training_updates=updates, formal_detector_training_updates=29650 + progress['optimizer_updates'],
                 capacity_only_optimizer_updates=19, detector_optimizer_updates_including_capacity=updates,
                 formal_training_epochs=36 + progress['completed_epochs'], training_counts_terminal=False,
                 counts_terminal=False, counters_observed_lower_bounds_while_job_live=True,
                 current_native_job_terminal=False, full_forward_cap_closed=False,
                 cap=49905, cap_scope='A43 only:30477formalstudent+19416formalteacher+8capacitystudent+4capacityteacher. Older phase caps stay closed.',
                 training_update_scope='Architecture cumulative actual updates include prior29664+newcapacity5+observedformalstudent updates; frozen teacher has0updates.',
                 full_model_forwards_scope='Architecture cumulative prior34010 plus A43 student AND teacher full forward entries; teacher separately reported. Saved live lower bound.',
                 teacher_forwards_A43=4, current_user_task_status='RUNNING_A43_THREE_ARM_PILOT',
                 current_goal_turn=event, current_user_task_phase=event, latest_authoritative_phase=event,
                 latest_report=str(report), current_priority_question=question, A43_execution=event)
    state.setdefault('authoritative_phase_history', []).append(event)
    write(board_path, board);write(state_path, state)
    evidence = [report, board_path, state_path, D / 'startup_download_verification.json',
                D / 'startup_live_observation.json', D / 'historical_A42_canonical_remap.json',
                D / 'explicit_goal_creation_receipt.json', D / 'startup_evidence/execution_protocol_FROZEN.json',
                D / 'startup_evidence/capacity_preflight_result.json']
    receipt = dict(event);receipt.update(previous_A42_files_verified=len(previous['files_sha256']),
        previous_canonical_snapshot_remap=historical, files_sha256={str(p): sha(p) for p in evidence})
    write(receipt_path, receipt);write(D / 'startup_state.json', event);write(D / 'goal_live_latest.json', event)
    print(json.dumps({'status': event['status'], 'goal_status': 'active', 'B_updates': progress['optimizer_updates'],
                      'report': str(report), 'receipt': str(receipt_path), 'evidence_files_hashed': len(evidence)}))


if __name__ == '__main__':main()
