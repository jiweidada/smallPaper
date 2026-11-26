"""Bring the mutable acceptance checklist up to date, preserving its history."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REMOTE = HERE.parent
PAPER = REMOTE.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


pair_path = REMOTE / '断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
pair = read(pair_path)
assert pair['status'] == 'RESUMED_PAIR_ENDPOINTS_VERIFIED_PENDING_MANUSCRIPT_REVIEW'
late_path = HERE / 'NO_MPS/lineage_through_round309.json'
late = read(late_path)
assert late['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and late['completed_rounds'] == 309
assert late['rolling_state_restored_after_evaluation'] is True
assert late['learning_rates']['recorded_attempts'] == 499653
assert late['learning_rates']['maximum_absolute_difference'] == 0
source_path = HERE / 'module_shared_conditions_verified_20261108.json'
assert read(source_path)['status'] == 'FROZEN_REMOVAL_CONDITIONS_AND_INITIAL_TENSORS_VERIFIED'
diagnostic_path = REMOTE / '诊断准备_20261105/remote_final/diagnostic_result.json'
diagnostic = read(diagnostic_path)
assert diagnostic['status'] == 'DIAGNOSTIC_COMPLETE_NOT_A_FAIR_PAIR'
assert diagnostic['optimizer_updates'] == 17787 and diagnostic['full_validation_passes'] == 14
master = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
assert digest(master) == 'b3aae17cf12667f34797179f399dcfd608edfab724633f023921c9703b0d4cee'
checklist = REMOTE / '结果并入与验收清单.txt'
backup = REMOTE / '结果验收清单_更新前_20261108.txt'
assert not backup.exists()
old_hash = digest(checklist)
shutil.copy2(checklist, backup)
assert digest(backup) == old_hash
header = (
    '2026-11-08 当前结果并入与验收状态\n\n'
    '一、已完成且已并入正文\n'
    'BASE/FULL 两组均已完成 320 个保留轮次并真实退出 0。实际第 320 轮 EMA 主端点和第 303/305 轮最高 AP 次端点已通过配对核验，'
    '不能继续把 FULL 标成未完成。权重、预测、评价数组、十二指标、恢复轨迹及退出回执的总证据为：\n'
    '断点续训_20261105/BASE_FULL_pair_verified_20261108.json\n'
    '持续维护的英文稿为第三十三轮，SHA-256 b3aae17cf12667f34797179f399dcfd608edfab724633f023921c9703b0d4cee，8 页逐页渲染检查通过。'
    '主结果为 Table 1，图 6 展示全部 320 轮 AP/APs；原有六张历史表及其完整上下文已移出正文并留档，原稿与逐轮备份保留。'
    '主端点 AP 提升 3.529724 个百分点、APs 提升 3.232599 个百分点；同时说明参数 3,737,611→7,540,942，约 2.02 倍。\n\n'
    '二、模块移除条件及当前证据\n'
    'NO_MPS/NO_MSCA 已获授权，各 320 轮，seed 0、batch 4、FP32。共同条件补核已通过：227 份共享源码、全部 101 项依赖版本、'
    '展开配方、冻结数据清单与真实首 100 批一致。初始权重直接逐张量比较通过：NO_MPS 684 个张量，其中 40 个替换 stem 张量来自 BASE，'
    '其余来自 FULL；NO_MSCA 673 个保留张量来自 FULL。NO_MPS/NO_MSCA 完整冻结输入包均已本机归档并逐文件核对。'
    '证据见模块消融/module_shared_conditions_verified_20261108.json 与 module_conditions_evidence_supplement_20261108.json。\n'
    'NO_MPS 使用原始 BASE StemBlock 并改其激活为 SiLU，其余 FULL 结构保留；NO_MSCA 仅移除两处 Lite 门和一处 Dilated 门，'
    '保留投影、ESE、SimAM、MPS 和 SiLU。不能把它解释成整个分层注意力或聚合路径的移除。BASE 不等同于双移除组。\n'
    'NO_MPS 第 309 轮的实际后期恢复已独立核验，499,653 条学习率匹配，13 次恢复记录衔接正确；本轮评测后滚动状态恢复到阶段一，'
    '因此滚动 next_epoch.pt 不能冒充本轮评测权重。证据为模块消融/NO_MPS/lineage_through_round309.json。'
    'NO_MSCA 前 5 轮状态和轨迹已核验，8,085 条学习率匹配。最新完成轮次以逐轮回执和 current_execution_state.json 为准。\n\n'
    '三、尚未完成的实证验收\n'
    '1. 等待 NO_MPS 和 NO_MSCA 各自完成实际第 320 轮。每臂的监督器必须真实 exit 0，并保存 completed_run_receipt.json 与 '
    'completed_supervisor.json；仅有 RUNNING、锁文件或终轮目录不算完成。\n'
    '2. 使用 audit_ablation_lineage.py 对各臂零基索引 319 执行完整核验，检查 517,440 条实际学习率、全部 320 行、共享初始化、'
    '真实后期恢复、规范化 EMA 指纹、预测与评价数组及十二指标、最终回执和真实进程身份。只分析既有产物，不新增推理或评价。\n'
    '3. 主端点仍为实际第 320 轮、回滚前已评测 EMA，必须取 terminal_epoch0319.pt 及 terminal.*。次端点为全程 AP 最高的完整同一行，'
    '平局选最早轮；APs 和其他指标随同一权重，不能各自择峰。不能按结果修改主端点。\n'
    '4. 汇总 FULL 与各移除组的绝对结果、带符号的 FULL 减移除组百分点差以及参数代价。原始数据为比例，差值乘 100 后以百分点报告，'
    '不写成相对百分比。改善、退步或混合结果均保留，不以历史 N0–N4、s 或诊断轨迹替代新臂。\n'
    '5. 正文按结果修改 4.4 节、相应表格、摘要与结论，核对全部数字、图文、引用，再渲染检查所有页面。新实验、调参、改精度、'
    '改预算、加组或加 seed 均不从现有授权推导；有必要时另行说明问题、证据不足和最小方案后取得授权。\n\n'
    '四、解释与验收边界\n'
    '保留的 320 轮轨迹不等于两组物理总计算消耗完全相等：BASE 从 214 轮锚点恢复时排除了原中断分支之后 11 行，后来还重跑了部分第 309 轮。'
    'BASE/FULL 首次恢复缺主进程随机状态，FULL 第 1 轮缺逐批学习率；两个移除组从零开始。共同配方及初始共享张量相同，不能扩写为全程随机轨迹相同或连续从头精确复现。\n'
    '验证集同时用于恢复决策和次端点选择；单 seed 不提供独立测试或跨运行不确定性。移除对照的观察只适用于其余 FULL 设置存在时，'
    '包括被移除部件自身的容量变化，不能证明 MPS×MSCA 协同、显著性、机制独立性或稳定性。\n'
    '此前单独授权的 11 轮基线诊断已经完成 17,787 次更新及 14 次验证，没有复现原骤降；它不是主对照，也不能证明根因或排除某种机制。'
    '它继续保留为独立诊断记录，不进入主/消融结果表。\n'
    '当前本地 COCO-style 协议与官方 VisDrone ignore/maxDets 不同；旧 s 速度不能绑定到新 n 权重，正文已不作速度或参数效率优势结论。'
    '选会后的模板和页数适配按用户要求后置，目前没有宣称满足特定会议格式。\n\n'
    '当前状态：两个模块终点与最终贡献分析尚未完成，不能宣布可投稿。以下为 2026-11-06 及更早历史记录；其中原串行/双机队列及预留文件退出规则不适用于当前已完成的独立续训监督器。\n\n')
old = checklist.read_text(encoding='utf-8-sig')
old = old.replace('2026-11-06 当前续训验收', '2026-11-06 历史续训验收', 1)
checklist.write_text(header + old, encoding='utf-8')
state_path = HERE / 'current_execution_state.json'
state = read(state_path)
state['NO_MPS']['independent_lineage_audit_round'] = 309
state['NO_MPS']['late_post_restore_state_verified_round'] = 309
state['acceptance_checklist_sha256'] = digest(checklist)
state['observed_local_utc'] = datetime.now(timezone.utc).isoformat()
temporary = state_path.with_suffix('.acceptance.tmp')
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, state_path)
with (HERE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-08 真实后期恢复核验与验收清单更新\n\n'
                 'NO_MPS 第 309 轮经 audit_ablation_lineage 独立核验：已完成 309 轮，499,653 条学习率误差为 0，'
                 '13 次恢复记录正确，评测后滚动状态确实恢复为阶段一。已有诊断档案再次确认只是一条未复现旧骤降的独立轨迹，不混入主结果。'
                 '结果并入与验收清单已更新至当前主对照/稿件状态与模块待验收步骤，旧清单完整备份。没有新推理、评价或训练。\n')
receipt = {'status': 'ACCEPTANCE_CHECKLIST_UPDATED_MODULE_TERMINALS_PENDING',
           'observed_local_utc': state['observed_local_utc'],
           'checklist_before_sha256': old_hash, 'preserved_backup_sha256': digest(backup),
           'checklist_after_sha256': digest(checklist),
           'late_round309_audit_sha256': digest(late_path),
           'diagnostic_result_sha256': digest(diagnostic_path),
           'manuscript_unchanged_sha256': digest(master), 'submission_ready': False}
with (HERE / 'acceptance_checklist_update_20261108.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'late_post_restore_verified_round': 309,
                  'manuscript_unchanged_sha256': receipt['manuscript_unchanged_sha256']}))
