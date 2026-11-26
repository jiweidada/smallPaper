"""Bind actual recovery progress, native backup and independent saved-state audit."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

here = Path(__file__).resolve().parent
arm = here.parent
bundle = here.parents[1]
paper = here.parents[3]
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return sha256(path.read_bytes()).hexdigest()
audit_path = here / 'lineage_recovered_through_round030.json'
audit = read(audit_path)
assert audit['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and audit['completed_rounds'] == 30
assert audit['learning_rates']['recorded_attempts'] == 48510 and audit['learning_rates']['maximum_absolute_difference'] == 0
assert audit['validation_images'] == 548 and audit['predictions'] == 164400
assert audit['maximum_latest_metric_error'] == 0
assert audit['recovery']['new_completed_rounds'] == 1 and audit['recovery']['new_optimizer_update_attempts'] == 1617
native = read(here / 'first_recovered_round030_native_acceptance.json')
assert native['status'] == 'FIRST_RECOVERED_ROUND_NATIVE_BACKUP_ACCEPTED' and native['round'] == 30
live_path = here / 'live_training_progress_05.json'
live = json.loads(read(live_path)['stdout'])
receipt = live['records']['run/receipt.json']
assert receipt['pid'] == 1779501 and receipt['actual_loop_epoch'] == 30 and receipt['optimizer_step_attempts'] >= 48511
assert live['processes']['pid']['start_ticks'] == 18098010 and live['processes']['pid']['exists']
files = [arm / 'execution_protocol.json', here / 'recovery_addendum.json',
    here / 'recovery_package_verified.json', here / 'target_preflight_readonly.json',
    here / 'resume_ast_equivalence.json', here / 'launch_readiness_verified.json',
    here / 'recovery_contract_round029_verified_v2.json',
    here / 'first_recovered_round030_native_acceptance.json', audit_path, live_path,
    here / 'installation_python38_failure_and_fix.json']
evidence = {'status': 'ROUND030_RECOVERY_STATE_LINEAGE_AND_NATIVE_BACKUP_VERIFIED',
    'observed_local_utc': datetime.now(timezone.utc).isoformat(), 'user_facing_local_date': '2026-11-08',
    'arm': 'NO_MSCA', 'retained_completed_rounds': 30, 'live_round_at_snapshot': 31,
    'source_bindings': [{'file': str(path), 'sha256': sha(path)} for path in files],
    'current_target': {'ssh_port': 30219, 'gpu_uuid': 'GPU-06ccf9c8-5f92-5aad-3458-e162506dd1dc',
        'boot_id': live['boot_id'], 'supervisor_pid': 1779500, 'trainer_pid': 1779501,
        'trainer_start_ticks': 18098010, 'backup_client_pid': 18924, 'backup_client_session_id': 8984},
    'learning_rates': audit['learning_rates'], 'validation_images': 548, 'predictions': 164400,
    'maximum_saved_metric_error': 0, 'rolling_checkpoint_sha256': native['rolling_sha256'],
    'progress_metrics_only': audit['latest_metrics'], 'scientific_endpoint_available': False,
    'recovery_limitation': audit['recovery']['physical_compute_limitation'],
    'local_audit_model_forwards': 0, 'local_audit_evaluator_runs': 0,
    'new_completed_training_rounds': 1, 'remaining_retained_rounds_after30': 290,
    'remaining_retained_optimizer_updates_after30': 468930,
    'next_independent_audit_round': 40,
    'next_auditor': str(here / 'audit_recovered_bootstrap.py'),
    'paper_revision': 35, 'submission_ready': False}
out = bundle / 'recovery_round030_evidence_supplement_20261108.json'
with out.open('x', encoding='utf-8') as handle:
    json.dump(evidence, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
state_path = bundle / 'current_execution_state.json'
state = read(state_path)
current = state['NO_MSCA']
current.update(independent_state_audit_round=30, independent_lineage_audit_round=30,
    latest_recovered_lineage_audit=str(audit_path), latest_recovered_lineage_audit_sha256=sha(audit_path),
    latest_recovery_evidence_supplement=str(out), latest_recovery_evidence_supplement_sha256=sha(out),
    next_planned_lineage_audit_round=40, next_lineage_auditor=str(here / 'audit_recovered_bootstrap.py'))
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = '''【第30轮恢复与独立核验通过：2026-11-08本地日期】
NO_MSCA已在30219从第29轮完整断点恢复。第30轮完成548图验证、164400条预测，并通过原生manifest/ACK/accepted核验备份到本地；第31轮已实际开始。完整保存状态及30轮轨迹独立核验通过，48510条四组学习率误差0，保存预测/评价数组/12指标误差0。本地审计没有执行模型或评价器。
原冻结协议、配置、248个冻结输入不变；目标GPU更换由独立恢复附录精确绑定，原始/EMA/优化器/预热/全部随机状态恢复，新增段只补第30–320轮。失联后未保留的物理计算量未知，不声称从未中断或同等物理计算。安装的Python3.8准备失败记录已保留；随后用原Python3.10完成安装，该失败没有启动训练。
已完成BASE/FULL/NO_MPS均320轮，另有Excel/CSV/JSON与完整逐轮来源整理；NO_MSCA的320轮终点仍缺。当前稿件第三十五轮，尚不能投稿。接下来需要完整MSCA终点、全轨迹和真实退出核验，再修订结果分析与全文。目标工具旧blocked记录与本次已恢复的实际运行分别记载，不能按旧入口失联状态报告当前实验。
证据：模块消融/recovery_round030_evidence_supplement_20261108.json；新恢复审计器为NO_MSCA/recovery30219_r29_20261108/audit_recovered_bootstrap.py，后续不得仅用要求从零训练的旧终点审计器。
'''
for path in [paper / '当前修订入口.txt', paper / '修订记录_20261103/目标验收与阻塞记录.txt',
             paper / '远程公平对照_20261103/结果并入与验收清单.txt']:
    path.write_text(note + '\n' + path.read_text(encoding='utf-8'), encoding='utf-8')
with (bundle / '方案与执行记录.md').open('a', encoding='utf-8') as handle:
    handle.write('\n\n## 2026-11-08本地日期：恢复第30轮完整验证/备份/独立轨迹核验\n\n' + note + '\n')
print(json.dumps({'evidence': str(out), 'independent_verified_round': 30, 'live_round': 31,
                  'paper_revision': 35, 'submission_ready': False}))
