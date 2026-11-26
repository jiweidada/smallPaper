"""Bind the verified200-round delivery to current monitoring and unchanged revision37."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
import psutil

HERE = Path(__file__).resolve().parent
BUNDLE, PAPER = HERE.parents[1], HERE.parents[3]
DATA = PAPER / '实验数据汇总_20261108'
SNAPSHOT = DATA / 'verified_snapshots/NO_MSCA_round200_verified_20261110'
OBSERVER = HERE / 'resume_monitor_20261110_003'
RECORDS = PAPER / '修订记录_20261103'
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text('utf-8-sig'))
relative = lambda p: p.relative_to(PAPER).as_posix()

def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')

expected = {
    DATA / '交付清单.json': '99fa19019d0d989da06cea9606d31f6a2413665dec0e32848151c660149d9b74',
    SNAPSHOT / 'snapshot_manifest.json': '43ffa891fc777ebf5c3800c043394a5e745d64c12699cec2f2b874d8d3b7b239',
    DATA / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx': 'f8a07aae1245e6f6708f5b8090be8f34aa0e41eeb91caf98ceb82c3de3eb1d1a',
    DATA / '逐轮指标.csv': '737c985b7b399898c66d99451a13c8b8254359435e4a8385a81165e1c587cbf3',
    DATA / '第200轮更新核查.json': 'fa212b024138b3d7c815d209a7458b232a34ca82e1736ff54a1d91a78faf7c18',
    DATA / '_build/round200_workbook_checks.json': '784add4c08b140091cf035682e282c3c1c79ca19aad636c2e4ab33df4ff3293a',
    HERE / 'lineage_second_recovery_through_round200.json': 'c451348bb627f7be74fd9b56deaa36b7299742537e63aaed01505bed9dfaeb11',
    PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx': '53060de71535e78591fd5856ff537cf65c12a9463bd532e7525de490eb3c2910',
}
for path, value in expected.items(): assert digest(path) == value, path
goal_receipt = HERE / 'goal_status_at_round200_handoff_20261110.json'
assert read(goal_receipt)['goal']['status'] == 'active'
for pid, created in [(8216, 1794292724.8624065), (40816, 1794312084.181138)]:
    assert abs(psutil.Process(pid).create_time() - created) < .01
status = read(OBSERVER / 'observer_status.json')
live_path = OBSERVER / 'latest_live.json'
raw = live_path.read_bytes(); live = json.loads(json.loads(raw)['stdout'])
receipt = live['records']['run/receipt.json']
assert time.time() - live_path.stat().st_mtime < 100
assert status['status'] == 'REMOTE_WORKER_CONFIRMED_LIVE' and status['backup_client_alive']
assert live['boot_id'] == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
for key, ticks in [('pid', 26597622), ('supervisor_pid', 26597399)]:
    assert live['processes'][key]['exists'] and live['processes'][key]['start_ticks'] == ticks
snapshot_live = HERE / 'live_round200_data_handoff_20261110.json'
qa_path = HERE / 'parent_data200_readback_20261110.json'
acceptance_path = HERE / 'goal_acceptance_round200_20261110.json'
handoff_path = HERE / 'round200_data_handoff_verified_20261110.json'
supplement = RECORDS / '主要结论证据索引/主要结论证据索引_第三十七轮_数据200补充_20261110.json'
for path in [snapshot_live, qa_path, acceptance_path, handoff_path, supplement]: assert not path.exists()
with snapshot_live.open('xb') as f: f.write(raw)
now = datetime.now(timezone.utc).isoformat()
previews = [
    ('_build/previews/round200_progress_after.png', '9ea92cbf4de9180545d5fc04e054341b529adbfdc172dce14a423a85e41682e8'),
    ('_build/previews/round200_endpoints_full_after.png', '84543e008356f3dfb295085c9cac443cc35afaceb59f686c957d7aa9bcc961f7'),
    ('_build/previews/round200_sources_full_after.png', 'e9aecc70047e9c1dc0dbb8c21ac1aaa550c1f9658b09807aea41920f789f17a7'),
]
for path, value in previews: assert digest(SNAPSHOT / path) == value
write(qa_path, {
    'status': 'PARENT_DATA200_READBACK_AND_VISUAL_QA_PASSED', 'observed_utc': now,
    'current_delivery_payload_files_byte_verified': 25, 'immutable_snapshot_payload_files_byte_verified': 26,
    'old165_payload_files_byte_verified_unchanged': 21,
    'per_epoch_CSV_rows': 1160, 'CSV_by_arm': {'BASE': 320, 'FULL': 320, 'NO_MPS': 320, 'NO_MSCA': 200},
    'old1125_rows_exact_byte_prefix_preserved': True, 'new_rows_against_native_JSONL': 35,
    'new_metric_values_independently_compared': 420,
    'new_native_loss_counters_LR_metadata_tensor_prediction_array_hashes_equal': True,
    'progress_XLSX_metric_cells_verified': 12, 'maximum_workbook_metric_cell_error': 0.0,
    'final_NO_MSCA_cells_C6_D17_remain_empty': 24, 'formula_cells_preserved': 48,
    'negative_FULL_minus_NO_MPS_ARl_preserved': True,
    'all3_full_sheet_previews_visually_inspected_by_parent': True,
    'visual_result': 'Progress200 and partial best199 are distinct; 24 final cells empty. Text and all values readable with no clipping or overlap. Endpoint and source sheet renders remain byte-identical.',
    'parent_reader_note': 'Initial read-only OOXML reader joined an absolute relationship target incorrectly; package URI resolution was corrected, then all worksheet and source checks passed. No artifact edit resulted.',
    'permanent_preview_sources': [{'file': relative(SNAPSHOT / p), 'sha256': h} for p, h in previews],
    'permanent_delivery_manifest': {'file': relative(SNAPSHOT / '交付清单.json'), 'sha256': digest(SNAPSHOT / '交付清单.json')},
    'permanent_snapshot_manifest': {'file': relative(SNAPSHOT / 'snapshot_manifest.json'), 'sha256': digest(SNAPSHOT / 'snapshot_manifest.json')},
    'model_forwards': 0, 'evaluator_runs': 0,
})
remaining = [
    'NO_MSCA320 retained native backups and genuine successful supervisor/worker exit/final capture',
    'Independent complete320 lineage and actual terminal/secondary endpoint verification',
    'MSCA conditional result analysis and final abstract/conclusion update',
    'Complete manuscript evidence and visual acceptance review',
]
write(acceptance_path, {'status': 'PARTIAL_EVIDENCE_ACCEPTED_FULL_TASK_REMAINS_ACTIVE',
    'observed_utc': now, 'goal_objective_text': 'resume', 'original_monitoring_and_manuscript_scope_preserved': True,
    'paper_revision': 37, 'paper_sha256': expected[PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'],
    'independent_NO_MSCA_rounds': 200, 'recorded_LR_attempts': 323400, 'maximum_LR_error': 0,
    'maximum_saved_metric_error': 0, 'data_rows': 1160,
    'parent_data_QA': relative(qa_path), 'parent_data_QA_sha256': digest(qa_path),
    'NO_MSCA320_final_endpoint_fields_filled': False, 'important_remaining_work': remaining,
    'unchanged_caveats': ['single seed and reused validation set', 'local non-official VisDrone evaluation',
                         'conditional contribution and capacity limits', 'both checkpoint recoveries and unquantified unretained physical work'],
    'venue_selection_and_template_adaptation_deferred_by_user': True, 'submission_ready': False,
    'goal_status': 'active', 'goal_receipt': relative(goal_receipt), 'goal_receipt_sha256': digest(goal_receipt)})
previous_index = RECORDS / '主要结论证据索引/主要结论证据索引_第三十七轮.json'
write(supplement, {'status': 'IMMUTABLE_ROUND200_DATA_EVIDENCE_SUPPLEMENT_NO_NEW_MANUSCRIPT_CLAIMS',
    'revision': 37, 'manuscript_sha256': expected[PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'],
    'previous_index': relative(previous_index), 'previous_index_sha256': digest(previous_index),
    'evidence': [{'file': relative(p), 'sha256': digest(p)} for p in [
        HERE / 'lineage_second_recovery_through_round200.json', SNAPSHOT / 'snapshot_manifest.json',
        SNAPSHOT / '交付清单.json', qa_path, acceptance_path]],
    'new_scientific_endpoints_or_external_sources': False, 'progress_cutoff': 200,
    'all_permanent_data_references_use_frozen_snapshot_paths': True, 'submission_ready': False})
state_path = BUNDLE / 'current_execution_state.json'; state = read(state_path); arm = state['NO_MSCA']
arm['data_snapshot_independently_verified_round'] = 200
arm['data_handoff'] = {'path': str(DATA / '交付清单.json'), 'sha256': digest(DATA / '交付清单.json'),
    'path_role': 'Current mutable delivery pointer; permanent proof is the separately bound snapshot',
    'data_rows': 1160, 'NO_MSCA_verified_round': 200,
    'immutable_snapshot_manifest': str(SNAPSHOT / 'snapshot_manifest.json'),
    'immutable_snapshot_manifest_sha256': digest(SNAPSHOT / 'snapshot_manifest.json')}
arm['last_accepted_round'] = status['native_accepted_round']; arm['last_observed_running_round'] = receipt['actual_loop_epoch'] + 1
arm['last_remote_live_observation'] = {'file': str(snapshot_live), 'wrapper_sha256': digest(snapshot_live),
    'receipt': receipt, 'gpu': live.get('gpu'), 'observed_local_utc': status['observed_local_utc'],
    'scope': 'Fresh exact live handles; metric evidence remains cut at independently verified200'}
state.update(observed_local_utc=status['observed_local_utc'], goal_status='active', goal_objective='resume',
    last_goal_turn_classification='progress', current_goal_acceptance_audit=str(acceptance_path),
    current_goal_acceptance_audit_sha256=digest(acceptance_path), latest_evidence_index_supplement=str(supplement),
    latest_evidence_index_supplement_sha256=digest(supplement), data200_refresh_pending=False)
state['next_action']['kind'] = 'verified_wait_for_native_round240_then_full_CPU_audit_and_data_update'
state['goal_tool_status_observed_receipt'] = str(goal_receipt)
pending = state_path.with_name(state_path.name + '.round200_partial')
pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'); pending.replace(state_path)
heading = ('【200轮独立核验与数据整理：2026-11-10】\n'
    'NO_MSCA已独立核验200轮，323400条学习率与保存评价指标误差0；Excel/CSV/JSON共1160条逐轮记录，'
    '原1125行原字节前缀保持，24个320轮端点格仍空。当前交付25项及清单共26项固定到200轮不可变快照，'
    '三张完整工作表预览已由父代理逐页检查。\n'
    '003只读观察器PID40816/session29581跟到240轮，原生逐轮备份PID8216/session56233继续。'
    '训练现场进度另有带时间的快照，未混入200轮指标。持续稿仍为第三十七轮；'
    '完整320轮及论文最终验收尚待完成，尚不能投稿。以下为历史记录。\n\n')
for p in [PAPER / '当前修订入口.txt', PAPER / '目标自主执行入口_20261109.txt',
          RECORDS / '目标验收与阻塞记录.txt', PAPER / '远程公平对照_20261103/结果并入与验收清单.txt']:
    p.write_text(heading + p.read_text('utf-8-sig'), encoding='utf-8')
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as f:
    f.write('\n\n## 2026-11-10 第200轮独立核验及数据交接\n\n' + heading)
write(handoff_path, {'status': 'ROUND200_INDEPENDENT_AUDIT_DATA_AND_PARENT_VISUAL_QA_ACCEPTED',
    'observed_utc': now, 'previous_goal_turn_classification': 'verified_wait', 'this_goal_turn_classification': 'progress',
    'audit_sha256': digest(HERE / 'lineage_second_recovery_through_round200.json'),
    'data_snapshot_manifest_sha256': digest(SNAPSHOT / 'snapshot_manifest.json'),
    'workbook_sha256': digest(DATA / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx'),
    'data_rows': 1160, 'metric_cutoff_round': 200, 'partial_best_round': 199,
    'current_native_accepted_round': status['native_accepted_round'],
    'current_observed_training_round': receipt['actual_loop_epoch'] + 1,
    'backup_pid': 8216, 'backup_create_time': 1794292724.8624065, 'backup_session_id': 56233,
    'observer_pid': 40816, 'observer_create_time': 1794312084.181138, 'observer_session_id': 29581,
    'next_independent_audit_round': 240, 'next_audit_epoch': 239,
    'binding_sha256': '6de82b5b4bcbf8bbdbd3a61a2fb7ecbd70905b98da0eceefc55c7b451435a41e',
    'parent_QA_sha256': digest(qa_path), 'acceptance_audit_sha256': digest(acceptance_path),
    'evidence_index_supplement_sha256': digest(supplement), 'paper_revision': 37,
    'paper_sha256': expected[PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'],
    'goal_status': 'active', 'full_task_complete': False, 'submission_ready': False,
    'training_actions': 0, 'model_forwards': 0, 'evaluator_runs': 0})
print(json.dumps({'status': 'ROUND200_HANDOFF_VERIFIED', 'native_accepted_round': status['native_accepted_round'],
    'running_round': receipt['actual_loop_epoch'] + 1, 'independent_round': 200, 'data_rows': 1160,
    'next_audit_round': 240, 'paper_revision': 37, 'goal_status': 'active', 'submission_ready': False}))
