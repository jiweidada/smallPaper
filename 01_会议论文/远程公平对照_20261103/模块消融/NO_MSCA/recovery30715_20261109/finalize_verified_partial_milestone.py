"""Promote separately read-back and visually inspected partial evidence, never final ML results."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
import psutil

HERE = Path(__file__).resolve().parent
BUNDLE, PAPER = HERE.parents[1], HERE.parents[3]
DATA = PAPER / '实验数据汇总_20261108'
RECORDS = PAPER / '修订记录_20261103'
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text('utf-8-sig'))
relative = lambda p: p.relative_to(PAPER).as_posix()

def write_new(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

p = argparse.ArgumentParser()
p.add_argument('--round', type=int, required=True)
p.add_argument('--stamp', required=True)
args = p.parse_args()
assert 0 < args.round < 320 and args.stamp.isdigit() and len(args.stamp) == 8
round_, stamp = args.round, args.stamp
qa_path = HERE / f'parent_data{round_}_readback_{stamp}.json'
visual_path = HERE / f'parent_visual_round{round_}_{stamp}.json'
goal_path = HERE / f'goal_status_at_round{round_}_handoff_{stamp}.json'
audit_path = HERE / f'lineage_second_recovery_through_round{round_}.json'
qa, visual, goal, audit = [read(q) for q in [qa_path, visual_path, goal_path, audit_path]]
assert qa['status'] == 'INDEPENDENT_PARENT_DATA_READBACK_PASSED_PENDING_PARENT_VISUAL_QA'
assert qa['cutoff_round'] == visual['cutoff_round'] == audit['completed_rounds'] == round_
assert visual['status'] == 'PARENT_FULL_SHEET_VISUAL_QA_PASSED'
assert visual['all3_full_sheet_previews_visually_inspected_by_parent'] is True
assert goal['goal']['status'] == 'active'
assert audit['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE'
assert audit['learning_rates']['recorded_attempts'] == round_ * 1617
assert audit['learning_rates']['maximum_absolute_difference'] == audit['maximum_latest_metric_error'] == 0
assert qa['audit_sha256'] == digest(audit_path)
assert qa['verification_script_sha256'] == digest(HERE / 'verify_milestone_data_readback.py')
assert qa['per_epoch_csv_rows'] == round_ + 960
assert qa['final_NO_MSCA_cells_C6_D17_remain_empty'] == 24
assert qa['formula_cells_preserved'] == 48 and qa['negative_FULL_minus_NO_MPS_ARl_preserved']
assert qa['protected_endpoint_and_source_sheet_cell_maps_equal']
snapshot_manifest = Path(qa['permanent_snapshot_manifest'])
snapshot = snapshot_manifest.parent
assert snapshot.resolve().is_relative_to(DATA.resolve())
assert qa['permanent_snapshot_manifest_sha256'] == digest(snapshot_manifest)
assert qa['delivery_manifest_sha256'] == digest(snapshot / '交付清单.json') == digest(DATA / '交付清单.json')
xlsx = Path('outputs/实验数据汇总_20261108/实验数据汇总.xlsx')
assert qa['workbook_sha256'] == digest(snapshot / xlsx) == digest(DATA / xlsx)
assert qa['CSV_sha256'] == digest(snapshot / '逐轮指标.csv') == digest(DATA / '逐轮指标.csv')
for image in visual['previews']:
    assert digest(snapshot / image['file']) == image['sha256']
    assert digest(DATA / image['file']) == image['sha256']
assert len(visual['previews']) == 3

state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
arm, obs = state['NO_MSCA'], state['read_only_observer']
assert state['paper_revision'] == 37
paper_path = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
assert state['paper_sha256'] == digest(paper_path) == '53060de71535e78591fd5856ff537cf65c12a9463bd532e7525de490eb3c2910'
assert arm['independent_lineage_audit_round'] == round_
for pid, created in [(arm['monitor_pid'], arm['monitor_create_time']), (obs['observer_pid'], obs['observer_create_time'])]:
    assert abs(psutil.Process(pid).create_time() - created) < .01
observer_folder = Path(obs['status_path']).parent
status = read(observer_folder / 'observer_status.json')
live_path = observer_folder / 'latest_live.json'
raw = live_path.read_bytes()
live = json.loads(json.loads(raw)['stdout'])
receipt = live['records']['run/receipt.json']
assert time.time() - live_path.stat().st_mtime < 100
assert status['status'] == 'REMOTE_WORKER_CONFIRMED_LIVE' and status['backup_client_alive']
assert live['boot_id'] == arm['remote_boot_id']
for key, ticks in [('pid', arm['remote_trainer_start_ticks']), ('supervisor_pid', arm['remote_supervisor_start_ticks'])]:
    assert live['processes'][key]['exists'] and live['processes'][key]['start_ticks'] == ticks
assert status['native_accepted_round'] >= round_
live_snapshot = HERE / f'live_round{round_}_data_handoff_{stamp}.json'
acceptance_path = HERE / f'goal_acceptance_round{round_}_{stamp}.json'
handoff_path = HERE / f'round{round_}_data_handoff_verified_{stamp}.json'
supplement = RECORDS / f'主要结论证据索引/主要结论证据索引_第三十七轮_数据{round_}补充_{stamp}.json'
for path in [live_snapshot, acceptance_path, handoff_path, supplement]:
    assert not path.exists()
with live_snapshot.open('xb') as f:
    f.write(raw)
now = datetime.now(timezone.utc).isoformat()
remaining = [
    'NO_MSCA320 retained native backups and genuine successful supervisor/worker exit/final capture',
    'Independent complete320 lineage and actual terminal/secondary endpoint verification',
    'MSCA conditional result analysis and final abstract/conclusion update',
    'Complete manuscript evidence and visual acceptance review',
]
write_new(acceptance_path, {
    'status': 'PARTIAL_EVIDENCE_ACCEPTED_FULL_TASK_REMAINS_ACTIVE', 'observed_utc': now,
    'goal_objective_text': goal['goal']['objective'], 'original_monitoring_and_manuscript_scope_preserved': True,
    'paper_revision': 37, 'paper_sha256': digest(paper_path), 'independent_NO_MSCA_rounds': round_,
    'recorded_LR_attempts': round_ * 1617, 'maximum_LR_error': 0, 'maximum_saved_metric_error': 0,
    'data_rows': qa['per_epoch_csv_rows'], 'parent_data_QA': relative(qa_path), 'parent_data_QA_sha256': digest(qa_path),
    'parent_visual_QA': relative(visual_path), 'parent_visual_QA_sha256': digest(visual_path),
    'NO_MSCA320_final_endpoint_fields_filled': False, 'important_remaining_work': remaining,
    'unchanged_caveats': ['single seed and reused validation set', 'local non-official VisDrone evaluation',
        'conditional contribution and capacity limits', 'both checkpoint recoveries and unquantified unretained physical work'],
    'venue_selection_and_template_adaptation_deferred_by_user': True, 'submission_ready': False,
    'goal_status': 'active', 'goal_receipt': relative(goal_path), 'goal_receipt_sha256': digest(goal_path),
})
previous_index = Path(state['current_evidence_index'])
previous_supplement = Path(state['latest_evidence_index_supplement'])
assert digest(previous_index) == state['current_evidence_index_sha256']
assert digest(previous_supplement) == state['latest_evidence_index_supplement_sha256']
write_new(supplement, {
    'status': f'IMMUTABLE_ROUND{round_}_DATA_EVIDENCE_SUPPLEMENT_NO_NEW_MANUSCRIPT_CLAIMS',
    'revision': 37, 'manuscript_sha256': digest(paper_path), 'progress_cutoff': round_,
    'previous_index': relative(previous_index), 'previous_index_sha256': digest(previous_index),
    'previous_data_supplement': relative(previous_supplement), 'previous_data_supplement_sha256': digest(previous_supplement),
    'evidence': [{'file': relative(q), 'sha256': digest(q)} for q in [audit_path,
        snapshot_manifest, snapshot / '交付清单.json', qa_path, visual_path, acceptance_path]],
    'new_scientific_endpoints_or_external_sources': False,
    'all_permanent_data_references_use_frozen_snapshot_paths': True, 'submission_ready': False,
})
arm['data_snapshot_independently_verified_round'] = round_
arm['data_handoff'] = {
    'path': str(DATA / '交付清单.json'), 'sha256': digest(DATA / '交付清单.json'),
    'path_role': 'Current mutable delivery pointer; permanent proof is the separately bound snapshot',
    'data_rows': qa['per_epoch_csv_rows'], 'NO_MSCA_verified_round': round_,
    'immutable_snapshot_manifest': str(snapshot_manifest), 'immutable_snapshot_manifest_sha256': digest(snapshot_manifest),
}
arm['current_data_delivery_manifest'] = {
    'path': str(DATA / '交付清单.json'), 'sha256': digest(DATA / '交付清单.json'),
    'scope': 'Current delivery at this acceptance; permanent proof in separately bound immutable snapshot',
}
arm['latest_data_snapshot_manifest'] = {'file': str(snapshot_manifest), 'sha256': digest(snapshot_manifest)}
arm['last_accepted_round'] = status['native_accepted_round']
arm['last_observed_running_round'] = arm['current_running_round_at_snapshot'] = receipt['actual_loop_epoch'] + 1
arm['last_remote_live_observation'] = {
    'file': str(live_snapshot), 'wrapper_sha256': digest(live_snapshot), 'receipt': receipt,
    'gpu': live.get('gpu'), 'observed_local_utc': status['observed_local_utc'],
    'scope': f'Fresh exact live handles; metric and data evidence independently verified through{round_}',
}
arm['latest_status_evidence'] = {'file': str(live_snapshot), 'sha256': digest(live_snapshot)}
state.update(observed_local_utc=status['observed_local_utc'], goal_status='active', goal_objective=goal['goal']['objective'],
    last_goal_turn_classification='progress', current_goal_acceptance_audit=str(acceptance_path),
    current_goal_acceptance_audit_sha256=digest(acceptance_path), latest_evidence_index_supplement=str(supplement),
    latest_evidence_index_supplement_sha256=digest(supplement), goal_tool_status_observed_receipt=str(goal_path),
    goal_status_observed_local_utc=now, operational_work_status=f'NO_MSCA_RUNNING_ROUND{round_}_AUDIT_AND_DATA_VERIFIED')
state[f'data{round_}_refresh_pending'] = False
pending = state_path.with_name(state_path.name + f'.data{round_}_partial')
assert not pending.exists()
pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
pending.replace(state_path)
next_round = arm['next_independent_audit_round']
heading = (
    f'第{round_}轮独立核查及数据交付（{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}）\n'
    f'NO_MSCA已独立核验{round_}轮、{round_ * 1617}次学习率与保存数组指标，最大差异0；'
    f'Excel/CSV/JSON共{qa["per_epoch_csv_rows"]}条逐轮记录，原{qa["old_csv_exact_byte_prefix_preserved_rows"]}行字节前缀保留。'
    f'当前交付{qa["delivery_payload_files_byte_verified"]}项及清单（{qa["immutable_snapshot_payload_files_byte_verified"]}个payload，另含snapshot_manifest）固定于不可变快照，'
    '三张完整工作表由父代理实际查看，48公式、负ARl及24个空缺320轮端点格均核对。\n'
    f'原训练进程继续，现场在本次固定快照中为第{receipt["actual_loop_epoch"] + 1}轮；'
    f'只读观察器PID{obs["observer_pid"]}/session{obs["exec_session_id"]}到{next_round}轮，'
    f'原逐轮备份PID{arm["monitor_pid"]}/session{arm["monitor_session_id"]}保持。'
    '论文仍为第三十七轮；完整320轮、最终MSCA比较和稿件验收尚待完成，尚不能投稿。以下为历史记录。\n\n'
)
for entry in [PAPER / '当前修订入口.txt', PAPER / '目标自主执行入口_20261109.txt',
              RECORDS / '目标验收与阻塞记录.txt', PAPER / '远程公平对照_20261103/结果并入与验收清单.txt']:
    entry.write_text(heading + entry.read_text('utf-8-sig'), encoding='utf-8')
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as f:
    f.write(f'\n\n## {stamp[:4]}-{stamp[4:6]}-{stamp[6:]} 第{round_}轮独立验收及数据交付\n\n' + heading)
write_new(handoff_path, {
    'status': f'ROUND{round_}_INDEPENDENT_AUDIT_DATA_AND_PARENT_VISUAL_QA_ACCEPTED',
    'observed_utc': now, 'this_goal_turn_classification': 'progress',
    'audit_sha256': digest(audit_path), 'data_snapshot_manifest_sha256': digest(snapshot_manifest),
    'workbook_sha256': qa['workbook_sha256'], 'data_rows': qa['per_epoch_csv_rows'],
    'metric_cutoff_round': round_, 'partial_best_round': audit['best_epoch_index'] + 1,
    'current_native_accepted_round': status['native_accepted_round'],
    'current_observed_training_round': receipt['actual_loop_epoch'] + 1,
    'backup_pid': arm['monitor_pid'], 'backup_create_time': arm['monitor_create_time'], 'backup_session_id': arm['monitor_session_id'],
    'observer_pid': obs['observer_pid'], 'observer_create_time': obs['observer_create_time'], 'observer_session_id': obs['exec_session_id'],
    'next_independent_audit_round': next_round, 'next_audit_epoch': next_round - 1,
    'binding_sha256': arm['next_lineage_audit_binding_sha256'],
    'parent_QA_sha256': digest(qa_path), 'parent_visual_QA_sha256': digest(visual_path),
    'acceptance_audit_sha256': digest(acceptance_path), 'evidence_index_supplement_sha256': digest(supplement),
    'paper_revision': 37, 'paper_sha256': digest(paper_path),
    'goal_status': 'active', 'full_task_complete': False, 'submission_ready': False,
    'training_actions': 0, 'model_forwards': 0, 'evaluator_runs': 0,
})
print(json.dumps({'status': f'ROUND{round_}_HANDOFF_VERIFIED', 'native_accepted_round': status['native_accepted_round'],
    'running_round': receipt['actual_loop_epoch'] + 1, 'independent_round': round_,
    'data_rows': qa['per_epoch_csv_rows'], 'next_audit_round': next_round,
    'paper_revision': 37, 'goal_status': 'active', 'submission_ready': False}))
