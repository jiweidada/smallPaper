"""Record the observed missing processes and preserve the distinction between partial and final results."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import os

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
RECORDS = PAPER / '修订记录_20261103'
LEDGER = ROOT / 'parallel_execution_ledger.json'
PREVIOUS = ROOT / 'parallel_evidence/observation_lost_ledger_20261105.json'
REPORT = ROOT / 'interrupted_runs_report_20261105.json'
SUMMARY = ROOT / '实验中断与证据保全_20261105.txt'
PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def remote(path):
    obj = load(path)
    assert obj['exit_code'] == 0 and not obj['stderr']
    return json.loads(obj['stdout'])

def exclusive(path, data):
    with path.open('xb') as f:
        f.write(data)

def ref(path):
    return {'path': str(path), 'sha256': sha(path)}

assert not any(p.exists() for p in [PREVIOUS, REPORT, SUMMARY])
previous_bytes = LEDGER.read_bytes()
previous = json.loads(previous_bytes)
assert previous['status'] == 'OBSERVATION_CONNECTION_LOST_TRAINING_UNVERIFIED'
primary_path = ROOT / 'reconnected_primary_inspection_20261105.json'
secondary_path = ROOT / 'reconnected_secondary_inspection_20261105.json'
primary, secondary = remote(primary_path), remote(secondary_path)
prior_secondary = remote(ROOT / 'secondary_inventory.json')
assert secondary['hostname'] == prior_secondary['hostname']
assert 'GPU-27a58478-346d-c441-dda0-26a5877409c9' in primary['gpu']['stdout']
assert 'GPU-4bd279c5-4e39-bd08-618e-20da276d13ed' in secondary['gpu']['stdout']
assert 'GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a' in prior_secondary['gpu']['stdout']
for obj, ids in [(primary, [1193, 1194]), (secondary, [564, 567])]:
    for pid in ids:
        process = next(p for p in obj['previous_processes'] if p['pid'] == pid)
        assert not process['live'] and not process['matches_prior_identity']
primary_paths = {p['path']: p for p in primary['paths']}
secondary_paths = {p['path']: p for p in secondary['paths']}
remote_root = '/25085400120/mpsma_fair_20261103_v1'
assert primary_paths[remote_root + '/execution_protocol.json']['sha256'] == PROTOCOL
assert not secondary_paths[remote_root]['exists']
primary_receipt = primary_paths[remote_root + '/runs/BASE/receipt.json']['selected_fields']
assert primary_receipt['completed_epochs'] == 225 and primary_receipt['actual_loop_epoch'] == 225
assert primary_receipt['optimizer_step_attempts'] == 363926
full_snapshot_path = ROOT / 'parallel_evidence/FULL_before_connection_loss_20261104.json'
full = remote(full_snapshot_path)
assert full['arms']['FULL']['receipt']['completed_epochs'] == 128
assert full['arms']['FULL']['complete_epoch_rows'] == 128
download_path = ROOT / 'interrupted_base_download_verified_20261105.json'
download = load(download_path)
assert download['status'] == 'INCOMPLETE_BASE_ARTIFACTS_PRESERVED_HASH_MATCHED'
assert len(download['files']) == 25 and download['total_bytes'] == 207491171
analysis_path = ROOT / 'interrupted_base_analysis_20261105.json'
analysis = load(analysis_path)
audit_path = ROOT / 'interrupted_base_saved_file_audit_20261105.json'
audit = load(audit_path)
lr_path = ROOT / 'interrupted_base_lr_interval_20261105.json'
lr = load(lr_path)
assert audit['status'] == 'BEST_CHECKPOINT_BOUND_LAST_ARRAYS_ONLY'
assert audit['saved_file_results']['best']['metrics']['AP'] == analysis['best_saved']['AP']
assert audit['saved_file_results']['latest']['metrics']['AP'] == analysis['last_complete']['AP']
assert lr['logged_steps_in_interval'] == 24255
assert analysis['saved_lr_stream']['gzip_missing_end_marker']
now = datetime.now(timezone.utc).isoformat()
report = {'status': 'ORIGINAL_PROCESS_IDENTITIES_ABSENT_AFTER_RECONNECTION', 'recorded_local_utc': now,
          'protocol_sha256': PROTOCOL, 'primary_reconnected_inspection': ref(primary_path),
          'secondary_reconnected_inspection': ref(secondary_path),
          'secondary_preparation_inventory': ref(ROOT / 'secondary_inventory.json'),
          'last_verified_full_snapshot': ref(full_snapshot_path),
          'preserved_base_files': ref(download_path), 'base_saved_file_analysis': ref(analysis_path),
          'base_saved_file_audit': ref(audit_path), 'base_lr_interval_analysis': ref(lr_path),
          'BASE': {'status': 'ORIGINAL_SUPERVISOR_AND_TRAIN_PID_ABSENT_IN_RECONNECTED_ENVIRONMENT',
                   'old_gpu_uuid': 'GPU-85eca9af-10f3-4279-d72a-05ff1ecdcd3a',
                   'current_gpu_uuid': 'GPU-27a58478-346d-c441-dda0-26a5877409c9',
                   'last_complete_epochs_on_disk': 225, 'next_partial_epoch_zero_based': 225,
                   'last_reported_partial_batch': 101, 'saved_best_epoch_zero_based': 213,
                   'saved_best_AP': analysis['best_saved']['AP'],
                   'last_complete_epoch_zero_based': 224, 'last_complete_AP': analysis['last_complete']['AP'],
                   'latest_matching_checkpoint_saved': False},
          'FULL': {'status': 'ORIGINAL_SUPERVISOR_AND_TRAIN_PID_ABSENT_AND_PROJECT_PATH_NOT_FOUND_IN_RECONNECTED_ENVIRONMENT',
                   'old_gpu_uuid': 'GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a',
                   'current_gpu_uuid': 'GPU-4bd279c5-4e39-bd08-618e-20da276d13ed',
                   'last_confirmed_complete_epochs': 128,
                   'later_progress_or_external_backup': 'Unknown; ask for original instance or storage location.'},
          'limits': ['The planned 320-epoch matched pair and four final endpoint audits do not exist in the recovered evidence.',
                     'A saved validation-best checkpoint from epoch 213 does not establish a terminal or paired result.',
                     'The original checkpoint state does not record a complete RNG/sampler state, so restarting from it cannot be claimed to reproduce the interrupted trajectory.',
                     'A sharp saved validation AP decline is verified but its cause is not established.',
                     'A missing directory on the current second host does not prove that no external backup exists.'],
          'new_training_authorized': False, 'model_executions': 0, 'evaluator_executions': 0,
          'remote_artifacts_modified': False, 'submission_ready': False}
exclusive(PREVIOUS, previous_bytes)
exclusive(REPORT, (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
new_ledger = {'status': report['status'], 'updated_local_utc': now,
              'previous_observation_ledger': ref(PREVIOUS), 'interruption_report': ref(REPORT),
              'arms': {'BASE': report['BASE'], 'FULL': report['FULL']}, 'submission_ready': False,
              'authority_note': 'Original PIDs/start-tick identities are absent in the reconnected environments; queue and receipt RUNNING fields are stale. Do not restart, resume or infer final results.'}
tmp = LEDGER.with_name('parallel_execution_ledger.interrupted.tmp')
assert not tmp.exists()
exclusive(tmp, (json.dumps(new_ledger, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
os.replace(tmp, LEDGER)

summary = ('实验中断与证据保全（2026-11-05）\n\n'
           '1. 两台服务器恢复SSH后，当前环境中的原监督器与训练进程身份均不存在，GPU UUID均已变化。'
           '旧queue/receipt中的RUNNING为遗留状态，不代表仍在训练；造成环境变化的具体原因尚未证实。\n'
           '2. 基线盘上保留225个完整轮次（索引0–224），中断于索引225的第101批记录附近。'
           '25个文件共207,491,171字节已下载并逐项与远端清单SHA256核对；位置：' + str(download['local_root']) + '。'
           '学习率gzip缺少结束标记，但已读到225个完整轮次的363,825条记录，最后一个完整记录为索引224第1616批。\n'
           '3. 基线保存的验证最优索引213 AP=0.1765698785；最后完整索引224 AP=0.0014737044。'
           '索引213的权重、EMA张量、预测、保存的评价数组及指标绑定通过；索引224的预测、数组和指标一致，但没有对应的该轮权重。'
           '210–224轮的24,255条已保存学习率在各参数组内均恒定，期间无配置的增强策略边界；这些观察仍不能确定AP骤降的原因。\n'
           '4. 第二台重连后同一hostname下原实验目录不存在；本地仅保留之前核实的128个完整轮次的状态快照及早期端点证据。'
           '是否存在服务器快照、原实例或外部备份尚待用户或管理员确认。基线未完成320轮；第二台现有记录不足以证明完成320轮，不能作为公平主对照写入论文。\n'
           '5. 保存的基线best权重是索引213的验证最优权重，不能冒充320轮终点；checkpoint未保存完整随机数及采样器状态，'
           '从它继续无法声称精确重现原轨迹。任何继续、重训、诊断性训练或新增模型评测须先说明方案并获得用户明确授权。\n'
           '6. 详细哈希与范围见interrupted_runs_report_20261105.json、interrupted_base_download_verified_20261105.json、'
           'interrupted_base_analysis_20261105.json、interrupted_base_saved_file_audit_20261105.json及interrupted_base_lr_interval_20261105.json。'
           '没有执行新模型、评价器或优化步骤，没有修改远端训练文件。目标仍未完成，尚不能投稿。\n')
exclusive(SUMMARY, summary.encode('utf-8'))
entry = PAPER / '当前修订入口.txt'
text = entry.read_text(encoding='utf-8-sig')
old = 'BASE/FULL的最后确认记录及当前连接状态见远程公平对照_20261103/parallel_execution_ledger.json；观察连接中断不代表训练停止。'
assert text.count(old) == 1
text = text.replace(old, '原BASE/FULL进程在重连环境中均不存在；基线部分证据已保存，第二台原目录待查。详情见远程公平对照_20261103/实验中断与证据保全_20261105.txt。', 1)
entry.write_text(text, encoding='utf-8')
note = ('\n2026-11-05 实验中断核查：重连后原训练及监督器进程身份均不在当前环境，GPU UUID均变化；'
        '基线磁盘保留225轮，25份文件已下载并核验，最后完整AP发生骤降，原因未确定；'
        '第二端原目录缺失，最后本地确认128轮，外部备份待查。旧RUNNING字段不能作为当前状态。'
        '详情见实验中断与证据保全_20261105.txt及interrupted_runs_report_20261105.json；'
        '未启动或恢复训练，公平主对照仍不完整，尚不能投稿。\n')
for path in [ROOT / '操作恢复说明.txt', ROOT / '第二服务器并行执行_20261104.txt',
             ROOT / '结果并入与验收清单.txt', RECORDS / '目标验收与阻塞记录.txt',
             RECORDS / '修订说明与未解决问题.txt']:
    with path.open('a', encoding='utf-8') as f:
        f.write(note)
print(json.dumps({'status': report['status'], 'base_complete_epochs': 225,
                  'full_last_confirmed_complete_epochs': 128, 'submission_ready': False}))
