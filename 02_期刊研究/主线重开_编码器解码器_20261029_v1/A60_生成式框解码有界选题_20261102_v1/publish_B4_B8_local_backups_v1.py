"""Publish a local-checkpoint preservation milestone without changing training."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
B = P.parent
D = B / 'A54_场景尺度约束候选_20261101_v1'
PARENT = P / 'A60_B8_paired_coverage_preservation_receipt_v1.json'
SNAP = P / 'before_B4_B8_local_backup_publish_v1'
REPORT = P / '11_B4与B8原始权重本机备份.txt'
HANDOFF = P / 'A60_turn_handoff_B_checkpoints_backup_v1.json'
OBSERVATION = P / 'formal_live_observation_163_v1.json'
RECEIPT = P / 'A60_B4_B8_local_backup_preservation_receipt_v1.json'
ROOTS = [B / '01_候选判别表.json', B / '15_当前状态与结果校验.json', P / 'CURRENT_A60.json']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def replace_json(path, value):
    temporary = path.with_name(path.name + '.B4_B8_backup_publish_tmp')
    write_new(temporary, value)
    temporary.replace(path)


def metadata(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    assert all(not path.exists() for path in [SNAP, REPORT, HANDOFF, OBSERVATION, RECEIPT])
    assert all(not path.with_name(path.name + '.B4_B8_backup_publish_tmp').exists() for path in ROOTS)
    parent = read(PARENT)
    for name, expected in parent['files'].items():
        path = Path(name)
        assert path.is_file() and metadata(path) == expected, f'Parent evidence changed: {path}'
    for path in ROOTS:
        assert str(path) in parent['files']

    verified = read(P / 'B_epoch4_8_checkpoint_backup_local_verification_v1.json')
    assert verified['status'] == 'PASS_B4_B8_RAW_CHECKPOINT_LOCAL_BACKUP'
    for item in verified['verified_files']:
        path = Path(item['path'])
        assert path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    assert [item['epoch'] for item in verified['verified_files']] == [4, 8]

    command = read(D / 'A60_formal_observe_163.json')
    assert command['status'] == 'EXECUTED' and command['exit'] == 0
    obs = json.loads(command['stdout'])
    assert obs['source_binding'] and obs['supervisor']['identity_matched'] and obs['supervisor']['live']
    arm = obs['arms']['B']
    assert arm['process']['identity_matched'] and arm['process']['live']
    assert arm['completed_epochs'] == 9 and arm['progress']['epoch'] == 10
    assert not obs['arms']['M']['exists'] and not obs['arms']['Q']['exists']
    counts = arm['journal_counts']
    assert counts['started'] >= counts['completed'] >= 0 and counts['updates'] > 0

    originals = {str(path): metadata(path) for path in ROOTS}
    SNAP.mkdir()
    resolution = {}
    for path in ROOTS:
        target = SNAP / path.name
        shutil.copy2(path, target)
        assert metadata(target) == originals[str(path)]
        resolution[str(path)] = {'resolved_path': str(target), **originals[str(path)]}

    write_new(OBSERVATION, obs)
    status = 'A60_B4_B8_LOCAL_BACKUP_VERIFIED_ORIGINAL_B10_LIVE'
    backup = {
        'status': verified['status'],
        'verification': str(P / 'B_epoch4_8_checkpoint_backup_local_verification_v1.json'),
        'scope': str(P / 'B_epoch4_8_checkpoint_backup_scope_FROZEN_v1.json'),
        'report': str(REPORT),
        'files': verified['verified_files'],
        'checkpoint_loads': 0,
        'new_forwards': 0,
        'new_updates': 0,
        'scope_note': 'Byte integrity only; not an independent model load or evaluation. B12 checkpoint does not yet exist.',
    }
    report = (
        'B4与B8原始权重本机备份及运行快照\n'
        '为避免远程实例丢失导致阶段权重无法恢复，已各下载一次B4与B8原始模型checkpoint。'
        '本地SHA-256与已核验的原始epoch记录、远程文件及传输回执一致；仅校验二进制字节，未加载模型。\n\n'
        + ''.join(f"B{item['epoch']}：{item['path']}\n字节：{item['bytes']}；SHA-256：{item['sha256']}\n" for item in verified['verified_files'])
        + f"\n运行快照：服务器UTC {obs['recorded_utc']}；原监督器及B进程身份匹配且存活，"
        + f"B已完成{arm['completed_epochs']}轮，正在第{arm['progress']['epoch']}轮；M/Q尚未启动。"
        + '快照仅为当时下界，不是终点结果。\n'
        + 'B12原始权重尚未产生；固定终点产生后另行备份。当前无候选收益结论，'
        + '本次不新增训练、模型前向、优化器更新、COCO评估或改变协议。\n'
    )
    with REPORT.open('x', encoding='utf-8', newline='\n') as out:
        out.write(report)

    prev = read(ROOTS[2])
    prev.update({
        'status': status, 'recorded_utc': obs['recorded_utc'], 'snapshot_is_live_lower_bound': True,
        'original_supervisor': obs['supervisor'], 'arms': obs['arms'], 'source_binding_verified': obs['source_binding'],
        'observed_formal_model_forwards': counts['started'], 'observed_formal_updates': counts['updates'],
        'completed_formal_epochs': arm['completed_epochs'],
        'global_observed_model_forwards': 276643 + counts['started'],
        'global_observed_all_optimizer_updates': 233606 + counts['updates'],
        'global_observed_formal_optimizer_updates': 233518 + counts['updates'],
        'global_completed_formal_epochs': 288 + arm['completed_epochs'],
        'previous_goal_turn_classification': 'verified wait',
        'current_goal_turn_classification': 'progress: original B4/B8 raw weights backed up and verified locally',
        'B4_B8_local_checkpoint_backups': backup,
        'latest_handoff': str(HANDOFF),
        'next_readonly_receipt': 'A60_formal_observe_164.json',
    })
    handoff = {
        'status': status, 'goal_status': 'active', 'mainline_validated': False,
        'previous_goal_turn_classification': 'verified wait', 'this_goal_turn_classification': 'progress',
        'recorded_remote_utc': obs['recorded_utc'],
        'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
        'snapshot': str(OBSERVATION), 'canonical': str(ROOTS[2]), 'backup': backup,
        'original_supervisor': obs['supervisor'], 'original_B': arm['process'],
        'actual_journal_lower_bound': counts, 'B_completed_epochs': arm['completed_epochs'],
        'B_current_epoch': arm['progress']['epoch'], 'B12_gate_completed': False,
        'M_started': False, 'Q_started': False, 'candidate_efficacy_tested': False,
        'next_readonly_receipt_hint': 'A60_formal_observe_164.json',
        'numbering_rule': 'Derive next receipt from actual files; hints can already be consumed.',
        'bridge_session_id': 19671,
        'remaining_action': 'Observe original B fixed12 gate and original M/Q queue; back up B12 after it exists. Never relaunch or retry.',
        'training_protocol_modified': False,
    }
    write_new(HANDOFF, handoff)
    board = read(ROOTS[0]); state = read(ROOTS[1])
    board.update({'status': status, 'latest_update_utc': obs['recorded_utc'], 'training_running': True,
                  'A60_execution': prev, 'A60_B4_B8_local_checkpoint_backups': backup})
    state.update({'status': status, 'recorded_utc': obs['recorded_utc'], 'latest_update_utc': obs['recorded_utc'],
                  'training_running': True, 'A60_execution': prev, 'A60_B4_B8_local_checkpoint_backups': backup,
                  'model_forwards': prev['global_observed_model_forwards'],
                  'training_updates': prev['global_observed_all_optimizer_updates'],
                  'detector_optimizer_updates_including_capacity': prev['global_observed_all_optimizer_updates'],
                  'formal_detector_training_updates': prev['global_observed_formal_optimizer_updates'],
                  'formal_optimizer_updates_observed': prev['global_observed_formal_optimizer_updates'],
                  'formal_training_epochs': prev['global_completed_formal_epochs'],
                  'previous_goal_turn_classification': 'verified wait',
                  'current_goal_turn': 'PROGRESS_B4_B8_CHECKPOINT_LOCAL_BACKUP_AND_ORIGINAL_B10_LIVE'})
    for path, value in zip(ROOTS, [board, state, prev]):
        replace_json(path, value)

    files = dict(parent['files'])
    new_paths = [*ROOTS, *[Path(v['resolved_path']) for v in resolution.values()], PARENT,
                 Path(__file__), REPORT, HANDOFF, OBSERVATION,
                 P / 'B_epoch4_8_checkpoint_backup_scope_FROZEN_v1.json',
                 P / 'B_epoch4_8_checkpoint_backup_local_verification_v1.json',
                 P / 'verify_B_epoch4_8_checkpoint_backup_v1.py',
                 D / 'A60_B_backup_readiness_ro_v1.json',
                 D / 'A60_B_epoch4_8_checkpoint_backup_export_v1.json',
                 D / 'A60_B_epoch4_8_checkpoint_backup_download_v1.json',
                 D / 'A60_downloads/B_epoch4_8_checkpoint_backup_ready_v1.json',
                 *[Path(item['path']) for item in verified['verified_files']],
                 *[D / f'A60_formal_observe_{i:03d}.json' for i in range(161, 164)]]
    for path in new_paths:
        assert path.is_file(), str(path)
        files[str(path)] = metadata(path)
    receipt = {'status': status, 'goal_status': 'active', 'mainline_validated': False,
               'parent_receipt': str(PARENT), 'parent_receipt_sha256': sha(PARENT),
               'parent_files_verified_before_publish': len(parent['files']),
               'parent_path_resolution': resolution, 'files': files,
               'scope': 'Append-only local backup milestone; original weights, training, decisions and research evidence retained.'}
    write_new(RECEIPT, receipt)
    print(json.dumps({'status': status, 'parent_files_verified': len(parent['files']),
                      'receipt_files': len(files), 'B_completed_epochs': arm['completed_epochs'],
                      'B_current_epoch': arm['progress']['epoch'], 'backups': len(verified['verified_files'])}))


if __name__ == '__main__':
    main()
