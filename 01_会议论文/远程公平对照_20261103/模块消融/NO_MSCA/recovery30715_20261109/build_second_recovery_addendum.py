"""Seal the user-authorized164 continuation after actual preflight and archive verification."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'recovery30219_r29_20261108'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


assert digest(HERE / 'resume30715_train_runner.py') == 'f5f7ef88ace026c5236734dfeeb29cdbc4688617f90fc32c3d6abb70b3fe0e20'
assert digest(HERE / 'resume30715_launch_arm.py') == '2bfa6a7f8db57e5e62d09e9a83d03d6f92e1d6464275108dc949dc304580f99a'
assert digest(PREVIOUS / 'recovery_addendum.json') == 'e2db4f390756e33e742684e8f1eff526047403ba2db6bad17d5c9e21132e9d54'
assert read(HERE / 'prior_segment_archive_local_verified.json')['files'] == 16
restored = read(HERE / 'restored_source_records.json')
assert restored['status'] == 'ALL_12_ACCEPTED164_SOURCE_BYTES_RESTORED_AND_ANCHOR_PINNED'
assert len(restored['files']) == 12 and restored['optimizer_step_attempts'] == 265188
preflight = read(HERE / 'runtime_and_state_preflight.json')
assert preflight['source_restore_required'] == ['run/receipt.json'] and not preflight['extra_selected_packages']
new_names = ['resume30715_train_runner.py', 'resume30715_launch_arm.py', 'resume30715_ast_equivalence.json',
    'build_resume30715.py', 'prefix164_metadata_reference.json', 'runtime_and_state_preflight.json',
    'restored_source_records.json', 'prior_segment_archive_manifest.json', 'prior_segment_archive.zip',
    'read_runtime_and_state_preflight.py', 'restore_exact_data_cache.py', 'archive_prior_segment.py',
    'verify_restored_sources_and_anchor.py', 'backup_monitor_30715_adapter.py', 'backup_port_adapter_verified.json']
legacy = ['recovery_addendum.json', 'resume_train_runner.py', 'resume_launch_arm.py', 'resume_ast_equivalence.json']
files = [{'path': 'recovery30715_anchor/next_epoch.pt',
    'sha256': digest(HERE / 'recovery30715_anchor/next_epoch.pt')}]
files += [{'path': name, 'sha256': digest(HERE / name)} for name in new_names]
files += [{'path': name, 'sha256': digest(PREVIOUS / name)} for name in legacy]
result = {'schema': 'authorized_no_msca_second_recovery_from_round164_v1',
    'authorized': True, 'arm': 'NO_MSCA', 'created_utc': datetime.now(timezone.utc).isoformat(),
    'user_authorization': '服务器到期了 又新申请了 数据应该还在上面 接着训练吧',
    'authorization_context': 'User supplied the new30715 instance and requested continuation of the existing authorized experiment after expiry,without a new arm,seed or recipe.',
    'original_protocol_sha256': '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc',
    'original_protocol_unchanged': True,
    'configuration_sha256': '8b32bc2c5a620dbed5a758a64c3e891f76acf78ef8deab615da7146f6c83a438',
    'original_run_id': 'no_msca_20261107_v1',
    'checkpoint_sha256': '512dfe64c5b99479ffc2235054ee1bc1b2b5378ac776e7bead4d57395b3e545d',
    'first_epoch_index': 164, 'final_epoch_index': 319, 'retained_round_limit': 320,
    'retained_optimizer_update_limit': 517440, 'new_segment_evaluation_limit': 156,
    'new_segment_optimizer_update_limit': 252252, 'target_ssh_port': 30715,
    'target_gpu_uuid': 'GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a',
    'target_boot_id': '2e5d4a32-c510-4f86-8b5a-eb66f03b864a',
    'prior_recovery_addendum_sha256': digest(PREVIOUS / 'recovery_addendum.json'),
    'prior_round164_lineage_audit_sha256': digest(PREVIOUS / 'lineage_recovered_through_round164.json'),
    'resume_runner_sha256': digest(HERE / 'resume30715_train_runner.py'),
    'runtime_and_data_audit_sha256': digest(HERE / 'runtime_and_state_preflight.json'),
    'source_restoration_sha256': digest(HERE / 'restored_source_records.json'),
    'prior_segment_archive_sha256': digest(HERE / 'prior_segment_archive.zip'),
    'recovery_files': files,
    'unchanged': ['seed0', 'batch4', 'FP32', 'TF32off', 'frozen model/data/loss/evaluator source',
        '7021 dataset files', '101 dependency versions and torch build', 'optimizer/EMA/LR/warmup/late restore policy',
        'original backup manifest/probe/nonce/ACK verification protocol'],
    'observed_discarded_partial_round165_updates': 1401,
    'physical_compute_limitation': 'The old instance expired after accepted round164. Its last observed partial round165 had1401 attempted updates beyond that checkpoint; further unretained work is unknown. Earlier unretained work after round29 also remains unknown. This continuation restores the accepted164 full state and does not claim equal physical compute or bitwise equivalence to uninterrupted training.',
    'automatic_training_retry': False, 'additional_arms': [], 'additional_seeds': [], 'tuning': False}
with (HERE / 'recovery30715_addendum.json').open('x', encoding='utf-8') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
plan = {'status': 'SECOND_RECOVERY_EXACT_FILE_STAGING_PLAN',
    'addendum_sha256': digest(HERE / 'recovery30715_addendum.json'),
    'files': files + [{'path': 'recovery30715_addendum.json', 'sha256': digest(HERE / 'recovery30715_addendum.json')}],
    'new_flat_names': new_names + ['recovery30715_addendum.json'], 'legacy_names': legacy}
with (HERE / 'new_recovery_file_plan.json').open('x', encoding='utf-8') as handle:
    json.dump(plan, handle, indent=2)
    handle.write('\n')
print(json.dumps({'status': 'SECOND_RECOVERY_ADDENDUM_SEALED',
    'addendum_sha256': plan['addendum_sha256'], 'files': len(plan['files']),
    'remaining_evaluations': 156, 'remaining_optimizer_updates': 252252,
    'training_started': False}))
