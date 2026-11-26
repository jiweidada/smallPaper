"""Retain immutable second-recovery evidence beside the existing experiment backup."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'recovery30219_r29_20261108'
BASE = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/recovery_references')
DESTINATION = BASE / 'recovery30715_20261109'
NATIVE_ANCHOR = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups/epochs/epoch0163/next_epoch.pt')


def digest(path):
    value = sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024 ** 2):
            value.update(block)
    return value.hexdigest()


assert DESTINATION.resolve().is_relative_to(BASE.resolve()) and not DESTINATION.exists()
binding = json.loads((HERE / 'segment30715_audit_binding.json').read_text(encoding='utf-8'))
assert digest(HERE / 'segment30715_audit_binding.json') == '6de82b5b4bcbf8bbdbd3a61a2fb7ecbd70905b98da0eceefc55c7b451435a41e'
files = {value['relative_path']: HERE / value['relative_path'] for value in binding['artifacts'].values()}
names = ['segment30715_audit_binding.json', 'audit_resume30715_lineage.py', 'audit_resume30715_bootstrap.py',
    'audit_resume30715_input_contract.json', 'audit_resume30715_static_checks.json',
    'resume30715_train_runner.py', 'resume30715_launch_arm.py', 'resume30715_ast_equivalence.json',
    'build_resume30715.py', 'build_second_recovery_addendum.py', 'new_recovery_file_plan.json',
    'prefix164_metadata_reference.json', 'recovery_contract_verified.json',
    'lineage_second_recovery_through_round165.json', 'verify_restored_sources_and_anchor.py',
    'install_recovery30715_bindings.py', 'installed_recovery30715_execution.json',
    'prior_segment_archive_local_verified.json', 'genuine_second_recovery_progress.json',
    'running_observation_recorded.json', 'initial_readonly.json', 'preserved_inputs_readonly.json',
    'data_restoration_execution.json', 'failed_cache_preparation_preserved.json',
    'data_restoration_detached_launch.json', 'data_preparation_latest.json',
    'resume_monitor_20261109_001/milestone_round165_live.json',
    'resume_monitor_20261109_001/first_verified_update_live.json']
files.update({name: HERE / name for name in names})
for name in ['recovery_addendum.json', 'resume_train_runner.py', 'resume_launch_arm.py', 'resume_ast_equivalence.json',
             'lineage_recovered_through_round164.json']:
    files['previous_segment/' + name] = PREVIOUS / name
files['recovery30715_anchor/next_epoch.pt'] = HERE / 'recovery30715_anchor/next_epoch.pt'
assert all(path.is_file() for path in files.values())
DESTINATION.mkdir(parents=True, exist_ok=False)
records = []
for relative, source in sorted(files.items()):
    destination = DESTINATION / relative
    assert destination.resolve().is_relative_to(DESTINATION.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = digest(source)
    if relative == 'recovery30715_anchor/next_epoch.pt':
        assert expected == digest(NATIVE_ANCHOR) == '512dfe64c5b99479ffc2235054ee1bc1b2b5378ac776e7bead4d57395b3e545d'
        os.link(NATIVE_ANCHOR, destination)
    else:
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 ** 2)
    assert digest(destination) == expected and destination.stat().st_size == source.stat().st_size
    records.append({'file': relative, 'source': str(source), 'bytes': destination.stat().st_size,
        'sha256': expected, 'native_anchor_hard_link': relative == 'recovery30715_anchor/next_epoch.pt'})
report = {'status': 'IMMUTABLE_SECOND_RECOVERY_REFERENCES_BYTE_VERIFIED',
    'observed_utc': datetime.now(timezone.utc).isoformat(), 'files': records,
    'destination': str(DESTINATION), 'binding_sha256': digest(HERE / 'segment30715_audit_binding.json'),
    'prior_full_recovery_reference_directory': str(BASE / 'recovery30219_r29_20261108'),
    'native_backup_store_changed': False, 'frozen_native_source_changed': False,
    'model_forwards': 0, 'training_actions': 0}
with (DESTINATION / 'archive_index.json').open('x', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
summary = {'status': report['status'], 'files': len(records),
    'total_logical_bytes': sum(item['bytes'] for item in records),
    'archive_index': str(DESTINATION / 'archive_index.json'),
    'archive_index_sha256': digest(DESTINATION / 'archive_index.json'), 'training_actions': 0}
with (HERE / 'second_recovery_references_D_archive_verified.json').open('x', encoding='utf-8') as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps(summary, ensure_ascii=False))
