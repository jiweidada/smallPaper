"""Preserve the exact authorized recovery inputs beside the existing D-drive backups."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
base = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA').resolve(strict=True)
destination = base / 'recovery_references/recovery30219_r29_20261108'
assert destination.resolve().is_relative_to(base) and not destination.exists()
assert shutil.disk_usage(base).free >= 2 * 1024**3
names = ['no_msca_round29_recovery_package.zip', 'recovery_addendum.json',
    'resume_train_runner.py', 'resume_launch_arm.py', 'recovery_package_verified.json',
    'target_preflight_readonly.json', 'resume_ast_equivalence.json',
    'launch_readiness_verified.json', 'recovery_launch_result.json',
    'installation_python38_failure_and_fix.json', 'installation_state_followup.json',
    'first_recovered_round030_native_acceptance.json', 'lineage_recovered_through_round030.json',
    'audit_recovered_ablation_lineage.py', 'audit_recovered_bootstrap.py',
    'recovered_auditor_validation.json', 'recovery_contract_round029_verified_v2.json',
    'live_training_progress_05.json']
files = [(name, here / name) for name in names]
files.append(('recovery_round030_evidence_supplement_20261108.json',
              here.parents[1] / 'recovery_round030_evidence_supplement_20261108.json'))
def sha(path):
    h = sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024**2): h.update(block)
    return h.hexdigest()
manifest = [{'name': name, 'bytes': source.stat().st_size, 'sha256': sha(source), 'source': str(source)}
            for name, source in files]
assert manifest[0]['sha256'] == '7dd93c476068cd8a85a865bac41d01223b51917c9850d9188f8a4fde62504f1b'
destination.mkdir(parents=True, exist_ok=False)
for (name, source), item in zip(files, manifest):
    target = destination / name
    with source.open('rb') as src, target.open('xb') as dest:
        shutil.copyfileobj(src, dest, length=4 * 1024**2)
    assert target.stat().st_size == item['bytes'] and sha(target) == item['sha256']
report = {'status': 'RECOVERY_REFERENCES_BYTE_VERIFIED_ON_D',
    'created_local_utc': datetime.now(timezone.utc).isoformat(), 'destination': str(destination),
    'files': manifest, 'native_backup_protocol_directory_changed': False}
with (destination / 'archive_index.json').open('x', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
with (here / 'recovery_references_D_archive_verified.json').open('x', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps({'status': report['status'], 'destination': str(destination),
                  'files': len(manifest), 'bytes': sum(x['bytes'] for x in manifest)}))
