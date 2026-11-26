"""Verify and safely unpack the completed, read-only baseline backup."""
from pathlib import Path, PurePosixPath
import hashlib, json, shutil, tarfile

folder = Path(__file__).resolve().parent / '完成结果_20261029_v1'
archive = folder / 'completed_baseline_results.tar.gz'
receipt = json.loads((folder / 'completed_archive_receipt.json').read_text(encoding='utf-8'))
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
assert archive.stat().st_size == receipt['bytes']
assert sha(archive) == receipt['sha256']
target = (folder / 'raw').resolve()
target.mkdir(exist_ok=True)
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar.getmembers():
        name = PurePosixPath(member.name)
        assert not name.is_absolute() and '..' not in name.parts
        assert '\\' not in member.name and ':' not in member.name
        assert member.isdir() or member.isfile(), member.name
        resolved = target.joinpath(*name.parts).resolve()
        assert resolved == target or target in resolved.parents
        if member.isdir():
            resolved.mkdir(parents=True, exist_ok=True)
            continue
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as source, resolved.open('wb') as dest:
            shutil.copyfileobj(source, dest)
checks = {}
for arm in ['D0', 'DA']:
    complete = json.loads((target / 'results' / arm / 'completed.json').read_text())
    checkpoint = target / 'results' / arm / 'weights/last.pt'
    digest = sha(checkpoint)
    assert digest == complete['primary_checkpoint_sha256']
    assert complete['epoch'] == 100 and complete['actual_optimizer_updates'] == 10073 and complete['amp_skips'] == 0
    checks[arm] = dict(local_checkpoint=str(checkpoint), sha256=digest, epoch=100, updates=10073, amp_skips=0)
result = dict(status='ARCHIVE_AND_BOTH_CHECKPOINT_HASHES_VERIFIED_LOCALLY', archive_sha256=receipt['sha256'], archive_bytes=receipt['bytes'], checkpoints=checks)
(folder / 'local_backup_verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
audit = json.loads((folder / 'baseline_integrity_and_curve_audit.json').read_text())
print(json.dumps(dict(backup=result, audit_keys=list(audit), curves=audit.get('training_curves', audit.get('models',{}))), ensure_ascii=False))
