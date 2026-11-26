"""Verify the transferred interrupted BASE files against the remote read-only manifest."""
from pathlib import Path, PurePosixPath
from hashlib import sha256
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'interrupted_base_remote_manifest_20261105.json'
DEST = ROOT / 'remote_evidence/interrupted_BASE_20261105'
RECEIPT = ROOT / 'interrupted_base_download_verified_20261105.json'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

assert not RECEIPT.exists()
wrapper = json.loads(SOURCE.read_text(encoding='utf-8-sig'))
assert wrapper['exit_code'] == 0 and not wrapper['stderr']
manifest = json.loads(wrapper['stdout'])
assert manifest['status'] == 'INTERRUPTED_BASE_FILES_HASHED_READ_ONLY'
assert not manifest['old_supervisor_live'] and not manifest['old_train_process_live']
verified = []
for item in manifest['files']:
    rel = PurePosixPath(item['relative_path'])
    assert not rel.is_absolute() and '..' not in rel.parts
    p = DEST.joinpath(*rel.parts)
    assert p.is_file() and not p.is_symlink() and p.stat().st_size == item['bytes']
    actual = sha(p)
    assert actual == item['sha256'], str(rel)
    verified.append({'relative_path': str(rel), 'bytes': item['bytes'], 'sha256': actual})
actual_files = {str(p.relative_to(DEST)).replace('\\', '/') for p in DEST.rglob('*') if p.is_file()}
assert actual_files == {item['relative_path'] for item in verified}
assert sum(x['bytes'] for x in verified) == manifest['total_bytes']
receipt = {'status': 'INCOMPLETE_BASE_ARTIFACTS_PRESERVED_HASH_MATCHED',
           'checked_local_utc': datetime.now(timezone.utc).isoformat(),
           'source_manifest_sha256': sha(SOURCE), 'protocol_sha256': manifest['protocol_sha256'],
           'local_root': str(DEST), 'files': verified, 'total_bytes': manifest['total_bytes'],
           'completed_training_endpoint': False, 'remote_files_modified': False,
           'model_executions': 0, 'evaluator_executions': 0}
with RECEIPT.open('x', encoding='utf-8') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': receipt['status'], 'files': len(verified), 'bytes': receipt['total_bytes']}))
