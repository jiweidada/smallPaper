"""Verify the one-time local backups of B4/B8; never load or run a model."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
D = P.parent / 'A54_场景尺度约束候选_20261101_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    scope_file = P / 'B_epoch4_8_checkpoint_backup_scope_FROZEN_v1.json'
    result_file = P / 'B_epoch4_8_checkpoint_backup_local_verification_v1.json'
    assert not result_file.exists(), 'Verification receipt is immutable.'
    scope = read(scope_file)
    ready = read(D / 'A60_downloads/B_epoch4_8_checkpoint_backup_ready_v1.json')
    receipt = read(D / 'A60_B_epoch4_8_checkpoint_backup_download_v1.json')
    assert receipt['status'] == 'TRANSFERRED' and receipt['operation'] == 'get'
    assert ready['scope_sha256'] == sha(scope_file)
    assert ready['status'] == 'IMMUTABLE_EXISTING_B4_B8_LINKS_READY'
    received = {Path(item['local']).name: item for item in receipt['files']}
    remote = {item['epoch']: item for item in ready['files']}
    verified = []
    for item in scope['files']:
        epoch = item['epoch']
        path = Path(item['local_backup'])
        transfer = received[path.name]
        original = remote[epoch]
        row = read(P / f'epoch{epoch}_B_remote_v1/B/epoch{epoch:02d}.json')
        assert Path(transfer['local']).resolve() == path.resolve()
        assert transfer['remote'] == item['remote_bridge_hardlink'] == original['hardlink']
        assert original['original'] == item['remote_source'] and original['same_inode_verified']
        assert path.is_file() and path.stat().st_size == item['bytes'] == original['bytes']
        assert transfer['bytes'] == item['bytes']
        actual = sha(path)
        assert actual == item['sha256'] == original['sha256'] == row['checkpoint_sha256']
        assert transfer['local_sha256'] == actual
        verified.append({'epoch': epoch, 'path': str(path), 'bytes': item['bytes'], 'sha256': actual})
    assert [x['epoch'] for x in verified] == [4, 8]
    result = {
        'status': 'PASS_B4_B8_RAW_CHECKPOINT_LOCAL_BACKUP',
        'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
        'scope_sha256': sha(scope_file),
        'download_receipt_sha256': sha(D / 'A60_B_epoch4_8_checkpoint_backup_download_v1.json'),
        'verified_files': verified,
        'checkpoint_loads': 0,
        'new_model_forwards': 0,
        'new_optimizer_updates': 0,
        'new_COCO_evaluations': 0,
        'scope': 'Original raw model checkpoint bytes only; no model load, inference, or independent parameter inspection.',
    }
    with result_file.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
        out.write('\n')
    print(json.dumps({'status': result['status'], 'files': verified}, ensure_ascii=False))


if __name__ == '__main__':
    main()
