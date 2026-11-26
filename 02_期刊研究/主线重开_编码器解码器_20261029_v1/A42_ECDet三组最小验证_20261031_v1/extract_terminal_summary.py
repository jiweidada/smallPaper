"""Verify the small terminal archive against the plain SSH receipt log."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

D = Path(__file__).resolve().parent
LOG = D.parent.parent / '前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(receipt_id):
    found = []
    for line in LOG.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('id') == receipt_id and row.get('action') == 'exec':
            found.append(row)
    assert len(found) == 1 and found[0]['exit_code'] == 0
    metadata = json.loads(found[0]['stdout'])
    assert metadata['status'] == 'PACKAGED_TERMINAL_SUMMARY'
    archive = D / 'terminal_summary_bundle.zip'
    content = archive.read_bytes()
    assert len(content) == metadata['bytes'] and sha(content) == metadata['sha256']
    dest = D / 'terminal_summary'
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('member_sha256.json'))
        assert manifest == metadata['members']
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == set(manifest) | {'member_sha256.json'}
        for name, digest in manifest.items():
            assert Path(name).name == name and name not in ['.', '..']
            data = z.read(name)
            assert sha(data) == digest
            target = dest / name
            if target.exists():
                assert target.read_bytes() == data
            else:
                target.write_bytes(data)
    for arm in ['B', 'S', 'L']:
        source = dest / ('independent_checkpoint_' + arm + '.json')
        result = json.loads(source.read_text(encoding='utf-8'))
        assert result['status'] == 'PASS'
        target = D / 'result_verification_v1' / (arm + '_epoch12_checkpoint_cpu_verification.json')
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
    processes = json.loads((dest / 'final_process_observation.json').read_text(encoding='utf-8'))
    assert processes['status'] == 'ALL_FROZEN_JOB_IDENTITIES_ABSENT'
    assert all(row.get('missing') or not row.get('identity_matches') for row in processes['processes'].values())
    receipt = {'status': 'PASS', 'metadata': metadata, 'member_files_verified': len(manifest),
               'checkpoint_cpu_receipts_exact_bytes_recovered': ['B', 'S', 'L'],
               'all_frozen_job_identities_absent': True, 'original_ssh_receipt': found[0]}
    (D / 'terminal_summary_download_verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ['status', 'member_files_verified', 'all_frozen_job_identities_absent']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-id', required=True)
    main(parser.parse_args().receipt_id)
