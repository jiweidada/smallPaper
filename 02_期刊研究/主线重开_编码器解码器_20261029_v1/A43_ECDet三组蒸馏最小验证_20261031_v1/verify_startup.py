import hashlib
import json
from pathlib import Path
import zipfile

D = Path(__file__).resolve().parent
LOG = D.parent.parent / '前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rows = []
    with LOG.open(encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('id') == 'A43_package_startup_evidence' and r.get('action') == 'exec':
                rows.append(r)
    assert len(rows) == 1 and rows[0]['exit_code'] == 0
    meta = json.loads(rows[0]['stdout']);archive = D / 'startup_evidence_v1.zip'
    assert archive.stat().st_size == meta['bytes'] and sha(archive) == meta['sha256']
    dest = D / 'startup_evidence';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('member_sha256.json'))
        assert manifest == meta['files']
        for name, digest in manifest.items():
            assert Path(name).name == name
            data = z.read(name);assert hashlib.sha256(data).hexdigest() == digest
            p = dest / name
            if p.exists():
                assert sha(p) == digest
            else:
                p.write_bytes(data)
    plan = json.loads((dest / 'execution_protocol_FROZEN.json').read_text())
    assert plan['arms'] == ['B', 'M', 'Q'] and plan['epochs_per_group'] == 12
    checked = []
    for name, digest in plan['files_sha256'].items():
        if name == 'teacher_B_epoch12.pth':
            p = D.parent / 'A42_ECDet三组最小验证_20261031_v1/completed_results/B/arm_B/epoch12.pth'
        elif (D / name).exists():
            p = D / name
        else:
            p = dest / name
        assert sha(p) == digest, name
        checked.append(name)
    cap = json.loads((dest / 'capacity_preflight_result.json').read_text())
    assert cap['status'] == 'PASS' and cap['new_student_forwards'] == 8 and cap['new_teacher_forwards'] == 4 and cap['optimizer_updates'] == 5
    assert all(row['peak_allocated_bytes'] < 22 * 2**30 for row in cap['arms'].values())
    result = {'status': 'PASS', 'archive': meta, 'startup_members_verified': len(manifest),
              'frozen_files_verified_against_local_sources': len(checked), 'checked_files': checked,
              'protocol_sha256': sha(dest / 'execution_protocol_FROZEN.json'),
              'new_model_forwards': 0, 'new_optimizer_updates': 0}
    (D / 'startup_download_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'startup_members_verified', 'frozen_files_verified_against_local_sources', 'protocol_sha256']}))


if __name__ == '__main__':
    main()
