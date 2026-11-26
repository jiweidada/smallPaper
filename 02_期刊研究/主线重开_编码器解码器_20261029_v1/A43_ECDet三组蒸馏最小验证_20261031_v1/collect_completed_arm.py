"""Immutable terminal artifact collection, no training action."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

R = Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):h.update(b)
    return h.hexdigest()


def main(arm):
    folder = R / ('arm_' + arm)
    assert read(folder / 'exit_receipt.json')['exit_code'] == 0
    result = read(folder / 'result.json')
    assert result['status'] == 'COMPLETED' and result['epochs'] == 12 and result['optimizer_updates'] == 9708
    plan = read(R / 'execution_protocol_FROZEN.json')
    for name, digest in plan['files_sha256'].items():assert sha(R / name) == digest, name
    epochs = [json.loads(row) for row in (folder / 'epochs.jsonl').read_text().splitlines()]
    final = [row for row in epochs if row['epoch'] == 12]
    assert len(final) == 1 and sha(folder / 'epoch12.pth') == final[0]['checkpoint_sha256']
    archive = R / ('completed_arm_' + arm + '_bundle.zip');receipt = archive.with_suffix('.receipt.json')
    if archive.exists():
        meta = read(receipt);assert sha(archive) == meta['sha256'];print(json.dumps(meta));return
    excluded = ['common_init_10class.pth', 'teacher_B_epoch12.pth']
    paths = [p for p in folder.iterdir() if p.is_file() and (p.suffix != '.pth' or p.name == 'epoch12.pth')]
    paths += [R / name for name in plan['files_sha256'] if name not in excluded]
    paths += [R / 'execution_protocol_FROZEN.json', R / 'supervisor_started.json', R / 'launch_receipt.json']
    partial = archive.with_suffix('.partial.zip');manifest = {}
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(set(paths)):
            name = p.relative_to(R).as_posix();manifest[name] = sha(p);z.write(p, name)
        z.writestr('member_sha256.json', json.dumps(manifest, indent=2))
    os.replace(partial, archive)
    meta = {'status': 'PACKAGED_TERMINAL_ARM', 'arm': arm, 'archive': str(archive),
            'bytes': archive.stat().st_size, 'sha256': sha(archive), 'member_files': len(manifest),
            'epoch12_checkpoint_sha256': final[0]['checkpoint_sha256'],
            'excluded_but_pinned': {n: plan['files_sha256'][n] for n in excluded}, 'training_action': 'none'}
    receipt.write_text(json.dumps(meta, indent=2) + '\n');print(json.dumps(meta))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--arm', choices=['B', 'M', 'Q'], required=True)
    main(p.parse_args().arm)
