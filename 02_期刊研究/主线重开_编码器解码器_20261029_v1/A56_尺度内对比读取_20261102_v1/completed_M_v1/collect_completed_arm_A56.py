"""Preserve an exited fixed12 arm without model/evaluator execution."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import zipfile

P = Path(__file__).resolve().parent
PROTOCOL_SHA = 'da3ae8ac399483307eeb7e221627d0bdaef4222021dde468cb41459d674915a9'
INITIAL_SHA = '582fd6c57aae3725afe7af469024cd0b259a63a5a3694d36a2b1ce56ad2bcd96'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text('utf-8'))


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def not_owned_live(record):
    path = Path('/proc')/str(record['child_PID'])/'stat'
    if not path.exists():
        return {'pid': record['child_PID'], 'owned_live': False, 'reason': 'missing'}
    fields = path.read_text().split(') ', 1)[1].split()
    same = int(fields[19]) == record['child_start_ticks']
    return {'pid': record['child_PID'], 'owned_live': same and fields[0] not in ['X', 'Z'],
            'identity_matches': same, 'state': fields[0]}


def exact_initial_branches():
    target = P/'new_branches_untrained_initial_state.pth'
    receipt = P/'new_branches_untrained_initial_state_receipt.json'
    if target.exists():
        assert sha(target) == load(receipt)['file_sha256']
        assert load(receipt)['tensor_sha256'] == INITIAL_SHA
        return [target, receipt]
    import torch
    from contrastive_read import ContrastiveRead, NEW_SEED
    states = []
    h = hashlib.sha256()
    for i in range(4):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(NEW_SEED+i)
            state = ContrastiveRead(192, 8, [3, 6, 3], 'M').state_dict()
        states.append(state)
        for name, tensor in state.items():
            h.update(f'{i}/{name}'.encode())
            h.update(tensor.contiguous().numpy().tobytes())
    assert h.hexdigest() == INITIAL_SHA
    torch.save({'states': states, 'reconstructed_after_training_started': True,
                'scope': 'CPU-only reconstruction with exact formal startup tensor digest'}, target)
    save(receipt, {'status': 'EXACT_STARTUP_TENSOR_DIGEST_MATCH', 'file_sha256': sha(target),
                   'tensor_sha256': h.hexdigest(), 'new_detector_forwards': 0,
                   'new_detector_optimizer_updates': 0, 'synthetic_optimizer_steps': 0})
    return [target, receipt]


def main(arm):
    folder = P/f'arm_{arm}'
    result = load(folder/'result.json')
    assert result['status'] == 'COMPLETED' and result['epochs'] == 12
    assert result['optimizer_updates'] == 9708 and result['full_model_forwards'] == 10159
    assert result['model_parameters'] == 9877256 and result['extra_parameters'] == 74144
    record = next(x for x in load(P/'formal_processes.json')['records'] if x['arm'] == arm)
    assert record['status'] == 'EXITED' and record['child_exit_code'] == 0 and not record['timed_out']
    live = not_owned_live(record)
    assert not live['owned_live']
    assert sha(P/'protocol_FROZEN_v1.json') == result['protocol_sha256'] == PROTOCOL_SHA
    plan = load(P/'protocol_FROZEN_v1.json')
    own = {n: sha(P/n) == value for n, value in plan['files_sha256'].items()}
    external = {n: sha(Path(n)) == value for n, value in plan['external_sha256'].items()}
    assert all(own.values()) and all(external.values())
    init = load(folder/'initialization.json')
    assert init['new_module_initialization_sha256'] == INITIAL_SHA and init['scratch_no_capacity_weights']
    audit = P/f'{arm}_completed_source_audit.json'
    save(audit, {'status': 'PASS', 'arm': arm, 'owned_process': live,
                 'exited_arm_process_snapshot': record, 'server_observation_unix': time.time(),
                 'own_frozen_files': own, 'external_frozen_files': external,
                 'new_detector_forwards': 0, 'new_optimizer_updates': 0})
    process_snapshot = P/f'{arm}_completed_process_snapshot.json'
    save(process_snapshot, record)
    files = list(folder.rglob('*'))
    files = [p for p in files if p.is_file()]
    files += [P/n for n in plan['files_sha256']]
    files += [P/n for n in ['protocol_FROZEN_v1.json', 'capacity_protocol_FROZEN_v1.json',
                             'capacity_result.json', 'capacity_exit.json', 'capacity_worker.log',
                             f'train_{arm}.log', 'collect_completed_arm_A56.py']]
    files += [audit, process_snapshot] + exact_initial_branches()
    files = sorted(set(files))
    assert all(not p.is_symlink() and p.resolve().is_relative_to(P) for p in files)
    manifest = {'arm': arm, 'status': 'COMPLETED_EXIT0',
                'files': {f.relative_to(P).as_posix(): {'bytes': f.stat().st_size, 'sha256': sha(f)}
                          for f in files}, 'source_hashes_unchanged': True,
                'new_detector_forwards': 0, 'new_optimizer_updates': 0}
    target = P/f'{arm}_completed_bundle_v1.zip'
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        z.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
        for f in files:
            z.write(f, f.relative_to(P).as_posix())
    assert all(sha(P/n) == row['sha256'] for n, row in manifest['files'].items())
    export = Path('/root/uav_a54_20261101_v1/A56_exports')
    export.mkdir(exist_ok=True)
    link = export/target.name
    assert not link.exists()
    os.link(target, link)
    ready = {'status': 'BUNDLE_READY', 'arm': arm, 'path': str(target),
             'bridge_export_hardlink': str(link), 'bytes': target.stat().st_size,
             'sha256': sha(target), 'payload_files': len(files), 'endpoint': result['endpoint_metrics'],
             'source_hashes_unchanged': True, 'new_detector_forwards': 0, 'new_optimizer_updates': 0}
    assert link.stat().st_ino == target.stat().st_ino
    save(P/f'{arm}_bundle_ready.json', ready)
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['M', 'Q'], required=True)
    main(parser.parse_args().arm)
