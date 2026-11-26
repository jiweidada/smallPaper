"""Collect final queue receipts only after every frozen process has exited."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

R = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proc(identity):
    pid = identity['pid']
    try:
        fields = Path('/proc/%s/stat' % pid).read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'state': fields[0],
                'expected_start_ticks': identity['start_ticks'],
                'actual_start_ticks': int(fields[19]),
                'identity_matches': int(fields[19]) == identity['start_ticks']}
    except FileNotFoundError:
        return {'pid': pid, 'missing': True, 'expected_start_ticks': identity['start_ticks']}


def main():
    supervisor = read(R / 'supervisor_result.json')
    assert supervisor['status'] == 'COMPLETED' and supervisor['exit_code'] == 0
    processes = {'supervisor': proc(read(R / 'supervisor_started.json'))}
    for arm in ['B', 'S', 'L']:
        assert read(R / ('arm_' + arm) / 'exit_receipt.json')['exit_code'] == 0
        processes[arm] = proc(read(R / ('arm_' + arm) / 'child_identity.json'))
    assert all(v.get('missing') or not v.get('identity_matches') for v in processes.values()), processes
    archive = R / 'terminal_summary_bundle.zip'
    receipt_path = R / 'terminal_summary_bundle.receipt.json'
    if archive.exists():
        existing = read(receipt_path)
        assert sha(archive) == existing['sha256']
        print(json.dumps(existing))
        return
    observation = {'status': 'ALL_FROZEN_JOB_IDENTITIES_ABSENT',
                   'remote_unix': time.time(), 'processes': processes}
    for key, query in [('nvidia_compute_processes', '--query-compute-apps=pid,process_name,used_gpu_memory'),
                       ('nvidia_gpu', '--query-gpu=name,memory.used,memory.total,utilization.gpu')]:
        result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=10)
        observation[key] = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    (R / 'final_process_observation.json').write_text(json.dumps(observation, indent=2) + '\n')
    names = ['three_arm_result.json', 'supervisor_result.json', 'queue_progress.json',
             'supervisor_started.json', 'final_process_observation.json',
             '\u4e09\u7ec4\u7ed3\u679c\u6458\u8981.txt',
             'independent_checkpoint_B.json', 'independent_checkpoint_S.json', 'independent_checkpoint_L.json']
    manifest = {name: sha(R / name) for name in names}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(R / name, name)
        z.writestr('member_sha256.json', json.dumps(manifest, indent=2) + '\n')
    receipt = {'status': 'PACKAGED_TERMINAL_SUMMARY', 'archive': str(archive),
               'bytes': archive.stat().st_size, 'sha256': sha(archive), 'members': manifest,
               'training_action': 'none', 'model_forwards': 0, 'optimizer_updates': 0}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
