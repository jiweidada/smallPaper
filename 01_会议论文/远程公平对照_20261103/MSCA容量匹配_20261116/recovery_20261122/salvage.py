"""Archive a completed pending epoch without repeating training or evaluation."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pair


def save_once(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Existing evidence differs: {path}'
    else:
        pair.save(path, value)


def main(port):
    server = next(s for s in pair.SERVERS if s.port == port)
    control = pair.BACKUP / server.arm
    sys.path.insert(0, str(control))
    import backup_monitor as monitor
    import psutil

    monitor.CHECKPOINT_PYTHON = pair.CPU_PYTHON
    monitor.CHECKPOINT_TORCH_SITE = pair.TORCH_SITE
    evidence = HERE / server.arm
    evidence.mkdir(exist_ok=True)
    prior_launch = json.loads((HERE.parent / 'recovery_20261118' / server.arm / 'launch.json').read_text())
    local_monitor = json.loads((control / 'monitor_launch.json').read_text())
    try:
        process = psutil.Process(local_monitor['pid'])
        assert process.cmdline()[1:] != local_monitor['command'][1:], 'Backup client is already live'
    except psutil.NoSuchProcess:
        pass
    store = monitor.BackupStore(control / 'backups', control)
    with monitor.protect_local_lock(store.root), pair.ssh.connect(port) as client:
        for remote, name in [('run/receipt.json', 'original_failed_receipt.json'),
                             ('supervisor.json', 'original_supervisor.json')]:
            if name == 'original_failed_receipt.json' and (evidence / name).exists():
                continue  # The canonical receipt may already be the exact pending bytes.
            pair.download(client, server.control / remote, evidence / name)
        failed_bytes = (evidence / 'original_failed_receipt.json').read_bytes()
        failed = json.loads(failed_bytes)
        supervisor = json.loads((evidence / 'original_supervisor.json').read_text())
        assert supervisor['status'] == 'STOPPED' and supervisor['exit_code'] == 1
        assert supervisor['supervisor_pid'] == prior_launch['supervisor_pid']
        assert supervisor['supervisor_start_ticks'] == prior_launch['start_ticks']
        assert supervisor['protocol_sha256'] == prior_launch['protocol_sha256'] == monitor.protocol.PROTOCOL
        assert failed == supervisor['final_receipt']
        assert failed['status'] == 'FAILED' and failed['phase'] == 'awaiting_local_backup'
        assert 'Local backup acknowledgement missing' in failed['traceback']
        check = '''
import json, subprocess
from pathlib import Path
sup = json.loads((Path(ROOT)/'supervisor.json').read_text())
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for pid, expected in [(sup['pid'],sup['start_ticks']), (sup['supervisor_pid'],sup['supervisor_start_ticks'])]:
    path=Path('/proc')/str(pid)/'stat'
    if path.is_file():
        fields=path.read_text().rsplit(') ',1)[1].split()
        assert not (boot==sup['boot_id'] and int(fields[19])==expected and fields[0]!='Z'), 'Prior process remains live'
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert not active, 'GPU has another compute process'
print(json.dumps({'status':'STOPPED_PROCESS_AND_IDLE_GPU_VERIFIED','boot_id':boot}))
'''
        result = pair.ssh.run_script(client, f'ROOT={str(server.control)!r}\n' + check, timeout=60)
        assert result['exit_code'] == 0, result
        save_once(evidence / 'stopped_process_check.json', json.loads(result['stdout']))
        store.setup()
        epoch = store.verify_progress(server.arm)
        assert epoch == failed['completed_epochs'] == failed['actual_loop_epoch'] < 296
        assert failed['next_epoch'] == epoch + 1
        assert failed['optimizer_step_attempts'] == (epoch + 1) * monitor.protocol.BATCHES
        archive = server.control / 'recovery_20261122_files'
        pair.run(client, pair.cmd(['mkdir', '-p', archive]))
        with client.open_sftp() as sftp:
            gate = server.control / 'backup_gate' / monitor.protocol.phase_name(epoch)
            manifest_bytes = monitor.remote_metadata(sftp, gate / 'manifest.json')
            manifest = monitor.protocol.parse_manifest(manifest_bytes, server.arm, epoch)
            assert manifest['rolling_sha256'] == failed['rolling_checkpoint_sha256']
            expected = next(f for f in manifest['files'] if f['source'] == 'run/receipt.json')
            pending = dict(failed)
            pending['status'] = 'RUNNING'
            del pending['ended_utc']
            del pending['traceback']
            pending_bytes = (json.dumps(pending, indent=2, allow_nan=False) + '\n').encode()
            assert len(pending_bytes) == expected['bytes']
            assert monitor.protocol.digest(pending_bytes) == expected['sha256']
            current_receipt = monitor.remote_metadata(sftp, server.control / 'run/receipt.json', stable=False)
            assert current_receipt in (failed_bytes, pending_bytes), 'Unexpected remote receipt mutation'
            for name in ('original_failed_receipt.json', 'original_supervisor.json'):
                pair.upload(sftp, archive / name, (evidence / name).read_bytes())
            pair.upload(sftp, archive / 'original_pending_receipt.json', pending_bytes)
            save_once(evidence / 'pending_receipt_reconstruction.json', {
                'status': 'EXACT_ORIGINAL_MANIFEST_BYTES_RECONSTRUCTED',
                'manifest_sha256': monitor.protocol.digest(manifest_bytes),
                'failed_receipt_sha256': monitor.protocol.digest(failed_bytes),
                'pending_receipt_sha256': monitor.protocol.digest(pending_bytes),
                'failure_preserved_locally_and_remotely': True,
            })
            temporary = server.control / 'run/recovery_20261122_receipt.partial'
            pair.upload(sftp, temporary, pending_bytes)
            sftp.posix_rename(str(temporary), str(server.control / 'run/receipt.json'))
            assert monitor.backup_once(sftp, server.control, store, epoch) is False
        accept = '''
import json,sys
from pathlib import Path
root=Path(ROOT);sys.path.insert(0,str(root))
import backup_protocol as p
gate=p.EpochGate(root/'run')
data=p.local_metadata(gate.stage(EPOCH)/'manifest.json')
print(json.dumps(gate.verify_ack(data,p.parse_manifest(data,root.name,EPOCH),0)))
'''
        result = pair.ssh.run_script(client, f'ROOT={str(server.control)!r}\nEPOCH={epoch}\n' + accept,
                                     python=str(server.python), timeout=120)
        assert result['exit_code'] == 0, result
        with client.open_sftp() as sftp:
            assert monitor.backup_once(sftp, server.control, store, epoch)
        report = {'arm': server.arm, 'accepted_rounds': epoch + 1,
                  'checkpoint_sha256': manifest['rolling_sha256'], 'manifest': manifest,
                  'repeated_updates': 0, 'repeated_evaluations': 0,
                  'remote_root': str(server.control), 'accepted_at_utc': monitor.protocol.utc_now()}
        pair.save(evidence / 'salvaged_boundary.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'manifest'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int, choices=pair.ssh.PORTS)
    main(parser.parse_args().port)
