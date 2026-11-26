"""Accept the last completed epoch after a backup acknowledgement timeout."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int, choices=pair.ssh.PORTS)
    args = parser.parse_args()
    server = next(s for s in pair.SERVERS if s.port == args.port)
    control = pair.BACKUP / server.arm
    sys.path.insert(0, str(control))
    import backup_monitor as monitor

    monitor.CHECKPOINT_PYTHON = pair.CPU_PYTHON
    monitor.CHECKPOINT_TORCH_SITE = pair.TORCH_SITE
    store = monitor.BackupStore(control / 'backups', control)
    evidence = HERE / server.arm
    evidence.mkdir(exist_ok=True)
    with monitor.protect_local_lock(store.root), pair.ssh.connect(server.port) as client:
        live = pair.live_status(client, server)
        assert not live['worker_live'] and not live['supervisor_live'] and not live['backup_client_live']
        assert live['worker_status'] == 'FAILED' and live['phase'] == 'awaiting_local_backup'
        assert 'Local backup acknowledgement missing' in live['details_on_failure']
        if not (evidence / 'observed_failure.json').exists():
            pair.save(evidence / 'observed_failure.json', live)
        for remote, name in [('run/receipt.json', 'original_failed_receipt.json'),
                             ('supervisor.json', 'original_supervisor.json')]:
            pair.download(client, server.control / remote, evidence / name)
        store.setup()
        epoch = store.verify_progress(server.arm)
        assert epoch == live['completed_rounds'] < 296
        with client.open_sftp() as sftp:
            gate = server.control / 'backup_gate' / monitor.protocol.phase_name(epoch)
            manifest = json.loads(monitor.remote_metadata(sftp, gate / 'manifest.json'))
            item = next(f for f in manifest['files'] if f['source'] == 'run/receipt.json')
            failed = (evidence / 'original_failed_receipt.json').read_bytes()
            pending = json.loads(failed)
            pending['status'] = 'RUNNING'
            del pending['ended_utc']
            del pending['traceback']
            payload = (json.dumps(pending, indent=2, allow_nan=False) + '\n').encode()
            assert len(payload) == item['bytes'] and monitor.protocol.digest(payload) == item['sha256']
            pair.upload(sftp, server.control / 'recovery_original_failed_receipt.json', failed)
            temporary = server.control / 'run/recovery_receipt.partial'
            pair.upload(sftp, temporary, payload)
            sftp.posix_rename(str(temporary), str(server.control / 'run/receipt.json'))
            assert monitor.backup_once(sftp, server.control, store, epoch) is False
        script = '''
import json, sys
from pathlib import Path
here = Path(ROOT)
sys.path.insert(0, str(here))
import backup_protocol as protocol
gate = protocol.EpochGate(here / 'run')
data = protocol.local_metadata(gate.stage(EPOCH) / 'manifest.json')
manifest = protocol.parse_manifest(data, here.name, EPOCH)
print(json.dumps(gate.verify_ack(data, manifest, 0)))
'''
        result = pair.ssh.run_script(client, f'ROOT={str(server.control)!r}\nEPOCH={epoch}\n' + script,
                                     python=str(server.python), timeout=120)
        assert result['exit_code'] == 0, result
        with client.open_sftp() as sftp:
            assert monitor.backup_once(sftp, server.control, store, epoch)
        stage = store.stage(epoch)
        manifest = json.loads((stage / 'manifest.json').read_text())
        report = {'arm': server.arm, 'accepted_rounds': epoch + 1,
                  'checkpoint_sha256': manifest['rolling_sha256'], 'repeated_updates': 0,
                  'repeated_evaluations': 0, 'manifest': manifest,
                  'remote_root': str(server.control), 'accepted_at_utc': monitor.protocol.utc_now()}
        pair.save(evidence / 'salvaged_boundary.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'manifest'}), flush=True)


if __name__ == '__main__':
    main()
