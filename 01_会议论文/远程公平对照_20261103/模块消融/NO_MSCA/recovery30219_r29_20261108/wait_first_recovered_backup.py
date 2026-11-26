"""Observe the first new accepted backup only; never controls the training process."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import psutil

sys.dont_write_bytecode = True
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
import backup_protocol as protocol
backups = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
stage = backups / 'epochs/epoch0029'
deadline = time.monotonic() + 1200
while True:
    process = psutil.Process(18924)
    assert abs(process.create_time() - 1794207481.6664748) < .01, 'Backup-client identity changed'
    status = json.loads((backups / 'monitor_status.json').read_text(encoding='utf-8'))
    assert status['status'] not in ['MONITOR_FAILED_NO_FURTHER_ACKS', 'MONITOR_STOPPED_BY_OPERATOR'], status
    accepted_path = stage / 'accepted.json'
    if accepted_path.is_file():
        payload = protocol.local_metadata(stage / 'manifest.json')
        manifest = protocol.parse_manifest(payload, 'no_msca_20261107_v1', 29)
        assert protocol.local_metadata(stage / 'ack.json') == protocol.ack_bytes(payload, manifest)
        assert protocol.local_metadata(accepted_path) == protocol.accepted_bytes(payload, manifest)
        assert manifest['next_epoch'] == 30 and manifest['optimizer_step_attempts'] == 48510
        result = {'status': 'FIRST_RECOVERED_ROUND_NATIVE_BACKUP_ACCEPTED',
            'observed_local_utc': datetime.now(timezone.utc).isoformat(), 'round': 30,
            'manifest_sha256': sha256(payload).hexdigest(), 'rolling_sha256': manifest['rolling_sha256'],
            'original_protocol_sha256': manifest['protocol_sha256'], 'optimizer_step_attempts': 48510,
            'backup_client_pid': process.pid, 'backup_client_create_time': process.create_time(),
            'native_monitor_status': status, 'scope': 'Native gate and existing client acceptance; independent state audit follows.'}
        with (here / 'first_recovered_round030_native_acceptance.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
        print(json.dumps(result), flush=True)
        break
    assert time.monotonic() < deadline, 'Observation deadline reached; training was not stopped or retried'
    time.sleep(5)
