"""Bind the observed local backup client and exact SFTP round trip before one launch."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import psutil

here = Path(__file__).resolve().parent
base = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA')
status_path = base / 'backups/monitor_status.json'
status = json.loads(status_path.read_text(encoding='utf-8'))
assert status['status'] == 'PINNED_SFTP_CONNECTED' and status['next_backup_epoch'] == 29
process = psutil.Process(18924)
assert abs(process.create_time() - 1794207481.6664748) < .01
argv = process.cmdline()
assert str(base / 'backup_monitor.py') in argv and argv[argv.index('--port') + 1] == '30219'
expected = '4a4a67671cdb6a9419ab47b11db4225cf8642a8096ca4dc6f501148d27cd92f1'
assert sha256((here / 'round29_sftp_readback.pt').read_bytes()).hexdigest() == expected
installation = json.loads((here / 'installation_state_followup.json').read_text(encoding='utf-8'))
assert installation['exit_code'] == 0
installed = json.loads(installation['stdout'])['installation_report']
assert installed['status'] == 'RECOVERY_INSTALLED_NO_TRAINING'
assert installed['rolling_checkpoint_sha256'] == expected
review = here.parents[3] / '实验数据汇总_20261108/_build/recovery_static_review.json'
assert json.loads(review.read_text(encoding='utf-8'))['status'] == 'READ_ONLY_STATIC_REVIEW_PASSED'
result = {'status': 'READY_FOR_ONE_EXPLICITLY_AUTHORIZED_RECOVERY_LAUNCH',
    'observed_local_utc': datetime.now(timezone.utc).isoformat(),
    'monitor_pid': process.pid, 'monitor_create_time': process.create_time(), 'monitor_session_id': 8984,
    'backup_status': status, 'sftp_roundtrip_checkpoint_sha256': expected,
    'installation': installed, 'independent_static_review': str(review),
    'independent_static_review_sha256': sha256(review.read_bytes()).hexdigest(),
    'original_protocol_sha256': '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc',
    'recovery_addendum_sha256': 'e2db4f390756e33e742684e8f1eff526047403ba2db6bad17d5c9e21132e9d54',
    'first_new_round': 30, 'last_retained_round': 320, 'training_started': False}
with (here / 'launch_readiness_verified.json').open('x', encoding='utf-8') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps({'status': result['status'], 'monitor_pid': process.pid, 'first_new_round': 30}))
