"""Read the current ablation process identity and progress without changing it."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_mps_20261107_v1')


def read_optional(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


supervisor = read_optional(ROOT / 'supervisor.json')
receipt = read_optional(ROOT / 'run/receipt.json')
result = {'observed_server_utc': datetime.now(timezone.utc).isoformat(),
          'protocol_sha256': hashlib.sha256((ROOT / 'execution_protocol.json').read_bytes()).hexdigest(),
          'supervisor': supervisor, 'receipt': receipt}
if supervisor and supervisor.get('pid'):
    pid = supervisor['pid']
    try:
        fields = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(') ', 1)[1].split()
        result['trainer'] = {
            'pid': pid, 'process_state': fields[0], 'start_ticks': int(fields[19]),
            'start_ticks_match': int(fields[19]) == supervisor['start_ticks'],
            'boot_id_match': Path('/proc/sys/kernel/random/boot_id').read_text().strip() == supervisor['boot_id'],
            'wchan': Path('/proc/{}/wchan'.format(pid)).read_text().strip(),
        }
    except FileNotFoundError:
        result['trainer'] = {'pid': pid, 'process_gone': True}
result['gpu'] = subprocess.check_output([
    'nvidia-smi', '--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu',
    '--format=csv,noheader,nounits'], text=True).strip()
log = ROOT / 'training.log'
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 4000))
        result['training_log_tail'] = stream.read().decode('utf-8', errors='replace').splitlines()[-4:]
print(json.dumps(result, allow_nan=False))
