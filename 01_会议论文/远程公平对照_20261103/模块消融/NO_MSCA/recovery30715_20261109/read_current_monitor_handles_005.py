"""Read existing monitor outputs and exact process handles; never controls training."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil

HERE = Path(__file__).resolve().parent
OBS = HERE / 'resume_monitor_20261110_005'
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
value = {'observed_local_utc': datetime.now(timezone.utc).isoformat(), 'training_actions': 0}
try:
    status = json.loads((OBS / 'observer_status.json').read_text('utf-8-sig'))
    wrapper_path = OBS / 'latest_live.json'
    live = json.loads(json.loads(wrapper_path.read_text('utf-8-sig'))['stdout'])
    receipt = live['records']['run/receipt.json']
    native = json.loads((BACKUPS / 'monitor_status.json').read_text('utf-8-sig'))
    handles = {}
    for label, pid, created in [('backup', 8216, 1794292724.8624065), ('observer', 31100, 1794350415.3048158)]:
        try:
            p = psutil.Process(pid)
            handles[label] = p.is_running() and abs(p.create_time() - created) < .01
        except psutil.NoSuchProcess:
            handles[label] = False
    age = time.time() - wrapper_path.stat().st_mtime
    remote_handles = (live['boot_id'] == '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
        and all(live['processes'][k]['exists'] and live['processes'][k]['start_ticks'] == t
                for k, t in [('pid', 26597622), ('supervisor_pid', 26597399)]))
    native_round = native['epoch'] + 1 if native['status'] == 'EPOCH_BACKUP_VERIFIED' else (
        320 if native['status'] == 'COMPLETED_AND_BACKED_UP' else status['native_accepted_round'])
    native_terminal_failure = native['status'] in ['MONITOR_FAILED_NO_FURTHER_ACKS', 'MONITOR_STOPPED_BY_OPERATOR']
    milestone = native_round >= 296 and (OBS / 'milestone_round296_live.json').is_file()
    value.update(native_accepted_round=native_round, observer_native_accepted_round=status['native_accepted_round'],
        remote_round=receipt['actual_loop_epoch'] + 1, phase=receipt.get('phase'),
        optimizer_step_attempts=receipt['optimizer_step_attempts'], batch=receipt.get('batch'),
        observer_status=status['status'], live_age_seconds=round(age, 2), local_handles=handles,
        remote_handles_match=remote_handles, native_status=native['status'],
        milestone296_available=milestone,
        verification=('REQUIRES_FRESH_INSPECTION' if native_terminal_failure else
          'MILESTONE296_READY_FOR_INDEPENDENT_AUDIT' if milestone else
          'VERIFIED_WAIT' if handles['backup'] and handles['observer'] and age < 100 and remote_handles
              and status['status'] == 'REMOTE_WORKER_CONFIRMED_LIVE' else 'REQUIRES_FRESH_INSPECTION'))
except Exception as error:
    value.update(verification='REQUIRES_FRESH_INSPECTION', error_type=type(error).__name__, error=str(error))
print(json.dumps(value, ensure_ascii=True))
