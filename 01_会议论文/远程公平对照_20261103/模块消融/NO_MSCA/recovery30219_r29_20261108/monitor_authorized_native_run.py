"""Read-only monitoring of the existing NO_MSCA worker and verified local backup client.

Never launches, modifies, retries or stops training; passwords remain in process memory.
An observation timeout is not a terminal training result.
"""
import argparse
from datetime import datetime, timezone
import getpass
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

import psutil
warnings.filterwarnings('ignore', category=DeprecationWarning)
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
import paramiko

HERE = Path(__file__).resolve().parent
ARM = HERE.parent
REMOTE_ROOT = HERE.parents[2]
BACKUPS = Path('D:/MPSMA_DEIM_experiment_backups/20261107/NO_MSCA/backups')
HOST, PORT = '172.17.120.160', 30219
BACKUP_PID, BACKUP_CREATED = 18924, 1794207481.6664748
WORKER_PID, WORKER_TICKS = 1779501, 18098010
SUPERVISOR_PID, SUPERVISOR_TICKS = 1779500, 18097864
BOOT = '2e5d4a32-c510-4f86-8b5a-eb66f03b864a'
PROTOCOL = '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
RUNNER = '7a3934c214f6b196180554c254864ecdc0555dc439bbb555c35b3e610f7e4e93'
ADDENDUM = 'e2db4f390756e33e742684e8f1eff526047403ba2db6bad17d5c9e21132e9d54'
REMOTE_HERE = '/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1'
OBSERVER_SOURCE_SHA = sha256(Path(__file__).read_bytes()).hexdigest()
sys.dont_write_bytecode = True
sys.path.insert(0, str(ARM))
import backup_protocol as protocol

def now(): return datetime.now(timezone.utc).isoformat()
def require(value, reason):
    if not value: raise RuntimeError(reason)
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def atomic(path, data):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    os.replace(temporary, path)

def accepted_backup(status, previous):
    candidate = previous
    if status['status'] == 'EPOCH_BACKUP_VERIFIED':
        candidate = status['epoch'] + 1
    elif status['status'] == 'COMPLETED_AND_BACKED_UP':
        candidate = 320
    elif status['status'] in ['MONITOR_READY', 'PINNED_SFTP_CONNECTED', 'RECONNECTING']:
        candidate = max(candidate, status.get('next_backup_epoch', candidate))
    if candidate > 0:
        folder = BACKUPS / f'epochs/epoch{candidate-1:04d}'
        payload = protocol.local_metadata(folder / 'manifest.json')
        manifest = protocol.parse_manifest(payload, 'no_msca_20261107_v1', candidate - 1)
        require(protocol.local_metadata(folder / 'ack.json') == protocol.ack_bytes(payload, manifest),
                'Local backup acknowledgement differs')
        require(protocol.local_metadata(folder / 'accepted.json') == protocol.accepted_bytes(payload, manifest),
                'Local backup acceptance differs')
    return candidate

def validate_live(value):
    require(value['boot_id'] == BOOT, 'Remote boot identity changed; do not adopt a replacement instance')
    supervisor = value['records']['supervisor.json']
    receipt = value['records']['run/receipt.json']
    require(supervisor['supervisor_pid'] == SUPERVISOR_PID and supervisor['supervisor_start_ticks'] == SUPERVISOR_TICKS,
            'Supervisor identity changed')
    require(supervisor['pid'] == WORKER_PID and supervisor['start_ticks'] == WORKER_TICKS,
            'Worker identity changed')
    require(supervisor['protocol_sha256'] == receipt['protocol_sha256'] == PROTOCOL,
            'Protocol binding changed')
    require(supervisor['runner_sha256'] == receipt['runner_sha256'] == RUNNER,
            'Runner binding changed')
    require(supervisor['recovery_addendum_sha256'] == receipt['recovery_addendum_sha256'] == ADDENDUM,
            'Recovery binding changed')
    require(receipt['pid'] == WORKER_PID and receipt['first_epoch_in_this_segment'] == 29,
            'Receipt worker/segment changed')
    require(receipt['arm'] == 'NO_MSCA' and receipt['optimizer_step_attempts'] <= 517440,
            'Arm or update budget changed')
    if supervisor['status'] in ['STOPPED', 'HARD_TIMEOUT'] or receipt['status'] == 'FAILED':
        return 'REMOTE_RECORDED_TERMINAL_FAILURE'
    if supervisor['status'] == 'COMPLETED' and receipt['status'] == 'COMPLETED':
        require(supervisor['exit_code'] == 0 and supervisor['final_receipt'] == receipt,
                'Genuine successful exit is missing')
        return 'REMOTE_COMPLETED_WAITING_FINAL_LOCAL_CAPTURE'
    if receipt['status'] == 'COMPLETED' and supervisor['status'] == 'RUNNING':
        require(receipt['completed_epochs'] == 320 and receipt['optimizer_step_attempts'] == 517440,
                'Unexpected completed receipt budget')
        item = value['processes']['supervisor_pid']
        if not item['exists']: return 'REMOTE_PROCESS_HANDLE_MISSING'
        require(item['start_ticks'] == SUPERVISOR_TICKS, 'Supervisor PID reused during completion')
        return 'REMOTE_COMPLETING_SUPERVISOR_PENDING'
    for key, ticks in [('supervisor_pid', SUPERVISOR_TICKS), ('pid', WORKER_TICKS)]:
        item = value['processes'][key]
        if not item['exists']: return 'REMOTE_PROCESS_HANDLE_MISSING'
        require(item['start_ticks'] == ticks, 'A remote PID was reused')
    require(supervisor['status'] == receipt['status'] == 'RUNNING', 'Unexpected nonterminal remote state')
    return 'REMOTE_WORKER_CONFIRMED_LIVE'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-round', type=int, required=True)
    parser.add_argument('--observation-name', required=True)
    parser.add_argument('--max-observation-seconds', type=int, default=5400)
    args = parser.parse_args()
    require(1 <= args.target_round <= 320 and args.observation_name.replace('_', '').isalnum(),
            'Invalid observation scope')
    destination = HERE / args.observation_name
    require(not destination.exists(), 'Observation already exists; inspect its process instead of duplicating it')
    require(sys.stdin.isatty(), 'Use an interactive terminal for the password')
    known_hosts = REMOTE_ROOT / 'known_hosts_30219'
    require(known_hosts.is_file(), 'Missing pinned known-host file')
    require(sha256((ARM / 'execution_protocol.json').read_bytes()).hexdigest() == PROTOCOL,
            'Original protocol changed')
    reader = HERE / 'read_live_recovery.py'
    reader_bytes = reader.read_bytes()
    reader_text = reader_bytes.decode('utf-8')
    with warnings.catch_warnings():
        warnings.simplefilter('error', getpass.GetPassWarning)
        password = getpass.getpass('SSH monitoring password (not saved): ')
    require(bool(password), 'No password supplied')
    destination.mkdir(exist_ok=False)
    myself = psutil.Process()
    identity = {'observer_pid': os.getpid(), 'observer_create_time': myself.create_time(),
        'observer_source_sha256': OBSERVER_SOURCE_SHA,
        'target_round': args.target_round, 'started_utc': now(), 'remote_ssh_port': PORT,
        'remote_reader_sha256': sha256(reader_bytes).hexdigest(),
        'backup_client_pid': BACKUP_PID, 'backup_client_create_time': BACKUP_CREATED,
        'remote_operations': 'Read-only process, receipt and GPU queries', 'training_control_actions': 0}
    atomic(destination / 'observer_identity.json', identity)
    client = None
    last_round = read(HERE.parents[1] / 'current_execution_state.json')['NO_MSCA']['last_accepted_round']
    last_progress = None
    last_progress_at = started = time.monotonic()
    offline_since = None
    first_sample = True
    def event(kind, **fields):
        value = {'event': kind, 'observed_local_utc': now(), **fields}
        with (destination / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        return value
    try:
        while True:
            native = read(BACKUPS / 'monitor_status.json')
            backup_alive = psutil.pid_exists(BACKUP_PID)
            if backup_alive:
                require(abs(psutil.Process(BACKUP_PID).create_time() - BACKUP_CREATED) < .01,
                        'Backup PID reused')
            if not backup_alive and native['status'] != 'COMPLETED_AND_BACKED_UP':
                raise RuntimeError('Backup client process handle is missing; remote state must be inspected')
            require(native['status'] not in ['MONITOR_FAILED_NO_FURTHER_ACKS', 'MONITOR_STOPPED_BY_OPERATOR'],
                    'Backup client recorded a terminal state; do not restart training')
            observed_round = accepted_backup(native, last_round)
            if observed_round != last_round:
                event('NATIVE_ACCEPTED_BACKUP_ADVANCED', previous_round=last_round, round=observed_round)
                last_round = observed_round
            try:
                if client is None:
                    client = paramiko.SSHClient()
                    client.load_host_keys(str(known_hosts))
                    client.set_missing_host_key_policy(paramiko.RejectPolicy())
                    client.connect(HOST, port=PORT, username='root', password=password, allow_agent=False,
                        look_for_keys=False, timeout=15, banner_timeout=15, auth_timeout=15)
                    client.get_transport().set_keepalive(15)
                    event('PINNED_READ_ONLY_SSH_CONNECTED')
                stream, stdout, stderr = client.exec_command('/usr/bin/python3 -', timeout=30)
                stream.write(reader_text)
                stream.channel.shutdown_write()
                out, err = stdout.read().decode(), stderr.read().decode(errors='replace')
                code = stdout.channel.recv_exit_status()
                require(code == 0, 'Read-only remote observation failed: ' + err[-700:])
                live = json.loads(out)
                live_state = validate_live(live)
                atomic(destination / 'latest_live.json', {'exit_code': code, 'stdout': out, 'stderr': err})
                receipt = live['records']['run/receipt.json']
                progress = (receipt['actual_loop_epoch'], receipt['optimizer_step_attempts'], receipt.get('phase'))
                if progress != last_progress:
                    last_progress_at = time.monotonic()
                    last_progress = progress
                sample = {**identity, 'observed_local_utc': now(), 'status': live_state,
                    'native_accepted_round': last_round, 'backup_client_alive': backup_alive,
                    'remote_round': receipt['actual_loop_epoch'] + 1, 'phase': receipt.get('phase'),
                    'optimizer_step_attempts': receipt['optimizer_step_attempts'],
                    'remote_worker_handle_confirmed_live': live_state == 'REMOTE_WORKER_CONFIRMED_LIVE',
                    'seconds_since_observed_progress': time.monotonic() - last_progress_at}
                atomic(destination / 'observer_status.json', sample)
                offline_since = None
                if first_sample:
                    print(json.dumps(event('INITIAL_REMOTE_AND_BACKUP_PROCESSES_VERIFIED', **sample)), flush=True)
                    first_sample = False
                if live_state in ['REMOTE_RECORDED_TERMINAL_FAILURE', 'REMOTE_PROCESS_HANDLE_MISSING']:
                    print(json.dumps(event(live_state, evidence=str(destination / 'latest_live.json'))), flush=True)
                    return 2
                if native['status'] == 'COMPLETED_AND_BACKED_UP' and live_state.startswith('REMOTE_COMPLETED'):
                    print(json.dumps(event('FINAL_CAPTURE_AVAILABLE_FOR_INDEPENDENT_AUDIT', round=320,
                        evidence=str(destination / 'latest_live.json'))), flush=True)
                    return 0
                if last_round >= args.target_round:
                    atomic(destination / f'milestone_round{args.target_round:03d}_live.json',
                           {'exit_code': code, 'stdout': out, 'stderr': err})
                    print(json.dumps(event('OBSERVED_BACKUP_MILESTONE', round=last_round,
                        remote_round=sample['remote_round'], evidence=str(destination))), flush=True)
                    return 0
                if time.monotonic() - last_progress_at > 1800:
                    print(json.dumps(event('PROGRESS_STALLED_BUT_PROCESS_LIVE', evidence=str(destination))), flush=True)
                    return 3
            except (paramiko.AuthenticationException, paramiko.BadHostKeyException):
                raise
            except (paramiko.SSHException, EOFError, TimeoutError, ConnectionError, OSError) as error:
                if client is not None: client.close()
                client = None
                offline_since = time.monotonic() if offline_since is None else offline_since
                atomic(destination / 'observer_status.json', {**identity, 'observed_local_utc': now(),
                    'status': 'OBSERVATION_TRANSPORT_RETRY_REMOTE_STATE_UNKNOWN',
                    'native_accepted_round': last_round, 'backup_client_alive': backup_alive,
                    'error_type': type(error).__name__, 'training_terminal_asserted': False})
                event('OBSERVATION_TRANSPORT_RETRY', error_type=type(error).__name__)
                if time.monotonic() - offline_since >= 600:
                    print(json.dumps(event('OBSERVABILITY_UNAVAILABLE_REQUIRES_INSPECTION',
                        native_accepted_round=last_round, training_terminal_asserted=False)), flush=True)
                    return 3
            if time.monotonic() - started >= args.max_observation_seconds:
                print(json.dumps(event('OBSERVATION_DEADLINE_ONLY', native_accepted_round=last_round,
                    training_stopped=False, terminal_training_result_asserted=False)), flush=True)
                return 0
            time.sleep(45)
    except Exception as error:
        value = event('OBSERVER_REQUIRES_ATTENTION', error_type=type(error).__name__,
                      message=str(error), training_control_actions=0)
        atomic(destination / 'observer_terminal.json', value)
        print(json.dumps(value), flush=True)
        return 2
    finally:
        if client is not None: client.close()
        del password

if __name__ == '__main__':
    raise SystemExit(main())
