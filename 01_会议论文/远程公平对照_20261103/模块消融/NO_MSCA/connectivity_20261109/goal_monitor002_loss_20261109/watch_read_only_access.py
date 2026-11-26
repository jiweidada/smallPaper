"""Retry only the existing pinned SSH entry and read the original job once accessible."""
from datetime import datetime, timezone
from hashlib import sha256
import getpass
import json
import logging
from pathlib import Path
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import paramiko
import psutil

HERE = Path(__file__).resolve().parent
ARM = HERE.parents[1]
REMOTE_ROOT = ARM.parents[1]
READER = ARM / 'recovery30219_r29_20261108/read_live_recovery.py'
assert sha256(READER.read_bytes()).hexdigest() == 'ebd4075d17fb884f808a594a49511ba9c37f64eaa5b67915354674362ee332c2'
logging.getLogger('paramiko.transport').setLevel(logging.CRITICAL)
assert sys.stdin.isatty()
assert not (HERE / 'access_watch_identity.json').exists(), 'Inspect existing observer instead of duplicating it'
password = getpass.getpass('SSH read-only recovery password (not saved): ')
assert password
started = time.monotonic()
identity = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'host': '172.17.120.160', 'port': 30219, 'maximum_observation_seconds': 900,
    'source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
    'reader_sha256': sha256(READER.read_bytes()).hexdigest(), 'training_actions': 0}
with (HERE / 'access_watch_identity.json').open('x', encoding='utf-8') as handle:
    json.dump(identity, handle, indent=2)


def event(status, **fields):
    value = {'status': status, 'observed_local_utc': datetime.now(timezone.utc).isoformat(), **fields}
    with (HERE / 'access_watch_events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value) + '\n')
    temporary = HERE / 'access_watch_status.json.partial'
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, HERE / 'access_watch_status.json')
    return value


attempt = 0
try:
    while True:
        attempt += 1
        client = paramiko.SSHClient()
        client.load_host_keys(str(REMOTE_ROOT / 'known_hosts_30219'))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect('172.17.120.160', port=30219, username='root', password=password,
                allow_agent=False, look_for_keys=False, timeout=10, banner_timeout=10, auth_timeout=10)
            stream, stdout, stderr = client.exec_command('/usr/bin/python3 -', timeout=30)
            stream.write(READER.read_text(encoding='utf-8'))
            stream.channel.shutdown_write()
            wrapper = {'exit_code': stdout.channel.recv_exit_status(),
                'stdout': stdout.read().decode('utf-8'), 'stderr': stderr.read().decode('utf-8', errors='replace')}
            destination = HERE / 'restored_access_read_only_snapshot.json'
            with destination.open('x', encoding='utf-8') as handle:
                json.dump(wrapper, handle, ensure_ascii=False, indent=2)
            print(json.dumps(event('SSH_ACCESS_RESTORED_READ_ONLY_SNAPSHOT_CAPTURED', attempt=attempt,
                snapshot=str(destination), remote_reader_exit_code=wrapper['exit_code'], training_actions=0)), flush=True)
            break
        except (paramiko.BadHostKeyException, paramiko.AuthenticationException) as error:
            print(json.dumps(event('PINNING_OR_AUTHENTICATION_REQUIRES_ATTENTION',
                error_type=type(error).__name__, training_actions=0)), flush=True)
            raise SystemExit(2)
        except (paramiko.SSHException, EOFError, TimeoutError, ConnectionError, OSError) as error:
            value = event('READ_ONLY_ACCESS_RETRY_REMOTE_TRAINING_UNKNOWN', attempt=attempt,
                error_type=type(error).__name__, training_actions=0)
            if attempt == 1:
                print(json.dumps(value), flush=True)
            if time.monotonic() - started >= 900:
                print(json.dumps(event('READ_ONLY_OBSERVATION_WINDOW_EXPIRED_REMOTE_TRAINING_UNKNOWN',
                    attempt=attempt, training_actions=0)), flush=True)
                raise SystemExit(3)
        finally:
            client.close()
        time.sleep(45)
finally:
    del password
