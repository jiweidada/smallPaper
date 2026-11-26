"""Interactive SSH/SFTP session. Password stays in process memory only."""
from pathlib import Path, PurePosixPath
import getpass
import hashlib
import json
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
import paramiko

LOCAL = Path(__file__).resolve().parent
REMOTE = '/25085400120/mpsma_fair_20261103_v1'
client = paramiko.SSHClient()
client.load_host_keys(str(LOCAL / 'known_hosts_30848'))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
password = sys.stdin.readline().rstrip('\r\n') if '--pipe' in sys.argv else getpass.getpass('SSH password (not saved): ')
if not password:
    raise SystemExit('No password received; refusing connection.')
client.connect('172.17.120.160', port=30848, username='root', password=password,
               allow_agent=False, look_for_keys=False, timeout=15, auth_timeout=15)
# Retain only in this live process for explicitly requested reconnects; never
# serialize it, put it in command lines, or replay an interrupted operation.
client.get_transport().set_keepalive(20)
print(json.dumps({'connected': True, 'host': '172.17.120.160', 'port': 30848}), flush=True)
sftp = client.open_sftp()
for line in sys.stdin:
    try:
        request = json.loads(line)
        if request['op'] == 'close':
            break
        if request['op'] == 'reconnect':
            try: sftp.close()
            except Exception: pass
            client.close()
            client.connect('172.17.120.160', port=30848, username='root', password=password,
                           allow_agent=False, look_for_keys=False, timeout=15, auth_timeout=15)
            client.get_transport().set_keepalive(20)
            sftp=client.open_sftp()
            result={'reconnected':True,'interrupted_operation_replayed':False}
        elif request['op'] == 'put':
            source = Path(request['local']).resolve()
            dest = request['remote']
            if not (dest.startswith(REMOTE + '/') and '..' not in PurePosixPath(dest).parts):
                raise ValueError('Upload destination outside isolated experiment directory')
            started = time.time()
            sftp.put(str(source), dest)
            result = {'put': dest, 'bytes': source.stat().st_size, 'seconds': time.time() - started}
        elif request['op'] == 'get':
            source = request['remote']
            dest = Path(request['local']).resolve()
            if not (source.startswith(REMOTE + '/') and dest.is_relative_to(LOCAL)):
                raise ValueError('Download outside experiment scope')
            dest.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(source, str(dest))
            result = {'get': source, 'bytes': dest.stat().st_size, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
        elif request['op'] in ['exec', 'python']:
            command = request['command'] if request['op'] == 'exec' else request.get('python', '/opt/conda/envs/esm/bin/python') + ' -'
            stdin, stdout, stderr = client.exec_command(command, timeout=request.get('timeout', 60))
            if request['op'] == 'python':
                stdin.write(Path(request['script']).read_text(encoding='utf-8'))
            stdin.channel.shutdown_write()
            out = stdout.read().decode('utf-8', errors='replace')
            err = stderr.read().decode('utf-8', errors='replace')
            result = {'exit_code': stdout.channel.recv_exit_status(), 'stdout': out, 'stderr': err}
            if request.get('save'):
                dest = Path(request['save']).resolve()
                if not dest.is_relative_to(LOCAL):
                    raise ValueError('Receipt outside local experiment directory')
                dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                result = {'exit_code': result['exit_code'], 'saved': str(dest), 'stdout_chars': len(out),
                          'stderr_chars': len(err), 'stdout_tail': out[-1800:], 'stderr_tail': err[-1000:]}
        else:
            raise ValueError('Unsupported operation')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False), flush=True)
sftp.close()
client.close()

