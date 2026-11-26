from pathlib import Path, PurePosixPath
import getpass, hashlib, json, sys
import paramiko

ROOT = Path(__file__).resolve().parent
REMOTE = PurePosixPath('/root/uav_a48_20261101_v1')
PIN = '5ef65ca83140d90523aa0aeaf4398863beeaa26068a00b4d0cee8f29068bd5ef'

class PinPolicy(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        if hashlib.sha256(key.asbytes()).hexdigest() != PIN:
            raise paramiko.SSHException('Server public key differs from observed key')

def local_path(relative):
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT)
    return path

password = getpass.getpass('SSH password (memory only): ')
print('{"status":"READY"}', flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req['op'] == 'close':
        break
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinPolicy())
    try:
        client.connect('172.17.120.160', port=30172, username='root', password=password,
                       timeout=15, auth_timeout=15, banner_timeout=15,
                       allow_agent=False, look_for_keys=False)
        assert hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest() == PIN
        client.get_transport().set_keepalive(10)
        if req['op'] == 'exec':
            stdin, stdout, stderr = client.exec_command(req['cmd'], timeout=req.get('timeout', 45))
            result = dict(status='EXECUTED', exit=stdout.channel.recv_exit_status(),
                          stdout=stdout.read().decode('utf-8', errors='replace'),
                          stderr=stderr.read().decode('utf-8', errors='replace'))
        elif req['op'] in ['put', 'get']:
            receipts = []
            with client.open_sftp() as sftp:
                sftp.get_channel().settimeout(60)
                for item in req['files']:
                    dest = REMOTE / item['remote']
                    assert dest.is_relative_to(REMOTE) and '..' not in dest.parts
                    path = local_path(item['local'])
                    if req['op'] == 'put':
                        sftp.put(str(path), str(dest), confirm=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        sftp.get(str(dest), str(path))
                    receipts.append(dict(local=str(path), remote=str(dest), bytes=path.stat().st_size,
                                         local_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            result = dict(status='TRANSFERRED', operation=req['op'], files=receipts)
        else:
            raise ValueError('Unsupported operation')
    except Exception as exc:
        result = dict(status='ERROR', type=type(exc).__name__, error=str(exc))
    finally:
        client.close()
    if req.get('receipt'):
        local_path(req['receipt']).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
password = None
