"""Interactive SSH bridge; credential is read hidden and never persisted."""
import base64
import getpass
import json
from pathlib import Path
import sys
import paramiko

sys.stdout.reconfigure(encoding='utf8')
workspace = Path(__file__).resolve().parents[3]
client = paramiko.SSHClient()
client.load_host_keys(str(workspace/'计算机工程投稿/.codex-cache/server_known_hosts'))
known = client.get_host_keys()['[172.17.120.160]:30836']
for name, key in known.items():
    client.get_host_keys().add('[172.17.120.160]:30552', name, key)
client.set_missing_host_key_policy(paramiko.RejectPolicy())
password = getpass.getpass('SSH password (hidden): ')
try:
    client.connect('172.17.120.160', port=30552, username='root', password=password,
                   allow_agent=False, look_for_keys=False, timeout=20)
finally:
    password = None
print('AUTHENTICATED', flush=True)
try:
    for line in sys.stdin:
        action = json.loads(line)
        if action['type'] == 'exit':
            break
        if action['type'] in ('upload', 'download'):
            with client.open_sftp() as sftp:
                if action['type'] == 'upload':
                    try:
                        sftp.stat(action['remote'])
                    except FileNotFoundError:
                        sftp.put(action['local'], action['remote'])
                    else:
                        raise RuntimeError('Remote upload exists; refusing overwrite')
                else:
                    if Path(action['local']).exists():
                        raise RuntimeError('Local download exists; refusing overwrite')
                    sftp.get(action['remote'], action['local'])
            print(json.dumps({'status': 'transfer_complete'}), flush=True)
            continue
        command = action.get('command')
        if action['type'] == 'script':
            content = Path(action['path']).read_bytes()
            encoded = base64.b64encode(content).decode('ascii')
            command = "printf '%s' '"+encoded+"' | base64 -d | PYTHONDONTWRITEBYTECODE=1 python3 -"
        _, out, err = client.exec_command(command, timeout=120)
        stdout, stderr = out.read().decode('utf8', 'replace'), err.read().decode('utf8', 'replace')
        result = {'exit_code': out.channel.recv_exit_status(), 'stdout': stdout, 'stderr': stderr}
        if action.get('receipt'):
            Path(action['receipt']).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(result, ensure_ascii=False), flush=True)
except Exception as error:
    print(json.dumps({'status': 'bridge_error', 'type': type(error).__name__, 'message': str(error)}), flush=True)
finally:
    client.close()
