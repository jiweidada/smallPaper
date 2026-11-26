"""Verify a reallocated endpoint against existing pins; accept credentials on stdin only."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'phase_parallel_20261014_v3'))
import scheduled_monitor_v4 as secure
import paramiko

def main():
    supplied = json.load(sys.stdin)
    port = int(supplied['port'])
    password = supplied.pop('password')
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    dest = HERE.parents[1] / 'data/07_第二数据集_20261017/GPU重申请恢复_20261018'
    dest.mkdir(parents=True, exist_ok=True)
    transport = paramiko.Transport(socket.create_connection(('172.17.120.160', port), timeout=15))
    try:
        transport.start_client(timeout=20)
        key = transport.get_remote_server_key()
        matches = []
        pin_root = secure.DATA / '恢复执行_20261015_v4'
        for old in (30512, 30059):
            keys = paramiko.HostKeys(str(pin_root / f'known_hosts_{old}'))
            if any(k.asbytes() == key.asbytes() for h in keys.values() for k in h.values()):
                matches.append(old)
        assert matches, 'Server key differs from all previously verified endpoint pins; no authentication or persistence performed'
        transport.auth_password('root', password, fallback=False)
        assert transport.is_authenticated()
        transport.set_keepalive(15)
        sftp = paramiko.SFTPClient.from_transport(transport)
        accounts = []
        for account, storage, root in [
            ('A', '/25085400120', '/25085400120/DEIM/experiments/phase_second_dataset_uavdt_20261017_v1'),
            ('B', '/24085403014', '/24085403014/experiments/phase_second_dataset_uavdt_20261017_v1')]:
            try:
                with sftp.open(root + '/plan.json', 'rb') as f:
                    raw = f.read()
                plan = json.loads(raw)
                assert plan['account'] == account
                accounts.append({'account': account, 'storage': storage, 'root': root,
                                 'seed': plan['seed'], 'plan_sha256': hashlib.sha256(raw).hexdigest()})
            except FileNotFoundError:
                pass
        assert len(accounts) == 1, 'Cannot identify exactly one account from its frozen plan'
        account = accounts[0]
        channel = transport.open_session(timeout=15)
        channel.exec_command('nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits')
        gpu = channel.makefile('r').read().decode(errors='replace')
        error = channel.makefile_stderr('r').read().decode(errors='replace')
        rc = channel.recv_exit_status()
        receipt = dict(account, port=port, checked_utc=now.isoformat(),
                       matched_previous_pin_ports=matches,
                       host_key_sha256=base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('='),
                       gpu={'returncode': rc, 'stdout': gpu, 'stderr': error})
        # Preserve all encrypted old endpoints; no plaintext secret is written.
        credentials = json.loads(secure.credentials_bytes())
        try:
            credentials[str(port)] = password
            raw = json.dumps(credentials).encode()
            encrypted = secure.crypt(raw, True)
            assert secure.crypt(encrypted, False) == raw
            vault = secure.VAULT
            backup = vault.with_name(vault.stem + '.' + stamp + '.backup.json')
            with backup.open('xb') as f:
                f.write(vault.read_bytes())
            record = {'version': 1, 'scope': 'Windows CurrentUser',
                      'ports': sorted(map(int, credentials)),
                      'ciphertext': base64.b64encode(encrypted).decode('ascii')}
            temp = vault.with_suffix('.next')
            temp.write_text(json.dumps(record, indent=2), encoding='utf-8')
            os.replace(temp, vault)
        finally:
            credentials.clear()
            password = None
        hostkeys = paramiko.HostKeys()
        hostkeys.add(f'[172.17.120.160]:{port}', key.get_name(), key)
        hostkeys.save(str(pin_root / f'known_hosts_{port}'))
        registry_path = HERE.parents[1] / 'data/07_第二数据集_20261017/connection_registry.json'
        registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {}
        registry[account['account']] = receipt
        temp = registry_path.with_suffix('.next')
        temp.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, registry_path)
        path = dest / (stamp + '_endpoint_verified.json')
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'verified': receipt, 'receipt': str(path)}, ensure_ascii=False))
        sftp.close()
    finally:
        password = None
        transport.close()

if __name__ == '__main__':
    main()
