"""Task-scoped SSH uploads/relay with in-memory DPAPI credentials and pinned keys."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import stat
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import paramiko

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'phase_parallel_20261014_v3'))
import scheduled_monitor_v4

ROOTS = {30512: '/25085400120/datasets_verified_20261017', 30059: '/24085403014/datasets_verified_20261017'}
PYTHONS = {30512: '/25085400120/envs/deim/bin/python', 30059: '/24085403014/deim_migration_runtime/envs/deim/bin/python'}
CONNECTION_REGISTRY = HERE.parents[1] / 'data/07_第二数据集_20261017/connection_registry.json'
if CONNECTION_REGISTRY.exists():
    for _account, _endpoint in json.loads(CONNECTION_REGISTRY.read_text(encoding='utf-8')).items():
        _old_port = {'A': 30512, 'B': 30059}[_account]
        ROOTS[int(_endpoint['port'])] = ROOTS[_old_port]
        PYTHONS[int(_endpoint['port'])] = PYTHONS[_old_port]


def connect(port, passwords):
    client = paramiko.SSHClient()
    known = scheduled_monitor_v4.DATA / '恢复执行_20261015_v4' / ('known_hosts_' + str(port))
    client.load_host_keys(str(known)); client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('172.17.120.160', port=port, username='root', password=passwords[str(port)],
                   look_for_keys=False, allow_agent=False, timeout=15, auth_timeout=20, banner_timeout=20)
    client.get_transport().set_keepalive(15)
    return client


def prepare_parent(sftp, port, path):
    root = PurePosixPath(ROOTS[port]); target = PurePosixPath(path)
    if not target.is_relative_to(root): raise RuntimeError('Destination outside authorized new dataset directory')
    current = PurePosixPath('/')
    for component in target.parent.parts[1:]:
        current /= component
        try:
            info = sftp.stat(str(current)); assert stat.S_ISDIR(info.st_mode)
        except FileNotFoundError: sftp.mkdir(str(current))
    for candidate in [str(target), str(target) + '.part']:
        try: sftp.stat(candidate)
        except FileNotFoundError: continue
        raise FileExistsError('Refusing to replace transfer target: ' + candidate)


def remote_sha(client, port, path):
    program = "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb');" \
              "[(h.update(b)) for b in iter(lambda:f.read(4194304),b'')];print(h.hexdigest())"
    command = 'OMP_NUM_THREADS=1 ' + shlex.quote(PYTHONS[port]) + ' -B -c ' + shlex.quote(program) + ' ' + shlex.quote(path)
    _, stdout, stderr = client.exec_command(command, timeout=180)
    value = stdout.read().decode().strip(); error = stderr.read().decode()
    if stdout.channel.recv_exit_status() != 0: raise RuntimeError('Remote SHA verification failed: ' + error[:300])
    return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, choices=ROOTS, required=True)
    parser.add_argument('--local', type=Path); parser.add_argument('--from-b')
    parser.add_argument('--remote', required=True); parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert (args.local is None) != (args.from_b is None)
    credentials = json.loads(scheduled_monitor_v4.credentials_bytes())
    receiver = sender = None; started = time.monotonic()
    try:
        receiver = connect(args.port, credentials); outgoing = receiver.open_sftp()
        prepare_parent(outgoing, args.port, args.remote)
        if args.local:
            assert args.local.resolve().is_relative_to(HERE.parents[2].resolve()), 'Local file must stay in this workspace'
            incoming = args.local.open('rb'); size = args.local.stat().st_size
        else:
            assert args.port == 30512 and args.from_b.startswith(ROOTS[30059] + '/')
            sender = connect(30059, credentials); source_sftp = sender.open_sftp()
            incoming = source_sftp.open(args.from_b, 'rb'); size = source_sftp.stat(args.from_b).st_size
        digest = hashlib.sha256(); total = 0; last_print = started
        with incoming, outgoing.open(args.remote + '.part', 'wx') as target:
            target.set_pipelined(True)
            while True:
                block = incoming.read(2 * 1024 * 1024)
                if not block: break
                target.write(block); digest.update(block); total += len(block)
                now = time.monotonic()
                if now - last_print >= 20:
                    print(json.dumps({'event': 'transfer_progress', 'port': args.port, 'bytes': total, 'total': size}), flush=True)
                    last_print = now
        assert total == size
        assert remote_sha(receiver, args.port, args.remote + '.part') == digest.hexdigest()
        outgoing.rename(args.remote + '.part', args.remote)
        receipt = {'port': args.port, 'source': str(args.local or args.from_b), 'remote': args.remote,
                   'bytes': total, 'sha256': digest.hexdigest(), 'remote_sha_verified': True,
                   'elapsed_seconds': time.monotonic() - started}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open('x', encoding='utf-8') as handle: json.dump(receipt, handle, ensure_ascii=False, indent=2)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    finally:
        credentials.clear()
        if receiver: receiver.close()
        if sender: sender.close()


if __name__ == '__main__': main()
