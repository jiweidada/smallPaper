"""Password is provided on stdin in memory only; never store or print it."""
import argparse
import base64
import hashlib
import json
import logging
import pathlib
import shlex
import sys
import time
import paramiko

# Errors are returned as concise JSON below; do not duplicate Paramiko's tracebacks.
logging.getLogger('paramiko').setLevel(logging.CRITICAL)

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, required=True)
p.add_argument('--python', default='python3')
p.add_argument('--script', type=pathlib.Path)
p.add_argument('--script-arg', action='append', default=[])
p.add_argument('--command')
p.add_argument('--output', type=pathlib.Path)
p.add_argument('--timeout', type=int, default=120)
args = p.parse_args()
credentials = json.load(sys.stdin)
known = pathlib.Path(__file__).resolve().parent.parents[2]/'英文小论文/data/06_第二轮_20261014/恢复执行_20261015_v4'/('known_hosts_'+str(args.port))
client = paramiko.SSHClient()
if known.is_file():
    client.load_host_keys(str(known))
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
stdout = bytearray()
stderr = bytearray()
operation_stage = 'connecting_and_authenticating'
try:
    client.connect('172.17.120.160', port=args.port, username='root', password=credentials[str(args.port)],
                   look_for_keys=False, allow_agent=False, timeout=15, auth_timeout=20, banner_timeout=20)
    credentials.clear()
    transport = client.get_transport()
    transport.set_keepalive(15)
    host_key_sha256 = base64.b64encode(hashlib.sha256(transport.get_remote_server_key().asbytes()).digest()).decode()
    client.save_host_keys(str(known))
    operation_stage = 'opening_command_channel'
    command = ('OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '
               +shlex.quote(args.python)+' -u -'+''.join(' '+shlex.quote(v) for v in args.script_arg)) if args.script else args.command
    assert command
    channel = client.get_transport().open_session(timeout=20)
    operation_stage = 'requesting_command'
    channel.exec_command(command)
    if args.script:
        operation_stage = 'sending_remote_script'
        channel.sendall(args.script.read_bytes())
    channel.shutdown_write()
    operation_stage = 'receiving_command_output'
    deadline = time.monotonic()+args.timeout
    while True:
        while channel.recv_ready():
            stdout.extend(channel.recv(65536))
        while channel.recv_stderr_ready():
            stderr.extend(channel.recv_stderr(65536))
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        if time.monotonic()>deadline:
            raise TimeoutError('Read-only/authorized remote command timeout; no credentials logged')
        time.sleep(.03)
    code = channel.recv_exit_status()
    if args.output:
        with args.output.open('xb') as handle:
            handle.write(stdout)
        with args.output.with_suffix(args.output.suffix+'.stderr.txt').open('xb') as handle:
            handle.write(stderr)
        print(json.dumps({'port': args.port, 'returncode': code, 'output': str(args.output),
                          'bytes': len(stdout), 'stderr_tail': stderr.decode(errors='replace')[-1200:],
                          'server_host_key_sha256': host_key_sha256}, ensure_ascii=False))
    else:
        sys.stdout.buffer.write(stdout)
        sys.stderr.buffer.write(stderr)
    sys.exit(code)
except Exception as exc:
    if args.output:
        partial = args.output.with_suffix(args.output.suffix+'.partial')
        with partial.open('xb') as handle:
            handle.write(stdout)
    print(json.dumps({'port': args.port, 'stage': operation_stage, 'error_type': type(exc).__name__, 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
    sys.exit(1)
finally:
    credentials.clear()
    client.close()
