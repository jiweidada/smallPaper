"""Use the existing experiment SSH identity; never read or log credentials."""
import argparse
import pathlib
import shlex
import subprocess
import sys

ENDPOINTS = {'A': 30770, 'B': 30576}
KEY = 'C:/Users/jiwei/.ssh/id_ed25519_lab'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('account', choices=ENDPOINTS)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--command')
    group.add_argument('--script', type=pathlib.Path)
    group.add_argument('--put', nargs=2)
    group.add_argument('--get', nargs=2)
    p.add_argument('--python', default='python3')
    p.add_argument('--timeout', type=int, default=90)
    args = p.parse_args()
    ssh = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
           '-p', str(ENDPOINTS[args.account]), 'root@172.17.120.160']
    payload = None
    command = args.command
    if args.script:
        payload = args.script.read_bytes()
        command = 'PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(args.python) + ' -u -'
    if args.put:
        local, remote = args.put
        payload = pathlib.Path(local).read_bytes()
        code = 'import sys,pathlib; p=pathlib.Path(' + repr(remote) + '); f=p.open("xb"); f.write(sys.stdin.buffer.read()); f.close()'
        command = shlex.quote(args.python) + ' -c ' + shlex.quote(code)
    if args.get:
        remote, local = args.get
        if pathlib.Path(local).exists():
            raise RuntimeError('Refusing to overwrite existing local file')
        code = 'import sys,pathlib; sys.stdout.buffer.write(pathlib.Path(' + repr(remote) + ').read_bytes())'
        command = shlex.quote(args.python) + ' -c ' + shlex.quote(code)
    completed = subprocess.run(ssh + [command], input=payload, capture_output=True, timeout=args.timeout)
    sys.stderr.buffer.write(completed.stderr)
    if args.get and completed.returncode == 0:
        with pathlib.Path(args.get[1]).open('xb') as output:
            output.write(completed.stdout)
    else:
        sys.stdout.buffer.write(completed.stdout)
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
