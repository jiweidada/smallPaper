"""Windows-user encrypted credentials for the existing read-only monitor.

Provisioning accepts JSON on stdin, never on argv. Default action reads progress;
--remote delegates arguments to remote_session_v4.py for authorized collection.
"""
import argparse
import base64
import ctypes
from ctypes import wintypes
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[2] / '英文小论文/data/06_第二轮_20261014'
VAULT = DATA / '每半小时跟进_v4/auth/ssh_credentials.dpapi.json'
ENTROPY = b'phase-monitor:automation:01a13311-a4ab-7631-905f-428f2dcdc0d2:v4'


class Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]


def crypt(payload, protect):
    if sys.platform != 'win32':
        raise RuntimeError('Windows current-user DPAPI is required')
    source_buffer = ctypes.create_string_buffer(payload)
    entropy_buffer = ctypes.create_string_buffer(ENTROPY)
    source = Blob(len(payload), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    entropy = Blob(len(ENTROPY), ctypes.cast(entropy_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = Blob()
    library = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    method = library.CryptProtectData if protect else library.CryptUnprotectData
    description_type = wintypes.LPCWSTR if protect else ctypes.c_void_p
    method.argtypes = [ctypes.POINTER(Blob), description_type, ctypes.POINTER(Blob),
                       ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    method.restype = wintypes.BOOL
    try:
        description = 'Phase monitor v4' if protect else None
        if not method(ctypes.byref(source), description, ctypes.byref(entropy),
                      None, None, 1, ctypes.byref(output)):
            raise OSError(ctypes.get_last_error(), 'DPAPI operation failed; no credential contents logged')
        return ctypes.string_at(output.data, output.size)
    finally:
        ctypes.memset(source_buffer, 0, len(payload))
        if output.data:
            ctypes.memset(output.data, 0, output.size)
            kernel.LocalFree(ctypes.cast(output.data, ctypes.c_void_p))


def credentials_bytes():
    record = json.loads(VAULT.read_text(encoding='utf-8'))
    if record['version'] != 1 or record['scope'] != 'Windows CurrentUser':
        raise RuntimeError('Unexpected encrypted credential format')
    return crypt(base64.b64decode(record['ciphertext']), False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provision', action='store_true')
    parser.add_argument('--credential-pipe')
    parser.add_argument('--remote', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.provision:
        if args.credential_pipe:
            if not args.credential_pipe.startswith('\\\\.\\pipe\\phase-monitor-provision-'):
                raise RuntimeError('Unexpected local credential pipe')
            with open(args.credential_pipe, 'r', encoding='utf-8') as source:
                credentials = json.load(source)
        else:
            credentials = json.load(sys.stdin)
        try:
            if set(credentials) != {'30059', '30512'} or not all(isinstance(v, str) and v for v in credentials.values()):
                raise RuntimeError('Expected credentials for the two approved endpoints')
            raw = json.dumps(credentials).encode('utf-8')
            encrypted = crypt(raw, True)
            if crypt(encrypted, False) != raw:
                raise RuntimeError('DPAPI round-trip verification failed')
            VAULT.parent.mkdir(parents=True, exist_ok=True)
            with VAULT.open('x', encoding='utf-8') as handle:
                json.dump({'version': 1, 'scope': 'Windows CurrentUser', 'ports': [30512, 30059],
                           'ciphertext': base64.b64encode(encrypted).decode('ascii')}, handle, indent=2)
            print('Encrypted monitor credentials provisioned and verified; no plaintext file created.')
        finally:
            credentials.clear()
        return
    for port in [30512, 30059]:
        pinned = DATA / '恢复执行_20261015_v4' / ('known_hosts_' + str(port))
        if not pinned.is_file():
            raise RuntimeError('Verified host-key file missing; refusing unpinned scheduled connection')
    script = 'remote_session_v4.py' if args.remote is not None else 'monitor_round2_v4.py'
    command = [sys.executable, '-W', 'ignore', '-X', 'utf8', '-B', str(HERE / script), *(args.remote or [])]
    result = subprocess.run(command, input=credentials_bytes(), timeout=130 if args.remote is None else 600)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
