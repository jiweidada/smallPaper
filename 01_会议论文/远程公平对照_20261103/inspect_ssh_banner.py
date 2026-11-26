"""Read only the SSH banners of the two explicitly supplied endpoints."""
from concurrent.futures import ThreadPoolExecutor
import json
import socket

HOST = '172.17.120.160'
PORTS = [30166, 30989]

def probe(port):
    try:
        s = socket.create_connection((HOST, port), timeout=4)
    except OSError as error:
        return {'port': port, 'tcp_connected': False, 'stage': 'connect',
                'error': type(error).__name__, 'message': str(error)}
    with s:
        try:
            s.settimeout(4)
            banner = s.recv(256)
            return {'port': port, 'tcp_connected': True,
                    'banner': banner.decode('utf-8', errors='replace')}
        except OSError as error:
            return {'port': port, 'tcp_connected': True, 'stage': 'banner',
                    'error': type(error).__name__, 'message': str(error)}

with ThreadPoolExecutor(max_workers=2) as pool:
    print(json.dumps(list(pool.map(probe, PORTS))))
