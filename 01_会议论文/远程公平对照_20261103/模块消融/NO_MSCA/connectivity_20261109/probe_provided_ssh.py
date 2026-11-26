"""Probe only the two user-provided SSH endpoints, without authentication."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--ports', nargs='+', type=int, choices=[30219, 30848], required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = (HERE / args.output).resolve()
assert output.is_relative_to(HERE) and not output.exists()
result = {'observed_local_utc': datetime.now(timezone.utc).isoformat(),
          'authentication_attempted': False, 'training_action': None, 'endpoints': []}
for port in args.ports:
    started = time.monotonic()
    item = {'host': '172.17.120.160', 'port': port}
    try:
        with socket.create_connection((item['host'], port), timeout=5) as connection:
            item['tcp_connected'] = True
            connection.settimeout(7)
            data = connection.recv(128)
            item['received_bytes'] = len(data)
            item['status'] = ('SSH_BANNER_OBSERVED' if data.startswith(b'SSH-') else
                              'CONNECTION_CLOSED_BEFORE_SSH_BANNER' if not data else
                              'UNEXPECTED_SERVICE_RESPONSE')
            if data.startswith(b'SSH-'):
                item['ssh_banner'] = data.decode('ascii', errors='backslashreplace').strip()
    except Exception as error:
        item.update(status='CONNECTION_OR_BANNER_ERROR', error_type=type(error).__name__,
                    error=str(error))
    item['elapsed_seconds'] = time.monotonic() - started
    result['endpoints'].append(item)
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result))
