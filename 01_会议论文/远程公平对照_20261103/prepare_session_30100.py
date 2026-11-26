"""Use the user-supplied new port with the original MSCA host-key pin."""
import base64
from hashlib import sha256
import json
from pathlib import Path
import paramiko

HERE = Path(__file__).resolve().parent
old_keys = paramiko.HostKeys(str(HERE / 'known_hosts_30848'))
old_label = '[172.17.120.160]:30848'
new_label = '[172.17.120.160]:30100'
keys = old_keys.lookup(old_label)
assert keys
target_keys = HERE / 'known_hosts_30100'
assert not target_keys.exists()
new_keys = paramiko.HostKeys()
fingerprints = {}
for kind, key in keys.items():
    new_keys.add(new_label, kind, key)
    fingerprints[kind] = 'SHA256:' + base64.b64encode(sha256(key.asbytes()).digest()).decode('ascii').rstrip('=')
new_keys.save(str(target_keys))
source = (HERE / 'remote_session_30848.py').read_text(encoding='utf-8')
assert source.count('30848') == 4
target = HERE / 'remote_session_30100.py'
with target.open('x', encoding='utf-8') as stream:
    stream.write(source.replace('30848', '30100'))
print(json.dumps({'new_port': 30100, 'host_key_policy': 'Reject keys differing from the previously pinned MSCA key',
                  'expected_fingerprints': fingerprints, 'credential_saved': False}))
