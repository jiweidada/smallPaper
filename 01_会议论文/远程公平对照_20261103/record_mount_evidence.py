"""Record the currently observed mount distinction without inferring prior mounts."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'mount_persistence_observation_20261105.json'
PRIMARY = ROOT / 'current_mount_primary_20261105.json'
SECONDARY = ROOT / 'current_mount_secondary_20261105.json'
INITIAL_SECONDARY = ROOT / 'secondary_inventory.json'
OLD_PRIMARY_MOUNT = ROOT / 'primary_replication_inventory.json'

def sha(path): return sha256(path.read_bytes()).hexdigest()
def remote(path):
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    assert obj['exit_code'] == 0 and not obj['stderr']
    return json.loads(obj['stdout'])

assert not REPORT.exists()
one, two, initial = remote(PRIMARY), remote(SECONDARY), remote(INITIAL_SECONDARY)
p1, p2 = [{x['path']: x for x in obj['paths']} for obj in [one, two]]
target = '/25085400120'
experiment = target + '/mpsma_fair_20261103_v1'
assert p1[target]['exists'] and p1[experiment]['exists']
assert p1[target]['backing_mount']['mountpoint'] == target
assert p1[target]['backing_mount']['filesystem'] == 'nfs'
assert p1[experiment]['backing_mount'] == p1[target]['backing_mount']
assert not p2[target]['exists'] and not p2[experiment]['exists']
assert p2[target]['nearest_existing_path'] == '/' and p2[target]['backing_mount']['filesystem'] == 'overlay'
assert not initial['paths'][target]['exists'] and initial['hostname'] == two['hostname']
assert all(p['/tmp']['backing_mount']['filesystem'] == 'overlay' for p in [p1, p2])
old_text = json.loads(OLD_PRIMARY_MOUNT.read_text(encoding='utf-8-sig'))
assert old_text['exit_code'] == 0 and ' /25085400120' in old_text['stdout']

report = {'status': 'CURRENT_MOUNT_DISTINCTION_RECORDED',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'current_primary': {'path': str(PRIMARY), 'sha256': sha(PRIMARY)},
          'current_secondary': {'path': str(SECONDARY), 'sha256': sha(SECONDARY)},
          'secondary_before_preparation': {'path': str(INITIAL_SECONDARY), 'sha256': sha(INITIAL_SECONDARY)},
          'primary_pre_interruption_mount_observation': {'path': str(OLD_PRIMARY_MOUNT), 'sha256': sha(OLD_PRIMARY_MOUNT)},
          'findings': ['The current primary experiment path is on the mounted NFS filesystem at /25085400120.',
                       'The current secondary does not have /25085400120; the nearest existing parent is its root overlay.',
                       'The same secondary hostname did not have /25085400120 in the recorded pre-preparation inventory.',
                       'At this observation /tmp is on the root overlay in both environments.'],
          'limits': ['The secondary mount layout at the actual time of training was not captured by these observations.',
                     'An NFS mount is not itself proof of backup, retention, permissions or user quota.',
                     'No claim is made that the secondary artifacts are irrecoverable.',
                     'No mount, write test, file modification, model execution or training was performed.'],
          'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'], 'primary_filesystem': p1[target]['backing_mount']['filesystem'],
                  'secondary_target_exists': p2[target]['exists'], 'secondary_parent_filesystem': p2[target]['backing_mount']['filesystem']}))
