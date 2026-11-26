"""Recover only never-started queues; use atomic directory locks on local /tmp."""
from pathlib import Path
import hashlib
import json
import os
import time

ARMS = ('backbone_silu', 'p2_skip', 'backbone_attention')
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assert_recoverable(v):
    assert v['frozen28_source_exact'], 'Source mismatch'
    assert v['old_dispatcher_stopped'], 'Old dispatcher remains live'
    assert not v['old_children'], 'Dispatcher has children'
    assert v['predecessor_warm_trainer_live'], 'No protected predecessor during handover'
    assert v['actual_configurations_started'] == v['actual_completed_epochs'] == 0
    assert not v['queue_failure'] and not v['existing_recovery']
    assert v['local_tmp_filesystem'] not in ('nfs', 'nfs4', 'cifs', 'ceph', 'fuse.glusterfs')


def same_live(handle, pid='pid', ticks='start_ticks'):
    try:
        fields = Path('/proc') / str(handle[pid]) / 'stat'
        fields = fields.read_text().split()
        return fields[21] == str(handle[ticks]) and fields[2] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def mount_type(path):
    path = str(Path(path).resolve())
    matches = []
    for line in Path('/proc/mounts').read_text().splitlines():
        f = line.split(); point = f[1].replace('\\040', ' ')
        if path == point or path.startswith(point.rstrip('/') + '/'):
            matches.append((len(point), f[2]))
    assert matches
    return max(matches)[1]


def snapshot(root, seed, old):
    manifest = read(root / 'recovery_v2/recovery_manifest.json')
    actual = {p.name: sha(p) for p in (root / 'code').iterdir() if p.is_file()}
    warm = root.parent / 'warm_stem12_20261011_v1' / f'seed{seed}_warm_stem/candidate_start.json'
    started = epochs = 0
    for arm in ARMS:
        screen = root / f'seed{seed}_{arm}'
        started += int(any((screen / n).exists() for n in (
            'detector_preflight_dispatch.json', 'detector_preflight_completion.json',
            'supervisor_dispatch.json', 'candidate_start.json', 'controller_start.json',
            'detector_preflight_v1/report.json')))
        log = screen / 'run/log.txt'
        epochs += len([s for s in log.read_text().splitlines() if s]) if log.exists() else 0
    pid = str(old['pid'])
    try:
        children = (Path('/proc') / pid / 'task' / pid / 'children').read_text().split()
    except FileNotFoundError:
        children = []
    return dict(frozen28_source_exact=actual == manifest['frozen28_files'],
        old_dispatcher_stopped=not same_live(old), old_children=children,
        predecessor_warm_trainer_live=warm.exists() and same_live(read(warm), 'child_pid', 'child_start_ticks'),
        actual_configurations_started=started, actual_completed_epochs=epochs,
        queue_failure=(root / f'queue_seed{seed}/failure.json').exists(),
        existing_recovery=(root / f'queue_seed{seed}/recovery_start_v2.json').exists(),
        local_tmp_filesystem=mount_type('/tmp'))


def guard_recovery(root, seed):
    old = read(root / f'queue_dispatch_seed{seed}.prelock_v1.json')
    state = snapshot(root, seed, old)
    assert_recoverable(state)
    manifest = read(root / 'recovery_v2/recovery_manifest.json')
    for name, digest in manifest['recovery_files'].items():
        assert sha(root / 'recovery_v2' / name) == digest
    return state


def publish_replacement(root, seed, owner):
    target = root / f'queue_dispatch_seed{seed}.json'
    assert read(target) == read(root / f'queue_dispatch_seed{seed}.prelock_v1.json')
    value = dict(owner, command=[os.sys.executable, '-B', str(root / 'recovery_v2/dispatch.py'),
        '--prefix', str(root.parent.parent.parent), '--seed', str(seed)],
        recovery='local_lock_before_any_configuration_started',
        original_dispatcher=read(root / f'queue_dispatch_seed{seed}.prelock_v1.json'),
        training_commands_in_launcher=0, automatic_retry=False,
        queue_directory=str(root / f'queue_seed{seed}'))
    temp = target.with_name(target.name + '.recovery_tmp')
    with temp.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2)
    temp.replace(target)


class LocalLaunchLock:
    def __init__(self, root, seed):
        uuid = read(root / 'code/training_protocol.json')['GPU_uuid'][str(seed)]
        assert uuid.startswith('GPU-') and all(c.isalnum() or c == '-' for c in uuid)
        assert mount_type('/tmp') not in ('nfs', 'nfs4', 'cifs', 'ceph', 'fuse.glusterfs')
        self.path = Path('/tmp') / f'deim-userbackbone12-{uuid}-seed{seed}.launch'
        self.owner = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21])
        self.acquired = False

    def acquire(self):
        self.path.mkdir(exist_ok=False)  # Fail immediately on conflict; no stale-owner takeover.
        with (self.path / 'owner.json').open('x', encoding='utf8') as f:
            json.dump(self.owner, f)
        self.acquired = True

    def release(self):
        assert self.acquired and read(self.path / 'owner.json') == self.owner
        (self.path / 'owner.json').unlink()
        self.path.rmdir()
        self.acquired = False
