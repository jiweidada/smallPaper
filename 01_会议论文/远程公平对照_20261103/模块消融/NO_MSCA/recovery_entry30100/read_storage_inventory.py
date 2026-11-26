"""Read mount and known storage-path inventory, without mounting or changing files."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

result = {'observed_server_utc': datetime.now(timezone.utc).isoformat(), 'mounts': [], 'storage_paths': {}}
for line in Path('/proc/self/mountinfo').read_text().splitlines():
    left, right = line.split(' - ', 1)
    a, b = left.split(), right.split()
    target, filesystem, source = a[4], b[0], b[1]
    if filesystem in ['overlay', 'ext4', 'xfs', 'nfs', 'nfs4', 'ceph', 'fuse.ceph', 'fuseblk'] or target.startswith(('/25085400120', '/mnt', '/data', '/workspace')):
        result['mounts'].append({'target': target, 'filesystem': filesystem, 'source': source})
for name in ['/25085400120', '/workspace', '/mnt', '/data', '/root/autodl-tmp', '/autodl-tmp',
             '/root/miniconda3', '/opt/conda', '/25085400120/mpsma_fair_20261103_v1']:
    path = Path(name)
    info = {'exists': path.exists(), 'directory': path.is_dir()}
    if path.is_dir():
        info['entries'] = [p.name for p in sorted(path.iterdir())][:40]
        usage = shutil.disk_usage(path)
        info['free_bytes'] = usage.free
        info['total_bytes'] = usage.total
    result['storage_paths'][name] = info
result['known_original_processes'] = {}
for pid in [1774398, 1774399]:
    path = Path('/proc') / str(pid)
    info = {'exists': path.exists()}
    if path.exists():
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            info.update(state=fields[0], start_ticks=int(fields[19]),
                        command=(path / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace').strip())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    result['known_original_processes'][str(pid)] = info
for name, command in [('df', ['df', '-hT', '/', '/25085400120']), ('lsblk', ['lsblk', '-J', '-o', 'NAME,TYPE,SIZE,MOUNTPOINT,FSTYPE'])]:
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=10)
        result[name] = {'exit_code': output.returncode, 'stdout': output.stdout, 'stderr': output.stderr}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        result[name] = {'error_type': type(error).__name__}
print(json.dumps(result))
