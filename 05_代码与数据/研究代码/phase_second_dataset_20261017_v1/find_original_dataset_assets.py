"""Bounded search for original dataset assets, excluding known converted images and environments."""
import datetime
import json
import os
from pathlib import Path
import shutil

root = Path('/24085403014')
found = []
skip = {'envs', 'deim_migration_runtime', '.git', '.cache', '__pycache__', 'site-packages',
        'images', 'labels', 'outputs', 'logs', 'weights', 'node_modules', 'experiments',
        'TinyPerson_Datasets', 'UAVDT_Datasets'}
for current, dirs, files in os.walk(root):
    path = Path(current)
    depth = len(path.relative_to(root).parts)
    dirs[:] = [name for name in dirs if name not in skip and not name.startswith('.')]
    for name in dirs + files:
        low = name.lower()
        if any(token in low for token in ['tiny', 'uavdt', 'uav-benchmark', 'gt_ignore', 'gt_whole']):
            item = path / name
            found.append({'path': str(item), 'is_dir': item.is_dir(),
                          'bytes': item.stat().st_size if item.is_file() else None})
    if depth >= 5:
        dirs[:] = []
usage = shutil.disk_usage(root)
print(json.dumps({'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'found': found, 'disk_bytes': {'total': usage.total, 'free': usage.free},
                  'searched_root': str(root), 'excluded_directories': sorted(skip)}, ensure_ascii=False))
