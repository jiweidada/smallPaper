#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/c10_region_restore_20261029_v1
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export YOLO_CONFIG_DIR="$ROOT/cache/yolo"
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
cd "$ROOT"
/root/anaconda3/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
root=Path('/root/c10_region_restore_20261029_v1')
receipt=json.loads((root/'assets/offline_bundle_archive.json').read_text())
path=root/'assets'/receipt['filename'];h=hashlib.sha256()
with path.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==receipt['sha256']
with tarfile.open(path) as tar:
    for m in tar.getmembers():
        p=(root/m.name).resolve()
        assert str(p).startswith(str(root)+'/') and m.isfile() and not m.issym() and not m.islnk()
    tar.extractall(root)
manifest=json.loads((root/'offline_bundle_manifest.json').read_text())
for row in manifest['packages']:
    path=root/'wheelhouse'/row['filename'];h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert h.hexdigest()==row['sha256'] and path.stat().st_size==row['bytes']
print('Transferred wheels verified:',len(manifest['packages']))
PY
PY="$ROOT/venv/bin/python"
"$PY" -m pip install --no-index --find-links "$ROOT/wheelhouse" --no-deps -r "$ROOT/offline_all_lock.txt"
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/receipts/pip_freeze.txt"
"$PY" - <<'PY'
import torch,torchvision,ultralytics,numpy,PIL,json,sys
from pathlib import Path
from PIL import features
root=Path('/root/c10_region_restore_20261029_v1')
assert torch.cuda.is_available()
r={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,
   'numpy':numpy.__version__,'pillow':PIL.__version__,'libjpeg':features.version('jpg'),'libjpeg_turbo':features.version_feature('libjpeg_turbo'),
   'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'vram':torch.cuda.get_device_properties(0).total_memory,
   'installation':'local download -> SFTP -> no-index offline install'}
(root/'receipts/environment.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
PY
printf '0\n' > "$ROOT/receipts/offline_setup_exit_code.txt"
