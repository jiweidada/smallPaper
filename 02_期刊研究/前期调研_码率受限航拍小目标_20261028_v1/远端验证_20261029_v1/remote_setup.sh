#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/c10_region_restore_20261029_v1
mkdir -p "$ROOT"/logs "$ROOT"/assets "$ROOT"/receipts "$ROOT"/code
export PIP_CACHE_DIR="$ROOT/pip_cache"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export YOLO_CONFIG_DIR="$ROOT/cache/yolo"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
/root/anaconda3/bin/python -m venv "$ROOT/venv"
PY="$ROOT/venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install ultralytics==8.4.191 numpy==1.26.4 pillow==11.3.0 opencv-python==4.11.0.86 pycocotools==2.0.11
"$PY" -m pip freeze > "$ROOT/receipts/pip_freeze.txt"
"$PY" - <<'PY'
import torch,ultralytics,numpy,PIL,json,sys
from pathlib import Path
from PIL import features
root=Path('/root/c10_region_restore_20261029_v1')
assert torch.cuda.is_available()
r={'python':sys.version,'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'ultralytics':ultralytics.__version__,
   'numpy':numpy.__version__,'pillow':PIL.__version__,'libjpeg':features.version('jpg'),
   'libjpeg_turbo':features.version_feature('libjpeg_turbo'),'cuda_runtime':torch.version.cuda,
   'gpu':torch.cuda.get_device_name(0),'vram':torch.cuda.get_device_properties(0).total_memory}
(root/'receipts/environment.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r))
PY
printf '0\n' > "$ROOT/receipts/setup_exit_code.txt"
