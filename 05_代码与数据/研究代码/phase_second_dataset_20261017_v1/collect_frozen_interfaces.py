"""Copy scoped frozen implementation interfaces for a separate dataset adapter."""
import base64
import hashlib
import json
from pathlib import Path

code = Path('/24085403014/experiments/phase_mechanism_20261012/artifacts/round2_20261014_v1/execution_v3/code')
paths = ['engine/data/coco/coco_dataset.py', 'engine/data/coco/coco_eval.py',
         'engine/solver/det_engine.py', 'engine/solver/_solver.py', 'engine/solver/det_solver.py',
         'engine/core/yaml_config.py', 'engine/core/workspace.py', 'engine/data/dataloader.py',
         'engine/data/__init__.py', 'engine/data/dataset.py', 'train.py',
         'configs/experiments/phase/common.yml', 'configs/experiments/phase/proto96.yml',
         'configs/experiments/phase/baseline_full96.yml', 'configs/experiments/phase/c_full96.yml',
         'configs/experiments/phase/c_mean_full96.yml', 'configs/experiments/phase/lowpass_full96.yml']
files = []
for name in paths:
    path = code / name
    if path.is_file():
        raw = path.read_bytes()
        files.append({'relative': name, 'source': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'base64': base64.b64encode(raw).decode('ascii')})
    else: files.append({'relative': name, 'missing': True})
print(json.dumps({'code': str(code), 'files': files}))
