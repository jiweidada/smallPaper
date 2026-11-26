"""Copy only required native modules; never edit the shared training repository."""
from pathlib import Path
import hashlib
import json
import datetime

HERE = Path(__file__).resolve().parent
SOURCE = Path(r'D:\BaiduNetdiskDownload\DEIM_1572862469')
FILES = [
    'core/workspace.py', 'backbone/common.py', 'backbone/hgnetv2.py',
    'deim/deim.py', 'deim/dfine_decoder.py', 'deim/dfine_utils.py',
    'deim/denoising.py', 'deim/utils.py', 'deim/box_ops.py',
    'deim/rtdetrv3_supervision.py', 'deim/o2m_c.py', 'deim/hybrid_encoder.py',
    'deim/matcher.py', 'deim/deim_criterion.py', 'deim/postprocessor.py',
    'misc/dist_utils.py', 'misc/flash_attention.py', 'misc/deformable_detr_msda.py',
    'data/dataloader.py', 'logger_module/__init__.py', 'optim/ema.py',
    'extre_module/custom_nn/attention/ema.py',
    'extre_module/custom_nn/attention/simam.py',
]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    dest = HERE / 'native' / 'engine'
    if dest.exists():
        raise RuntimeError('Snapshot already exists; do not replace it silently.')
    rows = []
    for rel in FILES:
        p = SOURCE / 'engine' / rel
        data = p.read_bytes()
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        rows.append(dict(source=str(p), local=out.relative_to(HERE).as_posix(),
                         sha256=sha(data), bytes=len(data)))
    # Avoid unrelated optional detectors/evaluators at import time. Leaf model,
    # matching, loss, and attention modules above remain byte-identical.
    initializers = {
        '': '', 'core': 'from .workspace import register, create, GLOBAL_CONFIG\n',
        'backbone': '', 'deim': '', 'misc': '', 'optim': '',
        'data': 'from .dataloader import DataLoader\n', 'extre_module': '',
        'extre_module/custom_nn': '', 'extre_module/custom_nn/attention': '',
    }
    for package, content in initializers.items():
        p = dest / package / '__init__.py'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf8')
    recipe_path = SOURCE / 'configs/experiments/d1_zoomout_full96_20261002_v1/d1_full96_p05.yml'
    data = recipe_path.read_bytes()
    (HERE / 'native_recipe.yml').write_bytes(data)
    rows.append(dict(source=str(recipe_path), local='native_recipe.yml',
                     sha256=sha(data), bytes=len(data)))
    (HERE / 'native_source_manifest.json').write_text(json.dumps({
        'created_local': datetime.datetime.now().astimezone().isoformat(),
        'status': 'isolated_local_source_snapshot_not_a_frozen_experiment',
        'files': rows, 'package_initializers': initializers,
        'shared_source_changed': False,
        'scope': 'Explicit model construction only; do not run shared train.py with these candidate configs.',
    }, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'Copied {len(rows)} source files without modifying the shared repository.')

if __name__ == '__main__':
    main()
