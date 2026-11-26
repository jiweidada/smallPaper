"""Record known original inputs/support identities locally, without model loads."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent/'fine_residual_geometry_20261008_v5'
INDEX = HERE.parent.parent/'data'/'00_实验计划与证据索引'
BASELINES = INDEX/'正式132轮基线模型下载_20261008_v1'


def sha(path, lf=False):
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n', b'\n') if lf else raw).hexdigest()


targets = {}
for seed in (1, 2):
    root = BASELINES/f'seed{seed}'
    receipt = json.loads((root/'download_receipt.json').read_text(encoding='utf-8'))
    args_sha, log_sha = sha(root/'run'/'args.json', True), sha(root/'run'/'log.txt', True)
    assert args_sha == receipt['args_sha256'] and log_sha == receipt['log_sha256']
    targets[str(seed)] = {'original_args_LF_sha256': args_sha, 'original_log_LF_sha256': log_sha,
                         'checkpoint_sha256': receipt['files']['last.pth']['sha256'],
                         'val_ann_sha256': '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd',
                         'original_stats': receipt['final_EMA_COCO_max_dets100'],
                         'download_receipt_sha256': sha(root/'download_receipt.json')}
binding = {'targets': targets,
           'original_Linux_mask_sha256': 'f490eedd34ec079dd49f27f2d7859f2dbc8f95f64f5bd067fcfd1371c556a1d4',
           'experiment_files': {name: sha(V5/name) for name in ('runtime.py', 'training_source_binding.json',
                                                              'evaluator_binding.py', 'evaluator_sources.json')},
           'Linux_mask_identity_boundary': 'Identified archived/seed2 original backend; other hosts must match or stop, not silently substitute.',
           'preparation_model_or_evaluator_calls': 0, 'preparation_checkpoint_loads': 0}
with (HERE/'source_binding.json').open('x', encoding='utf-8') as stream:
    json.dump(binding, stream, ensure_ascii=False, indent=2)
print('Prepared two original-run identities and bound native/evaluator support')
