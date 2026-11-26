"""Construction and strict initialization API for the isolated experiment."""
from pathlib import Path
import hashlib, json, sys

HERE = Path(__file__).resolve().parent
COMMON_INIT_SHA = '85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
IMPLEMENTATION = ('losses.py', 'native_adapter.py', 'runtime.py', 'experiment.py')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def implementation_hashes():
    return {name: sha(HERE/name) for name in IMPLEMENTATION}


def verify_training_snapshot(root):
    root = Path(root).resolve()
    binding = json.loads((HERE/'training_source_binding.json').read_text(encoding='utf-8'))
    for rel, expected in binding['files'].items():
        data = (root/rel).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f'Training source changed: {rel}')
    actual = {str(p.relative_to(root)).replace('\\', '/') for p in (root/'engine').rglob('*.py')}
    if actual != set(binding['files']):
        raise RuntimeError('Training Python source file set differs from the bound snapshot')
    return binding


def activate_training_snapshot(root):
    root = Path(root).resolve()
    verify_training_snapshot(root)
    if 'engine' in sys.modules:
        raise RuntimeError('Training requires a fresh interpreter')
    sys.path.insert(0, str(root))


def register_components():
    from engine.core import register
    from native_adapter import BudgetNormDEIM, BudgetNormCriterion
    register()(BudgetNormDEIM)
    register()(BudgetNormCriterion)

def verify_sources():
    sources = json.loads((HERE/'source_binding.json').read_text(encoding='utf-8'))
    for path, digest in sources.items():
        if sha(path) != digest:
            raise RuntimeError(f'Bound source changed: {path}')
    return sources

def activate_snapshot():
    verify_sources()
    root = HERE.parent/'local_candidates_20261004_v2'
    if 'engine' in sys.modules and root/'native' not in Path(sys.modules['engine'].__file__).resolve().parents:
        raise RuntimeError('Use a fresh interpreter; a different engine is loaded')
    sys.path.insert(0, str(root/'native'))
    if sys.platform == 'win32': sys.path.insert(0, str(root/'local_deps'))

def build_components(plan):
    from native_adapter import BudgetNormDEIM, BudgetNormCriterion
    from engine.deim.postprocessor import PostProcessor
    model = BudgetNormDEIM(plan['model_recipe'], plan['arm'])
    criterion = BudgetNormCriterion(plan['criterion_recipe'], plan['arm'], **plan['budget_loss'])
    postprocessor = PostProcessor(num_classes=10, num_top_queries=300, use_focal_loss=True,
                                  remap_mscoco_category=False, score_threshold=.001)
    return model, criterion, postprocessor

def load_common_initialization(model, path):
    import torch
    if sha(path) != COMMON_INIT_SHA:
        raise RuntimeError('Expected the bound common initialization, not a trained checkpoint')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['ema']['module'] if 'ema' in payload else payload['model']
    current = model.state_dict()
    if set(current) != set(state):
        raise RuntimeError('Initialization state keys do not match the native model')
    for key in state:
        if current[key].shape != state[key].shape or current[key].dtype != state[key].dtype:
            raise RuntimeError(f'Initialization tensor mismatch: {key}')
    model.load_state_dict(state, strict=True)
    return {'common_init_sha': COMMON_INIT_SHA, 'native_tensor_count': len(state),
            'new_zero_keys': []}
