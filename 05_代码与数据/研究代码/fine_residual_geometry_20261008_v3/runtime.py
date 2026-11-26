"""Construction and strict initialization API for the isolated experiment."""
from pathlib import Path
import hashlib, json, sys

HERE = Path(__file__).resolve().parent
COMMON_INIT_SHA = '85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
IMPLEMENTATION = ('geometry.py', 'residual.py', 'runtime.py', 'experiment.py', 'results.py', 'stage_trace.py')

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
    from geometry import ResidualGeometryDEIM
    register()(ResidualGeometryDEIM)


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
    from copy import deepcopy
    from geometry import ResidualGeometryDEIM
    from engine.deim.deim_criterion import DEIMCriterion
    from engine.deim.matcher import HungarianMatcher
    from engine.deim.postprocessor import PostProcessor
    model = ResidualGeometryDEIM(plan['model_recipe'], plan['arm'])
    config = deepcopy(plan['criterion_recipe'])
    matcher = config.pop('matcher')
    matcher.pop('type', None)
    criterion = DEIMCriterion(matcher=HungarianMatcher(**matcher, use_focal_loss=True), num_classes=10, **config)
    postprocessor = PostProcessor(num_classes=10, num_top_queries=300, use_focal_loss=True,
                                  remap_mscoco_category=False, score_threshold=.001)
    return model, criterion, postprocessor


def load_common_initialization(model, path):
    import torch
    if sha(path) != COMMON_INIT_SHA:
        raise RuntimeError('Expected the bound common initialization')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['ema']['module'] if 'ema' in payload else payload['model']
    current = model.state_dict()
    expected_extra = set() if model.geometry_arm == 'baseline' else {
        'decoder.geometry_residual.' + x for x in
        ('kernel','mix_weight','mix_bias','output_weight','output_bias')}
    if set(current) - set(state) != expected_extra or set(state) - set(current):
        raise RuntimeError('Native keys or extra residual keys differ')
    for key in state:
        if state[key].shape != current[key].shape or state[key].dtype != current[key].dtype:
            raise RuntimeError('Native initialization tensor mismatch: ' + key)
    if expected_extra:
        for suffix in ('output_weight','output_bias'):
            if torch.count_nonzero(current['decoder.geometry_residual.'+suffix]).item():
                raise RuntimeError('Residual output must start at zero')
    result = model.load_state_dict(state, strict=not expected_extra)
    if set(result.missing_keys) != expected_extra or result.unexpected_keys:
        raise RuntimeError('Unexpected common initialization load result')
    return {'common_init_sha': COMMON_INIT_SHA, 'native_tensor_count': len(state),
            'geometry_arm': model.geometry_arm, 'new_parameter_count': 35396 if expected_extra else 0,
            'initial_output_equality': 'Zero residual retains the native forward'}
