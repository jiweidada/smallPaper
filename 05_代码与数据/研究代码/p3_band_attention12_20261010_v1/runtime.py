"""Strict native initialization and source bindings; no training on import."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
COMMON_INIT_SHA = '85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
IMPLEMENTATION = ('model.py', 'band_attention.py', 'runtime.py', 'experiment.py',
    'results.py', 'stage_trace.py', 'evaluator_binding.py', 'evaluator_sources.json',
    'training_audit.py', 'attention_contract.json', 'preflight_module.py', 'preflight_model.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hashes():
    return {name: sha(HERE/name) for name in IMPLEMENTATION}


def attention_contract():
    value = json.loads((HERE/'attention_contract.json').read_text(encoding='utf-8'))
    if value.get('confirmed') is not True:
        raise RuntimeError('Attention mechanism contract has not been confirmed')
    if not value.get('identity_zero_parameters'):
        raise RuntimeError('Missing explicit zero-initialization parameter contract')
    return value


def verify_training_snapshot(root):
    root = Path(root).resolve()
    binding = json.loads((HERE/'training_source_binding.json').read_text(encoding='utf-8'))
    for rel, expected in binding['files'].items():
        data = (root/rel).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('Training source changed: ' + rel)
    actual = {p.relative_to(root).as_posix() for p in (root/'engine').rglob('*.py')}
    if actual != set(binding['files']):
        raise RuntimeError('Training Python source inventory differs')
    return binding


def activate_training_snapshot(root):
    verify_training_snapshot(root)
    if 'engine' in sys.modules:
        raise RuntimeError('A fresh native interpreter is required')
    sys.path.insert(0, str(Path(root).resolve()))


def register_components():
    from engine.core import register
    from model import P3BandAttentionDEIM
    register()(P3BandAttentionDEIM)


def build_components(plan):
    from copy import deepcopy
    from model import P3BandAttentionDEIM
    from engine.deim.deim_criterion import DEIMCriterion
    from engine.deim.matcher import HungarianMatcher
    from engine.deim.postprocessor import PostProcessor
    model = P3BandAttentionDEIM(plan['model_recipe'], plan['arm'])
    recipe = deepcopy(plan['criterion_recipe'])
    matcher = recipe.pop('matcher'); matcher.pop('type', None)
    criterion = DEIMCriterion(matcher=HungarianMatcher(**matcher, use_focal_loss=True),
                              num_classes=10, **recipe)
    postprocessor = PostProcessor(num_classes=10, num_top_queries=300, use_focal_loss=True,
                                  remap_mscoco_category=False, score_threshold=.001)
    return model, criterion, postprocessor


def load_common_initialization(model, path):
    import torch
    contract = attention_contract()
    if sha(path) != COMMON_INIT_SHA:
        raise RuntimeError('Expected the bound shared initialization bytes')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['ema']['module'] if 'ema' in payload else payload['model']
    current = model.state_dict()
    extras = {key for key in current if key.startswith('band_attention.')}
    if model.attention_arm == 'baseline':
        if model.band_attention is not None or extras:
            raise RuntimeError('Baseline must contain no attention module or extra state')
    elif model.attention_arm in ('wavelet', 'spatial'):
        if model.band_attention is None or not extras:
            raise RuntimeError('Candidate attention state is absent')
        parameters = dict(model.band_attention.named_parameters())
        for name in contract['identity_zero_parameters']:
            if name not in parameters or torch.count_nonzero(parameters[name]).item():
                raise RuntimeError('Identity parameter missing or nonzero: ' + name)
        if sum(p.numel() for p in parameters.values()) != contract['trainable_parameters_per_attention_arm']:
            raise RuntimeError('Attention trainable parameter count differs')
    else:
        raise RuntimeError('Unknown attention arm')
    if set(current) - set(state) != extras or set(state) - set(current):
        raise RuntimeError('Native keys or attention extra-key namespace differ')
    for name, tensor in state.items():
        if tensor.shape != current[name].shape or tensor.dtype != current[name].dtype:
            raise RuntimeError('Native tensor shape/dtype differs: ' + name)
    result = model.load_state_dict(state, strict=not extras)
    if set(result.missing_keys) != extras or result.unexpected_keys:
        raise RuntimeError('Unexpected native initialization load result')
    return dict(common_init_sha=COMMON_INIT_SHA, native_tensor_count=len(state),
                attention_arm=model.attention_arm, new_state_keys=sorted(extras),
                new_parameter_count=sum(p.numel() for n,p in model.named_parameters()
                                        if n.startswith('band_attention.')),
                identity_zero_parameters=contract['identity_zero_parameters'],
                initial_output_equality='Zero identity parameters checked; complete forward equality requires preflight')
