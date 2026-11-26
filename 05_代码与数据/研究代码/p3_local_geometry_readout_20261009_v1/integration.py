"""Source-bound construction, strict loading, and native registry integration.

Importing this file does not construct a model, optimizer, or training loop.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXTRA_PREFIX = 'decoder.local_readout.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    binding = json.loads((HERE/'source_binding.json').read_text(encoding='utf-8'))
    for path, digest in binding.items():
        if sha(path) != digest:
            raise RuntimeError('Bound source changed: ' + path)
    return binding


def activate_local_snapshot():
    verify_sources()
    root = HERE.parent/'local_candidates_20261004_v2'
    if 'engine' in sys.modules and root/'native' not in Path(sys.modules['engine'].__file__).resolve().parents:
        raise RuntimeError('Use a fresh interpreter: a different engine is loaded')
    sys.path.insert(0, str(root/'native'))
    if sys.platform == 'win32':
        sys.path.insert(0, str(root/'local_deps'))


def register_components():
    # Call after activating the bound training engine, before YAMLConfig.model.
    from engine.core import register
    from model import P3ReadoutDEIM
    register()(P3ReadoutDEIM)


def load_baseline_state(model, state):
    """Only added readout keys may be missing; never silently ignore native keys."""
    import torch
    current = model.state_dict()
    extras = {key for key in current if key.startswith(EXTRA_PREFIX)}
    if set(current) - set(state) != extras or set(state) - set(current):
        raise RuntimeError('Checkpoint native key set differs')
    for key, value in state.items():
        if value.shape != current[key].shape or value.dtype != current[key].dtype:
            raise RuntimeError('Checkpoint tensor mismatch: ' + key)
    for suffix in ('output.weight', 'output.bias'):
        key = EXTRA_PREFIX + suffix
        if key in current and torch.count_nonzero(current[key]).item():
            raise RuntimeError('Baseline initialization requires fresh zero residual')
    result = model.load_state_dict(state, strict=not extras)
    if set(result.missing_keys) != extras or result.unexpected_keys:
        raise RuntimeError('Unexpected native load result')
    return len(extras)


def build_criterion(recipe):
    from copy import deepcopy
    from engine.deim.deim_criterion import DEIMCriterion
    from engine.deim.matcher import HungarianMatcher
    config = deepcopy(recipe)
    matcher = config.pop('matcher')
    matcher.pop('type', None)
    return DEIMCriterion(matcher=HungarianMatcher(**matcher, use_focal_loss=True),
                         num_classes=10, **config)
