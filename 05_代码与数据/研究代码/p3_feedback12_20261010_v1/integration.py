"""Source-bound construction, strict loading, and native registry integration.

Importing this file does not construct a model, optimizer, or training loop.
"""
EXTRA_PREFIX = 'decoder.local_readout.'


def register_components():
    # Call after activating the bound training engine, before YAMLConfig.model.
    from engine.core import register
    from model import P3ReadoutDEIM
    from feedback import P3FeedbackCriterion
    register()(P3ReadoutDEIM)
    register()(P3FeedbackCriterion)


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
