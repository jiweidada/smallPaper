"""Build the parameter-matched ordinary-convolution control for frozen FULL."""

import importlib
from pathlib import Path

import torch
from torch import nn


FULL_ARM = 'FULL_MSCA'
MATCHED_ARM = 'NO_MSCA_MATCHED'
FULL_INITIAL_SHA256 = '20624a8d3715ec0b31c14343e8a63f29ad90b695b4e6cd1296a7b1e2b971f598'
TRAINABLE_PARAMETERS = 7_540_942
REPLACEMENT_PARAMETERS = 3_841_280
INITIALIZATION_SEED = 0
LITE_GATE_PATHS = (
    'backbone.stages.2.blocks.0.aggregation.1',
    'backbone.stages.2.blocks.1.aggregation.1',
)
DILATED_GATE_PATH = 'backbone.stages.3.blocks.0.aggregation.1'
GATE_PATHS = (*LITE_GATE_PATHS, DILATED_GATE_PATH)
CONVOLUTIONS = ((1024, 320, 1), (320, 900, 3), (900, 1024, 1))
WEIGHT_KEYS = tuple(f'{DILATED_GATE_PATH}.{index}.weight' for index in (0, 2, 4))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def trainable_count(module):
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def apply(model):
    """Replace the three gates before the optimizer or EMA is constructed."""
    enhanced = importlib.import_module('engine.backbone.hgnetv2_MSCA_EnhancedStem')
    gate_types = (enhanced.MSCA, enhanced.MSCA_Lite, enhanced.MSCA_Dilated, enhanced.MSCA_Enhanced)
    actual_gate_paths = [
        f'backbone.stages.{stage_index}.blocks.{block_index}.aggregation.1'
        for stage_index, stage in enumerate(model.backbone.stages)
        for block_index, block in enumerate(stage.blocks)
        if isinstance(block.aggregation[1], gate_types)
    ]
    require(tuple(actual_gate_paths) == GATE_PATHS, 'Frozen FULL MSCA gate locations differ')
    expected_types = (enhanced.MSCA_Lite, enhanced.MSCA_Lite, enhanced.MSCA_Dilated)
    aggregations = [model.get_submodule(path.rsplit('.', 1)[0]) for path in GATE_PATHS]
    for aggregation, gate_type, channels in zip(aggregations, expected_types, (512, 512, 1024)):
        require(type(aggregation[1]) is gate_type and aggregation[0].conv.out_channels == channels,
                'Frozen FULL MSCA gate type or projection width differs')
    require(trainable_count(model) == TRAINABLE_PARAMETERS, 'Frozen FULL parameter count differs')
    require([trainable_count(aggregation[1]) for aggregation in aggregations] == [0, 0, REPLACEMENT_PARAMETERS],
            'Frozen FULL MSCA parameter count differs')

    # Seed only the CPU generator so constructing the replacement preserves the caller's RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(INITIALIZATION_SEED)
        layers = []
        for index, (input_channels, output_channels, kernel) in enumerate(CONVOLUTIONS):
            layers.append(nn.Conv2d(input_channels, output_channels, kernel, stride=1, padding=kernel // 2,
                                    dilation=1, groups=1, bias=False, device='cpu', dtype=torch.float32))
            if index < len(CONVOLUTIONS) - 1:
                layers.append(nn.SiLU())
        replacement = nn.Sequential(*layers)
    require(trainable_count(replacement) == REPLACEMENT_PARAMETERS, 'Ordinary convolution parameter count differs')

    # The existing projection, normalization, activation and all other blocks retain their identities.
    for aggregation, new_gate in zip(aggregations, (nn.Identity(), nn.Identity(), replacement)):
        aggregation[1] = new_gate.train(aggregation[1].training)
    require(trainable_count(model) == TRAINABLE_PARAMETERS, 'Matched detector parameter count differs')
    return model


def initialize(fair, model, archive_path):
    """Load and verify the frozen FULL initialization shared by both arms."""
    archive_path = Path(archive_path)
    archive_sha256 = fair.sha(archive_path)
    require(archive_sha256 == FULL_INITIAL_SHA256, 'Frozen FULL initial checkpoint SHA-256 differs')
    archived = torch.load(archive_path, map_location='cpu', weights_only=True)
    initial = model.state_dict()
    added = set(initial) - set(archived)
    removed = set(archived) - set(initial)
    if added or removed:
        expected_removed = {key for key in archived if key.startswith(DILATED_GATE_PATH + '.')}
        require(added == set(WEIGHT_KEYS) and removed == expected_removed and len(removed) == 48,
                'Matched state schema differs from the frozen FULL replacement')
        require(all(type(model.get_submodule(path)) is nn.Identity for path in LITE_GATE_PATHS),
                'Matched detector retains a Lite gate')
        arm = MATCHED_ARM
    else:
        arm = FULL_ARM

    retained = sorted(set(initial) & set(archived))
    require(len(retained) == (673 if arm == MATCHED_ARM else 721), 'Retained tensor count differs')
    new_weights = {key: initial[key].detach().clone() for key in WEIGHT_KEYS if key in added}
    for key in retained:
        source, target = archived[key], initial[key]
        require(target.shape == source.shape and target.dtype == source.dtype,
                'Retained initialization layout differs at ' + key)
        initial[key] = source
    model.load_state_dict(initial, strict=True)
    current = model.state_dict()
    for key in retained:
        require(torch.equal(current[key].detach().cpu(), archived[key]), 'Retained initialization differs at ' + key)
    for key, value in new_weights.items():
        require(torch.equal(current[key], value), 'New convolution initialization changed at ' + key)
    require(trainable_count(model) == TRAINABLE_PARAMETERS, 'Initialized detector parameter count differs')
    return {
        'arm': arm,
        'trainable_parameters': TRAINABLE_PARAMETERS,
        'reference_initial_sha256': archive_sha256,
        'retained_tensor_count': len(retained),
        'retained_tensor_sha256': fair.tensor_hash({key: current[key] for key in retained}),
        'all_retained_tensors_verified': True,
        'matched_only_tensor_keys': list(new_weights),
        'matched_only_tensor_sha256': fair.tensor_hash(new_weights) if new_weights else None,
        'identity_gate_paths': list(LITE_GATE_PATHS) if new_weights else [],
        'replaced_gate_paths': list(GATE_PATHS) if new_weights else [],
        'replacement_parameter_count': REPLACEMENT_PARAMETERS if new_weights else 0,
        'replacement_init_seed': INITIALIZATION_SEED if new_weights else None,
        'replacement_initializer': 'torch.nn.Conv2d.reset_parameters, CPU float32' if new_weights else None,
    }


def verify_gradients(model):
    """Verify that every inserted convolution participates in the preflight loss."""
    parameters = dict(model.named_parameters())
    gradient_norms = {}
    for key, (input_channels, output_channels, kernel) in zip(WEIGHT_KEYS, CONVOLUTIONS):
        parameter = parameters.get(key)
        require(parameter is not None and parameter.requires_grad
                and tuple(parameter.shape) == (output_channels, input_channels, kernel, kernel),
                'Missing or incompatible matched convolution at ' + key)
        gradient = parameter.grad
        require(gradient is not None and bool(torch.isfinite(gradient).all()) and bool(torch.count_nonzero(gradient)),
                'Matched convolution gradient is missing, nonfinite or zero at ' + key)
        gradient_norms[key] = float(gradient.detach().norm())
    return gradient_norms
