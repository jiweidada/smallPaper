"""Apply one declared structural removal and bind its initial tensors."""

import hashlib
import importlib
from pathlib import Path

import torch


EXPECTED_PARAMETERS = {'NO_MPS': 7535617, 'NO_MSCA': 3699662}
INITIAL_SHA = {
    'FULL': '20624a8d3715ec0b31c14343e8a63f29ad90b695b4e6cd1296a7b1e2b971f598',
    'BASE': '4fbd01639df2f19e9178bc536805649c5edb0c5221968fe73cc7f8351d642517',
}


def apply(model, arm, reference):
    if arm not in EXPECTED_PARAMETERS:
        raise ValueError('Unknown ablation arm')
    reference = Path(reference)
    states = {}
    for name, expected in INITIAL_SHA.items():
        path = reference / f'{name}.initial.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Reference initialization changed')
        states[name] = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(states['FULL'], strict=True)
    enhanced = importlib.import_module('engine.backbone.hgnetv2_MSCA_EnhancedStem')
    removed = []
    stem_keys = []
    if arm == 'NO_MPS':
        original = importlib.import_module('engine.backbone.hgnetv2')
        model.backbone.stem = original.StemBlock(3, 16, 16, use_lab=True)
        for layer in model.backbone.stem.modules():
            if isinstance(layer, original.ConvBNAct) and isinstance(layer.act, torch.nn.ReLU):
                layer.act = torch.nn.SiLU()
        model.backbone.stem_type = 'baseline_stem_silu'
        mapped = model.state_dict()
        for key, target in mapped.items():
            if key.startswith('backbone.stem.'):
                source = states['BASE'][key]
                stem_keys.append(key)
            else:
                source = states['FULL'][key]
            if target.shape != source.shape:
                raise RuntimeError('Initialization shape mismatch at ' + key)
            mapped[key] = source.clone()
        model.load_state_dict(mapped, strict=True)
    else:
        for stage_index, stage in enumerate(model.backbone.stages):
            for block_index, block in enumerate(stage.blocks):
                gate = block.aggregation[1]
                if isinstance(gate, (enhanced.MSCA_Lite, enhanced.MSCA_Dilated)):
                    removed.append(f'backbone.stages.{stage_index}.blocks.{block_index}.aggregation.1')
                    block.aggregation[1] = torch.nn.Identity()
        if len(removed) != 3:
            raise RuntimeError('Expected two MSCA-Lite gates and one MSCA-Dilated gate')
    current = model.state_dict()
    for key, value in current.items():
        source = states['BASE'][key] if key in stem_keys else states['FULL'][key]
        if not torch.equal(value, source):
            raise RuntimeError('Shared initialization differs at ' + key)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError('Unexpected ablation parameter count')
    return {'arm': arm, 'trainable_parameters': parameter_count,
            'reference_initial_sha256': INITIAL_SHA, 'stem_keys_from_base': stem_keys,
            'identity_gate_paths': removed, 'retained_tensor_count': len(current),
            'activation': 'silu', 'all_retained_tensors_verified': True}
