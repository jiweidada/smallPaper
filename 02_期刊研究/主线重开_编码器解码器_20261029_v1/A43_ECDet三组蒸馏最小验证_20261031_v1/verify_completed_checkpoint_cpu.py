"""Terminal-only CPU checkpoint audit; zero detector forward/backward/update."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import torch
from runtime_common import R, MODEL_COUNTER, create_components, make_optimizer, save_json, digest
from distillation import module_state_digest, native_state


def main(arm):
    torch.set_num_threads(2)
    folder = R / ('arm_' + arm)
    result = json.loads((folder / 'result.json').read_text())
    assert result['status'] == 'COMPLETED' and result['epochs'] == 12 and result['optimizer_updates'] == 9708
    assert json.loads((folder / 'exit_receipt.json').read_text())['exit_code'] == 0
    plan = json.loads((R / 'execution_protocol_FROZEN.json').read_text())
    for name, expected in plan['files_sha256'].items():
        assert digest(R / name) == expected, name
    checkpoint = torch.load(folder / 'epoch12.pth', map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 12 and checkpoint['arm'] == arm and checkpoint['optimizer_updates'] == 9708
    assert checkpoint['ema']['updates'] == 9708 and checkpoint['inference_auxiliary_removed']
    assert checkpoint['protocol_sha256'] == digest(R / 'execution_protocol_FROZEN.json')
    assert checkpoint['teacher_checkpoint_sha256'] == plan['teacher_checkpoint_sha256']
    model, _, _ = create_components('B', role='cpu_check')
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    checked = {}
    for label, state in [('raw', checkpoint['model']), ('EMA', checkpoint['ema']['module'])]:
        assert not any(k.startswith('aux_projector.') for k in state)
        model.load_state_dict(state, strict=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert sum(p.numel() for p in model.parameters()) == 9803112
        anchors = state['decoder.anchors'];valid = state['decoder.valid_mask'].expand_as(anchors)
        assert torch.isfinite(anchors[valid]).all() and torch.isposinf(anchors[~valid]).all()
        bad = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
        assert set(bad) <= {'decoder.anchors'}
        checked[label] = {'strict_native_load': True, 'all_parameters_finite': True,
                          'expected_masked_anchor_infinity_only': True, 'inference_parameters': 9803112,
                          'auxiliary_keys_absent': True,
                          'changed_float_keys_from_initialization': sum(not torch.equal(v, initial[k]) for k, v in state.items() if v.is_floating_point())}
    training, _, _ = create_components(arm, role='cpu_check')
    info = json.loads((folder / 'initialization.json').read_text())
    auxiliary = {}
    if arm != 'B':
        assert module_state_digest(training.aux_projector) == info['auxiliary_initial_tensor_digest']
        initial_aux = {k: v.detach().clone() for k, v in training.aux_projector.state_dict().items()}
        for label, key in [('raw', 'model'), ('EMA', 'ema')]:
            state = checkpoint['training_auxiliary'][key]
            training.aux_projector.load_state_dict(state, strict=True)
            assert all(torch.isfinite(v).all() for v in state.values())
            assert all(not torch.equal(v, initial_aux[k]) for k, v in state.items())
            auxiliary[label] = {'finite': True, 'changed_from_common_initialization': True,
                                 'parameters': sum(v.numel() for v in state.values())}
        training.aux_projector.load_state_dict(checkpoint['training_auxiliary']['model'], strict=True)
        assert result['teacher_frozen_unchanged'] and result['teacher_state_start_digest'] == result['teacher_state_end_digest']
    else:
        assert checkpoint['training_auxiliary'] == {'model': {}, 'ema': {}}
        assert result['teacher_forwards'] == 0
    full = dict(checkpoint['model'])
    if arm != 'B':
        full.update({'aux_projector.' + k: v for k, v in checkpoint['training_auxiliary']['model'].items()})
    training.load_state_dict(full, strict=True)
    optimizer = make_optimizer(training);optimizer.load_state_dict(checkpoint['optimizer'])
    names = {id(p): n for n, p in training.named_parameters()}
    actual_names = [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
    assert actual_names == checkpoint['optimizer_parameter_names']
    steps = [int(s['step']) for s in optimizer.state.values() if 'step' in s]
    assert steps and min(steps) > 0 and max(steps) == 9708
    aux_steps = {}
    if arm != 'B':
        for name, p in training.aux_projector.named_parameters():
            step = int(optimizer.state[p]['step']);assert step == 9708;aux_steps[name] = step
    assert MODEL_COUNTER['forwards'] == MODEL_COUNTER['teacher_forwards'] == 0
    assert not torch.cuda.is_initialized()
    output = {'status': 'PASS', 'arm': arm, 'checkpoint_sha256': digest(folder / 'epoch12.pth'),
              'raw_and_EMA_native_inference': checked, 'training_auxiliary': auxiliary,
              'reconstructed_initial_auxiliary_matches_saved_digest': True if arm != 'B' else None,
              'optimizer_parameter_names_match': True, 'optimizer_steps_min': min(steps),
              'optimizer_steps_max': max(steps), 'auxiliary_optimizer_steps': aux_steps,
              'new_student_forwards': 0, 'new_teacher_forwards': 0, 'new_optimizer_updates': 0,
              'cuda_initialized': False, 'scope': 'CPU construction/strict state and saved optimizer checks; no detector inference or retraining.'}
    save_json(R / ('independent_checkpoint_' + arm + '.json'), output)
    print(json.dumps(output))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--arm', choices=['B', 'M', 'Q'], required=True)
    main(p.parse_args().arm)
