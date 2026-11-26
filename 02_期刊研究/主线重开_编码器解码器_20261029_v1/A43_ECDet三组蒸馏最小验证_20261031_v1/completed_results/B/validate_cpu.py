"""CPU-only import, strict native checkpoint and common projection checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import gc
import json
import torch
from runtime_common import R, MODEL_COUNTER, create_components, build_teacher, save_json
from distillation import module_state_digest, native_state

torch.set_num_threads(4)
initial = torch.load(R / 'common_init_10class.pth', map_location='cpu', weights_only=True)['model']
aux_digests = [];counts = {}
for arm in ['B', 'M', 'Q']:
    model, criterion, post = create_components(arm, role='cpu_check')
    assert all(torch.equal(t, initial[k]) for k, t in native_state(model.state_dict()).items())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    counts[arm] = sum(p.numel() for p in model.parameters())
    if arm != 'B':
        aux_digests.append(module_state_digest(model.aux_projector))
    assert len(model.decoder.dec_score_head) == 4 and model.decoder.eval_idx == 3
    assert model.decoder.dec_score_head[3].in_features == 192
    assert not hasattr(model.decoder.decoder, 'detail_reader')
    del model, criterion, post;gc.collect()
assert aux_digests[0] == aux_digests[1]
assert counts == {'B': 9803112, 'M': 9812520, 'Q': 9812520}
teacher, matcher = build_teacher()
assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
assert all(torch.isfinite(p).all() for p in teacher.parameters())
state_digest = module_state_digest(teacher)
result = {'status': 'PASS', 'model_parameters': counts, 'M_Q_projector_initial_digest': aux_digests[0],
          'native_decoder_unmodified': True, 'native_checkpoint_strict_load': True,
          'teacher_is_frozen_A42_B_epoch12_EMA': True, 'teacher_state_digest': state_digest,
          'source_cpu_import': True, 'new_student_forwards': MODEL_COUNTER['forwards'],
          'new_teacher_forwards': MODEL_COUNTER['teacher_forwards'], 'optimizer_updates': 0,
          'cuda_initialized': torch.cuda.is_initialized()}
assert result['new_student_forwards'] == result['new_teacher_forwards'] == 0
assert not result['cuda_initialized']
save_json(R / 'cpu_interface_validation.json', result)
print(json.dumps(result))
