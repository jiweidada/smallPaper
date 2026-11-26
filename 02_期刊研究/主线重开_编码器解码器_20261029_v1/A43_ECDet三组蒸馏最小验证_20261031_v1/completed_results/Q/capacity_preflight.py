"""One fixed dense batch, bounded implementation/capacity checks; never resumes."""
from __future__ import annotations
import gc
import hashlib
import json
import os
import time
import traceback
import torch
from runtime_common import (R, SEED, MODEL_COUNTER, VisDroneDataset, collate, create_components,
                            build_teacher, fp32_tree, make_optimizer, save_json, seed_all, targets_to_device)
from distillation import (StudentCapture, TeacherCapture, auxiliary_loss, module_state_digest,
                          native_inference, native_state, teacher_targets, update_teacher_flow)


def main():
    with (R / 'capacity_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    assert json.loads((R / 'cpu_interface_validation.json').read_text())['status'] == 'PASS'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32 = True;torch.backends.cudnn.allow_tf32 = True
    updates = 0;aux_gradient_probes = 0;t0 = time.monotonic();results = {}
    try:
        dataset = VisDroneDataset('train')
        indices = sorted(range(len(dataset)), key=lambda i: (-len(dataset.rows[i]['valid']), i))[:8]
        images, targets = collate([dataset[i] for i in indices])
        image_names = [dataset.rows[i]['file'] for i in indices]
        images = images.cuda();targets = targets_to_device(targets, 'cuda')
        # The auxiliary module is not in the native forward graph, even while attached.
        probe, _, _ = create_components('M');probe = probe.cuda().eval()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with native_inference(probe):
                reference = probe(images[:1])
            attached = probe(images[:1])
            with native_inference(probe):
                stripped = probe(images[:1])
        assert all(torch.equal(reference[k], attached[k]) and torch.equal(reference[k], stripped[k])
                   for k in ['pred_logits', 'pred_boxes'])
        export_reference = probe.cpu()
        del probe, reference, attached, stripped;gc.collect();torch.cuda.empty_cache()
        teacher, matcher = build_teacher();teacher = teacher.cuda();teacher_capture = TeacherCapture(teacher)
        teacher_before = module_state_digest(teacher)
        teacher_streams = {}
        for arm, steps in [('B', 1), ('M', 2), ('Q', 2)]:
            model, criterion, _ = create_components(arm);model = model.cuda();criterion = criterion.cuda()
            capture = StudentCapture(model) if arm != 'B' else None
            optimizer = make_optimizer(model);seed_all(SEED)
            torch.cuda.reset_peak_memory_stats();flow = hashlib.sha256();loss_rows = []
            for step in range(steps):
                bundle = teacher_targets(teacher, matcher, teacher_capture, images, targets) if arm != 'B' else None
                if bundle is not None:
                    update_teacher_flow(flow, bundle)
                model.train();optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model(images, targets)
                for key in ['pred_logits', 'pred_boxes', 'pred_corners']:
                    assert torch.isfinite(output[key]).all()
                with torch.autocast(device_type='cuda', enabled=False):
                    losses = criterion(fp32_tree(output), targets, epoch=0, step=step, global_step=step, epoch_step=809)
                    native = sum(losses.values())
                    auxiliary = auxiliary_loss(model, capture.feature, bundle, arm) if arm != 'B' else native.new_zeros(())
                    total = native + auxiliary
                assert torch.isfinite(total)
                if arm != 'B' and step == 0:
                    g = torch.autograd.grad(auxiliary, capture.feature, retain_graph=True)[0]
                    assert torch.isfinite(g).all() and float(g.float().abs().sum()) > 0
                    aux_gradient_probes += 1;del g
                total.backward()
                auxiliary_gradient = None
                if arm != 'B':
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.aux_projector.parameters())
                    auxiliary_gradient = float(model.aux_projector.weight.grad.norm())
                    assert auxiliary_gradient > 0
                    assert not any(p.grad is not None for p in teacher.parameters())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                optimizer.step();updates += 1;optimizer.zero_grad(set_to_none=True)
                loss_rows.append({'native_loss': float(native.detach()), 'distillation_loss': float(auxiliary.detach()),
                                  'grad_norm_before_clip': float(norm), 'auxiliary_weight_gradient_norm': auxiliary_gradient,
                                  'teacher_matched_instances': bundle['record']['matched_instances'] if bundle else 0,
                                  'teacher_GT_over_300_unmatched': bundle['record']['unmatched_GT_due_to_300_query_limit'] if bundle else 0})
                if capture is not None:
                    capture.feature = None
                del output, losses, native, auxiliary, total, bundle
            peak = torch.cuda.max_memory_allocated()
            assert peak < 22 * 2**30, 'Fixed dense batch exceeds22GiB; no fallback sweep'
            if arm != 'B':
                teacher_streams[arm] = flow.hexdigest()
                # A trained native export strictly loads without auxiliary keys.
                exported = {k: v.detach().cpu() for k, v in native_state(model.state_dict()).items()}
                with native_inference(export_reference):
                    export_reference.load_state_dict(exported, strict=True)
                    assert sum(p.numel() for p in export_reference.parameters()) == 9803112
                del exported
                capture.remove()
            results[arm] = {'optimizer_updates': steps, 'peak_allocated_bytes': peak,
                            'losses': loss_rows, 'strict_native_export_after_update': True}
            del model, criterion, optimizer, capture;gc.collect();torch.cuda.empty_cache()
        assert teacher_streams['M'] == teacher_streams['Q']
        assert module_state_digest(teacher) == teacher_before
        assert not any(p.grad is not None for p in teacher.parameters())
        assert MODEL_COUNTER['forwards'] == 8 and MODEL_COUNTER['teacher_forwards'] == 4 and updates == 5
        result = {'status': 'PASS', 'fixed_microbatch': 8, 'gradient_accumulation': 1,
                  'dense_images': image_names, 'dense_GT_counts': [len(t['boxes']) for t in targets],
                  'arms': results, 'zero_output_influence_bitwise_equal': True,
                  'auxiliary_gradient_reaches_stem': True, 'auxiliary_gradient_probe_calls': aux_gradient_probes,
                  'teacher_frozen_unchanged': True, 'teacher_state_digest': teacher_before,
                  'M_Q_actual_teacher_stream_identical': True,
                  'new_student_forwards': MODEL_COUNTER['forwards'], 'new_teacher_forwards': MODEL_COUNTER['teacher_forwards'],
                  'optimizer_updates': updates, 'all_capacity_weights_discarded_before_formal_training': True,
                  'elapsed_seconds': time.monotonic() - t0, 'no_batch_search_or_fallback': True}
        save_json(R / 'capacity_preflight_result.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        result = {'status': 'FAILED', 'optimizer_updates': updates,
                  'new_student_forwards': MODEL_COUNTER['forwards'], 'new_teacher_forwards': MODEL_COUNTER['teacher_forwards'],
                  'auxiliary_gradient_probe_calls': aux_gradient_probes,
                  'error': repr(exc), 'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - t0,
                  'formal_training_started': False, 'automatic_retry': False}
        save_json(R / 'capacity_preflight_result.json', result);print(json.dumps(result), flush=True)
        raise


if __name__ == '__main__':
    main()
