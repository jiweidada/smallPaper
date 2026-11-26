import gc
import hashlib
import json
import os
import time
import traceback
import torch
from runtime_A56 import ROOT, ASSETS, native, components, validate_files
from training_utils_A49 import assert_finite_output
from contrastive_read import diagnostics

UPDATES = 0


def state_hash(branches):
    h = hashlib.sha256()
    for i, module in enumerate(branches):
        for k, v in module.state_dict().items():
            h.update(f'{i}/{k}'.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    global UPDATES
    validate_files('capacity_protocol_FROZEN_v1.json')
    with (ROOT / 'capacity_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dataset = native.VisDroneDataset('train')
    dataset.epoch = 0
    assert len(dataset) == 6471 and sum(len(r['valid']) for r in dataset.rows) == 343204
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(native.SEED)).tolist()
    batches = [order[i:i+8] for i in range(0, len(order), 8)]
    sizes = [len(r['valid']) for r in dataset.rows]
    chosen = [0]
    for key in [lambda j: max(sizes[i] for i in batches[j]),
                lambda j: sum(sizes[i] for i in batches[j])]:
        chosen.append(next(j for j in sorted(range(len(batches)), key=key, reverse=True)
                           if j not in chosen))
    selected = [{'batch_index': j, 'indices': batches[j],
                 'maximum_GT': max(sizes[i] for i in batches[j]),
                 'sum_GT': sum(sizes[i] for i in batches[j])} for j in chosen]
    native.save_json(ROOT / 'capacity_batch_selection.json', selected)
    eval_images = native.collate([dataset[i] for i in batches[0]])[0][:4].cuda()
    reference = None
    results = []
    new_init = None
    begin = time.monotonic()
    for arm in ['B', 'M', 'Q']:
        model, criterion, post, branches = components(arm)
        expected_params = 9803112 if arm == 'B' else 9877256
        assert sum(p.numel() for p in model.parameters()) == expected_params
        initial = torch.load(ASSETS / 'common_init_10class.pth', map_location='cpu', weights_only=True)['model']
        actual = model.state_dict()
        assert all(k in actual and torch.equal(v, actual[k]) for k, v in initial.items())
        del initial, actual
        initial_sha = state_hash(branches) if branches else None
        if initial_sha is not None:
            if new_init is None:
                new_init = initial_sha
            assert initial_sha == new_init
        model = model.cuda()
        criterion = criterion.cuda()
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            outputs = model(eval_images)
        assert all(torch.isfinite(outputs[k]).all() for k in ['pred_logits', 'pred_boxes'])
        error = {}
        if reference is None:
            reference = {k: outputs[k].clone() for k in ['pred_logits', 'pred_boxes']}
        else:
            error = {k: float((outputs[k] - reference[k]).abs().max()) for k in reference}
            assert all(torch.equal(outputs[k], reference[k]) for k in reference), error
        native.save_json(ROOT / 'capacity_progress.json',
                         {'status': 'RUNNING', 'stage': arm + '_INITIAL_EVAL',
                          'whole_model_forwards': native.MODEL_COUNTER['forwards'],
                          'capacity_only_optimizer_updates': UPDATES, 'initial_error': error,
                          'pid': os.getpid()})
        del outputs
        if arm == 'B':
            del model, criterion, post
            gc.collect()
            torch.cuda.empty_cache()
            continue
        optimizer = native.make_optimizer(model)
        model.train()
        criterion.train()
        native.seed_all()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for step, batch_index in enumerate(chosen):
            images, targets = native.collate([dataset[i] for i in batches[batch_index]])
            images = images.cuda()
            targets = native.targets_to_device(targets, 'cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images, targets)
            assert_finite_output(outputs)
            with torch.autocast(device_type='cuda', enabled=False):
                losses = criterion(native.fp32_tree(outputs), targets, epoch=0, step=step,
                                   global_step=step, epoch_step=809)
                loss = sum(losses.values())
            assert torch.isfinite(loss)
            loss.backward()
            grads = []
            for branch in branches:
                gg = branch.gain.grad
                bg = branch.negative_logits.weight.grad
                assert gg is not None and bg is not None
                assert torch.isfinite(gg).all() and torch.isfinite(bg).all()
                if step > 0:
                    assert gg.norm() > 0 and bg.norm() > 0
                grads.append({'gain': float(gg.float().norm()), 'negative_weight': float(bg.float().norm())})
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
            optimizer.step()
            UPDATES += 1
            row = {'step': step + 1, 'batch_index': batch_index,
                   'total_loss': float(loss.detach()), 'gradient_norm': float(gradient),
                   'new_parameter_gradients_before_clipping': grads,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'module': diagnostics(branches)}
            rows.append(row)
            native.save_json(ROOT / 'capacity_progress.json',
                             {'status': 'RUNNING', 'stage': arm + '_TEMPORARY_UPDATE',
                              'whole_model_forwards': native.MODEL_COUNTER['forwards'],
                              'capacity_only_optimizer_updates': UPDATES, 'pid': os.getpid(), **row})
            del images, targets, outputs, losses, loss
        peak = torch.cuda.max_memory_allocated()
        assert peak < 24 * (1 << 30)
        assert all(b.gain.detach().abs().max() > 0 for b in branches)
        results.append({'arm': arm, 'initial_FP32_output_bitwise_equal': True,
                        'initial_FP32_error': error, 'new_initial_tensors_sha256': initial_sha,
                        'model_parameters': expected_params, 'extra_parameters': 74144,
                        'peak_allocated_bytes': peak, 'temporary_updates': 3, 'steps': rows})
        del model, criterion, post, branches, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    result = {'status': 'PASS', 'scope': 'integration/capacity only; all updated states discarded',
              'protocol_sha256': native.digest(ROOT / 'capacity_protocol_FROZEN_v1.json'),
              'whole_model_forwards': native.MODEL_COUNTER['forwards'],
              'capacity_only_optimizer_updates': UPDATES,
              'new_module_initial_tensors_match': True, 'all_native_initial_tensors_match': True,
              'selected_batches': selected, 'results': results, 'seconds': time.monotonic()-begin,
              'formal_training_epochs': 0, 'task_effect_or_novelty_validated': False, 'pid': os.getpid()}
    assert result['whole_model_forwards'] == 9 and UPDATES == 6
    native.save_json(ROOT / 'capacity_result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        native.save_json(ROOT / 'capacity_result.json',
                         {'status': 'FAILED_STOPPED_NO_RETRY', 'pid': os.getpid(),
                          'whole_model_forwards': native.MODEL_COUNTER['forwards'],
                          'capacity_only_optimizer_updates': UPDATES,
                          'traceback': traceback.format_exc()})
        raise
