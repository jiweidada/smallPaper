from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from runtime_common import (R, SEED, MODEL_COUNTER, VisDroneDataset, collate, create_components,
                            build_teacher, digest, fp32_tree, make_optimizer, save_json,
                            seed_all, targets_to_device)
from distillation import (StudentCapture, TeacherCapture, auxiliary_loss, auxiliary_state,
                          module_state_digest, native_inference, native_state, teacher_targets,
                          tensor_digest, update_teacher_flow)
from evaluate import evaluate_model, latency_ms


class StateEMA:
    def __init__(self, model):
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()};self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1;decay = .9999 * (1 - math.exp(-self.updates / 1000))
        current = model.state_dict();dest = [];src = []
        for k, v in self.state.items():
            if v.is_floating_point() and not k.endswith('.anchors'):
                dest.append(v);src.append(current[k].detach())
            else:
                v.copy_(current[k])
        torch._foreach_mul_(dest, decay);torch._foreach_add_(dest, src, alpha=1 - decay)


def cpu_state(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def input_digest(images, targets):
    h = hashlib.sha256(images.contiguous().numpy().tobytes())
    for t in targets:
        for k in sorted(t):
            h.update(k.encode());h.update(t[k].contiguous().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(model, optimizer, ema, arm, epoch, updates, plan):
    raw = cpu_state(model.state_dict());ema_cpu = cpu_state(ema.state)
    checkpoint = {'model': native_state(raw), 'ema': {'module': native_state(ema_cpu), 'updates': ema.updates},
                  'epoch': epoch, 'arm': arm, 'optimizer_updates': updates,
                  'protocol_sha256': digest(R / 'execution_protocol_FROZEN.json'),
                  'teacher_checkpoint_sha256': plan['teacher_checkpoint_sha256'],
                  'inference_auxiliary_removed': True,
                  'training_auxiliary': {'model': auxiliary_state(raw), 'ema': auxiliary_state(ema_cpu)}}
    if epoch == 12:
        checkpoint['optimizer'] = optimizer.state_dict()
        names = {id(p): n for n, p in model.named_parameters()}
        checkpoint['optimizer_parameter_names'] = [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
    path = R / ('arm_' + arm) / ('epoch%02d.pth' % epoch)
    torch.save(checkpoint, path)
    return path, raw


def train(arm):
    plan = json.loads((R / 'execution_protocol_FROZEN.json').read_text())
    assert plan['epochs_per_group'] == 12 and plan['arms'] == ['B', 'M', 'Q']
    assert plan['microbatch'] == 8 and plan['gradient_accumulation'] == 1
    for file, sha in plan['files_sha256'].items():
        assert digest(R / file) == sha, file
    assert json.loads((R / 'capacity_preflight_result.json').read_text())['status'] == 'PASS'
    out = R / ('arm_' + arm);out.mkdir(exist_ok=True)
    with (out / 'run_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32 = True;torch.backends.cudnn.allow_tf32 = True
    trainset = VisDroneDataset('train');valset = VisDroneDataset('val')
    assert len(trainset) == 6471 and len(valset) == 548
    for split, dataset in [('train', trainset), ('val', valset)]:
        manifest = json.loads((R / (split + '_data_manifest.json')).read_text())
        for row in manifest['rows']:
            assert digest(dataset.images / row['file']) == row['image_sha256']
            assert digest(dataset.anns / (row['file'][:-4] + '.txt')) == row['annotation_sha256']
    model, criterion, post = create_components(arm);model = model.cuda();criterion = criterion.cuda()
    capture = StudentCapture(model) if arm != 'B' else None
    teacher = matcher = teacher_capture = None;teacher_start_digest = None
    if arm != 'B':
        teacher, matcher = build_teacher();teacher = teacher.cuda()
        teacher_start_digest = module_state_digest(teacher)
        teacher_capture = TeacherCapture(teacher)
    optimizer = make_optimizer(model);ema = StateEMA(model)
    initial = torch.load(R / 'common_init_10class.pth', map_location='cpu', weights_only=True)['model']
    assert all(torch.equal(v.cpu(), initial[k]) for k, v in native_state(model.state_dict()).items())
    init = {'common_init_sha256': digest(R / 'common_init_10class.pth'), 'all_base_tensors_equal': True,
            'auxiliary_initial_tensor_digest': module_state_digest(model.aux_projector) if arm != 'B' else None,
            'training_model_parameters': sum(p.numel() for p in model.parameters()),
            'native_inference_parameters': 9803112, 'training_seed': SEED,
            'teacher_checkpoint_sha256': plan['teacher_checkpoint_sha256'],
            'teacher_state_start_digest': teacher_start_digest,
            'protocol_sha256': digest(R / 'execution_protocol_FROZEN.json')}
    save_json(out / 'initialization.json', init);del initial
    seed_all(SEED);torch.cuda.reset_peak_memory_stats()
    updates = seen = completed_epochs = 0;t0 = time.monotonic();events = []
    micro_per_epoch = math.ceil(len(trainset) / 8);max_updates = 12 * micro_per_epoch
    assert micro_per_epoch == 809 and max_updates == 9708
    save_json(out / 'start.json', {'arm': arm, 'pid': os.getpid(), 'started_unix': time.time(),
                                  'seed': SEED, 'max_epochs': 12, 'max_updates': 9708,
                                  'teacher_forward_before_student': True, 'device': torch.cuda.get_device_name()})
    try:
        for epoch in range(12):
            trainset.epoch = epoch
            order = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(SEED + epoch * 1009)).tolist()
            loader = DataLoader(trainset, batch_size=8, sampler=order, num_workers=2, collate_fn=collate,
                                pin_memory=True, drop_last=False,
                                generator=torch.Generator().manual_seed(SEED + 100000 + epoch))
            order_sha = hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest()
            actual_inputs = [];teacher_first_eight = [];teacher_flow = hashlib.sha256()
            native_sum = aux_sum = 0.;matched = unmatched = 0;ts = time.monotonic()
            model.train();criterion.train();optimizer.zero_grad(set_to_none=True)
            for micro, (images, targets) in enumerate(loader):
                if micro < 8:
                    actual_inputs.append(input_digest(images, targets))
                images = images.cuda(non_blocking=True);targets = targets_to_device(targets, 'cuda')
                bundle = None
                if arm != 'B':
                    bundle = teacher_targets(teacher, matcher, teacher_capture, images, targets)
                    batch_hash = update_teacher_flow(teacher_flow, bundle)
                    if micro < 8:
                        teacher_first_eight.append(batch_hash)
                    matched += bundle['record']['matched_instances']
                    unmatched += bundle['record']['unmatched_GT_due_to_300_query_limit']
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images, targets)
                for key in ['pred_logits', 'pred_boxes', 'pred_corners']:
                    if not torch.isfinite(outputs[key]).all():
                        raise FloatingPointError('Nonfinite student ' + key)
                with torch.autocast(device_type='cuda', enabled=False):
                    losses = criterion(fp32_tree(outputs), targets, epoch=epoch, step=micro,
                                       global_step=seen, epoch_step=micro_per_epoch)
                    native_loss = sum(losses.values())
                    aux = auxiliary_loss(model, capture.feature, bundle, arm) if arm != 'B' else native_loss.new_zeros(())
                    loss = native_loss + aux
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite summed loss')
                loss.backward()
                ratio = (updates + 1) / 500 if updates < 500 else .1 + .9 * .5 * (1 + math.cos(math.pi * (updates - 500) / (max_updates - 500)))
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * ratio
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates += 1;seen += 1;ema.update(model)
                nv, av = float(native_loss.detach()), float(aux.detach());native_sum += nv;aux_sum += av
                if capture is not None:
                    capture.feature = None
                if micro % 50 == 0 or micro + 1 == len(loader):
                    progress = {'status': 'TRAINING', 'arm': arm, 'epoch': epoch + 1,
                                'microbatch': micro + 1, 'microbatches_per_epoch': len(loader),
                                'completed_epochs': completed_epochs, 'optimizer_updates': updates,
                                'full_model_forwards': MODEL_COUNTER['forwards'],
                                'teacher_forwards': MODEL_COUNTER['teacher_forwards'],
                                'loss': nv + av, 'native_loss': nv, 'distillation_loss': av,
                                'grad_norm_before_clip': float(grad_norm),
                                'elapsed_seconds': time.monotonic() - t0,
                                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'pid': os.getpid()}
                    save_json(out / 'progress.json', progress);print(json.dumps(progress), flush=True)
                del outputs, losses, native_loss, aux, loss, images, targets, bundle
            completed_epochs = epoch + 1
            event = {'epoch': completed_epochs, 'optimizer_updates': updates, 'order_sha256': order_sha,
                     'first_eight_actual_input_sha256': actual_inputs, 'mean_native_loss': native_sum / 809,
                     'mean_distillation_loss': aux_sum / 809, 'train_seconds': time.monotonic() - ts,
                     'teacher_forward_count_cumulative': MODEL_COUNTER['teacher_forwards'],
                     'teacher_matched_instances': matched, 'teacher_unmatched_GT_due_to_query_limit': unmatched,
                     'teacher_all_batches_flow_sha256': teacher_flow.hexdigest() if arm != 'B' else None,
                     'first_eight_teacher_batch_sha256': teacher_first_eight}
            if arm != 'B':
                baseline = json.loads((R / 'arm_B' / ('epoch%02d_input_flow.json' % completed_epochs)).read_text())
                assert baseline['order_sha256'] == order_sha and baseline['first_eight_actual_input_sha256'] == actual_inputs
            if arm == 'Q':
                control = json.loads((R / 'arm_M' / ('epoch%02d_input_flow.json' % completed_epochs)).read_text())
                assert control['teacher_all_batches_flow_sha256'] == event['teacher_all_batches_flow_sha256'], 'Teacher target/match stream differs between M/Q'
                assert control['teacher_matched_instances'] == matched
            save_json(out / ('epoch%02d_input_flow.json' % completed_epochs), event)
            if completed_epochs in [4, 8, 12]:
                path, raw = save_checkpoint(model, optimizer, ema, arm, completed_epochs, updates, plan)
                model.load_state_dict(ema.state, strict=True)
                save_json(out / 'progress.json', {'status': 'VALIDATING', 'arm': arm, 'epoch': completed_epochs,
                                                 'optimizer_updates': updates, 'full_model_forwards': MODEL_COUNTER['forwards'],
                                                 'teacher_forwards': MODEL_COUNTER['teacher_forwards'],
                                                 'elapsed_seconds': time.monotonic() - t0, 'pid': os.getpid()})
                with native_inference(model, capture):
                    assert sum(p.numel() for p in model.parameters()) == 9803112
                    event['EMA_validation'] = evaluate_model(model, post, valset, out, 'epoch%02d_EMA' % completed_epochs)
                model.load_state_dict(raw, strict=True)
                event['checkpoint_sha256'] = digest(path)
                del raw;gc.collect()
            events.append(event)
            with (out / 'epochs.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            save_json(out / 'completed_epochs.json', {'arm': arm, 'completed_epochs': completed_epochs,
                                                       'optimizer_updates': updates, 'epochs': events})
        teacher_end_digest = None
        if teacher is not None:
            assert not any(p.grad is not None for p in teacher.parameters())
            teacher_end_digest = module_state_digest(teacher)
            assert teacher_end_digest == teacher_start_digest
            teacher_capture.close();teacher_capture = None;teacher = None;matcher = None
            gc.collect();torch.cuda.empty_cache()
        model.load_state_dict(ema.state, strict=True)
        single, _ = collate([valset[0]])
        with native_inference(model, capture):
            latency = latency_ms(model, single.cuda())
        result = {'status': 'COMPLETED', 'arm': arm, 'epochs': completed_epochs,
                  'optimizer_updates': updates, 'full_model_forwards': MODEL_COUNTER['forwards'],
                  'teacher_forwards': MODEL_COUNTER['teacher_forwards'], 'elapsed_seconds': time.monotonic() - t0,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'endpoint_metrics': events[-1]['EMA_validation'], 'endpoint_checkpoint': str(out / 'epoch12.pth'),
                  'latency': latency, 'teacher_state_start_digest': teacher_start_digest,
                  'teacher_state_end_digest': teacher_end_digest, 'teacher_frozen_unchanged': teacher_end_digest == teacher_start_digest,
                  'inference_auxiliary_removed': True, 'pid': os.getpid()}
        assert updates == 9708 and MODEL_COUNTER['forwards'] == 10159
        assert MODEL_COUNTER['teacher_forwards'] == (0 if arm == 'B' else 9708)
        save_json(out / 'result.json', result);save_json(out / 'progress.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        result = {'status': 'FAILED', 'arm': arm, 'completed_epochs': completed_epochs,
                  'optimizer_updates': updates, 'full_model_forwards': MODEL_COUNTER['forwards'],
                  'teacher_forwards': MODEL_COUNTER['teacher_forwards'], 'elapsed_seconds': time.monotonic() - t0,
                  'error': repr(exc), 'traceback': traceback.format_exc(), 'pid': os.getpid()}
        save_json(out / 'result.json', result);save_json(out / 'progress.json', result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--arm', choices=['B', 'M', 'Q'], required=True)
    train(parser.parse_args().arm)
