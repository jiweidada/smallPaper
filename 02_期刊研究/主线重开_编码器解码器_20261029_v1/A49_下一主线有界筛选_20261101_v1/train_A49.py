import argparse
import gc
import hashlib
import json
import os
import time
import traceback
import torch
from torch.utils.data import DataLoader
from runtime_A49 import ROOT, ASSETS, native, plan_check
from ignore_query_supervision import IgnorePolicy, native_inference
from training_utils_A49 import StateEMA, cpu_state, input_digest, rng_digest, assert_finite_output, lr_ratio
from evaluate import evaluate_model, latency_ms


def train(arm):
    plan = plan_check('formal')
    capacity = json.loads((ROOT / 'capacity_result.json').read_text())
    assert capacity['status'] == 'PASS'
    assert capacity['protocol_sha256'] == native.digest(ROOT / 'capacity_protocol_FROZEN_v2.json') == plan['capacity_protocol_sha256']
    out = ROOT / ('arm_' + arm)
    out.mkdir(exist_ok=True)
    with (out / 'run_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    trainset = native.VisDroneDataset('train')
    valset = native.VisDroneDataset('val')
    assert len(trainset) == 6471 and len(valset) == 548
    assert sum(len(r['valid']) for r in trainset.rows) == 343204
    assert sum(len(r['valid']) for r in valset.rows) == 38759
    model, criterion, post = native.create_components('B')
    policy = IgnorePolicy(arm)
    sites = policy.install_model(model)
    policy.install_loss(criterion)
    initial = torch.load(ASSETS / 'common_init_10class.pth', map_location='cpu', weights_only=True)['model']
    assert tuple(model.state_dict()) == tuple(initial)
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters()) == 9803112
    assert all(m.__class__ is torch.nn.BatchNorm2d for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    native.save_json(out / 'initialization.json', {'common_init_sha256': native.digest(ASSETS / 'common_init_10class.pth'),
        'all_initial_tensors_identical': True, 'model_parameters': 9803112, 'seed': native.SEED,
        'protocol_sha256': native.digest(ROOT / 'protocol_FROZEN.json')})
    del initial
    model = model.cuda(); criterion = criterion.cuda()
    optimizer = native.make_optimizer(model)
    ema = StateEMA(model)
    native.seed_all(); torch.cuda.reset_peak_memory_stats()
    updates = completed = 0
    t0 = time.monotonic()
    events = []
    try:
        for epoch in range(12):
            trainset.epoch = epoch
            order = torch.randperm(6471, generator=torch.Generator().manual_seed(native.SEED + epoch * 1009)).tolist()
            loader = DataLoader(trainset, batch_size=8, sampler=order, num_workers=2, collate_fn=native.collate,
                pin_memory=True, drop_last=False, generator=torch.Generator().manual_seed(native.SEED + 100000 + epoch))
            assert len(loader) == 809
            input_hashes, rng_hashes = [], []
            target_flow = hashlib.sha256()
            loss_sum = 0.
            epoch_omissions = {'box_union_IoF': 0, 'read_mass': 0}
            ts = time.monotonic()
            model.train(); criterion.train(); optimizer.zero_grad(set_to_none=True)
            for micro, (images, targets) in enumerate(loader):
                if micro < 8:
                    input_hashes.append(input_digest(images, targets))
                for target in targets:
                    for key in ['boxes', 'labels', 'image_id', 'ignore_tiles']:
                        target_flow.update(key.encode())
                        target_flow.update(str(tuple(target[key].shape)).encode())
                        target_flow.update(target[key].contiguous().numpy().tobytes())
                images = images.cuda(non_blocking=True)
                targets = native.targets_to_device(targets, 'cuda')
                if micro < 8:
                    rng_hashes.append(rng_digest())
                with policy.training_step(targets):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(images, targets)
                    assert_finite_output(outputs)
                    policy.annotate_outputs(outputs)
                    with torch.autocast(device_type='cuda', enabled=False):
                        losses = criterion(native.fp32_tree(outputs), targets, epoch=epoch,
                            step=micro, global_step=updates, epoch_step=809)
                        loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite criterion loss')
                loss.backward()
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * lr_ratio(updates)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                updates += 1
                ema.update(model)
                for row in policy.loss_statistics:
                    epoch_omissions[row['rule']] += row['omitted_unmatched_rows']
                value = float(loss.detach())
                loss_sum += value
                if micro % 50 == 0 or micro == 808:
                    progress = {'status': 'TRAINING', 'arm': arm, 'epoch': epoch + 1, 'microbatch': micro + 1,
                        'microbatches_per_epoch': 809, 'completed_epochs': completed, 'optimizer_updates': updates,
                        'full_model_forwards': native.MODEL_COUNTER['forwards'], 'loss': value, 'grad_norm': float(grad),
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.monotonic() - t0,
                        'extra_mask_core_reads': policy.mask_core_calls, 'epoch_omitted_query_loss_rows': dict(epoch_omissions),
                        'pid': os.getpid()}
                    native.save_json(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                del outputs, losses, loss, images, targets
            completed = epoch + 1
            event = {'epoch': completed, 'optimizer_updates': updates, 'mean_loss': loss_sum / 809,
                'train_seconds': time.monotonic() - ts,
                'order_sha256': hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest(),
                'first_eight_actual_input_sha256': input_hashes, 'first_eight_pre_forward_rng_sha256': rng_hashes,
                'all_batches_targets_including_ignore_tiles_sha256': target_flow.hexdigest(),
                'extra_mask_core_reads': policy.mask_core_calls, 'epoch_omitted_query_loss_rows': dict(epoch_omissions)}
            if arm != 'B':
                prior = json.loads((ROOT / 'arm_B' / ('epoch%02d_input_flow.json' % completed)).read_text())
                for key in ['order_sha256', 'first_eight_actual_input_sha256', 'first_eight_pre_forward_rng_sha256',
                            'all_batches_targets_including_ignore_tiles_sha256']:
                    assert prior[key] == event[key], key
            native.save_json(out / ('epoch%02d_input_flow.json' % completed), event)
            if completed in [4, 8, 12]:
                raw = cpu_state(model.state_dict())
                checkpoint = {'model': raw, 'ema': {'module': cpu_state(ema.state), 'updates': ema.updates},
                    'arm': arm, 'epoch': completed, 'optimizer_updates': updates,
                    'protocol_sha256': native.digest(ROOT / 'protocol_FROZEN.json'),
                    'inference': 'Literal native sampling cores; no ignore masks or extra parameters.'}
                if completed == 12:
                    checkpoint['optimizer'] = optimizer.state_dict()
                path = out / ('epoch%02d.pth' % completed)
                torch.save(checkpoint, path); del checkpoint
                model.load_state_dict(ema.state, strict=True)
                native.save_json(out / 'progress.json', {'status': 'VALIDATING', 'arm': arm, 'epoch': completed,
                    'optimizer_updates': updates, 'full_model_forwards': native.MODEL_COUNTER['forwards'], 'pid': os.getpid()})
                before = policy.mask_core_calls
                with native_inference(model):
                    event['EMA_validation'] = evaluate_model(model, post, valset, out, 'epoch%02d_EMA' % completed)
                assert policy.mask_core_calls == before and policy.active_tiles is None
                model.load_state_dict(raw, strict=True)
                event['checkpoint_sha256'] = native.digest(path)
                del raw; gc.collect()
            events.append(event)
            with (out / 'epochs.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            native.save_json(out / 'completed_epochs.json', {'completed_epochs': completed,
                'optimizer_updates': updates, 'events': events})
        model.load_state_dict(ema.state, strict=True)
        single, _ = native.collate([valset[0]])
        before = policy.mask_core_calls
        with native_inference(model):
            latency = latency_ms(model, single.cuda())
        assert policy.mask_core_calls == before and policy.active_tiles is None
        if arm == 'Q':
            assert policy.mask_core_calls == updates * len(sites)
        else:
            assert policy.mask_core_calls == 0
        result = {'status': 'COMPLETED', 'arm': arm, 'epochs': completed, 'optimizer_updates': updates,
            'full_model_forwards': native.MODEL_COUNTER['forwards'], 'elapsed_seconds': time.monotonic() - t0,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'endpoint_metrics': events[-1]['EMA_validation'],
            'extra_mask_core_reads': policy.mask_core_calls, 'native_inference_no_mask_reads': True,
            'latency': latency, 'pid': os.getpid(), 'initialization_sha256': native.digest(ASSETS / 'common_init_10class.pth'),
            'protocol_sha256': native.digest(ROOT / 'protocol_FROZEN.json')}
        assert updates == 9708 and native.MODEL_COUNTER['forwards'] == 10159
        native.save_json(out / 'result.json', result)
        native.save_json(out / 'progress.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        result = {'status': 'FAILED_STOPPED_NO_RETRY', 'arm': arm, 'completed_epochs': completed,
            'optimizer_updates': updates, 'full_model_forwards': native.MODEL_COUNTER['forwards'],
            'extra_mask_core_reads': policy.mask_core_calls, 'elapsed_seconds': time.monotonic() - t0,
            'error': repr(exc), 'traceback': traceback.format_exc(), 'pid': os.getpid()}
        native.save_json(out / 'result.json', result)
        native.save_json(out / 'progress.json', result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['B', 'M', 'Q'], required=True)
    train(parser.parse_args().arm)
