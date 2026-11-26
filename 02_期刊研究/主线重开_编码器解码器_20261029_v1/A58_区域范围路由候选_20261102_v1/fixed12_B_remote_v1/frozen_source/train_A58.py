"""Fresh matched B/M/Q, exactly 12 epochs, no automatic rescue."""
import argparse
import gc
import hashlib
import json
import os
import time
import traceback
import torch
from torch.utils.data import DataLoader
from runtime_A58 import ROOT, INIT, native, plan_check, backend, components, ForwardJournal
from training_utils_A58 import StateEMA, cpu_state, input_digest, rng_digest, assert_finite_output, lr_ratio
from evaluate_A58 import evaluate_model, latency_ms

def train(arm):
    plan = plan_check('formal')
    capacity = json.loads((ROOT / 'capacity_result.json').read_text())
    assert capacity['status'] == 'PASS' and capacity['real_detector_forwards'] == 6
    assert native.digest(ROOT / 'capacity_protocol_FROZEN_v1.json') == capacity['protocol_sha256'] == plan['capacity_protocol_sha256']
    out = ROOT / ('arm_' + arm)
    out.mkdir(exist_ok=False)
    with (out / 'run_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    backend()
    journal = None
    updates = completed = 0
    start = time.monotonic()
    try:
        trainset, valset = native.VisDroneDataset('train'), native.VisDroneDataset('val')
        assert len(trainset) == 6471 and len(valset) == 548
        assert sum(len(r['valid']) for r in trainset.rows) == 343204
        assert sum(len(r['valid']) for r in valset.rows) == 38759
        model, criterion, post = components(arm)
        native.save_json(out / 'initialization.json', dict(common_init_sha256=native.digest(INIT),
             all_initial_tensors_identical=True, model_parameters=9951721,
             seed=native.SEED, arm_mode=arm, protocol_sha256=native.digest(ROOT / 'protocol_FROZEN_v1.json')))
        model.cuda(); criterion.cuda()
        optimizer = native.make_optimizer(model)
        ema = StateEMA(model)
        journal = ForwardJournal(model, out / 'actual_events.jsonl', limit=10159)
        native.seed_all()
        torch.cuda.reset_peak_memory_stats()
        events = []
        for epoch in range(12):
            assert time.monotonic()-start < plan['maximum_seconds_per_arm']
            trainset.epoch = epoch
            order = torch.randperm(6471, generator=torch.Generator().manual_seed(native.SEED + epoch * 1009)).tolist()
            loader = DataLoader(trainset, batch_size=8, sampler=order, num_workers=2, collate_fn=native.collate,
                pin_memory=True, drop_last=False, generator=torch.Generator().manual_seed(native.SEED + 100000 + epoch))
            assert len(loader) == 809
            input_hashes, rng_hashes = [], []
            target_flow = hashlib.sha256()
            loss_sum = 0.
            ts = time.monotonic()
            model.train(); criterion.train(); optimizer.zero_grad(set_to_none=True)
            journal.phase = 'train'
            for micro, (images, targets) in enumerate(loader):
                assert time.monotonic()-start < plan['maximum_seconds_per_arm']
                if micro < 8:
                    input_hashes.append(input_digest(images, targets))
                for target in targets:
                    for key in ['boxes', 'labels', 'image_id', 'orig_size']:
                        target_flow.update(key.encode())
                        target_flow.update(str(tuple(target[key].shape)).encode())
                        target_flow.update(target[key].contiguous().numpy().tobytes())
                images = images.cuda(non_blocking=True)
                targets = native.targets_to_device(targets, 'cuda')
                if micro < 8:
                    rng_hashes.append(rng_digest())
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images, targets)
                assert_finite_output(outputs)
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
                optimizer.step(); journal.optimizer_returned()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                ema.update(model)
                value = float(loss.detach())
                loss_sum += value
                if micro % 50 == 0 or micro == 808:
                    progress = dict(status='TRAINING', arm=arm, epoch=epoch+1, microbatch=micro+1,
                        microbatches_per_epoch=809, completed_epochs=completed, optimizer_updates=updates,
                        full_model_forwards=journal.started, loss=value, grad_norm=float(grad),
                        residual_gate=float(model.encoder.a58_context.gate.detach()),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        elapsed_seconds=time.monotonic()-start, pid=os.getpid())
                    native.save_json(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                del outputs, losses, loss, images, targets
            completed = epoch+1
            event = dict(epoch=completed, optimizer_updates=updates, mean_loss=loss_sum/809,
                train_seconds=time.monotonic()-ts,
                order_sha256=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest(),
                first_eight_actual_input_sha256=input_hashes, first_eight_pre_forward_rng_sha256=rng_hashes,
                all_batches_targets_sha256=target_flow.hexdigest(),
                residual_gate=float(model.encoder.a58_context.gate.detach()))
            if arm != 'B':
                prior = json.loads((ROOT / 'arm_B' / ('epoch%02d_input_flow.json' % completed)).read_text())
                for key in ['order_sha256','first_eight_actual_input_sha256',
                            'first_eight_pre_forward_rng_sha256','all_batches_targets_sha256']:
                    assert prior[key] == event[key], key
            native.save_json(out / ('epoch%02d_input_flow.json' % completed), event)
            if completed in [4,8,12]:
                raw = cpu_state(model.state_dict())
                checkpoint = dict(model=raw, ema=dict(module=cpu_state(ema.state),updates=ema.updates),
                    arm=arm,epoch=completed,optimizer_updates=updates,
                    protocol_sha256=native.digest(ROOT / 'protocol_FROZEN_v1.json'),
                    common_initialization_sha256=native.digest(INIT),
                    inference_mode=arm)
                if completed == 12:
                    checkpoint['optimizer'] = optimizer.state_dict()
                path = out / ('epoch%02d.pth' % completed)
                torch.save(checkpoint,path); del checkpoint
                model.load_state_dict(ema.state,strict=True)
                journal.phase = 'validation'
                native.save_json(out / 'progress.json',dict(status='VALIDATING',arm=arm,epoch=completed,
                    optimizer_updates=updates,full_model_forwards=journal.started,pid=os.getpid()))
                event['EMA_validation'] = evaluate_model(model,post,valset,out,'epoch%02d_EMA' % completed)
                assert event['EMA_validation']['evaluated_GT'] == 38602
                model.load_state_dict(raw,strict=True)
                event['checkpoint_sha256'] = native.digest(path)
                del raw; gc.collect()
            events.append(event)
            with (out / 'epochs.jsonl').open('a') as f:
                f.write(json.dumps(event)+'\n')
            native.save_json(out / 'completed_epochs.json',dict(completed_epochs=completed,
                optimizer_updates=updates,events=events))
        model.load_state_dict(ema.state,strict=True)
        single,_ = native.collate([valset[0]])
        journal.phase = 'latency'
        latency = latency_ms(model,single.cuda(),warmup=10,repeats=30)
        assert updates == journal.updates == 9708
        assert journal.started == journal.completed == native.MODEL_COUNTER['forwards'] == 10159
        result = dict(status='COMPLETED',arm=arm,epochs=completed,optimizer_updates=updates,
            full_model_forwards=journal.started,counts=journal.counts(),elapsed_seconds=time.monotonic()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),endpoint_metrics=events[-1]['EMA_validation'],
            latency=latency,pid=os.getpid(),initialization_sha256=native.digest(INIT),
            protocol_sha256=native.digest(ROOT / 'protocol_FROZEN_v1.json'),
            residual_gate=float(model.encoder.a58_context.gate.detach()))
        native.save_json(out / 'result.json',result)
        native.save_json(out / 'progress.json',result)
        print(json.dumps(result),flush=True)
    except BaseException:
        result = dict(status='FAILED_STOPPED_NO_RETRY',arm=arm,completed_epochs=completed,
            optimizer_updates=updates,counts=None if journal is None else journal.counts(),
            elapsed_seconds=time.monotonic()-start,traceback=traceback.format_exc(),pid=os.getpid())
        native.save_json(out / 'result.json',result)
        native.save_json(out / 'progress.json',result)
        raise

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['B','M','Q'],required=True)
    train(parser.parse_args().arm)
