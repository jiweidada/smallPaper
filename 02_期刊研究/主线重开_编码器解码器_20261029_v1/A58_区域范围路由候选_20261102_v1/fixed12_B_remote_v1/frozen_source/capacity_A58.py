"""Exactly two real batch8 forward/backward/updates per arm, discarded."""
import argparse
import json
import os
import time
import traceback
import torch
from torch.utils.data import DataLoader
from runtime_A58 import ROOT, native, plan_check, backend, components, ForwardJournal
from training_utils_A58 import input_digest, rng_digest, assert_finite_output, lr_ratio

def run(arm):
    plan = plan_check('capacity')
    backend()
    out = ROOT / ('capacity_' + arm)
    out.mkdir(exist_ok=False)
    with (out / 'run_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    journal = None
    start = time.monotonic()
    try:
        trainset = native.VisDroneDataset('train')
        assert len(trainset) == 6471
        assert sum(len(r['valid']) for r in trainset.rows) == 343204
        trainset.epoch = 0
        order = torch.randperm(6471, generator=torch.Generator().manual_seed(native.SEED)).tolist()
        loader = DataLoader(trainset, batch_size=8, sampler=order[:16], num_workers=0,
                            collate_fn=native.collate, drop_last=False)
        model, criterion, _ = components(arm)
        model.cuda().train(); criterion.cuda().train()
        optimizer = native.make_optimizer(model)
        journal = ForwardJournal(model, out / 'actual_events.jsonl', limit=2)
        journal.phase = 'capacity'
        native.seed_all()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for i, (images, targets) in enumerate(loader):
            assert i < 2 and len(targets) == 8
            inp = input_digest(images, targets)
            images = images.cuda()
            targets = native.targets_to_device(targets, 'cuda')
            rng = rng_digest()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            ts = time.monotonic()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images, targets)
            assert_finite_output(outputs)
            with torch.autocast(device_type='cuda', enabled=False):
                losses = criterion(native.fp32_tree(outputs), targets, epoch=0,
                                   step=i, global_step=i, epoch_step=809)
                loss = sum(losses.values())
            assert torch.isfinite(loss)
            loss.backward()
            context = model.encoder.a58_context
            grads = {k: (None if p.grad is None else float(p.grad.detach().abs().max()))
                     for k, p in context.named_parameters()}
            assert all(v is None or __import__('math').isfinite(v) for v in grads.values())
            if arm == 'B':
                assert all(v is None for v in grads.values())
            elif i == 1:
                assert grads['gate'] is not None and grads['gate'] > 0
                assert any(v is not None and v > 0 for k, v in grads.items() if k != 'gate')
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * lr_ratio(i)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
            optimizer.step()
            journal.optimizer_returned()
            torch.cuda.synchronize()
            rows.append(dict(step=i + 1, input_sha256=inp, pre_forward_rng_sha256=rng,
                             loss=float(loss.detach()), grad_norm=float(grad), context_gradient_max=grads,
                             residual_gate=float(context.gate.detach()),
                             total_train_step_seconds=time.monotonic()-ts,
                             allocated_peak=torch.cuda.max_memory_allocated(),
                             reserved_peak=torch.cuda.max_memory_reserved()))
            del outputs, loss, losses, images, targets
        assert journal.started == journal.completed == journal.updates == native.MODEL_COUNTER['forwards'] == 2
        if arm != 'B':
            prior = json.loads((ROOT / 'capacity_B/result.json').read_text())
            for row, other in zip(rows, prior['steps']):
                assert row['input_sha256'] == other['input_sha256']
                assert row['pre_forward_rng_sha256'] == other['pre_forward_rng_sha256']
        result = dict(status='PASS', arm=arm, steps=rows, counts=journal.counts(),
                      protocol_sha256=native.digest(ROOT / 'capacity_protocol_FROZEN_v1.json'),
                      elapsed_seconds=time.monotonic()-start, weights_discarded=True,
                      real_batch=8, AP_evaluations=0, model_checkpoint_saved=False,
                      torch=torch.__version__, GPU=torch.cuda.get_device_name(),
                      augmented_initialization_sha256=plan['augmented_initialization_sha256'])
        native.save_json(out / 'result.json', result)
        print(json.dumps(result), flush=True)
    except BaseException:
        native.save_json(out / 'result.json', dict(status='FAILED_STOPPED_NO_RETRY', arm=arm,
            counts=None if journal is None else journal.counts(), traceback=traceback.format_exc(),
            protocol_sha256=native.digest(ROOT / 'capacity_protocol_FROZEN_v1.json'),
            elapsed_seconds=time.monotonic()-start))
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['B', 'M', 'Q'], required=True)
    run(parser.parse_args().arm)
