"""Explicit capacity/train entry; no controller, no automatic retry or live-run edits."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import bootstrap
from bootstrap import ROOT
from factory import load_config, build_model, build_criterion, load_common_init, verify_snapshot, native_recipe
from dataset import PairedCocoImages, paired_collate, native_query_loader, move
from dual_view import paired_loss
from engine.optim.ema import ModelEMA
from evaluate import evaluate_dataset


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(2 ** 20), b''):
            digest.update(data)
    return digest.hexdigest()


def implementation_hash():
    rows = [(str(p.relative_to(ROOT)), hash_file(p)) for p in sorted(ROOT.glob('*.py'))]
    rows.append(('native_source_manifest.json', hash_file(ROOT / 'native_source_manifest.json')))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2 ** 32); torch.manual_seed(seed)


def optimizer_for(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = .0001 if name.startswith('backbone.') else .0002
        decay = 0. if any(s in name for s in ('bn', 'norm', 'bias')) else .0001
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{'params': p, 'lr': lr, 'initial_lr': lr, 'weight_decay': wd}
                              for (lr, wd), p in groups.items()], betas=(.9, .999))


def loader_for(cfg, images, annotations, dense=False):
    if cfg['kind'] == 'dual_view':
        dataset = PairedCocoImages(images, annotations, seed=cfg['seed'], stride=cfg['tile_stride'])
        loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True,
                            num_workers=cfg['workers'], collate_fn=paired_collate,
                            persistent_workers=False)
    else:
        loader = native_query_loader(images, annotations, workers=cfg['workers'])
    if dense:
        if cfg['kind'] == 'dual_view':
            indices = sorted(range(len(loader.dataset)),
                key=lambda i: (-len(loader.dataset.anns[loader.dataset.images[i]['id']]),
                               loader.dataset.images[i]['id']))[:8]
        else:
            dataset = loader.dataset
            indices = sorted(range(len(dataset)), key=lambda i: (-len(dataset.coco.imgToAnns[dataset.ids[i]]), dataset.ids[i]))[:8]
        if len(indices) < 8:
            raise ValueError('Capacity needs at least eight real images')
        loader = DataLoader(loader.dataset, batch_size=8, sampler=indices * 3, drop_last=True,
                            num_workers=cfg['workers'], collate_fn=loader.collate_fn,
                            persistent_workers=False)
    return loader


def set_epoch(loader, criterion, epoch):
    loader.dataset.set_epoch(epoch)
    if hasattr(loader.collate_fn, 'set_epoch'):
        loader.collate_fn.set_epoch(epoch)
    criterion.set_epoch(epoch)


def one_step(model, criterion, optimizer, scaler, samples, targets, cfg, device, ema=None):
    samples, targets = move(samples, device), move(targets, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        outputs = model(samples, targets)
        losses = paired_loss(criterion, outputs, targets) if cfg['kind'] == 'dual_view' else criterion(outputs, targets)
        loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss; preserve failure without retry')
    count = [0]
    def counted(*args): count[0] += 1
    hook = optimizer.register_step_post_hook(counted)
    try:
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        scaler.step(optimizer); scaler.update()
    finally:
        hook.remove()
    if count[0] and ema is not None:
        ema.update(model)
    return {'loss': float(loss.detach()), 'optimizer_step_calls': count[0],
            'amp_skips_proxy': 1 - count[0],
            'target_counts': {k: [len(t['labels']) for t in targets[k]] for k in ('global', 'local')}
                if cfg['kind'] == 'dual_view' else [len(t['labels']) for t in targets]}


def input_digest(samples, targets):
    digest = hashlib.sha256()
    def add(value):
        if torch.is_tensor(value):
            x = value.detach().cpu().contiguous()
            digest.update(str((tuple(x.shape), str(x.dtype))).encode()); digest.update(x.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value): digest.update(key.encode()); add(value[key])
        elif isinstance(value, list):
            for item in value: add(item)
    add(samples); add(targets)
    return digest.hexdigest()


def flat_cosine(step, batches, epochs, base):
    # Native schedule formula (engine/optim/lr_scheduler.py), with locked constants.
    total, flat, no_aug, warmup = batches * epochs, batches * 46, batches * 12, 1000
    if step <= warmup:
        return base * (step / warmup) ** 2
    if step <= flat:
        return base
    if step >= total - no_aug:
        return base * .5
    return base * .5 + base * .5 * .5 * (1 + math.cos(math.pi * (step - flat) / (total - flat - no_aug)))


def bindings(args, cfg, init):
    return {'config': cfg, 'config_sha256': hash_file(args.config), 'implementation_sha256': implementation_hash(),
            'annotation_sha256': hash_file(args.train_annotations), 'init': init,
            'torch': torch.__version__, 'GPU_uuid': args.gpu_uuid, 'batch8_originals': True,
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'TF32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'TF32_cudnn': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'candidate_recipe_distinct_from_frozen_C': True}


def capacity(args, cfg, model, criterion, optimizer, scaler, device, receipt):
    if device.type != 'cuda':
        raise ValueError('Real VRAM acceptance requires CUDA; CPU tests cannot pass this gate')
    rows = []
    for epoch in [0, 4, 7, 46, 84]:
        for dense, limit in [(False, 2), (True, 3)]:
            seed_all(cfg['seed'] + epoch * 100003)
            loader = loader_for(cfg, args.train_images, args.train_annotations, dense)
            set_epoch(loader, criterion, epoch)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            steps = []
            iterator = iter(loader)
            try:
                for _ in range(limit):
                    samples, targets = next(iterator)
                    digest = input_digest(samples, targets)
                    steps.append({**one_step(model, criterion, optimizer, scaler, samples, targets, cfg, device),
                                  'input_and_targets_sha256': digest})
            finally:
                if hasattr(iterator, '_shutdown_workers'): iterator._shutdown_workers()
            torch.cuda.synchronize()
            rows.append({'phase': epoch, 'case': 'dense' if dense else 'ordinary', 'steps': steps,
                         'allocated_peak_GiB': torch.cuda.max_memory_allocated() / 2 ** 30,
                         'reserved_peak_GiB': torch.cuda.max_memory_reserved() / 2 ** 30})
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
    passes = all(len(r['steps']) == (3 if r['case'] == 'dense' else 2) and
                  all(s['optimizer_step_calls'] == 1 for s in r['steps']) for r in rows)
    passes &= max(r['allocated_peak_GiB'] for r in rows) >= 10.
    passes &= max(r['reserved_peak_GiB'] for r in rows) < total * .95
    result = {**receipt, 'status': 'passed_capacity_only' if passes else 'capacity_not_accepted',
              'phases': rows, 'total_GPU_GiB': total, 'capacity_weights_discarded_for_formal_training': True,
              'future_peak_not_guaranteed': True}
    (args.output / 'capacity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    if not passes:
        raise RuntimeError('Capacity gate failed; do not automatically change batch or recipe')


def train(args, cfg, model, criterion, optimizer, scaler, device, receipt):
    accepted = json.loads(Path(args.capacity_receipt).read_text(encoding='utf8'))
    for key in ['config_sha256', 'implementation_sha256', 'annotation_sha256', 'init', 'torch', 'GPU_uuid']:
        if accepted[key] != receipt[key]: raise ValueError(f'Capacity binding differs: {key}')
    if accepted['status'] != 'passed_capacity_only': raise ValueError('No passed real capacity test')
    if cfg['epochs'] != 96: raise ValueError('Formal candidate budget is 96 epochs')
    loader = loader_for(cfg, args.train_images, args.train_annotations)
    if len(loader) == 0: raise ValueError('No full batch8 available')
    ema = ModelEMA(model, decay=.9999, warmups=1000)
    best_ap, best_epoch = -math.inf, None
    for epoch in range(96):
        seed_all(cfg['seed'] + epoch * 100003); set_epoch(loader, criterion, epoch); model.train()
        counts, records = 0, []
        for step, (samples, targets) in enumerate(loader):
            for group in optimizer.param_groups:
                group['lr'] = flat_cosine(epoch * len(loader) + step, len(loader), 96, group['initial_lr'])
            record = one_step(model, criterion, optimizer, scaler, samples, targets, cfg, device, ema)
            if cfg['kind'] == 'dual_view':
                record['image_ids'] = [int(t['image_id'].item()) for t in targets['global']]
                record['windows'] = targets['windows'].tolist()
            else:
                record['image_ids'] = [int(t['image_id'].item()) for t in targets]
            if step == 0: record['first_inputs_sha256'] = input_digest(samples, targets)
            counts += record['optimizer_step_calls']; records.append(record)
        result = evaluate_dataset(ema.module, cfg, args.val_images, args.val_annotations,
                                  args.output / 'val_epochs' / f'{epoch:03}', device)
        ap = result['metrics']['AP']
        # Highest overall AP, first epoch wins exact tie; never select by APs.
        if ap > best_ap: best_ap, best_epoch = ap, epoch
        checkpoint = {'model': model.state_dict(), 'ema': ema.state_dict(),
                      'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                      'epoch': epoch, 'best_epoch': best_epoch, 'best_AP': best_ap, 'bindings': receipt,
                      'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                      'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}
        torch.save(checkpoint, args.output / 'last.pth')
        if best_epoch == epoch: torch.save(checkpoint, args.output / 'best_ema.pth')
        with (args.output / 'training_audit.jsonl').open('a', encoding='utf8') as handle:
            handle.write(json.dumps({'epoch': epoch, 'batches': len(records), 'optimizer_steps': counts,
                'amp_skips_proxy': len(records) - counts, 'metrics': result['metrics'], 'steps': records}) + '\n')
    ck = torch.load(args.output / 'best_ema.pth', map_location='cpu', weights_only=False)
    ema.module.load_state_dict(ck['ema']['module'])
    evaluate_dataset(ema.module, cfg, args.val_images, args.val_annotations,
                     args.output / 'independent_val_selected_ema', device)
    (args.output / 'complete.json').write_text(json.dumps({'status': 'candidate96_pending_independent_acceptance',
        'best_epoch': best_epoch, 'best_AP': best_ap, 'bindings': receipt}, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['capacity', 'train'])
    parser.add_argument('--config', required=True); parser.add_argument('--common-init', required=True)
    parser.add_argument('--common-init-sha256', default='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91',
                        help='Reviewed frozen initialization hash; do not substitute trained weights')
    parser.add_argument('--train-images', required=True); parser.add_argument('--train-annotations', required=True)
    parser.add_argument('--val-images'); parser.add_argument('--val-annotations')
    parser.add_argument('--capacity-receipt'); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-uuid', required=True, help='Explicit physical GPU UUID; other GPUs may remain busy')
    args = parser.parse_args(); verify_snapshot(); cfg = load_config(args.config)
    if args.mode == 'train' and not all([args.val_images, args.val_annotations, args.capacity_receipt]):
        parser.error('Training requires full val paths and this arm\'s real capacity receipt')
    device = torch.device(args.device)
    if device.type != 'cuda' or device.index not in (None, 0):
        raise ValueError('Use cuda:0 within the explicitly selected UUID; CPU tests are separate')
    if torch.cuda.is_initialized(): raise RuntimeError('Select the UUID before initializing CUDA')
    # This preparatory runner refuses a GPU already used by another process.
    uuid = subprocess.check_output(['nvidia-smi', '--id', args.gpu_uuid, '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    if uuid != args.gpu_uuid: raise ValueError('Require the full actual GPU UUID, not an index/guess')
    busy = subprocess.check_output(['nvidia-smi', '--id', args.gpu_uuid, '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy: raise RuntimeError('GPU has existing compute processes; do not compete or stop them')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(4); seed_all(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if hash_file(args.common_init) != args.common_init_sha256:
            raise ValueError('Common initialization differs from the reviewed SHA256')
        model = build_model(cfg); init = load_common_init(model, args.common_init)
        model.to(device).train(); criterion = build_criterion().to(device); criterion.set_model(model)
        optimizer = optimizer_for(model)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        receipt = bindings(args, cfg, init)
        (args.output / 'execution.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
        if args.mode == 'capacity': capacity(args, cfg, model, criterion, optimizer, scaler, device, receipt)
        else: train(args, cfg, model, criterion, optimizer, scaler, device, receipt)
    except BaseException as error:
        import traceback
        (args.output / 'failure.json').write_text(json.dumps({'error': str(error),
            'traceback': traceback.format_exc(), 'preserved_no_automatic_retry': True}, indent=2), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
