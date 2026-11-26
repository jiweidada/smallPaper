"""Observe a fresh seed1 run; preserve the formal132 training recipe and RNG."""
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import train
import p2_candidate  # noqa: F401: register the frozen candidate implementation
from screen_runtime import install_runtime, tensor_digest


def digest(value):
    h = hashlib.sha256()
    tensor_digest(value, h)
    return h.hexdigest()


def rng_digest():
    return {
        'torch_CPU': digest(torch.get_rng_state()),
        'torch_CUDA': digest(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None,
        'python': hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        'numpy': hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
    }


def main(args):
    from engine.solver import det_solver
    from engine.solver.det_solver import DetSolver
    from engine.misc import dist_utils
    from engine.deim import dfine_decoder

    train.dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    train.check_cuda()
    assert args.seed == 1 and not args.resume and args.tuning and not args.test_only
    updates = train.yaml_utils.parse_cli(args.update)
    updates.update({k: v for k, v in vars(args).items() if k != 'update' and v is not None})
    cfg = train.YAMLConfig(args.config, **updates)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    assert cfg.epoches == 132 and cfg.flat_epoch == 64 and cfg.no_aug_epoch == 12
    assert cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['policy']['epoch'] == [4, 64, 120]
    assert cfg.yaml_cfg['train_dataloader']['collate_fn']['mixup_epochs'] == [4, 64]
    assert cfg.use_amp and cfg.use_ema
    root = Path(os.environ['SCREEN_RECEIPT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    original_stats = det_solver.stats
    install_runtime(cfg, root)
    if cfg.yaml_cfg['model'] == 'DEIM':
        # BASE must consume exactly its native profiler RNG, as the formal BASE did.
        det_solver.stats = original_stats
        assert cfg.yaml_cfg.get('profiling_rng_mode') is None
    else:
        assert cfg.yaml_cfg['model'] == 'P2CandidateDEIM'
        assert cfg.yaml_cfg['profiling_rng_mode'] == 'native_reference_replay'

    profiler = det_solver.stats
    def observed_stats(*a, **kw):
        before = rng_digest()
        result = profiler(*a, **kw)
        (root / 'profiler_RNG.json').write_text(json.dumps({'before': before, 'after': rng_digest()}, indent=2))
        return result
    det_solver.stats = observed_stats

    state = {'epoch': -1, 'batch': 0, 'inputs': 0, 'attempts': 0, 'updates': 0}
    native_train = DetSolver.train
    inputs_file = (root / 'all_inputs.jsonl').open('x', buffering=1)
    dn_file = (root / 'denoising.jsonl').open('x', buffering=1)
    optimizer_file = (root / 'all_optimizer_updates.jsonl').open('x', buffering=1)
    epoch_file = (root / 'epoch_RNG.jsonl').open('x', buffering=1)

    def audited_train(self):
        native_train(self)
        model = dist_utils.de_parallel(self.model)
        def prehook(module, a, kw):
            if not module.training:
                return
            assert 0 <= state['epoch'] < 6
            targets = kw.get('targets')
            image_ids = [int(t['image_id'].reshape(-1)[0]) for t in targets]
            record = {'index': state['inputs'], 'native_epoch': state['epoch'],
                      'completed_round': state['epoch'] + 1, 'batch_index': state['batch'],
                      'sha256': digest([a[0], targets]), 'shape': list(a[0].shape),
                      'image_ids': image_ids, 'RNG_before_forward': rng_digest()}
            inputs_file.write(json.dumps(record) + '\n')
            state['inputs'] += 1
            state['batch'] += 1
        model.register_forward_pre_hook(prehook, with_kwargs=True)
        self.optimizer.register_step_post_hook(lambda *_: state.__setitem__('updates', state['updates'] + 1))
        scaler_step = self.scaler.step
        def step(optimizer, *a, **kw):
            before = state['updates']
            scale = self.scaler.get_scale()
            result = scaler_step(optimizer, *a, **kw)
            state['attempts'] += 1
            optimizer_file.write(json.dumps({'attempt': state['attempts'], 'input_index': state['inputs'] - 1,
                'native_epoch': state['epoch'], 'actual_updates': state['updates'],
                'updated': state['updates'] > before, 'scale': scale,
                'lr': [g['lr'] for g in optimizer.param_groups]}) + '\n')
            return result
        self.scaler.step = step
    DetSolver.train = audited_train

    native_epoch_train = det_solver.train_one_epoch
    def epoch_train(*a, **kw):
        epoch = int(a[7]) if len(a) > 7 else int(kw['epoch'])
        assert 0 <= epoch < 6
        state['epoch'], state['batch'] = epoch, 0
        epoch_file.write(json.dumps({'native_epoch': epoch, 'before': rng_digest(), 'inputs_before': state['inputs']}) + '\n')
        result = native_epoch_train(*a, **kw)
        epoch_file.write(json.dumps({'native_epoch': epoch, 'after': rng_digest(), 'inputs_after': state['inputs'],
                                    'actual_updates': state['updates']}) + '\n')
        return result
    det_solver.train_one_epoch = epoch_train

    native_dn = dfine_decoder.get_contrastive_denoising_training_group
    def observed_dn(*a, **kw):
        result = native_dn(*a, **kw)
        # Do not hash learned class embeddings: their divergence is expected.
        dn_file.write(json.dumps({'input_index': state['inputs'] - 1, 'native_epoch': state['epoch'],
            'box_noise_sha256': digest(result[1]), 'mask_meta_sha256': digest(result[2:]),
            'RNG_after_DN': rng_digest()}) + '\n')
        return result
    dfine_decoder.get_contrastive_denoising_training_group = observed_dn

    solver = train.TASKS[cfg.yaml_cfg['task']](cfg)
    solver.fit(json.dumps(cfg.__dict__, indent=2, default=str))
    assert solver.last_epoch == 5
    assert state['inputs'] == 6 * len(solver.train_dataloader)
    assert state['attempts'] == state['inputs'] and state['updates'] > 0
    torch.save(solver.state_dict(), Path(cfg.output_dir) / 'checkpoint_fixed6.pth')
    (root / 'six_round_completion.json').write_text(json.dumps({
        'completed_rounds': 6, 'schedule_epochs': cfg.epoches, 'last_native_epoch': solver.last_epoch,
        'input_batches': state['inputs'], 'optimizer_attempts': state['attempts'],
        'actual_updates': state['updates'], 'fresh_common_initialization': True}, indent=2))
    for f in (inputs_file, dn_file, optimizer_file, epoch_file):
        f.close()
    train.dist_utils.cleanup()


if __name__ == '__main__':
    main(train.build_arg_parser().parse_args())
