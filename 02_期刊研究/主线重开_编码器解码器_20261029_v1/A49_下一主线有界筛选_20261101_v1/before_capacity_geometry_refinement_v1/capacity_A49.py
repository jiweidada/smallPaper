"""One bounded real-model capacity/invariant check. No formal training queue."""
import gc
import hashlib
import json
import os
import sys
import time
import traceback
import torch
from runtime_A49 import ROOT, ASSETS, A48, native, plan_check
from ignore_query_supervision import IgnorePolicy, native_inference
sys.path.insert(0, str(A48))
from training_utils_A48 import assert_finite_output


def tensor_tree_digest(value):
    digest = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            digest.update(str((tuple(x.shape), x.dtype)).encode())
            digest.update(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                digest.update(k.encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            for item in x:
                add(item)
        else:
            digest.update(repr(x).encode())
    add(value)
    return digest.hexdigest()


if __name__ == '__main__':
    plan = plan_check('capacity')
    with (ROOT / 'capacity_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    updates = 0
    records = []
    initialization_hashes = {}
    initial_output_hashes = {}
    first_rng_hashes = {}
    mask_core_reads = 0
    try:
        ds = native.VisDroneDataset('train')
        assert len(ds) == 6471
        order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(native.SEED)).tolist()
        for arm in ['B', 'M', 'Q']:
            # All arms use the original B host. No A48 BN installation is called.
            model, criterion, _ = native.create_components('B')
            common = torch.load(ASSETS / 'common_init_10class.pth', map_location='cpu', weights_only=True)['model']
            assert set(model.state_dict()) == set(common)
            assert all(torch.equal(t, common[k]) for k, t in model.state_dict().items())
            initialization_hashes[arm] = native.digest(ASSETS / 'common_init_10class.pth')
            del common
            policy = IgnorePolicy(arm)
            sites = policy.install_model(model)
            policy.install_loss(criterion)
            assert sum(p.numel() for p in model.parameters()) == 9803112
            assert all(m.__class__ is torch.nn.BatchNorm2d for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
            model = model.cuda(); criterion = criterion.cuda()
            optimizer = native.make_optimizer(model)
            native.seed_all()
            torch.cuda.reset_peak_memory_stats()
            model.train(); criterion.train()
            for step in range(3):
                images, targets = native.collate([ds[i] for i in order[8 * step:8 * (step + 1)]])
                images = images.cuda(); targets = native.targets_to_device(targets, 'cuda')
                if step == 0:
                    first_rng_hashes[arm] = (hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                        hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
                with policy.training_step(targets):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(images, targets)
                    assert_finite_output(outputs)
                    if step == 0:
                        initial_output_hashes[arm] = tensor_tree_digest(outputs)
                    policy.annotate_outputs(outputs)
                    with torch.autocast(device_type='cuda', enabled=False):
                        losses = criterion(native.fp32_tree(outputs), targets,
                            epoch=0, step=step, global_step=step, epoch_step=809)
                        loss = sum(losses.values())
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); updates += 1
                record = {'arm': arm, 'step': step, 'loss': float(loss.detach()), 'grad_norm': float(grad),
                          'mask_core_reads_so_far': policy.mask_core_calls,
                          'loss_validity_statistics': policy.loss_statistics,
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                records.append(record)
                native.save_json(ROOT / 'capacity_progress.json', {'status': 'RUNNING', 'arm': arm,
                    'step_completed': step + 1, 'optimizer_updates': updates,
                    'full_model_forwards': native.MODEL_COUNTER['forwards'], 'record': record, 'pid': os.getpid()})
                del images, targets, outputs, losses, loss
            if arm == 'Q':
                assert policy.mask_core_calls == 3 * len(sites)
            mask_core_reads += policy.mask_core_calls
            before = policy.mask_core_calls
            model.eval()
            with native_inference(model), torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                image, _ = ds[order[0]]
                result = model(image[None].cuda())
                assert torch.isfinite(result['pred_logits']).all() and torch.isfinite(result['pred_boxes']).all()
                del result, image
            assert policy.mask_core_calls == before and policy.active_tiles is None
            assert torch.cuda.max_memory_allocated() < 23 * 1024 ** 3
            del model, criterion, optimizer, policy
            gc.collect(); torch.cuda.empty_cache()
        assert len(set(initialization_hashes.values())) == len(set(initial_output_hashes.values())) == 1
        assert len(set(first_rng_hashes.values())) == 1
        assert updates == 9 and native.MODEL_COUNTER['forwards'] == 12
        final = {'status': 'PASS', 'records': records, 'initialization_sha256': initialization_hashes,
                 'initial_full_native_output_digest': initial_output_hashes,
                 'first_cpu_cuda_rng_sha256': first_rng_hashes, 'all_initial_native_predictions_identical': True,
                 'native_inference_no_GT_or_mask_reads': True, 'all_BN_native': True,
                 'model_forwards': native.MODEL_COUNTER['forwards'], 'capacity_optimizer_updates': updates,
                 'extra_mask_core_reads': mask_core_reads, 'formal_training_started': False,
                 'protocol_sha256': native.digest(ROOT / 'capacity_protocol_FROZEN_v1.json'),
                 'scope': 'Feasibility and actual code-path/invariant check, not efficacy, causality or novelty.',
                 'unix': time.time()}
        native.save_json(ROOT / 'capacity_result.json', final)
        native.save_json(ROOT / 'capacity_progress.json', final)
        print(json.dumps(final), flush=True)
    except BaseException as exc:
        final = {'status': 'FAILED_STOPPED_NO_RETRY', 'error': repr(exc), 'traceback': traceback.format_exc(),
                 'capacity_optimizer_updates': updates, 'model_forwards': native.MODEL_COUNTER['forwards'],
                 'formal_training_started': False, 'records': records, 'pid': os.getpid()}
        native.save_json(ROOT / 'capacity_result.json', final)
        native.save_json(ROOT / 'capacity_progress.json', final)
        raise
