"""Validate one declared ablation without any optimizer update."""

import json
import os
from pathlib import Path
import subprocess
import traceback

import fair_helpers as fair

HERE = Path(__file__).resolve().parent


def main():
    spec = fair.readj(HERE / 'arm_spec.json')
    arm = spec['arm']
    folder = HERE / 'inputs'
    folder.mkdir(exist_ok=False)
    report = {'status': 'INITIALIZING', 'arm': arm, 'optimizer_updates': 0,
              'started_utc': fair.now(), 'pid': os.getpid(), 'runner_sha256': fair.sha(Path(__file__)),
              'configuration_sha256': fair.sha(HERE / 'execution_configs' / f'{arm}.yml')}
    path = folder / 'preflight.json'
    fair.savej(path, report)
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert gpu[0] == spec['gpu_uuid'] and int(gpu[1]) >= 20000
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'], text=True).strip()
        assert not active, 'GPU is occupied'
        fair.controls()
        cfg, model, criterion, post, train, val, evaluator, lrfn = fair.build(arm)
        assert cfg.yaml_cfg['ablation']['arm'] == arm
        assert not train.persistent_workers and not val.persistent_workers
        optimizer = cfg.optimizer
        _ = cfg.lr_scheduler
        _ = cfg.lr_warmup_scheduler
        report.update(initialization=model.ablation_initialization,
                      optimizer_groups=len(optimizer.param_groups),
                      initial_executed_lrs=[group['lr'] for group in optimizer.param_groups])
        assert report['optimizer_groups'] == 4 and not optimizer.state
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        fair.savept(folder / 'initial.pt', initial)
        report.update(initial_tensor_sha256=fair.tensor_hash(initial),
                      initial_file_sha256=fair.sha(folder / 'initial.pt'))
        fair.seeds()
        fair.prepare_epoch(train, 0)
        batches = []
        first = None
        for step, (samples, targets) in enumerate(train):
            if step == 0:
                first = (samples, targets)
            batches.append({'step': step, 'sha256': fair.batch_hash(samples, targets),
                            'image_ids': [int(target['image_id']) for target in targets]})
            if step == 99:
                break
        assert batches == fair.readj(HERE / 'reference/reference.first100.json'), 'Training inputs differ'
        fair.savej(folder / 'first100.json', batches)
        model.cuda().train()
        criterion.cuda().train()
        post.cuda()
        samples, targets = first
        samples = samples.cuda()
        targets = [{key: value.cuda() for key, value in target.items()} for target in targets]
        fair.seeds()
        fair.torch.cuda.reset_peak_memory_stats()
        outputs = model(samples, targets=targets)
        losses = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1617)
        loss = sum(losses.values())
        assert bool(fair.torch.isfinite(loss))
        loss.backward()
        assert all(parameter.grad is None or bool(fair.torch.isfinite(parameter.grad).all())
                   for parameter in model.parameters())
        model.eval()
        with fair.torch.no_grad():
            predictions = post(model(samples), fair.torch.stack([target['orig_size'] for target in targets]))
        assert len(predictions) == 4
        assert all(len(item['scores']) == 300 and bool(fair.torch.isfinite(item['scores']).all())
                   for item in predictions)
        assert not optimizer.state
        report.update(status='PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE', completed_utc=fair.now(),
                      first100_batches_sha256=fair.sha(folder / 'first100.json'),
                      trainable_parameters=sum(parameter.numel() for parameter in model.parameters()
                                               if parameter.requires_grad),
                      first_batch_cuda_allocated_peak=fair.torch.cuda.max_memory_allocated(),
                      first_batch_cuda_reserved_peak=fair.torch.cuda.max_memory_reserved(),
                      finite_loss_and_gradients=True, loss=float(loss.detach()))
        fair.savej(path, report)
        print(json.dumps(report), flush=True)
    except BaseException:
        report.update(status='PREFLIGHT_FAILED_NO_TRAINING', traceback=traceback.format_exc(), ended_utc=fair.now())
        fair.savej(path, report)
        raise


if __name__ == '__main__':
    main()
