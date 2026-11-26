"""One authorized first-batch forward/backward, zero optimizer/evaluator updates.

Run as a separate process after stage_arm.py prepare. It never saves mutated
model/EMA weights. Training reconstructs and reloads the archived initial state.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import sys
import traceback

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm-dir',required=True,type=Path)
    parser.add_argument('--arm',required=True,choices=['BASE_FRESH','FULL_FRESH'])
    args=parser.parse_args()
    root=args.arm_dir.resolve(strict=True)
    staged=json.loads((root/'staged_inputs.json').read_text())
    assert staged['arm']==args.arm and staged['authorized'] is True
    os.environ['CUDA_VISIBLE_DEVICES']=str(staged['cuda_visible_device'])
    sys.path.insert(0,str(root))
    import fair_helpers as fair
    torch=fair.torch
    environment=fair.readj(root/'environment.current.json')
    assert str(torch.__version__)==environment['torch'] and torch.version.cuda==environment['cuda']
    assert torch.backends.cudnn.version()==environment['cudnn']
    destination=root/'inputs'/'preflight.json'
    assert not destination.exists(),'Preflight receipt exists; do not automatically repeat the authorized model check'
    report={'status':'INITIALIZING','arm':args.arm,'started_utc':fair.now(),
            'optimizer_updates':0,'model_optimizer_steps':0,'model_forwards':0,'full_validation_passes':0,
            'configuration_sha256':staged['configuration_sha256'],'initial_file_sha256':fair.sha(root/'inputs/initial.pt')}
    fair.savej(destination,report)
    try:
        fair.controls()
        cfg,model,criterion,post,train,val,evaluator,lrfn=fair.build(args.arm)
        original_arm=args.arm.removesuffix('_FRESH')
        archived=fair.readj(root/f'frozen_parent/preflight/{original_arm}.json')
        initial=torch.load(root/'inputs/initial.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(initial,strict=True)
        initial_hash=fair.tensor_hash(model.state_dict())
        assert initial_hash==archived['initial_tensor_sha256']
        assert report['initial_file_sha256']==archived['initial_file_sha256']
        common=fair.readj(root/'frozen_parent/preflight/common_initialization.json')
        assert common['shared_tensors']==len(common['keys'])==554
        shared={}
        for arm in ['BASE','FULL']:
            value=torch.load(root/f'frozen_parent/preflight/{arm}.initial.pt',map_location='cpu',weights_only=True)
            shared[arm]=fair.tensor_hash({key:value[key] for key in common['keys']})
            del value
        assert shared['BASE']==shared['FULL']==common['shared_tensor_sha256']
        optimizer=cfg.optimizer
        _=cfg.lr_scheduler
        _=cfg.lr_warmup_scheduler
        ema=cfg.ema
        assert fair.tensor_hash(ema.module.state_dict())==initial_hash and ema.updates==0
        assert [g['lr'] for g in optimizer.param_groups]==archived['initial_executed_lrs']
        assert not train.persistent_workers and not val.persistent_workers
        fair.seeds(0);fair.prepare_epoch(train,0)
        expected=fair.readj(root/f'frozen_parent/preflight/{original_arm}.first100.json')
        assert fair.readj(root/'frozen_parent/preflight/BASE.first100.json')==fair.readj(root/'frozen_parent/preflight/FULL.first100.json')
        data=[];first=None
        for step,(samples,targets) in enumerate(train):
            if step==0:first=(samples,targets)
            value={'step':step,'sha256':fair.batch_hash(samples,targets),
                   'image_ids':[int(t['image_id']) for t in targets]}
            assert value==expected[step],f'Frozen first100 input differs at batch{step}'
            data.append(value)
            if step==99:break
        assert len(data)==100
        fair.savej(root/'inputs/first100.json',data)
        model=model.cuda().train();criterion=criterion.cuda().train()
        samples,targets=first;samples=samples.cuda()
        targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
        fair.seeds(0);torch.cuda.reset_peak_memory_stats()
        report['model_forwards']=1
        fair.savej(destination,report)
        outputs=model(samples,targets=targets)
        losses=criterion(outputs,targets,epoch=0,step=0,global_step=0,epoch_step=1617)
        loss=sum(losses.values());assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert fair.sha(root/'inputs/initial.pt')==report['initial_file_sha256']
        report.update(status='PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE',completed_utc=fair.now(),
                      initial_tensor_sha256=initial_hash,initial_ema_tensor_sha256=initial_hash,
                      initial_ema_updates=0,loss_and_gradients_finite=True,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      first100_batches_sha256=fair.sha(root/'inputs/first100.json'),
                      shared_tensors_verified=554,shared_tensor_sha256=shared['BASE'],
                      initialization={'arm':args.arm,'source':'frozen archived initial tensors, strict loading; shared554 verified',
                                      'initialization_after_constructor_seed_reset':0},
                      preflight_process_pid=os.getpid(),training_process_must_reload_archived_initial_state=True)
        fair.savej(destination,report)
        print(json.dumps(report),flush=True)
    except BaseException:
        report.update(status='PREFLIGHT_FAILED_BEFORE_TRAINING',traceback=traceback.format_exc())
        fair.savej(destination,report)
        raise

if __name__=='__main__':main()
