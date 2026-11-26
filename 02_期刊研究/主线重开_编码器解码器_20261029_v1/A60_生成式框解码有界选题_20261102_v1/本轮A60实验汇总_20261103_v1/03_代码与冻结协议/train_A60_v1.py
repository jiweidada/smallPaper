"""Fresh matched B/M/Q: exactly <=12 epochs, no resume/retry/EMA/best selection."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
import torch
from torch.utils.data import Dataset,DataLoader
from runtime_A60_v1 import (EndpointDetector,backend,load_item,prepare_training,mode_inputs,
    loss_targets,criterion,optimizer,sha,save_json,tensor_sha)
from evaluate_A60_v1 import evaluate,latency

ROOT=Path('/root/uav_a60_20261102_v1/pilot_v1')
CODE=ROOT/'code'

def atomic_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temporary,path)

def epoch_order(n,epoch):
    rng=random.Random(20261102+100003*epoch);order=list(range(n));rng.shuffle(order)
    return [(index,rng.random()<.5) for index in order]

class TrainingImages(Dataset):
    def __init__(self,split,epoch):
        self.split=split;self.epoch=epoch;self.order=epoch_order(len(split['rows']),epoch)
    def __len__(self):return len(self.order)
    def __getitem__(self,position):
        index,flip=self.order[position];r=self.split['rows'][index]
        image_path=Path(self.split['image_dir'])/r['file'];st=image_path.stat()
        assert st.st_size==r['image_bytes'] and st.st_mtime_ns==r['image_mtime_ns'],r['file']
        image,target=load_item(r['file'],self.split['image_dir'],self.split['annotation_dir'],flip=flip)
        assert target['annotation_sha256']==r['annotation_sha256'] and len(target['labels'])==r['valid_GT']
        return image,target,position,flip

def collate_train(items):return items

def lr_factor(update,total):
    if update<500:return .1+.9*(update+1)/500
    v=(update-500)/(total-500-1)
    return .1+.9*.5*(1+math.cos(math.pi*v))

def assert_sources(scope):
    for name,h in scope['source_sha256'].items():assert sha(CODE/name)==h,name
    assert sha(ROOT/'dataset_manifest_v1.json')==scope['dataset_manifest_sha256']

def main(arm):
    scope=json.loads((ROOT/'protocol_FROZEN_v1.json').read_text());assert_sources(scope)
    assert arm in ['B','M','Q'] and scope['epochs_per_arm']==12
    out=ROOT/arm;assert not out.exists();out.mkdir()
    save_json(out/'run_once.json',dict(arm=arm,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],
        protocol_sha256=sha(ROOT/'protocol_FROZEN_v1.json'),maximum_forwards=78514,maximum_updates=9708,epochs=12,
        fresh_common_initialization=True,no_resume_or_retry=True))
    counts=dict(started=0,completed=0,updates=0,backwards=0,epochs=0,train_forwards=0,eval_forwards=0,latency_forwards=0)
    start=time.monotonic();phase='setup';epoch=0;position=-1
    events=(out/'actual_events.jsonl').open('x',buffering=1)
    samples=(out/'first8_input_flow.jsonl').open('x',buffering=1)
    def log(value):events.write(json.dumps(value,allow_nan=False)+'\n')
    def pre(module,args):
        assert counts['started']<78514
        counts['started']+=1;counts[phase+'_forwards']+=1
        log(dict(event='forward_start',count=counts['started'],phase=phase,epoch=epoch,position=position))
    def post(module,args,result):
        counts['completed']+=1;log(dict(event='forward_complete',count=counts['completed']))
    def progress(status,**extra):
        atomic_json(out/'progress.json',dict(status=status,arm=arm,epoch=epoch,position=position,counts=counts.copy(),
            elapsed_seconds=time.monotonic()-start,**extra))
    try:
        backend();torch.cuda.reset_peak_memory_stats()
        manifest=json.loads((ROOT/'dataset_manifest_v1.json').read_text())
        init=ROOT/'common_init_A60_v1.pth';assert sha(init)==scope['initialization_sha256']
        state=torch.load(init,map_location='cpu',weights_only=True)['model']
        model=EndpointDetector();model.load_state_dict(state,strict=True);del state
        model=model.cuda();model.register_forward_pre_hook(pre);model.register_forward_hook(post)
        opt=optimizer(model);crit=criterion().cuda()
        initial_probe={n:tensor_sha(p) for n,p in model.named_parameters() if n.startswith('head.time_mlp.') or '.class_logits.' in n}
        save_json(out/'initialization.json',dict(initialization_sha256=sha(init),parameters=sum(p.numel() for p in model.parameters()),
            time_and_class_parameter_sha256=initial_probe,checkpoint_loads=1,detector_constructors=1,
            backend='BF16 backbone/FP32 head and loss, TF32 off',gradient_accumulation='equal image weight; native losses normalized by their own matched-positive counts'))
        total_updates=9708
        for epoch in range(1,13):
            assert_sources(scope);model.train();phase='train';epoch_start=time.monotonic()
            dataset=TrainingImages(manifest['splits']['train'],epoch)
            gen=torch.Generator();gen.manual_seed(20261102+epoch)
            # Loader batches are CPU prefetch groups; GPU always processes one image at a time.
            loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=2,
                collate_fn=collate_train,pin_memory=True,generator=gen)
            raw_stream=hashlib.sha256();noisy_stream=hashlib.sha256();condition_stream=hashlib.sha256()
            epoch_loss=0.;epoch_images=0;min_time_gradient=None;last_norm=None
            for items in loader:
                opt.zero_grad(set_to_none=True)
                factor=lr_factor(counts['updates'],total_updates)
                for group in opt.param_groups:group['lr']=group['initial_lr']*factor
                group_loss=0.
                for image,target,position,flip in items:
                    seed=20261102+(epoch-1)*6471+position
                    prepared=prepare_training([target],seed)
                    noisy,condition,noisy_cpu,alpha=mode_inputs(arm,prepared,'cuda')
                    raw=dict(epoch=epoch,position=position,name=target['name'],flip=flip,queries=prepared['queries'],
                        image=tensor_sha(image),GT=tensor_sha(target['boxes']),labels=tensor_sha(target['labels']),
                        prior=tensor_sha(prepared['prior']),endpoint=tensor_sha(prepared['endpoint']),u=tensor_sha(prepared['progress']))
                    raw_stream.update(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()+b'\n')
                    noisy_hash=tensor_sha(noisy_cpu);condition_hash=tensor_sha(condition)
                    noisy_stream.update(bytes.fromhex(noisy_hash));condition_stream.update(bytes.fromhex(condition_hash))
                    if epoch==1 and position<8:
                        samples.write(json.dumps(dict(**raw,noisy=noisy_hash,time=condition_hash))+'\n')
                    output=model(image[None].cuda(non_blocking=True),noisy,condition)
                    losses=crit(output,loss_targets([target],'cuda'))
                    total=sum(v*crit.weight_dict[k] for k,v in losses.items() if k in crit.weight_dict)
                    assert torch.isfinite(total),'Nonfinite loss; stop without retry'
                    # The native criterion re-normalizes by matched positives. Average per-image
                    # normalized losses, including all images of the final seven-image group.
                    (total/len(items)).backward();counts['backwards']+=1
                    value=float(total.detach());group_loss+=value;epoch_loss+=value;epoch_images+=1
                    del output,losses,total,noisy,condition
                time_grad=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
                    if n.startswith('head.time_mlp.') and p.grad is not None)
                assert math.isfinite(time_grad) and time_grad>0
                min_time_gradient=time_grad if min_time_gradient is None else min(min_time_gradient,time_grad)
                last_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                assert counts['updates']<9708;opt.step();counts['updates']+=1
                log(dict(event='optimizer_returned',count=counts['updates'],epoch=epoch,position=position,
                    group_images=len(items),mean_loss=group_loss/len(items),gradient_norm=last_norm,time_MLP_gradient_L1=time_grad,lr_factor=factor))
                if counts['updates']%25==0:progress('TRAINING',last_group_loss=group_loss/len(items),gradient_norm=last_norm)
            assert epoch_images==6471 and counts['updates']==epoch*809
            counts['epochs']=epoch
            flow=dict(epoch=epoch,images=epoch_images,raw_stream_sha256=raw_stream.hexdigest(),
                noisy_stream_sha256=noisy_stream.hexdigest(),condition_stream_sha256=condition_stream.hexdigest())
            if arm!='B':
                reference=json.loads((ROOT/'B'/('epoch%02d_flow.json'%epoch)).read_text())
                assert flow['raw_stream_sha256']==reference['raw_stream_sha256'],'Shared raw input stream mismatch'
            if arm=='Q':
                reference=json.loads((ROOT/'M'/('epoch%02d_flow.json'%epoch)).read_text())
                assert flow['noisy_stream_sha256']==reference['noisy_stream_sha256'],'M/Q noisy input stream mismatch'
                assert flow['condition_stream_sha256']!=reference['condition_stream_sha256']
            save_json(out/('epoch%02d_flow.json'%epoch),flow)
            row=dict(epoch=epoch,mean_loss=epoch_loss/epoch_images,train_seconds=time.monotonic()-epoch_start,
                updates=counts['updates'],counts=counts.copy(),minimum_time_MLP_gradient_L1=min_time_gradient,
                last_gradient_norm=last_norm,flow=flow)
            if epoch in [4,8,12]:
                checkpoint=out/('epoch%02d_raw.pth'%epoch);assert not checkpoint.exists()
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,arm=arm,
                    counts=counts.copy(),initialization_sha256=scope['initialization_sha256'],
                    protocol_sha256=sha(ROOT/'protocol_FROZEN_v1.json')),checkpoint)
                row['checkpoint_sha256']=sha(checkpoint)
                phase='eval';progress('EVALUATING',evaluation_epoch=epoch)
                row['metrics']=evaluate(model,manifest['splits']['val'],out,'epoch%02d_raw'%epoch)
            save_json(out/('epoch%02d.json'%epoch),row);progress('EPOCH_COMPLETED',mean_loss=row['mean_loss'])
            print(json.dumps(dict(arm=arm,epoch=epoch,loss=row['mean_loss'],metrics=row.get('metrics'),
                elapsed_seconds=time.monotonic()-start,counts=counts)),flush=True)
        phase='latency';progress('TIMING_FIXED12_MODEL')
        timing=latency(model,manifest['splits']['val']);save_json(out/'fixed12_latency.json',timing)
        assert counts==dict(started=78514,completed=78514,updates=9708,backwards=77652,epochs=12,
            train_forwards=77652,eval_forwards=822,latency_forwards=40),counts
        assert all(torch.isfinite(p).all() for p in model.parameters())
        final_probe={n:tensor_sha(p) for n,p in model.named_parameters() if n in initial_probe}
        assert any(final_probe[n]!=initial_probe[n] for n in initial_probe if n.startswith('head.time_mlp.'))
        optimizer_path=out/'fixed12_optimizer.pth';assert not optimizer_path.exists();torch.save(opt.state_dict(),optimizer_path)
        metrics=json.loads((out/'epoch12_raw_metrics.json').read_text())
        result=dict(status='COMPLETED_FIXED12_RAW',arm=arm,counts=counts,metrics=metrics,latency=timing,
            initialization_sha256=scope['initialization_sha256'],protocol_sha256=sha(ROOT/'protocol_FROZEN_v1.json'),
            source_sha256=scope['source_sha256'],fixed12_checkpoint_sha256=sha(out/'epoch12_raw.pth'),
            optimizer_sha256=sha(optimizer_path),elapsed_seconds=time.monotonic()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            initial_time_and_class_parameters=initial_probe,final_time_and_class_parameters=final_probe,
            no_best_epoch_selection=True,training_epochs=12,limitations='One short seed; no statistical significance, convergence, novelty or onboard speed claim.')
        save_json(out/'result.json',result);progress('COMPLETED_FIXED12_RAW')
        save_json(out/'exit.json',dict(exit=0,counts=counts,no_retry=True))
    except BaseException:
        progress('FAILED_NO_RETRY')
        if not (out/'exit.json').exists():save_json(out/'exit.json',dict(exit=1,counts=counts,traceback=traceback.format_exc(),no_retry=True))
        raise
    finally:events.close();samples.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['B','M','Q'],required=True);args=p.parse_args();main(args.arm)
