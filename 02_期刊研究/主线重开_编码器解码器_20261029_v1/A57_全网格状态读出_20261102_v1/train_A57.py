"""One fresh arm, fixed12 raw endpoint; no continuation, EMA or retry."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import datetime,timezone
import torch
from torch.utils.data import DataLoader
from common_A57 import (ROOT,MODEL_COUNTER,make_model,make_criterion,make_optimizer,
    lr_ratio,weighted_loss,assert_output,cpu_tree,input_digest,rng_digest,
    validate_files,save_json,load,digest,VisDroneDataset,collate,targets_to_device,seed_all,
    set_event_journal,event)
from evaluate_A57 import bind_ground_truth,evaluate_model,latency_ms

STATE={'status':'NOT_STARTED','epochs_completed':0,'optimizer_updates':0,
       'optimizer_step_attempts':0,'train_forward_completions':0,'val_forward_completions':0,
       'latency_forward_completions':0,'no_nonfinite_skip':True}
MODEL=None
OPTIMIZER=None
OUT=None
START=None


def utc():return datetime.now(timezone.utc).isoformat()


def report(status,**extra):
    STATE.update(status=status,**extra)
    save_json(OUT/'progress.json',dict(STATE,model_counter=dict(MODEL_COUNTER),
              elapsed_seconds=time.monotonic()-START,utc=utc(),pid=os.getpid()))


def save_torch(path,value):
    assert not path.exists(),str(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp);os.replace(temp,path)
    return {'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path)}


def target_stream_update(h,targets):
    for row in targets:
        for k in sorted(row):
            v=row[k].contiguous()
            header=json.dumps([k,str(v.dtype),list(v.shape)],separators=(',',':')).encode()
            h.update(len(header).to_bytes(8,'little'));h.update(header);h.update(v.numpy().tobytes())


def main(arm):
    global MODEL,OPTIMIZER,OUT,START
    START=time.monotonic();OUT=ROOT/('arm_'+arm)
    OUT.mkdir(exist_ok=False)
    with (OUT/'run_once.lock').open('x') as f:f.write(str(os.getpid()))
    set_event_journal(OUT/'events.jsonl')
    STATE['arm']=arm
    report('VERIFYING_FROZEN_INPUTS')
    plan=validate_files('protocol_FROZEN_v1.json')
    protocol_sha=digest(ROOT/'protocol_FROZEN_v1.json')
    assert plan['integration']['whole_detector_forwards']==3
    assert plan['integration']['optimizer_updates']==0
    assert load(ROOT/'eval_integration_result.json')['status']=='PASS_EVAL_INTEGRATION_ONLY'
    assert arm in plan['arms'] and plan['epochs_per_arm']==12
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    train=VisDroneDataset('train');val=VisDroneDataset('val')
    assert len(train)==6471 and sum(len(r['valid']) for r in train.rows)==343204
    bind_ground_truth(val)
    MODEL=make_model(arm,19867).cuda().train()
    criterion=make_criterion().cuda()
    OPTIMIZER=make_optimizer(MODEL)
    assert all(p.requires_grad for p in MODEL.parameters())
    seed_all(20261101)
    torch.cuda.reset_peak_memory_stats()
    manifest={'arm':arm,'protocol_sha256':protocol_sha,
              'initialization_sha256':plan['initialization_sha256'],'checkpoints':[],
              'data_epoch_zero_based':True,'endpoint':'epoch12 raw','no_EMA':True}
    save_json(OUT/'run_manifest.json',manifest)
    all_metrics={}
    for e in range(12):
        train.epoch=e
        order=torch.randperm(6471,generator=torch.Generator().manual_seed(20261101+e*1009)).tolist()
        order_sha=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
        flow={'epoch':e+1,'data_epoch':e,'order_indices':order,'order_sha256':order_sha,
              'first8_microbatches':[],'target_stream_sha256':None,
              'orig_size_convention':'width,height','microbatches':1618,'images':6471,'GT':343204}
        baseline=None
        if arm!='B':
            baseline=load(ROOT/'arm_B'/('epoch%02d_flow.json'%(e+1)))
            assert baseline['order_indices']==order and baseline['order_sha256']==order_sha
        loader=DataLoader(train,batch_size=4,sampler=order,num_workers=2,pin_memory=True,
                          collate_fn=collate,drop_last=False,
                          generator=torch.Generator().manual_seed(20261101+10000000+e*1009))
        assert len(loader)==1618
        MODEL.train();OPTIMIZER.zero_grad(set_to_none=True)
        th=hashlib.sha256();seen=0;GT=0;epoch_loss=0.0;epoch_start=time.monotonic()
        report('TRAINING',epoch=e+1,microbatch_completed=0)
        for j,(images,targets) in enumerate(loader):
            target_stream_update(th,targets)
            seen+=len(targets);GT+=sum(len(t['labels']) for t in targets)
            if j<8:
                row={'microbatch':j,'image_ids':[int(t['image_id']) for t in targets],
                     'input_and_targets_sha256':input_digest(images,targets),'pre_forward_rng_sha256':rng_digest()}
                if baseline is not None:assert row==baseline['first8_microbatches'][j],('input/RNG mismatch',arm,e,j)
                flow['first8_microbatches'].append(row)
            images=images.cuda(non_blocking=True)
            targets=targets_to_device(targets,'cuda')
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=MODEL(images)
            STATE['train_forward_completions']+=1
            assert_output(outputs,len(targets),2400)
            assert outputs['num_groups']==6
            with torch.autocast(device_type='cuda',enabled=False):loss,losses=weighted_loss(criterion,outputs,targets)
            (loss/2).backward()
            loss_value=float(loss.detach());epoch_loss+=loss_value
            if (j+1)%2==0:
                for g in OPTIMIZER.param_groups:g['lr']=g['initial_lr']*lr_ratio(STATE['optimizer_updates'])
                norm=torch.nn.utils.clip_grad_norm_(MODEL.parameters(),1.0,error_if_nonfinite=True)
                STATE['optimizer_step_attempts']+=1
                event('optimizer_step_attempt',attempts=STATE['optimizer_step_attempts'],updates=STATE['optimizer_updates'])
                OPTIMIZER.step();STATE['optimizer_updates']+=1
                event('optimizer_step_returned',attempts=STATE['optimizer_step_attempts'],updates=STATE['optimizer_updates'])
                if STATE['optimizer_updates']==1:
                    active=sum(p.grad is not None for p in MODEL.parameters())
                    assert active==({'B':1898,'M':1892,'Q':1898}[arm]),active
                    STATE['active_gradient_parameter_tensors_first_update']=active
                OPTIMIZER.zero_grad(set_to_none=True)
                STATE['last_gradient_norm_before_clip']=float(norm)
                STATE['last_lr_backbone']=OPTIMIZER.param_groups[0]['lr']
                STATE['last_lr_rest']=OPTIMIZER.param_groups[1]['lr']
                peak=torch.cuda.max_memory_allocated()
                assert peak<=22*(1<<30),('allocated memory safety margin exceeded',arm,peak)
                STATE['peak_allocated_bytes']=peak
            if (j+1)%50==0 or j+1==len(loader):
                report('TRAINING',epoch=e+1,microbatch_completed=j+1,last_loss=loss_value)
            del outputs,loss,losses,images,targets
        assert seen==6471 and GT==343204
        assert STATE['optimizer_updates']==(e+1)*809
        flow['target_stream_sha256']=th.hexdigest()
        if baseline is not None:assert flow==baseline,('whole-epoch target/order mismatch',arm,e)
        save_json(OUT/('epoch%02d_flow.json'%(e+1)),flow)
        STATE['epochs_completed']=e+1
        event('training_epoch_completed',epoch=e+1,updates=STATE['optimizer_updates'])
        save_json(OUT/('epoch%02d_train.json'%(e+1)),{
            'epoch':e+1,'microbatch_mean_loss':epoch_loss/1618,'training_seconds':time.monotonic()-epoch_start,
            'counts':dict(STATE),'model_counter':dict(MODEL_COUNTER),'utc':utc()})
        if e+1 in [4,8,12]:
            report('SAVING_RAW_CHECKPOINT',epoch=e+1)
            cp=save_torch(OUT/('epoch%02d_RAW.pth'%(e+1)),{
                'model':cpu_tree(MODEL.state_dict()),'epoch':e+1,'optimizer_updates':STATE['optimizer_updates'],
                'arm':arm,'protocol_sha256':protocol_sha,'initialization_sha256':plan['initialization_sha256']})
            manifest['checkpoints'].append(cp);save_json(OUT/'run_manifest.json',manifest)
            report('VALIDATING',epoch=e+1)
            before=MODEL_COUNTER['completions']
            metrics=evaluate_model(MODEL,val,OUT,'epoch%02d_RAW'%(e+1))
            increment=MODEL_COUNTER['completions']-before
            assert increment==137
            STATE['val_forward_completions']+=increment
            all_metrics[str(e+1)]=metrics
            save_json(OUT/'metrics_by_epoch.json',all_metrics)
    assert STATE['epochs_completed']==12 and STATE['optimizer_updates']==9708
    report('SAVING_FIXED12_OPTIMIZER')
    manifest['optimizer']=save_torch(OUT/'epoch12_optimizer.pth',{
        'optimizer':cpu_tree(OPTIMIZER.state_dict()),'epoch':12,'optimizer_updates':9708,
        'CPU_RNG':torch.get_rng_state(),'CUDA_RNG':torch.cuda.get_rng_state(),
        'protocol_sha256':protocol_sha})
    report('FIXED12_LATENCY')
    image=val[0][0][None].cuda();before=MODEL_COUNTER['completions']
    timing=latency_ms(MODEL,image)
    STATE['latency_forward_completions']=MODEL_COUNTER['completions']-before
    assert STATE['latency_forward_completions']==40
    save_json(OUT/'fixed12_latency.json',timing)
    assert MODEL_COUNTER['entries']==MODEL_COUNTER['completions']==19867
    assert STATE['train_forward_completions']==19416 and STATE['val_forward_completions']==411
    assert STATE['optimizer_step_attempts']==STATE['optimizer_updates']==9708
    assert digest(ROOT/'common_initialization.pth')==plan['initialization_sha256']
    validate_files('protocol_FROZEN_v1.json')
    manifest.update(final_counts=dict(STATE),model_counter=dict(MODEL_COUNTER),metrics_fixed12=all_metrics['12'])
    save_json(OUT/'run_manifest.json',manifest)
    report('COMPLETED_FIXED12_RAW',epoch=12)
    save_json(OUT/'exit.json',{'status':'COMPLETED_FIXED12_RAW','exit_code':0,'arm':arm,
        'counts':dict(STATE),'model_counter':dict(MODEL_COUNTER),'metrics_fixed12':all_metrics['12'],
        'protocol_sha256':protocol_sha,'post_arm_queue':None,'utc':utc()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=['B','M','Q'])
    args=parser.parse_args()
    try:main(args.arm)
    except BaseException:
        trace=traceback.format_exc()
        if OUT is not None and OUT.exists():
            failure={'status':'FAILED_PRESERVED_NO_RETRY','traceback':trace,'counts':dict(STATE),
                     'model_counter':dict(MODEL_COUNTER),'post_arm_queue':None,'utc':utc()}
            save_json(OUT/'failure.json',failure)
            try:
                if MODEL is not None:
                    failure['partial_state']=save_torch(OUT/'failure_partial_state.pth',{
                        'model':cpu_tree(MODEL.state_dict()),
                        'optimizer':cpu_tree(OPTIMIZER.state_dict()) if OPTIMIZER is not None else None,
                        'counts':dict(STATE),'model_counter':dict(MODEL_COUNTER)})
            except BaseException:failure['partial_state_error']=traceback.format_exc()
            save_json(OUT/'failure.json',failure)
            save_json(OUT/'exit.json',dict(failure,exit_code=1))
        raise
