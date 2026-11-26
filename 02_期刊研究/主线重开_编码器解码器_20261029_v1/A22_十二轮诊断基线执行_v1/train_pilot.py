"""One fixed12epoch raw-checkpoint diagnostic baseline; no validation or followup job."""
from pathlib import Path
import hashlib,itertools,json,math,os,random,signal,sys,time,traceback
import numpy as np
from runtime_common import ROOT,TRAIN,host,torch,rd,sha,utc,event,atomic_json,verify_execution_manifest,process_identity,finite_tensors,tensor_hash
from data_pipeline import TrainDataset,collate

C={'forward_entries':0,'forward_completed':0,'optimizer_attempts':0,'optimizer_step_calls':0,
   'actual_updates':0,'AMP_skips':0,'consecutive_AMP_skips':0,'epochs_completed':0,'epoch':0,'microbatch':0}
START=time.monotonic();CURRENT=[]
def parameter_group_hashes(model):
    digests={k:hashlib.sha256() for k in ['backbone','encoder','decoder']}
    for name,p in model.named_parameters():
        v=p.detach().cpu().contiguous();h=digests[name.split('.')[0]]
        h.update(name.encode()+b'\0'+v.numpy().tobytes())
    return {k:h.hexdigest() for k,h in digests.items()}
def warmup_factor(actual_updates):return .001+.999*min(actual_updates/999.0,1.0)
def optimizer_for(model):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        lr=2e-5 if name.startswith('backbone.') else 2e-4
        decay=0.0 if p.ndim<=1 or name.endswith('.bias') else .000065
        groups.setdefault((lr,decay),[]).append(p)
    opt=torch.optim.AdamW([{'params':v,'lr':lr,'base_lr':lr,'weight_decay':wd} for (lr,wd),v in sorted(groups.items())],betas=(.9,.999),foreach=False)
    assert sum(len(g['params']) for g in opt.param_groups)==762
    assert sum(p.numel() for g in opt.param_groups for p in g['params'])==38708393
    return opt
def progress(status,**extra):
    atomic_json(ROOT/'training_progress.json',{'utc':utc(),'status':status,**C,'elapsed_seconds':time.monotonic()-START,
      'counts_are_live_not_terminal':status not in ['COMPLETED_12_EPOCH_PILOT','STOPPED_WITH_EVIDENCE_NO_AUTOMATIC_RESUME'],
      'validation_calls':0,'mainline_validated':False,**extra})
def check_stop():
    if (ROOT/'STOP_REQUESTED').exists():raise RuntimeError('Explicit task stop file observed; no automatic resume')
    if time.monotonic()-START>=48*3600:raise RuntimeError('Prespecified48hour wall-clock cap reached')
def signal_stop(signum,frame):raise RuntimeError('Received stop signal '+str(signum))
def enter(model,args):
    assert C['forward_entries']<38832,'Forward cap exhausted'
    C['forward_entries']+=1
    event('forward_events.jsonl',{'event':'DETECTOR_FORWARD_ENTER','index':C['forward_entries'],
      'epoch':C['epoch'],'microbatch':C['microbatch'],'images':CURRENT})
def verify_training_assets():
    m=rd(ROOT/'train_asset_manifest.json');aggregate=hashlib.sha256();total=0
    assert len(m['files'])==6471
    for r in m['files']:
        im=TRAIN/'images'/r['image'];an=TRAIN/'annotations'/r['annotation']
        ih,ah=sha(im),sha(an);assert ih==r['image_sha256'] and ah==r['annotation_sha256'],r['image']
        aggregate.update(r['image'].encode()+b'\0'+bytes.fromhex(ih)+bytes.fromhex(ah));total+=im.stat().st_size
    assert aggregate.hexdigest()==m['aggregate_sha256'] and total==m['image_bytes']
    atomic_json(ROOT/'training_start_data_integrity.json',{'status':'ALL6471_MATCH_AT_RUN_START','aggregate_sha256':aggregate.hexdigest(),'image_bytes':total,'utc':utc()})
def main():
    global CURRENT
    manifest=verify_execution_manifest();assert manifest['new_training_complete_forward_cap']==38832
    fd=os.open(str(ROOT/'training_one_shot.lock'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.write(fd,json.dumps({'utc':utc(),**process_identity(os.getpid()),'manifest_sha256':sha(ROOT/'execution_manifest_FROZEN.json')}).encode());os.close(fd)
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    progress('INITIALIZING');verify_training_assets();torch.set_num_threads(4)
    model,criterion,_,receipt=host.build_host();assert receipt['host_state_digest']==manifest['initial_state_digest']
    assert model.decoder.min_num_select==250 and model.decoder.max_num_select==500
    assert finite_tensors(model.parameters())
    initial_parameter_hashes=parameter_group_hashes(model)
    atomic_json(ROOT/'training_initial_host_receipt.json',receipt)
    free,total=torch.cuda.mem_get_info();assert free>=16*1024**3,free
    model=model.cuda();criterion=criterion.cuda();model.train();criterion.train()
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Native grid_sample backward is not claimed bitwise deterministic.
    torch.use_deterministic_algorithms(False)
    random.seed(42301);np.random.seed(42301);torch.manual_seed(42301);torch.cuda.manual_seed_all(42301)
    optimizer=optimizer_for(model);original_step=optimizer.step
    def counted_step(*args,**kwargs):
        assert C['optimizer_step_calls']<9708
        C['optimizer_step_calls']+=1
        event('optimizer_events.jsonl',{'event':'OPTIMIZER_STEP_ENTER',**C})
        value=original_step(*args,**kwargs);C['actual_updates']+=1
        event('optimizer_events.jsonl',{'event':'OPTIMIZER_STEP_COMPLETE',**C})
        return value
    optimizer.step=counted_step
    scaler=torch.amp.GradScaler('cuda',init_scale=128.0,growth_interval=2000)
    model.register_forward_pre_hook(enter)
    (ROOT/'checkpoints').mkdir(exist_ok=False)
    event('training_events.jsonl',{'event':'TRAINING_INITIALIZED','manifest_sha256':sha(ROOT/'execution_manifest_FROZEN.json'),
      'parameters':receipt['host_parameters'],'trainable_parameters':38708393,'optimizer_groups':[
      {'base_lr':g['base_lr'],'weight_decay':g['weight_decay'],'parameters':sum(p.numel() for p in g['params'])} for g in optimizer.param_groups],
      'GPU':torch.cuda.get_device_name(0),'validation_calls':0},True)
    for epoch in range(1,13):
        C['epoch']=epoch;C['microbatch']=0;check_stop()
        order=torch.randperm(6471,generator=torch.Generator().manual_seed(42301+epoch)).tolist()
        order_sha=hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest()
        dataset=TrainDataset(epoch)
        loader=torch.utils.data.DataLoader(dataset,batch_size=2,sampler=order,drop_last=False,num_workers=4,
          pin_memory=True,persistent_workers=False,multiprocessing_context='spawn',collate_fn=collate,prefetch_factor=2)
        iterator=iter(loader);epoch_loss=0.0;epoch_images=0
        event('training_events.jsonl',{'event':'EPOCH_START','epoch':epoch,'order_sha256':order_sha,**C},True)
        while True:
            group=list(itertools.islice(iterator,4))
            if not group:break
            check_stop();assert C['optimizer_attempts']<9708
            C['optimizer_attempts']+=1;group_images=sum(len(batch[1]) for batch in group)
            factor=warmup_factor(C['actual_updates'])
            for g in optimizer.param_groups:g['lr']=g['base_lr']*factor
            optimizer.zero_grad(set_to_none=True);loss_group=0.0;guard_total=0
            torch.cuda.reset_peak_memory_stats()
            for images,targets,meta in group:
                check_stop();C['microbatch']+=1;CURRENT=meta
                if C['forward_entries']<12:
                    event('input_flow_first12.jsonl',{'epoch':epoch,'microbatch':C['microbatch'],'images':meta,
                       'input_sha256':tensor_hash(images),'box_hashes':[tensor_hash(t['boxes']) for t in targets],
                       'label_hashes':[tensor_hash(t['labels']) for t in targets]})
                bs=len(targets);images=images.cuda(non_blocking=True)
                targets=[{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets]
                with torch.autocast('cuda',dtype=torch.float16,cache_enabled=True):outputs=model(images,targets=targets)
                C['forward_completed']+=1
                assert finite_tensors([outputs['pred_logits'],outputs['pred_boxes']]),'Nonfinite detector outputs'
                original_nan_to_num=torch.nan_to_num;guarded=[]
                def raw_loss_guard(value,*args,**kwargs):
                    assert bool(torch.isfinite(value).all()),'Nonfinite native raw loss before sanitation'
                    guarded.append(1);return original_nan_to_num(value,*args,**kwargs)
                torch.nan_to_num=raw_loss_guard
                try:
                    with torch.autocast('cuda',enabled=False):losses=criterion(outputs,targets,epoch=epoch-1)
                finally:torch.nan_to_num=original_nan_to_num
                loss=sum(losses.values());assert bool(torch.isfinite(loss)), 'Nonfinite loss'
                value=float(loss.detach());loss_group+=value*bs/group_images;epoch_loss+=value*bs;epoch_images+=bs;guard_total+=len(guarded)
                scaler.scale(loss*(bs/group_images)).backward()
                del images,targets,outputs,losses,loss
            scaler.unscale_(optimizer)
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            finite=finite_tensors(grads);norm=None
            if finite:norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True))
            before=C['actual_updates'];scaler.step(optimizer);scaler.update()
            if C['actual_updates']==before:
                assert not finite,'Finite gradients but optimizer unexpectedly skipped'
                C['AMP_skips']+=1;C['consecutive_AMP_skips']+=1
                event('training_events.jsonl',{'event':'AMP_OVERFLOW_SKIP',**C,'scale':scaler.get_scale()},True)
            else:
                assert finite and C['actual_updates']==before+1
                assert finite_tensors(model.parameters()),'Nonfinite model parameter after update'
                C['consecutive_AMP_skips']=0
                if C['actual_updates']==1:
                    after=parameter_group_hashes(model)
                    changed={k:after[k]!=initial_parameter_hashes[k] for k in after}
                    assert any(changed.values()),'Real optimizer returned but no model parameter changed'
                    atomic_json(ROOT/'initial_to_first_update_parameter_changes.json',{'utc':utc(),
                      'before':initial_parameter_hashes,'after':after,'groups_changed':changed,
                      'actual_update_index':1,'no_extra_forward_for_check':True})
            assert C['consecutive_AMP_skips']<3 and C['AMP_skips']<32,'Prespecified AMP skip limit'
            optimizer.zero_grad(set_to_none=True)
            info={'loss_group':loss_group,'raw_loss_guard_count':guard_total,'grad_norm_before_clip':norm,
                 'scale':scaler.get_scale(),'learning_rates':[g['lr'] for g in optimizer.param_groups],
                 'allocated_peak_MiB':torch.cuda.max_memory_allocated()/1048576,'reserved_peak_MiB':torch.cuda.max_memory_reserved()/1048576,
                 'group_images':group_images,'warmup_factor':factor}
            event('training_events.jsonl',{'event':'GROUP_COMPLETE',**C,**info})
            if C['optimizer_attempts']<=8 or C['optimizer_attempts']%10==0:
                progress('RUNNING',**info)
            del group,grads
        assert epoch_images==6471 and C['microbatch']==3236
        assert C['forward_entries']==C['forward_completed']==epoch*3236
        assert C['optimizer_attempts']==epoch*809
        assert C['actual_updates']+C['AMP_skips']==C['optimizer_attempts']
        verify_execution_manifest()
        assert finite_tensors(v for v in model.state_dict().values() if v.is_floating_point() or v.is_complex()),'Nonfinite epoch-end model state'
        C['epochs_completed']=epoch
        checkpoint={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'completed_epoch':epoch,
            'counters':dict(C),'execution_manifest_sha256':sha(ROOT/'execution_manifest_FROZEN.json'),
            'scientific_design_sha256':manifest['scientific_design_sha256'],'order_sha256':order_sha,
            'no_validation_selection':True}
        path=ROOT/'checkpoints'/f'epoch_{epoch:03d}.pth';assert not path.exists()
        temporary=path.with_suffix('.tmp');torch.save(checkpoint,temporary);os.replace(temporary,path)
        checkpoint_sha=sha(path);del checkpoint
        row={'event':'EPOCH_COMPLETE',**C,'checkpoint':str(path),'checkpoint_sha256':checkpoint_sha,
             'epoch_mean_loss_per_image':epoch_loss/6471,'elapsed_seconds':time.monotonic()-START}
        event('training_events.jsonl',row,True);atomic_json(ROOT/'latest_epoch.json',row)
        progress('RUNNING',last_completed_checkpoint=str(path),checkpoint_sha256=checkpoint_sha)
        del iterator,loader,dataset
    assert C['forward_entries']==38832 and C['optimizer_attempts']==9708
    result={'utc':utc(),'status':'COMPLETED_12_EPOCH_PILOT',**C,'elapsed_seconds':time.monotonic()-START,
        'fixed_checkpoint':str(path),'fixed_checkpoint_sha256':checkpoint_sha,
        'execution_manifest_sha256':sha(ROOT/'execution_manifest_FROZEN.json'),
        'validation_calls':0,'followup_jobs_launched':False,'mainline_validated':False,
        'scope':'Raw fixed12 pilot baseline only; no accuracy/convergence/architecture-defect conclusion'}
    atomic_json(ROOT/'training_result.json',result);progress(result['status'],fixed_checkpoint_sha256=checkpoint_sha)
    event('training_events.jsonl',{'event':'TRAINING_TERMINAL',**result},True)

if __name__=='__main__':
    assert sys.argv[1:]==['--run'],'Only explicitly frozen --run is supported; no resume'
    try:main()
    except BaseException:
        result={'utc':utc(),'status':'STOPPED_WITH_EVIDENCE_NO_AUTOMATIC_RESUME',**C,
          'elapsed_seconds':time.monotonic()-START,'traceback':traceback.format_exc(),'validation_calls':0,
          'last_step_may_be_partial':C['optimizer_step_calls']!=C['actual_updates'],'mainline_validated':False}
        atomic_json(ROOT/'training_result.json',result);progress(result['status'])
        event('training_events.jsonl',{'event':'TRAINING_TERMINAL_FAILURE',**result},True);raise
