from __future__ import annotations
import argparse,gc,hashlib,json,math,os,pathlib,time,traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from runtime_common import R,SEED,MODEL_COUNTER,VisDroneDataset,collate,create_components,digest,fp32_tree,make_optimizer,save_json,seed_all,targets_to_device
from evaluate import evaluate_model,latency_ms

class StateEMA:
    def __init__(self,model):
        self.state={k:v.detach().clone()for k,v in model.state_dict().items()};self.updates=0
    @torch.no_grad()
    def update(self,model):
        self.updates+=1;decay=.9999*(1-math.exp(-self.updates/1000))
        current=model.state_dict();dest=[];src=[]
        for k,v in self.state.items():
            if v.is_floating_point() and not k.endswith('.anchors'):dest.append(v);src.append(current[k].detach())
            else:v.copy_(current[k])
        torch._foreach_mul_(dest,decay);torch._foreach_add_(dest,src,alpha=1-decay)

def cpu_state(state):return {k:v.detach().cpu().clone()for k,v in state.items()}

def input_digest(images,targets):
    h=hashlib.sha256(images.contiguous().numpy().tobytes())
    for t in targets:
        for k in sorted(t):h.update(k.encode());h.update(t[k].contiguous().numpy().tobytes())
    return h.hexdigest()

def train(arm):
    plan=json.loads((R/'execution_protocol_FROZEN.json').read_text())
    assert plan['epochs_per_group']==12 and plan['arms']==['B','S','L']
    for file,sha in plan['files_sha256'].items():assert digest(R/file)==sha,file
    assert json.loads((R/'capacity_preflight_result.json').read_text())['status']=='PASS'
    out=R/('arm_'+arm);out.mkdir(exist_ok=True)
    with (out/'run_once.lock').open('x') as f:f.write(str(os.getpid()))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    trainset=VisDroneDataset('train');valset=VisDroneDataset('val')
    assert len(trainset)==6471 and len(valset)==548
    for name,ds in [('train',trainset),('val',valset)]:
        manifest=json.loads((R/(name+'_data_manifest.json')).read_text())
        for row in manifest['rows']:
            assert digest(ds.images/row['file'])==row['image_sha256']
            assert digest(ds.anns/(pathlib.Path(row['file']).stem+'.txt'))==row['annotation_sha256']
    model,criterion,post=create_components(arm);model=model.cuda();criterion=criterion.cuda()
    optimizer=make_optimizer(model);ema=StateEMA(model)
    seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    updates=0;seen_micro=0;completed_epochs=0;t0=time.monotonic();epoch_results=[]
    microbatch=plan['microbatch'];accumulation=plan['gradient_accumulation']
    micro_per_epoch=math.ceil(len(trainset)/microbatch);updates_per_epoch=math.ceil(micro_per_epoch/accumulation);max_updates=updates_per_epoch*12
    assert updates_per_epoch==809
    initial_base={k:v for k,v in model.state_dict().items()if '.detail_reader.'not in k}
    ref=torch.load(R/'common_init_10class.pth',map_location='cpu',weights_only=True)['model']
    assert all(torch.equal(v.cpu(),ref[k])for k,v in initial_base.items())
    branch_digest=None
    if arm!='B':
        h=hashlib.sha256()
        for k,v in model.decoder.decoder.detail_reader.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
        branch_digest=h.hexdigest()
    save_json(out/'initialization.json',{'common_init_sha256':digest(R/'common_init_10class.pth'),'all_base_tensors_equal':True,'branch_initial_tensor_digest':branch_digest,'model_parameters':sum(p.numel()for p in model.parameters()),'training_seed':SEED,'protocol_sha256':digest(R/'execution_protocol_FROZEN.json')})
    del initial_base,ref
    save_json(out/'start.json',{'arm':arm,'pid':os.getpid(),'started_unix':time.time(),'max_epochs':12,'max_optimizer_updates':max_updates,'device':torch.cuda.get_device_name(),'precision':'BF16 model forward; FP32 criterion; unscaled FP32 gradient accumulation','augmentation':'deterministic image/epoch horizontal flip; 640 stretch; ImageNet normalization; no mosaic/mixup','optimizer':'AdamW backbone2.5e-5 other5e-4 wd1e-4 except norm/bn/bias0; clip0.1; 500-update linear warmup + 12-epoch cosine floor0.1','ema':'state EMA .9999*(1-exp(-updates/1000)); nonfloat/anchor buffers copied'})
    try:
        for epoch in range(12):
            trainset.epoch=epoch
            order=torch.randperm(len(trainset),generator=torch.Generator().manual_seed(SEED+epoch*1009)).tolist()
            loader=DataLoader(trainset,batch_size=microbatch,sampler=order,num_workers=2,collate_fn=collate,pin_memory=True,drop_last=False,generator=torch.Generator().manual_seed(SEED+100000+epoch))
            order_digest=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
            first_inputs=[];loss_total=0.;running=[];ts=time.monotonic()
            model.train();criterion.train();optimizer.zero_grad(set_to_none=True)
            for micro,(images,targets) in enumerate(loader):
                if micro<8:first_inputs.append(input_digest(images,targets))
                images=images.cuda(non_blocking=True);targets=targets_to_device(targets,'cuda')
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=model(images,targets)
                for key in ['pred_logits','pred_boxes','pred_corners']:
                    if not torch.isfinite(outputs[key]).all():raise FloatingPointError('Nonfinite model '+key)
                with torch.autocast(device_type='cuda',enabled=False):
                    ld=criterion(fp32_tree(outputs),targets,epoch=epoch,step=micro,global_step=seen_micro,epoch_step=micro_per_epoch)
                    loss=sum(ld.values())
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite summed loss')
                group_start=(micro//accumulation)*accumulation;divisor=min(accumulation,micro_per_epoch-group_start)
                (loss/divisor).backward();value=float(loss.detach());loss_total+=value;running.append(value);seen_micro+=1
                if (micro+1)%accumulation==0 or micro+1==len(loader):
                    ratio=(updates+1)/500 if updates<500 else .1+.9*.5*(1+math.cos(math.pi*(updates-500)/max(1,max_updates-500)))
                    for group in optimizer.param_groups:group['lr']=group['initial_lr']*ratio
                    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;ema.update(model)
                if micro%50==0 or micro+1==len(loader):
                    progress={'status':'TRAINING','arm':arm,'epoch':epoch+1,'microbatch':micro+1,'microbatches_per_epoch':len(loader),'completed_epochs':completed_epochs,'optimizer_updates':updates,'seen_microbatches':seen_micro,'full_model_forwards':MODEL_COUNTER['forwards'],'loss':value,'recent_mean_loss':float(np.mean(running[-50:])),'elapsed_seconds':time.monotonic()-t0,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'pid':os.getpid()}
                    save_json(out/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
                del outputs,ld,loss,images,targets
            completed_epochs=epoch+1
            event={'epoch':epoch+1,'optimizer_updates':updates,'mean_loss':loss_total/micro_per_epoch,'train_seconds':time.monotonic()-ts,'order_sha256':order_digest,'first_eight_actual_input_sha256':first_inputs}
            save_json(out/('epoch%02d_input_flow.json'%(epoch+1)),event)
            if (epoch+1) in [4,8,12]:
                raw=cpu_state(model.state_dict())
                checkpoint={'model':raw,'ema':{'module':cpu_state(ema.state),'updates':ema.updates},'epoch':epoch+1,'arm':arm,'optimizer_updates':updates,'protocol_sha256':digest(R/'execution_protocol_FROZEN.json')}
                if epoch+1==12:checkpoint['optimizer']=optimizer.state_dict()
                path=out/('epoch%02d.pth'%(epoch+1));torch.save(checkpoint,path)
                model.load_state_dict(ema.state,strict=True)
                save_json(out/'progress.json',{'status':'VALIDATING','arm':arm,'epoch':epoch+1,'optimizer_updates':updates,'full_model_forwards':MODEL_COUNTER['forwards'],'elapsed_seconds':time.monotonic()-t0,'pid':os.getpid()})
                event['EMA_validation']=evaluate_model(model,post,valset,out,'epoch%02d_EMA'%(epoch+1))
                model.load_state_dict(raw,strict=True)
                event['checkpoint_sha256']=digest(path)
                del raw,checkpoint;gc.collect()
            epoch_results.append(event)
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            save_json(out/'completed_epochs.json',{'arm':arm,'completed_epochs':completed_epochs,'optimizer_updates':updates,'epochs':epoch_results})
        model.load_state_dict(ema.state,strict=True)
        single,_=collate([valset[0]])
        latency=latency_ms(model,single.cuda())
        result={'status':'COMPLETED','arm':arm,'epochs':completed_epochs,'optimizer_updates':updates,'seen_microbatches':seen_micro,'full_model_forwards':MODEL_COUNTER['forwards'],'elapsed_seconds':time.monotonic()-t0,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'endpoint_metrics':epoch_results[-1]['EMA_validation'],'endpoint_checkpoint':str(out/'epoch12.pth'),'latency':latency,'pid':os.getpid()}
        assert updates==9708
        save_json(out/'result.json',result);save_json(out/'progress.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        result={'status':'FAILED','arm':arm,'completed_epochs':completed_epochs,'optimizer_updates':updates,'seen_microbatches':seen_micro,'full_model_forwards':MODEL_COUNTER['forwards'],'elapsed_seconds':time.monotonic()-t0,'error':repr(exc),'traceback':traceback.format_exc(),'pid':os.getpid()}
        save_json(out/'result.json',result);save_json(out/'progress.json',result)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['B','S','L'],required=True)
    train(parser.parse_args().arm)
