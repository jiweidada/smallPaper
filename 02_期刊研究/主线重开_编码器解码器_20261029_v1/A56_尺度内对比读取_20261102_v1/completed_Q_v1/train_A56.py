import argparse,gc,hashlib,json,os,time,traceback
import torch
from torch.utils.data import DataLoader
from runtime_A56 import ROOT,A49,ASSETS,native,components,validate_files
from training_utils_A49 import StateEMA,cpu_state,input_digest,rng_digest,assert_finite_output,lr_ratio
from evaluate import evaluate_model,latency_ms
from contrastive_read import diagnostics


def module_diagnostics(module):
    return diagnostics(module)


def train(arm):
    assert arm in ('M','Q')
    plan=validate_files('protocol_FROZEN_v1.json')
    capacity=json.loads((ROOT/'capacity_result.json').read_text())
    assert capacity['status']=='PASS'
    assert capacity['protocol_sha256']==native.digest(ROOT/'capacity_protocol_FROZEN_v1.json')==plan['capacity_protocol_sha256']
    out=ROOT/('arm_'+arm);out.mkdir(exist_ok=True)
    with (out/'run_once.lock').open('x') as f:f.write(str(os.getpid()))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    trainset=native.VisDroneDataset('train');valset=native.VisDroneDataset('val')
    assert len(trainset)==6471 and len(valset)==548
    assert sum(len(r['valid']) for r in trainset.rows)==343204
    assert sum(len(r['valid']) for r in valset.rows)==38759
    model,criterion,post,module=components(arm)
    initial=torch.load(ASSETS/'common_init_10class.pth',map_location='cpu',weights_only=True)['model']
    actual=model.state_dict()
    assert all(torch.equal(v,actual[k]) for k,v in initial.items())
    assert sum(p.numel() for p in model.parameters())==9877256
    assert all(torch.count_nonzero(b.gain)==0 for b in module)
    assert all(m.__class__ is torch.nn.BatchNorm2d for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
    extra_sha=hashlib.sha256()
    for index,branch in enumerate(module):
        for key,value in branch.state_dict().items():extra_sha.update(f'{index}/{key}'.encode());extra_sha.update(value.contiguous().numpy().tobytes())
    native.save_json(out/'initialization.json',{'common_init_sha256':native.digest(ASSETS/'common_init_10class.pth'),
       'all_native_initial_tensors_identical':True,'new_module_initialization_sha256':extra_sha.hexdigest(),
       'model_parameters':9877256,'extra_parameters':74144,'seed':native.SEED,
       'protocol_sha256':native.digest(ROOT/'protocol_FROZEN_v1.json'),'scratch_no_capacity_weights':True})
    assert extra_sha.hexdigest()==capacity['results'][0]['new_initial_tensors_sha256']
    del initial,actual
    model=model.cuda();criterion=criterion.cuda();optimizer=native.make_optimizer(model);ema=StateEMA(model)
    native.seed_all();torch.cuda.reset_peak_memory_stats()
    updates=completed=0;events=[];start=time.monotonic()
    try:
        for epoch in range(12):
            trainset.epoch=epoch
            order=torch.randperm(6471,generator=torch.Generator().manual_seed(native.SEED+epoch*1009)).tolist()
            loader=DataLoader(trainset,batch_size=8,sampler=order,num_workers=2,collate_fn=native.collate,
              pin_memory=True,drop_last=False,generator=torch.Generator().manual_seed(native.SEED+100000+epoch))
            assert len(loader)==809
            input_hashes=[];rng_hashes=[];flow=hashlib.sha256();loss_sum=0.;ts=time.monotonic()
            model.train();criterion.train();optimizer.zero_grad(set_to_none=True)
            for micro,(images,targets) in enumerate(loader):
                if micro<8:input_hashes.append(input_digest(images,targets))
                for target in targets:
                    for key in ['boxes','labels','image_id','ignore_tiles']:
                        flow.update(key.encode());flow.update(str(tuple(target[key].shape)).encode());flow.update(target[key].contiguous().numpy().tobytes())
                images=images.cuda(non_blocking=True);targets=native.targets_to_device(targets,'cuda')
                if micro<8:rng_hashes.append(rng_digest())
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=model(images,targets)
                assert_finite_output(outputs)
                with torch.autocast(device_type='cuda',enabled=False):
                    losses=criterion(native.fp32_tree(outputs),targets,epoch=epoch,step=micro,global_step=updates,epoch_step=809)
                    loss=sum(losses.values())
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite criterion loss')
                loss.backward()
                for group in optimizer.param_groups:group['lr']=group['initial_lr']*lr_ratio(updates)
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;ema.update(model)
                value=float(loss.detach());loss_sum+=value
                if micro%50==0 or micro==808:
                    progress={'status':'TRAINING','arm':arm,'epoch':epoch+1,'microbatch':micro+1,
                      'completed_epochs':completed,'optimizer_updates':updates,'full_model_forwards':native.MODEL_COUNTER['forwards'],
                      'loss':value,'grad_norm':float(grad),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                      'elapsed_seconds':time.monotonic()-start,'pid':os.getpid(),
                      'contrast_diagnostics':module_diagnostics(module)}
                    native.save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                del images,targets,outputs,losses,loss
            completed=epoch+1
            event={'epoch':completed,'optimizer_updates':updates,'mean_loss':loss_sum/809,'train_seconds':time.monotonic()-ts,
               'order_sha256':hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest(),
               'first_eight_actual_input_sha256':input_hashes,'first_eight_pre_forward_rng_sha256':rng_hashes,
               'all_batches_targets_including_ignore_tiles_sha256':flow.hexdigest(),
               'contrast_diagnostics':module_diagnostics(module)}
            prior=json.loads((A49/'arm_B'/('epoch%02d_input_flow.json'%completed)).read_text())
            for key in ['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']:
                assert prior[key]==event[key],key
            native.save_json(out/('epoch%02d_input_flow.json'%completed),event)
            if completed in (4,8,12):
                raw=cpu_state(model.state_dict())
                checkpoint={'model':raw,'ema':{'module':cpu_state(ema.state),'updates':ema.updates},
                 'arm':arm,'epoch':completed,'optimizer_updates':updates,'protocol_sha256':native.digest(ROOT/'protocol_FROZEN_v1.json'),
                 'inference':'native ECDet plus fixed A56 contrast residual; no GT, crop, teacher or re-scoring at inference'}
                if completed==12:checkpoint['optimizer']=optimizer.state_dict()
                path=out/('epoch%02d.pth'%completed);torch.save(checkpoint,path);del checkpoint
                model.load_state_dict(ema.state,strict=True)
                native.save_json(out/'progress.json',{'status':'VALIDATING','arm':arm,'epoch':completed,
                 'optimizer_updates':updates,'full_model_forwards':native.MODEL_COUNTER['forwards'],'pid':os.getpid()})
                event['EMA_validation']=evaluate_model(model,post,valset,out,'epoch%02d_EMA'%completed)
                model.load_state_dict(raw,strict=True);event['checkpoint_sha256']=native.digest(path)
                del raw;gc.collect()
            events.append(event)
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            native.save_json(out/'completed_epochs.json',{'completed_epochs':completed,'optimizer_updates':updates,'events':events})
        model.load_state_dict(ema.state,strict=True)
        single,_=native.collate([valset[0]]);latency=latency_ms(model,single.cuda())
        result={'status':'COMPLETED','arm':arm,'epochs':completed,'optimizer_updates':updates,
         'full_model_forwards':native.MODEL_COUNTER['forwards'],'elapsed_seconds':time.monotonic()-start,
         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'endpoint_metrics':events[-1]['EMA_validation'],
         'model_parameters':9877256,'extra_parameters':74144,'latency':latency,'pid':os.getpid(),
         'initialization_sha256':native.digest(ASSETS/'common_init_10class.pth'),
         'protocol_sha256':native.digest(ROOT/'protocol_FROZEN_v1.json'),'contrast_diagnostics_EMA':module_diagnostics(module)}
        assert updates==9708 and native.MODEL_COUNTER['forwards']==10159
        native.save_json(out/'result.json',result);native.save_json(out/'progress.json',result);print(json.dumps(result),flush=True)
    except BaseException as e:
        result={'status':'FAILED_STOPPED_NO_RETRY','arm':arm,'completed_epochs':completed,'optimizer_updates':updates,
          'full_model_forwards':native.MODEL_COUNTER['forwards'],'error':repr(e),'traceback':traceback.format_exc(),'pid':os.getpid()}
        native.save_json(out/'result.json',result);native.save_json(out/'progress.json',result);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['M','Q'],required=True);train(p.parse_args().arm)
