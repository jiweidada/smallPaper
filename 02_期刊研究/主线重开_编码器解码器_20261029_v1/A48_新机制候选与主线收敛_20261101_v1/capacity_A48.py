import gc,json,math,os,time,traceback
import torch
from runtime_A48 import ROOT,ASSETS,native,plan_check
from instance_moment_norm import install,ordinary_inference,report_calls
from training_utils_A48 import assert_finite_output

def main():
    plan=plan_check()
    assert json.loads((ROOT/'preparation_result.json').read_text())['status']=='PASS'
    with (ROOT/'capacity_once.lock').open('x') as f:f.write(str(os.getpid()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    ds=native.VisDroneDataset('train')
    order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(native.SEED)).tolist()
    records=[];updates=0
    try:
        for arm in ['B','M','Q']:
            model,criterion,_=native.create_components('B')
            context=install(model,arm)
            model=model.cuda();criterion=criterion.cuda();optimizer=native.make_optimizer(model)
            native.seed_all();torch.cuda.reset_peak_memory_stats()
            model.train();criterion.train()
            for step in range(3):
                images,targets=native.collate([ds[i] for i in order[8*step:8*(step+1)]])
                images=images.cuda();targets=native.targets_to_device(targets,'cuda')
                if context is not None:context.set(targets,1618)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=model(images,targets)
                assert_finite_output(outputs)
                with torch.autocast(device_type='cuda',enabled=False):
                    losses=criterion(native.fp32_tree(outputs),targets,epoch=2,step=step,global_step=1618+step,epoch_step=809)
                    loss=sum(losses.values())
                assert torch.isfinite(loss)
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                if context is not None:context.clear()
                records.append({'arm':arm,'step':step,'loss':float(loss.detach()),'grad_norm':float(grad),
                                'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
                del images,targets,outputs,losses,loss
            if context is not None:
                assert all(v['weighted_training_calls']==3 for v in report_calls(model).values())
            model.eval()
            with ordinary_inference(model),torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                image,_=ds[order[0]]
                out=model(image[None].cuda())
                assert torch.isfinite(out['pred_logits']).all() and torch.isfinite(out['pred_boxes']).all()
                del out,image
            assert torch.cuda.max_memory_allocated()<23*1024**3
            del model,criterion,optimizer,context
            gc.collect();torch.cuda.empty_cache()
        result={'status':'PASS','records':records,'optimizer_updates':updates,'full_model_forwards':native.MODEL_COUNTER['forwards'],
                'weighted_mode_tested_at_full_mixture':True,'ordinary_BN_inference_no_GT':True,
                'protocol_sha256':native.digest(ROOT/'protocol_FROZEN.json'),'unix':time.time()}
        assert updates==9 and result['full_model_forwards']==12
        native.save_json(ROOT/'capacity_result.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        native.save_json(ROOT/'capacity_result.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),
                'optimizer_updates':updates,'full_model_forwards':native.MODEL_COUNTER['forwards']})
        raise

if __name__=='__main__':main()
