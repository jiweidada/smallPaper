import gc,json,os,time,traceback
from pathlib import Path
import torch
from runtime_A51 import ROOT,native,components,validate_files
from training_utils_A49 import assert_finite_output


def main():
    plan=validate_files('capacity_protocol_FROZEN_v1.json')
    with (ROOT/'capacity_once.lock').open('x') as f:f.write(str(os.getpid()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dataset=native.VisDroneDataset('train')
    images,targets=native.collate([dataset[i] for i in range(8)])
    images=images.cuda();targets=native.targets_to_device(targets,'cuda')
    models={};reference=None
    results=[]
    for arm in ('B','M','Q'):
        model,criterion,post,module=components(arm)
        assert sum(p.numel() for p in model.parameters())==(9803112 if arm=='B' else 9876858)
        model=model.cuda();criterion=criterion.cuda();model.eval()
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            output=model(images[:4])
        if reference is None:reference={k:output[k].clone() for k in ('pred_logits','pred_boxes')}
        else:assert all(torch.equal(reference[k],output[k]) for k in reference),arm
        if arm=='B':
            del model,criterion,post,output;gc.collect();torch.cuda.empty_cache();continue
        optimizer=native.make_optimizer(model)
        model.train();criterion.train();native.seed_all()
        torch.cuda.reset_peak_memory_stats()
        losses_record=[]
        before={k:v.detach().clone() for k,v in module.state_dict().items()}
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=model(images,targets)
            assert_finite_output(outputs)
            with torch.autocast(device_type='cuda',enabled=False):
                losses=criterion(native.fp32_tree(outputs),targets,epoch=0,step=step,global_step=step,epoch_step=809)
                loss=sum(losses.values())
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
            optimizer.step()
            losses_record.append(float(loss.detach()))
        changes={k:float((v-before[k]).abs().max()) for k,v in module.state_dict().items()}
        assert all(torch.isfinite(p).all() for p in module.parameters())
        assert all(v>0 for v in changes.values()),changes
        peak=torch.cuda.max_memory_allocated()
        assert peak<23*1024**3
        results.append({'arm':arm,'optimizer_updates':3,'losses':losses_record,
              'peak_allocated_bytes':peak,'new_state_max_abs_change':changes,
              'beta':[float(x.beta.detach()) for x in module.pairs],
              'orthogonal_frame_errors':[float((x.orthogonal_frame().T@x.orthogonal_frame()-torch.eye(192,device='cuda')).abs().max()) for x in module.pairs]})
        del outputs,losses,loss,model,criterion,post,module,optimizer;gc.collect();torch.cuda.empty_cache()
    assert native.MODEL_COUNTER['forwards']==9
    result={'status':'PASS','protocol_sha256':native.digest(ROOT/'capacity_protocol_FROZEN_v1.json'),
      'initial_whole_model_predictions_bitwise_equal_B_M_Q':True,'formal_training_epochs':0,
      'whole_model_forwards':9,'capacity_only_optimizer_updates':6,'results':results,
      'no_capacity_weights_reused_for_formal_training':True}
    native.save_json(ROOT/'capacity_result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        native.save_json(ROOT/'capacity_result.json',{'status':'FAILED_STOPPED_NO_AUTORETRY',
           'error':repr(e),'traceback':traceback.format_exc(),'whole_model_forwards':native.MODEL_COUNTER['forwards']})
        raise
