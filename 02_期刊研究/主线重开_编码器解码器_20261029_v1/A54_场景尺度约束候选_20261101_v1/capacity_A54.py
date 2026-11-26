import gc,hashlib,json,os,time,traceback
import torch
from runtime_A54 import ROOT,A49,ASSETS,native,components,validate_files
from training_utils_A49 import assert_finite_output

UPDATES=0
def state_hash(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    global UPDATES
    plan=validate_files('capacity_protocol_FROZEN_v1.json')
    with (ROOT/'capacity_once.lock').open('x') as f:f.write(str(os.getpid()))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dataset=native.VisDroneDataset('train');dataset.epoch=0
    assert len(dataset)==6471 and sum(len(r['valid']) for r in dataset.rows)==343204
    order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(native.SEED)).tolist();batches=[order[i:i+8] for i in range(0,len(order),8)]
    sizes=[len(r['valid']) for r in dataset.rows];chosen=[0]
    for fun in [lambda j:max(sizes[i] for i in batches[j]),lambda j:sum(sizes[i] for i in batches[j])]:
        chosen.append(next(j for j in sorted(range(len(batches)),key=fun,reverse=True) if j not in chosen))
    selected=[{'batch_index':j,'indices':batches[j],'maximum_GT':max(sizes[i] for i in batches[j]),'sum_GT':sum(sizes[i] for i in batches[j])} for j in chosen]
    native.save_json(ROOT/'capacity_batch_selection.json',selected)
    eval_images=native.collate([dataset[i] for i in batches[0]])[0][:4].cuda();reference=None;results=[];new_init_sha=None;start=time.monotonic()
    for arm in ['B','M','Q']:
        model,criterion,post,module=components(arm)
        assert sum(p.numel() for p in model.parameters())==(9803112 if arm=='B' else 9809322)
        original=torch.load(ASSETS/'common_init_10class.pth',map_location='cpu',weights_only=True)['model'];actual=model.state_dict()
        assert all(k in actual and torch.equal(actual[k],v) for k,v in original.items())
        del original,actual
        ih=state_hash(module) if module is not None else None
        if ih is not None:
            if new_init_sha is None:new_init_sha=ih
            assert ih==new_init_sha,'New M/Q tensors must match exactly'
        model=model.cuda();criterion=criterion.cuda();model.eval()
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):outputs=model(eval_images)
        assert_finite_output(outputs);error={}
        if reference is None:reference={k:outputs[k].clone() for k in ['pred_logits','pred_boxes']}
        else:
            error={k:float((outputs[k]-reference[k]).abs().max()) for k in reference}
            assert max(error.values())<=1e-6,error
        native.save_json(ROOT/'capacity_progress.json',{'status':'RUNNING','stage':arm+'_INITIAL_EVAL','pid':os.getpid(),'whole_model_forwards':native.MODEL_COUNTER['forwards'],'optimizer_updates':UPDATES,'initial_error':error})
        del outputs
        if arm=='B':del model,criterion,post;gc.collect();torch.cuda.empty_cache();continue
        opt=native.make_optimizer(model);model.train();criterion.train();native.seed_all();torch.cuda.reset_peak_memory_stats();loss_log=[]
        for step,j in enumerate(chosen):
            images,targets=native.collate([dataset[i] for i in batches[j]]);images=images.cuda();targets=native.targets_to_device(targets,'cuda');opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=model(images,targets)
            assert_finite_output(outputs)
            with torch.autocast(device_type='cuda',enabled=False):
                losses=criterion(native.fp32_tree(outputs),targets,epoch=0,step=step,global_step=step,epoch_step=809);loss=sum(losses.values())
            assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);opt.step();UPDATES+=1
            row={'step':step+1,'batch_index':j,'total_loss':float(loss.detach()),'scale_loss':float(losses['loss_a54_scene_scale'].detach()),'gradient_norm':float(grad),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'module':module.latest}
            loss_log.append(row);native.save_json(ROOT/'capacity_progress.json',{'status':'RUNNING','stage':arm+'_TEMPORARY_UPDATE','pid':os.getpid(),'whole_model_forwards':native.MODEL_COUNTER['forwards'],'optimizer_updates':UPDATES,**row})
            del images,targets,outputs,losses,loss
        assert module.map[-1].weight.detach().abs().max()>0,'No actual scale-head update'
        assert module.gain.detach().abs()>0,'No actual main-path gain update'
        peak=torch.cuda.max_memory_allocated();assert peak<24*(1<<30)
        results.append({'arm':arm,'initial_FP32_error':error,'new_initial_tensors_sha256':ih,'peak_allocated_bytes':peak,'updates':3,'losses':loss_log,'final_gate':float(module.gain.detach().tanh()),'head_weight_norm':float(module.map[-1].weight.detach().norm())})
        del model,criterion,post,module,opt;gc.collect();torch.cuda.empty_cache()
    result={'status':'PASS','scope':'capacity and native integration only; all states discarded; no formal training',
        'protocol_sha256':native.digest(ROOT/'capacity_protocol_FROZEN_v1.json'),'whole_model_forwards':native.MODEL_COUNTER['forwards'],'capacity_only_optimizer_updates':UPDATES,
        'new_module_initial_tensors_match':True,'all_native_initial_tensors_match':True,'batches':selected,'results':results,'seconds':time.monotonic()-start,'pid':os.getpid()}
    assert result['whole_model_forwards']==9 and UPDATES==6
    native.save_json(ROOT/'capacity_result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        native.save_json(ROOT/'capacity_result.json',{'status':'FAILED','pid':os.getpid(),'whole_model_forwards':native.MODEL_COUNTER['forwards'],'capacity_only_optimizer_updates':UPDATES,'traceback':traceback.format_exc()});raise
