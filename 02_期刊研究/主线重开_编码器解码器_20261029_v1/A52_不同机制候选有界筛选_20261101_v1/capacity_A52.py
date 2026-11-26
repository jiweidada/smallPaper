import gc,json,os,time,traceback
import torch
from runtime_A52 import ROOT,native,components,validate_files
from training_utils_A49 import assert_finite_output
from distribution_relation import physical_marginals,pair_features

UPDATES=0
def main():
    global UPDATES
    plan=validate_files('capacity_protocol_FROZEN_v1.json')
    with (ROOT/'capacity_once.lock').open('x') as f:f.write(str(os.getpid()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dataset=native.VisDroneDataset('train');dataset.epoch=0
    order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(native.SEED)).tolist()
    batches=[order[k:k+8] for k in range(0,len(order),8)]
    sizes=[len(r['valid']) for r in dataset.rows]
    chosen=[0]
    for key in [lambda j:max(sizes[k] for k in batches[j]),lambda j:sum(sizes[k] for k in batches[j])]:
        chosen.append(next(j for j in sorted(range(len(batches)),key=key,reverse=True) if j not in chosen))
    selected=[{'batch_index_zero_based':j,'dataset_indices':batches[j],
               'maximum_GT':max(sizes[k] for k in batches[j]),'sum_GT':sum(sizes[k] for k in batches[j])} for j in chosen]
    native.save_json(ROOT/'capacity_batch_selection.json',{'seed':native.SEED,'selection':selected,
      'scope':'Three actual epoch0 batches: first, largest per-image GT, largest distinct batch-total GT; no detector call used for selection.'})
    images,targets=native.collate([dataset[i] for i in batches[0]])
    eval_images=images[:4].cuda()
    reference=None;results=[];start=time.monotonic()
    for arm in ('B','M','Q'):
        model,criterion,post,module=components(arm)
        assert sum(p.numel() for p in model.parameters())==(9803112 if arm=='B' else 9809592)
        model=model.cuda();criterion=criterion.cuda();model.eval()
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            output=model(eval_images)
        assert_finite_output(output)
        err={}
        if reference is None:reference={k:output[k].clone() for k in ('pred_logits','pred_boxes')}
        else:
            err={k:float((reference[k]-output[k]).abs().max()) for k in reference}
            assert max(err.values())<=plan['initial_FP32_prediction_absolute_tolerance'],err
        native.save_json(ROOT/'capacity_progress.json',{'status':'RUNNING','stage':arm+'_INITIAL_EVAL','pid':os.getpid(),
             'whole_model_forwards':native.MODEL_COUNTER['forwards'],'capacity_only_optimizer_updates':UPDATES})
        if arm=='B':
            del model,criterion,post,output;gc.collect();torch.cuda.empty_cache();continue
        # Capacity-only observer of the feature basis, without changing the returned mask.
        basis_stats=[];base_mask=module.mask
        def observed_mask(index,context,original_mask,target):
            if len(basis_stats)<3:
                with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
                    corners,refs,project,boxes=context
                    pp,xx=physical_marginals(corners[:1,-300:],refs[:1,-300:],project,module.reg_scale)
                    m=pair_features(pp,xx,boxes[:1,-300:],'M')
                    q=pair_features(pp,xx,boxes[:1,-300:],'Q')
                    diff=q[...,4:]-m[...,4:]
                    basis_stats.append({'decoder_layer_1based':index+2,'categorical_minus_Gaussian_kernel_RMS':float(diff.square().mean().sqrt()),
                          'maximum_absolute_difference':float(diff.abs().max())})
            return base_mask(index,context,original_mask,target)
        module.mask=observed_mask
        optimizer=native.make_optimizer(model)
        model.train();criterion.train();native.seed_all()
        torch.cuda.reset_peak_memory_stats()
        before={k:v.detach().clone() for k,v in module.state_dict().items()}
        losses_record=[]
        for step,indices in enumerate([batches[j] for j in chosen]):
            imgs,tgts=native.collate([dataset[i] for i in indices])
            imgs=imgs.cuda();tgts=native.targets_to_device(tgts,'cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=model(imgs,tgts)
            assert_finite_output(outputs)
            with torch.autocast(device_type='cuda',enabled=False):
                losses=criterion(native.fp32_tree(outputs),tgts,epoch=0,step=step,global_step=step,epoch_step=809)
                loss=sum(losses.values())
            assert torch.isfinite(loss)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
            optimizer.step();UPDATES+=1
            losses_record.append(float(loss.detach()))
            native.save_json(ROOT/'capacity_progress.json',{'status':'RUNNING','stage':arm+'_TEMPORARY_UPDATE','step':step+1,
                 'whole_model_forwards':native.MODEL_COUNTER['forwards'],'capacity_only_optimizer_updates':UPDATES,
                 'loss':float(loss),'gradient_norm':float(grad),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'pid':os.getpid()})
            del imgs,tgts,outputs,losses,loss
        changes={k:float((v-before[k]).abs().max()) for k,v in module.state_dict().items()}
        assert all(torch.isfinite(p).all() for p in module.parameters()) and all(v>0 for v in changes.values()),changes
        assert len(basis_stats)==3 and all(s['maximum_absolute_difference']>1e-7 for s in basis_stats)
        peak=torch.cuda.max_memory_allocated();assert peak<23*1024**3
        results.append({'arm':arm,'optimizer_updates':3,'losses':losses_record,'peak_allocated_bytes':peak,
          'new_state_max_abs_change':changes,'initial_FP32_prediction_max_absolute_error':err,
          'real_native_distribution_basis_statistics':basis_stats,
          'patched_forward_sha256':module.patched_forward_sha256})
        del model,criterion,post,module,optimizer,output,base_mask;gc.collect();torch.cuda.empty_cache()
    assert native.MODEL_COUNTER['forwards']==9 and UPDATES==6
    result={'status':'PASS','protocol_sha256':native.digest(ROOT/'capacity_protocol_FROZEN_v1.json'),
       'formal_training_epochs':0,'whole_model_forwards':9,'capacity_only_optimizer_updates':6,'results':results,
       'no_capacity_weights_reused_for_formal_training':True,'elapsed_seconds':time.monotonic()-start,
       'scope':'Full-detector interface, finite training, zero-init tolerance, actual dense batch memory and branch activity; no AP efficacy result.'}
    native.save_json(ROOT/'capacity_result.json',result);native.save_json(ROOT/'capacity_progress.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        result={'status':'FAILED_STOPPED_NO_AUTORETRY','error':repr(e),'traceback':traceback.format_exc(),
          'whole_model_forwards':native.MODEL_COUNTER['forwards'],'capacity_only_optimizer_updates':UPDATES,'formal_training_epochs':0}
        native.save_json(ROOT/'capacity_result.json',result);native.save_json(ROOT/'capacity_progress.json',result);raise

