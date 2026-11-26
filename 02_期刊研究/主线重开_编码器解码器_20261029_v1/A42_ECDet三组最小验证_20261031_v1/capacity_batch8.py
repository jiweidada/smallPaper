import gc,json,time,torch
from runtime_common import R,MODEL_COUNTER,VisDroneDataset,collate,create_components,fp32_tree,make_optimizer,save_json,targets_to_device
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
ds=VisDroneDataset('train');idx=sorted(range(len(ds)),key=lambda i:(-len(ds.rows[i]['valid']),ds.rows[i]['file']))[:8]
batch,targets=collate([ds[i]for i in idx]);batch=batch.cuda();targets=targets_to_device(targets,'cuda')
rows=[];updates=0
try:
    for arm in ['L','S']:
        model,criterion,post=create_components(arm);model=model.cuda().train();criterion=criterion.cuda();opt=make_optimizer(model)
        torch.cuda.reset_peak_memory_stats();t0=time.monotonic();losses=[]
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch,targets)
            with torch.autocast('cuda',enabled=False):ld=criterion(fp32_tree(out),targets,epoch=0,step=step,global_step=step,epoch_step=809)
            loss=sum(ld.values());assert torch.isfinite(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
            opt.step();updates+=1;losses.append(float(loss))
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
        assert peak<22*1024**3
        rows.append({'arm':arm,'microbatch':8,'peak_allocated_bytes':peak,'seconds':time.monotonic()-t0,'updates':2,'losses':losses})
        del model,criterion,post,opt,out,ld,loss;gc.collect();torch.cuda.empty_cache()
    result={'status':'PASS','selected_microbatch':8,'selected_accumulation':1,'dense_GT':[len(ds.rows[i]['valid'])for i in idx],'rows':rows,'capacity_optimizer_updates':updates,'full_model_forwards':MODEL_COUNTER['forwards'],'training_starts_from_common_init':True}
except torch.cuda.OutOfMemoryError as exc:
    result={'status':'OOM_FALLBACK_TO_PREVIOUSLY_VERIFIED_BATCH2','selected_microbatch':2,'selected_accumulation':4,'rows':rows,'capacity_optimizer_updates':updates,'full_model_forwards':MODEL_COUNTER['forwards'],'error':str(exc),'no_training_resume':True}
save_json(R/'batch_selection_result.json',result)
print(json.dumps(result),flush=True)
