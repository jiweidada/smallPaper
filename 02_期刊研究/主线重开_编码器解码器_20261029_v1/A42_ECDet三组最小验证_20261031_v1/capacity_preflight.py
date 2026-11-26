from __future__ import annotations
import gc,hashlib,json,pathlib,time
import torch
from runtime_common import R,SEED,MODEL_COUNTER,VisDroneDataset,collate,create_components,data_manifest,digest,fp32_tree,make_optimizer,save_json,seed_all,targets_to_device
from evaluate import latency_ms

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
assert torch.cuda.is_bf16_supported()
train=VisDroneDataset('train');val=VisDroneDataset('val')
assert len(train)==6471 and len(val)==548,(len(train),len(val))
save_json(R/'train_data_manifest.json',data_manifest(train))
save_json(R/'val_data_manifest.json',data_manifest(val))
dense=sorted(range(len(train)),key=lambda i:(-len(train.rows[i]['valid']),train.rows[i]['file']))[:2]
batch,targets=collate([train[i] for i in dense]);batch=batch.cuda();targets=targets_to_device(targets,'cuda')
single,_=collate([train[0]]);single=single.cuda()
models={arm:create_components(arm)[0].cuda().eval() for arm in ['B','S','L']}
inputs={}
handles=[]
for arm,model in models.items():
    def capture(module,args,out,key=arm):inputs[key]=args[0].detach().clone()
    handles.append(model.decoder.dec_score_head[3].register_forward_hook(capture))
with torch.inference_mode():
    outputs={arm:model(single) for arm,model in models.items()}
for arm in ['S','L']:
    assert torch.equal(outputs['B']['pred_logits'],outputs[arm]['pred_logits']),arm
    assert torch.equal(outputs['B']['pred_boxes'],outputs[arm]['pred_boxes']),arm
zero_equal=True
with torch.no_grad():
    for arm in ['S','L']:models[arm].decoder.decoder.detail_reader.mlp[-1].bias.fill_(0.05)
with torch.inference_mode():
    altered={arm:model(single) for arm,model in models.items()}
assert torch.equal(inputs['B'],inputs['L'])
assert not torch.equal(inputs['B'],inputs['S'])
assert not torch.equal(altered['B']['pred_boxes'],altered['L']['pred_boxes'])
for h in handles:h.remove()
counts={a:sum(p.numel() for p in m.parameters())for a,m in models.items()}
assert counts['S']==counts['L']
del outputs,altered,inputs,models
gc.collect();torch.cuda.empty_cache()

capacity=[]
for arm in ['L','S']:
    model,criterion,post=create_components(arm)
    model=model.cuda().train();criterion=criterion.cuda()
    opt=make_optimizer(model);seed_all(SEED+50)
    torch.cuda.reset_peak_memory_stats();t0=time.monotonic();losses=[];grads=[]
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):out=model(batch,targets)
        for key in ['pred_logits','pred_boxes','pred_corners']:assert torch.isfinite(out[key]).all(),key
        with torch.autocast(device_type='cuda',enabled=False):ld=criterion(fp32_tree(out),targets,epoch=0,step=step,global_step=step,epoch_step=3236)
        loss=sum(ld.values());assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
        reader=model.decoder.decoder.detail_reader
        grads.append({'step':step,'first_weight_grad_sum':float(reader.mlp[0].weight.grad.abs().sum()),'last_weight_grad_sum':float(reader.mlp[-1].weight.grad.abs().sum())})
        opt.step();losses.append(float(loss))
    torch.cuda.synchronize()
    assert grads[0]['last_weight_grad_sum']>0 and grads[-1]['first_weight_grad_sum']>0
    peak=torch.cuda.max_memory_allocated();assert peak<22*1024**3
    capacity.append({'arm':arm,'dense_files':[train.rows[i]['file']for i in dense],'dense_GT':[len(train.rows[i]['valid'])for i in dense],'microbatch':2,'optimizer_updates':4,'losses':losses,'reader_gradient_checks':grads,'peak_allocated_bytes':peak,'seconds':time.monotonic()-t0})
    del out,ld,loss,model,criterion,post,opt,reader
    gc.collect();torch.cuda.empty_cache()

latencies={}
for arm in ['B','S','L']:
    model,criterion,post=create_components(arm);model=model.cuda().eval()
    latencies[arm]=latency_ms(model,single)
    del model,criterion,post;gc.collect();torch.cuda.empty_cache()

result={'status':'PASS','zero_output_equals_original_bitwise':zero_equal,'nonzero_L_keeps_classification_Linear_input_bitwise':True,'nonzero_S_changes_classification_input':True,'nonzero_L_changes_boxes':True,'parameter_counts':counts,'additional_reader_parameters':counts['L']-counts['B'],'capacity':capacity,'latencies':latencies,'train_images':len(train),'val_images':len(val),'train_empty_valid_GT_images':sum(not r['valid']for r in train.rows),'all_training_common_init_reset':True,'formal_training_epochs':0,'capacity_optimizer_updates':8,'full_model_calls_observed':MODEL_COUNTER['forwards'],'common_initialization_sha256':digest(R/'common_init_10class.pth')}
save_json(R/'capacity_preflight_result.json',result)
print(json.dumps({'status':'PASS','parameters':counts,'peak_GiB':[v['peak_allocated_bytes']/1024**3 for v in capacity],'dense_GT':capacity[0]['dense_GT'],'latency_ms':{a:v['median_ms']for a,v in latencies.items()},'capacity_updates':8}),flush=True)
