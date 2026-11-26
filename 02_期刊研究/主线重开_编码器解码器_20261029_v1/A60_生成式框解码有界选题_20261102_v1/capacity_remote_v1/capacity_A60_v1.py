"""One bounded B/M/Q real-data capacity check; discarded weights, no AP test."""
from pathlib import Path
import gc
import json
import os
import time
import traceback
import torch
from runtime_A60_v1 import (EndpointDetector,backend,load_item,prepare_training,mode_inputs,
    loss_targets,criterion,optimizer,sha,save_json,tensor_sha)
from box_progress_conditioning_v1 import sample_prior,single_step_inputs

ROOT=Path('/root/uav_a60_20261102_v1')
OUT=ROOT/'capacity_v1'

def main():
    scope=json.loads((ROOT/'capacity_scope_FROZEN_v1.json').read_text())
    for n,h in scope['source_sha256'].items():assert sha(ROOT/n)==h,n
    init=ROOT/'common_init_A60_v1.pth';assert sha(init)==scope['initialization_sha256']
    assert not OUT.exists();OUT.mkdir()
    save_json(OUT/'run_once.json',dict(maximum_detector_forwards=9,maximum_updates=6,formal_epochs=0))
    backend();start=time.monotonic();counts=dict(started=0,completed=0,updates=0)
    event=(OUT/'actual_events.jsonl').open('x',buffering=1)
    def log(value):event.write(json.dumps(value)+'\n')
    def pre(module,args):
        assert counts['started']<9
        counts['started']+=1;log(dict(event='forward_start',count=counts['started']))
    def post(module,args,result):
        counts['completed']+=1;log(dict(event='forward_complete',count=counts['completed']))
    data=[load_item(n) for n in scope['real_train_images']]
    prepared=[prepare_training([t],20261102+i,progress=.5) for i,(_,t) in enumerate(data)]
    initial=torch.load(init,map_location='cpu',weights_only=True)['model'];rows=[];reference={}
    for arm in ['B','M','Q']:
        model=EndpointDetector();model.load_state_dict(initial,strict=True);model=model.cuda().train()
        model.register_forward_pre_hook(pre);model.register_forward_hook(post)
        opt=optimizer(model);crit=criterion().cuda()
        torch.cuda.reset_peak_memory_stats();arm_start=time.monotonic()
        for i,((image,target),prepared_input) in enumerate(zip(data,prepared)):
            opt.zero_grad(set_to_none=True)
            noisy,condition,noisy_cpu,alpha=mode_inputs(arm,prepared_input,'cuda')
            image_batch=image[None].cuda()
            flow=dict(image=tensor_sha(image[None]),GT=tensor_sha(target['boxes']),labels=tensor_sha(target['labels']),
                prior=tensor_sha(prepared_input['prior']),u=tensor_sha(prepared_input['progress']),
                noisy=tensor_sha(noisy_cpu),time=tensor_sha(condition))
            if arm=='B':reference[i]=flow
            else:
                for k in ['image','GT','labels','prior','u']:assert flow[k]==reference[i][k]
                if arm=='M':reference[(i,'M_noisy')]=flow['noisy']
                else:assert flow['noisy']==reference[(i,'M_noisy')]
            output=model(image_batch,noisy,condition)
            losses=crit(output,loss_targets([target],'cuda'))
            total=sum(v*crit.weight_dict[k] for k,v in losses.items() if k in crit.weight_dict)
            assert torch.isfinite(total)
            total.backward()
            time_grad=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith('head.time_mlp.') and p.grad is not None)
            assert time_grad>0
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();counts['updates']+=1;log(dict(event='optimizer_returned',count=counts['updates'],arm=arm,case=i))
            assert all(torch.isfinite(p).all() for p in model.parameters())
            rows.append(dict(arm=arm,case=i,GT=len(target['labels']),queries=prepared_input['queries'],
                loss=float(total.detach()),gradient_norm=float(norm),time_MLP_gradient_L1=time_grad,input_hashes=flow))
        model.eval();eval_image=torch.stack([v[0] for v in data]).cuda()
        gen=torch.Generator(device='cpu');gen.manual_seed(20261102)
        p=sample_prior(2,300,'cpu',generator=gen);p[...,2:].clamp_min_(1e-4)
        box,t=single_step_inputs(p)
        with torch.inference_mode():o=model(eval_image,box.cuda(),t.cuda())
        assert o['pred_logits'].shape==(2,300,10) and o['pred_boxes'].shape==(2,300,4)
        assert torch.isfinite(o['pred_logits']).all() and torch.isfinite(o['pred_boxes']).all()
        torch.cuda.synchronize()
        rows.append(dict(arm=arm,GT_free_inference_batch=2,queries=300,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-arm_start))
        del o,model,opt,crit,output,losses,total,eval_image,image_batch,noisy,condition
        gc.collect();torch.cuda.empty_cache()
    assert counts==dict(started=9,completed=9,updates=6)
    result=dict(status='PASS_REAL_GPU_CAPACITY_AND_SHARED_INPUTS',counts=counts,rows=rows,
        formal_training_epochs=0,capacity_weights_reused=False,checkpoint_loads=1,GPU_detector_constructors=3,
        source_sha256=scope['source_sha256'],initialization_sha256=scope['initialization_sha256'],
        torch_version=torch.__version__,device=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,
        limits='Real finite-loss/gradient/memory/interface check. No AP test, scientific update, efficacy or novelty claim.')
    save_json(OUT/'result.json',result);save_json(OUT/'exit.json',dict(exit=0))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists() and not (OUT/'exit.json').exists():save_json(OUT/'exit.json',dict(exit=1,traceback=traceback.format_exc(),no_retry=True))
        traceback.print_exc();raise
