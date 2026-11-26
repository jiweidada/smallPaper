"""One six-forward/six-discarded-step native-code capacity check, no training queue."""
from pathlib import Path
import gc
import hashlib
import json
import os
import sys
import time
import traceback

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'host_compat_v2'))
COUNTS={'whole_model_forward_entries':0,'whole_model_forward_completions':0,
        'capacity_only_optimizer_updates':0,'decoder_layer_entries_including_checkpoint_recomputation':0,
        'formal_training_epochs':0}


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    temp=P/(name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temp,P/name)


def main():
    assert P==Path('/root/uav_a57_20261102_v1')
    plan=json.loads((P/'capacity_scope_FROZEN_v1.json').read_text())
    assert all(sha(P/n)==h for n,h in plan['files_sha256'].items())
    assert sha(P/'common_initialization.pth')==plan['common_initialization_sha256']
    with (P/'capacity_once.lock').open('x') as f:f.write(str(os.getpid()))
    import torch
    from torch.utils.checkpoint import checkpoint
    from model import MambaSOD
    from native_data import VisDroneDataset,seed_all,fp32_tree
    from native_criterion import HungarianMatcher,SetCriterion
    from native_variants_v2 import install
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    data=VisDroneDataset('train')
    assert len(data)==6471 and sum(len(r['valid']) for r in data.rows)==343204
    first=int(torch.randperm(len(data),generator=torch.Generator().manual_seed(20261101))[0])
    heavy=max(range(len(data)),key=lambda i:len(data.rows[i]['valid']))
    assert heavy!=first and len(data.rows[heavy]['valid'])==902
    selected=[]
    samples=[]
    for i in [first,heavy]:
        image,target=data[i]
        row=data.rows[i]
        selected.append({'index':i,'file':row['file'],'GT':len(row['valid']),
                         'image_sha256':sha(data.images/row['file']),
                         'annotation_sha256':sha(Path(row['annotation'])),
                         'processed_image_sha256':hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest(),
                         'target_boxes_sha256':hashlib.sha256(target['boxes'].numpy().tobytes()).hexdigest()})
        samples.append((image.unsqueeze(0),target))
    save('capacity_selected_samples.json',selected)
    initial=torch.load(P/'common_initialization.pth',map_location='cpu',weights_only=True)['model']
    records=[]
    reference_rng=[]
    for arm in ['B','M','Q']:
        save('capacity_progress.json',{'status':'CONSTRUCTING','arm':arm,'counts':COUNTS,'pid':os.getpid()})
        model=MambaSOD(img_size=640,patch_size=16,vim_depth=24,vim_embed_dim=192,d_state=16,
                       out_channels=256,bifpn_repeats=3,num_queries=400,num_decoder_layers=6,
                       d_ffn=1024,num_classes=10,dropout=0.0,drop_path_rate=0.1,
                       num_groups=6,use_ca=False,use_sa=False,use_fpn=False)
        model.load_state_dict(initial,strict=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
        install(model,arm)
        assert sum(p.numel() for p in model.parameters())==72697131
        # Same predeclared memory optimization for every arm.
        for layer in model.decoder.layers:
            original=layer.forward
            def wrapper(tgt,query_pos,memories,pos_embeds,num_groups=1,source=original):
                def core(a,b,c,d):
                    COUNTS['decoder_layer_entries_including_checkpoint_recomputation']+=1
                    return source(a,b,c,d,num_groups=num_groups)
                return checkpoint(core,tgt,query_pos,memories,pos_embeds,
                                  use_reentrant=False,preserve_rng_state=True)
            layer.forward=wrapper
        def count(module,args):
            COUNTS['whole_model_forward_entries']+=1
            assert COUNTS['whole_model_forward_entries']<=6
        model.register_forward_pre_hook(count)
        model=model.cuda().train()
        matcher=HungarianMatcher(cost_class=2,cost_bbox=2,cost_giou=5,num_classes=10,
                                 focal_alpha=.25,focal_gamma=1.5)
        criterion=SetCriterion(10,matcher,{'loss_ce':2,'loss_bbox':2,'loss_giou':5},
                               ['labels','boxes'],focal_alpha=.25,focal_gamma=1.5).cuda()
        optimizer=torch.optim.AdamW([
            {'params':[p for n,p in model.named_parameters() if n.startswith('encoder.backbone.')],'lr':1e-5},
            {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.backbone.')],'lr':1e-4}
        ],weight_decay=1e-4)
        seed_all()
        steps=[]
        for step,(image,target) in enumerate(samples):
            images=image.cuda()
            targets=[{k:v.cuda() for k,v in target.items()}]
            optimizer.zero_grad(set_to_none=True)
            rng={'CPU':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                 'CUDA':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
            if arm=='B':reference_rng.append(rng)
            else:assert rng==reference_rng[step],('RNG mismatch',arm,step)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start=time.monotonic()
            save('capacity_progress.json',{'status':'FORWARD_START','arm':arm,'step':step,
                                          'counts':COUNTS,'pid':os.getpid()})
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=model(images)
            COUNTS['whole_model_forward_completions']+=1
            assert outputs['pred_logits'].shape==(1,2400,10) and outputs['pred_boxes'].shape==(1,2400,4)
            assert all(torch.isfinite(v).all() for o in [outputs]+outputs['aux_outputs'] for k,v in o.items() if isinstance(v,torch.Tensor))
            with torch.autocast(device_type='cuda',enabled=False):
                losses=criterion(fp32_tree(outputs),targets)
                loss=sum(v*criterion.weight_dict[k.split('_aux')[0]] for k,v in losses.items())
            assert torch.isfinite(loss) and all(torch.isfinite(v) for v in losses.values())
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            COUNTS['capacity_only_optimizer_updates']+=1
            assert COUNTS['capacity_only_optimizer_updates']<=6
            torch.cuda.synchronize()
            seconds=time.monotonic()-start
            peak=torch.cuda.max_memory_allocated()
            assert peak<=22*(1<<30),('Allocated safety margin exceeded',arm,peak)
            row={'step':step,'GT':selected[step]['GT'],'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm),
                 'seconds_forward_loss_backward_step':seconds,'peak_allocated_bytes':peak,
                 'active_gradient_parameter_tensors':len(gradients),'RNG_before_forward':rng}
            steps.append(row)
            save('capacity_progress.json',{'status':'STEP_COMPLETE','arm':arm,'step':step,'last_step':row,
                                          'counts':COUNTS,'pid':os.getpid()})
            del outputs,loss,losses,gradients,images,targets
        records.append({'arm':arm,'steps':steps,'initial_state_exact':True,'weights_discarded':True})
        del model,criterion,matcher,optimizer
        gc.collect();torch.cuda.empty_cache()
    assert COUNTS['whole_model_forward_entries']==COUNTS['whole_model_forward_completions']==6
    assert COUNTS['capacity_only_optimizer_updates']==6
    estimates={r['arm']:r['steps'][1]['seconds_forward_loss_backward_step']*6471*12/3600 for r in records}
    result={'status':'PASS_REAL_BATCH1_NATIVE_CRITERION','records':records,'selected_samples':selected,
            'counts':COUNTS,'microbatch':1,'query_groups':6,'queries_per_group':400,
            'decoder_activation_checkpointing':'all6layers, nonreentrant, preserve_rng_state',
            'BF16_forward_FP32_native_criterion':True,'no_loss_clamp_or_skip':True,
            'rough_training_only_hours_per12_from_single_postwarmup_stress_step':estimates,
            'rough_three_arm_training_hours':sum(estimates.values()),
            'formal_training_launched':False,'formal_queue':None,'temporary_weights_saved':False,
            'protocol_sha256':sha(P/'capacity_scope_FROZEN_v1.json'),
            'limits':['Two images per arm; second timing is one stress image, not steady-state throughput benchmark.',
                      'No EMA memory included; this is raw-model capacity only.',
                      'Capacity success does not establish AP, novelty, or full300-epoch feasibility.']}
    save('capacity_result.json',result)
    save('capacity_exit.json',{'status':'COMPLETED_CAPACITY_ONLY','exit_code':0,'counts':COUNTS,'post_capacity_queue':None})
    print(json.dumps({'status':result['status'],'counts':COUNTS,'rough_three_arm_training_hours':sum(estimates.values()),
                      'steady_stress_steps':{r['arm']:r['steps'][1] for r in records},'formal_training_launched':False}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        save('capacity_failure.json',{'status':'FAILED_PRESERVED_NO_AUTOMATIC_RETRY','counts':COUNTS,'traceback':traceback.format_exc()})
        save('capacity_exit.json',{'status':'FAILED_CAPACITY_ONLY','exit_code':1,'counts':COUNTS,'post_capacity_queue':None})
        raise
