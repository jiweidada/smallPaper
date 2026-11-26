"""Bounded real-CUDA engineering preflight; discarded states, never continued as training."""
import argparse, copy, json, random, sys
from pathlib import Path
from experiment import verify_bundle,assert_optimizer_coverage,write_json
from runtime import activate_training_snapshot,register_components,load_common_initialization

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-bundle',required=True);p.add_argument('--candidate-bundle',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    left,lp,paths=verify_bundle(a.baseline_bundle);right,rp,rpaths=verify_bundle(a.candidate_bundle)
    if lp['arm']!='baseline' or rp['arm']!='fine_residual' or lp['seed']!=rp['seed']:raise ValueError('Unpaired bundles')
    activate_training_snapshot(paths['native_root'])
    from evaluator_binding import activate
    activate(paths['evaluator_root'],left['identity']['evaluator_binding'])
    import torch,numpy as np
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    register_components();dist_utils.setup_seed(lp['seed'])
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expected one existing CUDA device')
    device=torch.device('cuda:0')
    def move(x):
        if isinstance(x,torch.Tensor):return x.to(device)
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        if isinstance(x,list):return [move(v) for v in x]
        if isinstance(x,tuple):return tuple(move(v) for v in x)
        return x
    configs=[YAMLConfig(str(Path(b)/'resolved.yml')) for b in (a.baseline_bundle,a.candidate_bundle)]
    models=[];init=[]
    for cfg in configs:
        dist_utils.setup_seed(lp['seed'])
        m=cfg.model;init.append(load_common_initialization(m,paths['common_init']));models.append(m.to(device).eval())
    # A native validation batch checks WH postprocessing inputs and zero-init equality.
    val=move(next(iter(configs[0].val_dataloader)))
    with torch.no_grad():
        outputs=[m(val[0]) for m in models]
    differences={k:float((outputs[0][k]-outputs[1][k]).abs().max()) for k in ('pred_logits','pred_boxes')}
    if any(v>1e-6 for v in differences.values()):raise RuntimeError('Initial CUDA outputs differ: '+str(differences))
    del outputs,val
    models[0].cpu();torch.cuda.empty_cache()
    model=models[1].train();cfg=configs[1];criterion=cfg.criterion.to(device);criterion.set_model(model)
    optimizer=cfg.optimizer;coverage=assert_optimizer_coverage(model,optimizer)
    # Native training data, criterion metadata, AMP and clipping, at most three batches.
    dist_utils.setup_seed(lp['seed'])
    loader=cfg.train_dataloader;loader.set_epoch(0)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.0)
    rows=[];updates=[]
    handle=optimizer.register_step_post_hook(lambda *args:updates.append(True))
    for index,batch in enumerate(loader):
        images,targets=move(batch)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda',cache_enabled=True):pred=model(images,targets=targets)
        with torch.autocast(device_type='cuda',enabled=False):
            losses=criterion(pred,targets,epoch=0,step=index,global_step=index,epoch_step=len(loader))
            loss=sum(losses.values())
        if not torch.isfinite(loss):raise RuntimeError('Non-finite preflight loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        grad=model.decoder.geometry_residual.output_weight.grad
        row={'batch':index,'loss':float(loss.detach()),'output_grad_norm':float(grad.float().norm()),'output_grad_finite':bool(torch.isfinite(grad).all())}
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
        scaler.step(optimizer);scaler.update()
        row['actual_steps']=len(updates);row['output_weight_norm']=float(model.decoder.geometry_residual.output_weight.detach().norm());rows.append(row)
        if index==2:break
    handle.remove()
    if not updates or not rows[-1]['output_weight_norm']>0 or not all(r['output_grad_finite'] for r in rows):
        raise RuntimeError('Candidate failed finite-gradient/update check: '+str(rows))
    report={'status':'PASS_BOUNDED_CUDA_PREFLIGHT_NOT_AP_EVIDENCE','seed':lp['seed'],'batches':len(rows),
        'initial_output_max_abs_difference':differences,'initialization':init,'optimizer_coverage':coverage,'training':rows,
        'peak_memory_allocated':torch.cuda.max_memory_allocated(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'numpy':np.__version__,'torchvision':__import__('torchvision').__version__,'python':sys.version,
        'device_name':torch.cuda.get_device_name(0),'preflight_scaler_init':1024,'states_discarded':True,
        'formal_training_scaler':'unchanged native config; fresh interpreter and common initialization'}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
