"""Candidate full native loss/AMP/backward check, private states, zero updates."""
from pathlib import Path
import argparse, json, torch
from runtime import HERE, verify_bundle, activate, register_model, load_common, optimizer_coverage, read, write, sha

def move(x):
    if isinstance(x,torch.Tensor):return x.cuda()
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    if isinstance(x,list):return [move(v) for v in x]
    if isinstance(x,tuple):return tuple(move(v) for v in x)
    return x

def finite(x):
    if isinstance(x,torch.Tensor):return bool(torch.isfinite(x).all()) if x.is_floating_point() else True
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
    return True

def main(bundle):
    receipt,plan,paths=verify_bundle(bundle)
    from dispatch import resource
    state=resource(paths['screen_root'],plan)
    assert state['eligible'],state
    out=paths['screen_root']/'detector_preflight_v1';out.mkdir(exist_ok=False)
    write(out/'resource_receipt.json',state)
    activate(paths['native_root'])
    from evaluator_binding import activate as bind
    bind(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    register_model()
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
    cfg=YAMLConfig(str(Path(bundle)/'resolved.yml'))
    model=cfg.model;proof=load_common(model,paths['common_init'])
    assert proof==read(Path(bundle)/'cpu_initialization.json')['initialization']
    model=model.cuda().train();criterion=cfg.criterion.cuda();criterion.set_model(model);criterion.set_epoch(0)
    optimizer=cfg.optimizer;coverage=optimizer_coverage(model,optimizer,cfg.yaml_cfg)
    updates=[];step_hook=optimizer.register_step_post_hook(lambda *args:updates.append(True))
    loader=cfg.train_dataloader;loader.set_epoch(0)
    images,targets=move(next(iter(loader)))
    assert images.ndim==4 and images.shape[0]==len(targets)==4
    torch.cuda.reset_peak_memory_stats();calls=0
    def count(module,args):
        nonlocal calls
        calls+=1
    h=model.register_forward_pre_hook(count);cases=[]
    try:
        for index,side in enumerate((640,832)):
            model.zero_grad(set_to_none=True)
            x=torch.nn.functional.interpolate(images,size=(side,side),mode='bilinear',align_corners=False)
            with torch.autocast('cuda',dtype=torch.float16):pred=model(x,targets=targets)
            assert finite(pred)
            with torch.autocast('cuda',enabled=False):
                losses=criterion(pred,targets,epoch=0,step=index,global_step=index,epoch_step=len(loader))
                loss=sum(losses.values())
            assert torch.isfinite(loss) and finite(losses)
            loss.backward()
            gradients={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
            assert all(torch.isfinite(g).all() for g in gradients.values() if g is not None)
            branch=dict()
            if plan['arm']=='p2_skip':
                p2=model.backbone.p2_skip
                assert torch.count_nonzero(p2.restore.weight)==0
                assert p2.restore.weight.grad is not None and bool((p2.restore.weight.grad!=0).any())
                assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in p2.reduce.parameters())
                branch=dict(zero_restore_maintained=True,restore_gradient_active=True,upstream_bridge_gradients_initially_zero=True)
            clip=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
            assert torch.isfinite(clip)
            cases.append(dict(side=side,input_shape=list(x.shape),target_counts=[len(t['labels']) for t in targets],
                loss=float(loss.detach()),losses={k:float(v.detach()) for k,v in losses.items()},
                preclip_gradient_norm=float(clip),finite_gradients=True,
                missing_trainable_gradients=[n for n,g in gradients.items() if g is None],branch=branch))
            del x,pred,loss,losses
        torch.cuda.synchronize();assert calls==2 and not updates
        report=dict(status='PASS_USER_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES',seed=plan['seed'],arm=plan['arm'],
            identity=receipt['identity'],initialization=proof,optimizer_coverage=coverage,cases=cases,
            candidate_detector_forwards=calls,BASE_detector_forwards=0,evaluator_calls=0,
            optimizer_updates=0,training_epochs=0,states_discarded=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            data_scope='One actual native epoch0 training batch resized to640/832 with normalized targets unchanged. Private states discarded.',
            preflight_source_sha256=sha(__file__),protocol_sha256=sha(HERE/'training_protocol.json'),AP_effect_established=False)
        assert report['peak_reserved_bytes']/1024**2+512<=read(HERE/'training_protocol.json')['concurrency']['reserved_budget_MiB'][plan['arm']]
        write(out/'report.json',report)
        print(json.dumps(dict(status=report['status'],seed=plan['seed'],arm=plan['arm'],peak_reserved_MiB=report['peak_reserved_bytes']/1024**2)))
    finally:
        h.remove();step_hook.remove();dist_utils.cleanup()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args();main(a.bundle)
