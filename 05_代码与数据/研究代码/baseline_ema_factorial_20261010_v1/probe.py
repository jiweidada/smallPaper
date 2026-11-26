"""Six zero-update forwards isolate frozen flags, deepcopy and native EMA."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);out=root/'ema_factorial_seed1_v1';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(root/'code'))
    from runtime import verify_bundle,activate,resolved_config,write
    receipt,plan,paths=verify_bundle(root/'prepared_seed1');activate(paths['native_root'])
    from evaluator_binding import activate as bind
    bind(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    import torch,numpy as np
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.optim.ema import ModelEMA
    dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=1)
    data=resolved_config(plan,dict(paths,output=out),baseline=True);write(out/'config.yml',data);cfg=YAMLConfig(str(out/'config.yml'))
    model=cfg.model.cuda().eval();state=torch.load(paths['baseline_checkpoint'],map_location='cpu',weights_only=False)['ema']['module'];model.load_state_dict(state,strict=True)
    flags={n:p.requires_grad for n,p in model.named_parameters()}
    cfg.train_dataloader;loader=cfg.val_dataloader;samples,targets=next(iter(loader));samples=samples.cuda();ids=[int(x['image_id']) for x in targets];assert ids==[0,1,2,3]
    assert hashlib.sha256(samples.cpu().numpy().tobytes()).hexdigest()=='29a65726c6d1bb016c801b315e3cc8cae377cb5229ff969352f8d57facb51dd4'
    outputs={};models={};started=time.monotonic()
    def run(name,m):
        assert all(torch.equal(m.state_dict()[k].cpu(),v) for k,v in state.items())
        assert all(torch.equal(v,dict(m.named_buffers())[k]) for k,v in model.named_buffers())
        with torch.no_grad():outputs[name]={k:v.detach().cpu().numpy().copy() for k,v in m(samples).items() if torch.is_tensor(v)}
        models[name]={'trainable_parameter_tensors':sum(p.requires_grad for p in m.parameters()),'eval':not m.training}
    run('native_grad_flags',model)
    model.requires_grad_(False);run('native_all_frozen',model)
    for n,p in model.named_parameters():p.requires_grad_(flags[n])
    run('native_restored_flags',model)
    cloned=copy.deepcopy(model);cloned.requires_grad_(False);run('deepcopy_all_frozen',cloned)
    ema=ModelEMA(model,**{k:v for k,v in data['ema'].items() if k!='type'});ema.module.load_state_dict(state,strict=True);run('native_EMA',ema.module)
    old=paths['native_root'].parent/'code';sys.path.insert(0,str(old))
    from geometry import ResidualGeometryDEIM
    wrapper=ResidualGeometryDEIM(plan['model_recipe'],'baseline').cuda().eval();wrapper.load_state_dict(state,strict=True);wrapper.requires_grad_(False);run('wrapper_all_frozen',wrapper)
    comparisons={}
    for name,values in outputs.items():
        comparisons[name]={}
        for source,folder in [('current','seed1_fine/baseline_eval'),('EMA_full','complete_replay_seed1_v1')]:
            comparisons[name][source]={}
            for field in ('boxes','logits'):
                a=np.stack([np.load(root/f'{folder}/stages_epoch12/image_{iid:06d}.npz')['final_'+field] for iid in ids]);v=values['pred_'+field]
                comparisons[name][source][field]={'equal':bool(np.array_equal(v,a)),'max_abs':float(np.max(np.abs(v-a)))}
    np.savez_compressed(out/'outputs.npz',**{n+'_'+k:v for n,x in outputs.items() for k,v in x.items()})
    summary={'status':'SIX_FORWARD_EMA_FACTORIAL_COMPLETE','images':ids,'model_forwards':6,'optimizer_steps':0,'epochs_added':0,'weights_and_all_buffers_exact':True,'models':models,'comparisons':comparisons,'elapsed_seconds':time.monotonic()-started,'probe_sha256':sha(__file__),'claim_boundary':'Same four images only. Grad-flag, deepcopy and EMA contrasts do not prove population efficacy or a particular kernel cause.'};write(out/'summary.json',summary);write(out/'manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
