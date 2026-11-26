"""Frozen four-forward, zero-update constructor/repetition diagnostic."""
import argparse,base64,hashlib,json,sys,time
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--seed',type=int,choices=(1,2),required=True);args=a.parse_args();root=Path(args.root);seed=args.seed
    out=root/f'diagnosis_seed{seed}_v2';out.mkdir(exist_ok=False)
    code=root/'code';sys.path.insert(0,str(code))
    from runtime import verify_bundle,activate,resolved_config
    receipt,plan,paths=verify_bundle(root/f'prepared_seed{seed}');activate(paths['native_root'])
    from evaluator_binding import activate as bind
    bind(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    import torch,numpy as np
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=seed)
    cfgdata=resolved_config(plan,dict(paths,output=out),baseline=True);write(out/'native.yml',cfgdata);cfg=YAMLConfig(str(out/'native.yml'))
    native=cfg.model.cuda().eval();state=torch.load(paths['baseline_checkpoint'],map_location='cpu',weights_only=False)['ema']['module'];native.load_state_dict(state,strict=True)
    oldcode=paths['native_root'].parent/'code';sys.path.insert(0,str(oldcode))
    assert sha(oldcode/'geometry.py')=='f537d8ad50a54fd1b063db5e86d6c1cf4534304588db3c5008cfe69fac1100c9'
    from geometry import ResidualGeometryDEIM
    wrapper=ResidualGeometryDEIM(plan['model_recipe'],'baseline').cuda().eval();wrapper.load_state_dict(state,strict=True)
    assert all(torch.equal(native.state_dict()[k],wrapper.state_dict()[k]) for k in state)
    samples,targets=next(iter(cfg.val_dataloader));samples=samples.cuda();ids=[int(t['image_id']) for t in targets]
    def attrs(model):
        result={}
        for name,module in model.named_modules():
            vals={k:v for k,v in vars(module).items() if not k.startswith('_') and isinstance(v,(int,float,str,bool,type(None)))}
            result[name]=dict(type=type(module).__name__,values=vals)
        return result
    na,wa=attrs(native),attrs(wrapper);differences={k:{'native':na.get(k),'wrapper':wa.get(k)} for k in set(na)|set(wa) if na.get(k)!=wa.get(k)}
    outputs=[];t=time.monotonic()
    with torch.no_grad():
        for model in (native,native,wrapper,wrapper):outputs.append({k:v.detach().cpu().numpy().copy() for k,v in model(samples).items() if torch.is_tensor(v)})
    comparisons={}
    for lhs,rhs in ((0,1),(0,2),(2,3)):
        comparisons[f'{lhs}_{rhs}']={k:{'equal':bool(np.array_equal(outputs[lhs][k],outputs[rhs][k])),'max_abs':float(np.max(np.abs(outputs[lhs][k]-outputs[rhs][k])))} for k in outputs[lhs]}
    previous={}
    for bidx,iid in enumerate(ids):
        with np.load(paths['screen_root']/f'baseline_eval/stages_epoch12/image_{iid:06d}.npz') as z:
            previous[str(iid)]={field:{'equal':bool(np.array_equal(outputs[0]['pred_'+field][bidx],z['final_'+field])),'max_abs':float(np.max(np.abs(outputs[0]['pred_'+field][bidx]-z['final_'+field])))} for field in ('boxes','logits')}
    np.savez_compressed(out/'outputs.npz',**{f'forward_{i}_{k}':v for i,d in enumerate(outputs) for k,v in d.items()})
    summary={'status':'FOUR_FORWARD_CONSTRUCTOR_PROBE_COMPLETE','seed':seed,'images':ids,'samples_sha256':hashlib.sha256(samples.cpu().numpy().tobytes()).hexdigest(),'checkpoint_sha256':sha(paths['baseline_checkpoint']),'constructor_module_attribute_differences':differences,'comparisons':comparisons,'previous_fresh_process_comparisons':previous,'torch':str(torch.__version__),'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'model_forwards':4,'optimizer_constructions':0,'optimizer_steps':0,'epochs_added':0,'elapsed_seconds':time.monotonic()-t,'probe_sha256':sha(__file__)}
    write(out/'summary.json',summary);write(out/'manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
