"""Exactly two FP32 frozen-gradient common-initialization EVAL forwards."""
import argparse,hashlib,json,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);out=root/'zero_ema_preflight_v1';out.mkdir(exist_ok=False);sys.path.insert(0,str(root/'code'))
    from runtime import verify_bundle,activate,resolved_config,write,load_common,COMMON_INIT_SHA,sha
    receipt,plan,paths=verify_bundle(root/'prepared_seed1');activate(paths['native_root'])
    from evaluator_binding import activate as bind
    bind(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    import torch,numpy as np
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from small_object_model import SmallObjectDEIM
    dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=1)
    data=resolved_config(plan,dict(paths,output=out),baseline=True);write(out/'config.yml',data);cfg=YAMLConfig(str(out/'config.yml'))
    assert sha(paths['common_init'])==COMMON_INIT_SHA;payload=torch.load(paths['common_init'],map_location='cpu',weights_only=False);state=payload['ema']['module'] if 'ema' in payload else payload['model']
    native=cfg.model.cuda().eval().requires_grad_(False);native.load_state_dict(state,strict=True)
    candidate=SmallObjectDEIM(plan['model_recipe'],variant='fine',initialization_seed=713).cuda();init=load_common(candidate,paths['common_init']);candidate.eval().requires_grad_(False)
    samples,targets=next(iter(cfg.val_dataloader));samples=samples.cuda();outputs=[]
    with torch.no_grad():
        for m in (native,candidate):outputs.append({k:v.detach().cpu().numpy().copy() for k,v in m(samples).items() if torch.is_tensor(v)})
    comparisons={k:{'equal':bool(np.array_equal(outputs[0][k],outputs[1][k])),'max_abs':float(np.max(np.abs(outputs[0][k]-outputs[1][k])))} for k in outputs[0]}
    np.savez_compressed(out/'outputs.npz',**{f'model_{i}_{k}':v for i,d in enumerate(outputs) for k,v in d.items()})
    result={'status':'FROZEN_EMA_ZERO_OUTPUT_PREFLIGHT_COMPLETE','pass':all(v['equal'] for v in comparisons.values()),'comparisons':comparisons,'images':[int(t['image_id']) for t in targets],'FP32':True,'batch_size':4,'every_parameter_frozen':True,'init':init,'model_forwards':2,'optimizer_steps':0,'epochs_added':0,'candidate_AP':None,'model_source_sha256':receipt['identity']['implementation']['small_object_model.py'],'probe_sha256':sha(__file__)};write(out/'summary.json',result);write(out/'manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()});assert result['pass'];print(json.dumps(result),flush=True)
if __name__=='__main__':main()
