"""One independent full replay with the original baseline wrapper and native EMA."""
import argparse,hashlib,json,sys,time
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seed',required=True,type=int,choices=(1,2));a=p.parse_args();root=Path(a.root);out=root/f'complete_replay_seed{a.seed}_v1';out.mkdir(exist_ok=False);start=time.monotonic()
    sys.path.insert(0,str(root/'code'))
    from runtime import verify_bundle,activate,resolved_config
    receipt,plan,paths=verify_bundle(root/f'prepared_seed{a.seed}');activate(paths['native_root'])
    from evaluator_binding import activate as bind,loaded_fingerprint
    bind(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    import torch,numpy as np
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.optim.ema import ModelEMA
    from engine.solver.det_engine import evaluate
    from stage_trace import StageTrace,TracedLoader
    from results import capture
    dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=a.seed)
    data=resolved_config(plan,dict(paths,output=out),baseline=True);write(out/'config.yml',data);cfg=YAMLConfig(str(out/'config.yml'))
    old=paths['native_root'].parent/'code';assert sha(old/'geometry.py')=='f537d8ad50a54fd1b063db5e86d6c1cf4534304588db3c5008cfe69fac1100c9';sys.path.insert(0,str(old))
    from geometry import ResidualGeometryDEIM
    training_model=ResidualGeometryDEIM(plan['model_recipe'],'baseline').cuda()
    criterion=cfg.criterion.cuda();post=cfg.postprocessor.cuda();ema_options={k:v for k,v in data['ema'].items() if k!='type'};ema=ModelEMA(training_model,**ema_options).to('cuda')
    payload=torch.load(paths['baseline_checkpoint'],map_location='cpu',weights_only=False);assert payload['last_epoch']==11;ema.load_state_dict(payload['ema'],strict=True);del payload
    # Match original class-construction order without constructing an optimizer,
    # sampling training data, or performing any training/profiling forward.
    train_loader=cfg.train_dataloader;loader=cfg.val_dataloader;evaluator=cfg.evaluator
    criterion.set_model(training_model);criterion.set_epoch(11)
    model=ema.module;assert all(not x.requires_grad for x in model.parameters())
    state_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    trace=StageTrace(model,receipt['identity'],paths['val_ann'])
    with torch.no_grad():stats,obj=evaluate(model,criterion,post,TracedLoader(loader,trace),evaluator,torch.device('cuda:0'),test_only=False,output_dir=out,yolo_metrice=data['yolo_metrice'])
    trace.finish(out/'stages_epoch12');capture(obj.coco_eval['bbox'],stats,receipt['identity'],paths['val_ann'],out/'endpoint_epoch12',epoch=11,environment={'torch':str(torch.__version__),'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0)})
    assert all(torch.equal(state_before[k],v.cpu()) for k,v in model.state_dict().items()),'Evaluation mutated parameters/BN'
    first=paths['screen_root']/'baseline_eval';stage_comparison={};all_equal=True;maxdiff={}
    for iid in range(548):
        with np.load(out/f'stages_epoch12/image_{iid:06d}.npz') as x,np.load(first/f'stages_epoch12/image_{iid:06d}.npz') as y:
            assert set(x.files)==set(y.files)
            for field in x.files:
                eq=bool(np.array_equal(x[field],y[field]));all_equal=all_equal and eq
                error=float(np.max(np.abs(x[field]-y[field])));maxdiff[field]=max(error,maxdiff.get(field,0))
                if not eq:stage_comparison.setdefault(str(iid),{})[field]=error
    comparisons={}
    for name in ('precision_recall.npz','GT_hits.npz'):
        fp=first/'endpoint_epoch12'/name;sp=out/'endpoint_epoch12'/name
        # Field names are discovered from the frozen native exporter, not assumed.
        assert fp.exists() and sp.exists(),name
        with np.load(fp) as x,np.load(sp) as y:
            assert set(x.files)==set(y.files)
            comparisons[name]={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
    old_stats=json.loads((first/'historical_comparison.json').read_text())
    current=[float(x) for x in stats['coco_eval_bbox']];reference=old_stats['current_stats'];historical=old_stats['historical_stats']
    pass_current=all_equal and current==reference and all(all(v.values()) for v in comparisons.values())
    result={'status':'FULL_BASE12_NATIVE_EMA_WRAPPER_REPLAY_COMPLETE','seed':a.seed,'images':548,'model_forwards':137,'optimizer_constructions':0,'optimizer_steps':0,'epochs_added':0,'candidate_training_started':False,'all_stage_arrays_exactly_equal_to_first_replay':all_equal,'stage_max_absolute_differences':maxdiff,'stage_mismatches':stage_comparison,'endpoint_arrays_exact':comparisons,'current_stats':current,'first_replay_stats':reference,'current_reference_reproduction_pass':pass_current,'historical_stats':historical,'historical_max_absolute_fraction_difference':max(abs(x-y) for x,y in zip(current,historical)),'historical_gate_unchanged':1e-8,'historical_log_reproduction_pass':max(abs(x-y) for x,y in zip(current,historical))<=1e-8,'state_parameters_and_BN_unchanged':True,'CUDA_RNG_unchanged':bool(torch.equal(cuda_rng,torch.cuda.get_rng_state())),'CPU_RNG_unchanged':bool(torch.equal(cpu_rng,torch.get_rng_state())),'checkpoint_sha256':sha(paths['baseline_checkpoint']),'probe_sha256':sha(__file__),'loaded_evaluator':loaded_fingerprint(receipt['identity']['evaluator_binding'],paths['evaluator_root']),'elapsed_seconds':time.monotonic()-start,'claim_boundary':'Current deterministic replay verification only. Historical mismatch root cause remains unresolved; no candidate efficacy or historical equivalence claim.'}
    write(out/'completion.json',result);write(out/'manifest.json',{q.relative_to(out).as_posix():sha(q) for q in out.rglob('*') if q.is_file()});assert pass_current,'Independent full current replay differs';print(json.dumps({k:v for k,v in result.items() if k not in ('stage_mismatches','loaded_evaluator')}),flush=True)
if __name__=='__main__':main()
