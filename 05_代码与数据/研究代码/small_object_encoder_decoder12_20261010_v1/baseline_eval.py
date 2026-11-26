"""One full frozen BASE12 EMA evaluation; no optimizer or training."""
import argparse,copy,datetime,json,time
from pathlib import Path
from runtime import activate,read,write,sha,verify_bundle,resolved_config
def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args();bundle=Path(a.bundle)
    receipt,plan,paths=verify_bundle(bundle);out=paths['screen_root']/'baseline_eval';out.mkdir(exist_ok=False);start=time.monotonic()
    activate(paths['native_root'])
    from evaluator_binding import activate as bind_evaluator,loaded_fingerprint
    bind_evaluator(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    import torch
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver.det_engine import evaluate
    from stage_trace import StageTrace,TracedLoader
    from results import capture
    dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
    config=resolved_config(plan,dict(paths,output=out),baseline=True);write(out/'baseline.yml',config);cfg=YAMLConfig(str(out/'baseline.yml'))
    model=cfg.model.cuda();payload=torch.load(paths['baseline_checkpoint'],map_location='cpu',weights_only=False)
    assert payload['last_epoch']==11;model.load_state_dict(payload['ema']['module'],strict=True);del payload
    criterion=cfg.criterion.cuda();criterion.set_model(model);criterion.set_epoch(11)
    post=cfg.postprocessor.cuda();loader=cfg.val_dataloader;evaluator=cfg.evaluator
    identity=dict(receipt['identity'],arm='baseline',evaluated_checkpoint_sha256=receipt['identity']['baseline_checkpoint_sha256'])
    trace=StageTrace(model,identity,paths['val_ann']);model.eval()
    with torch.no_grad():stats,eval_obj=evaluate(model,criterion,post,TracedLoader(loader,trace),evaluator,torch.device('cuda:0'),test_only=False,output_dir=out,yolo_metrice=config['yolo_metrice'])
    assert eval_obj is evaluator
    trace.finish(out/'stages_epoch12');summary=capture(evaluator.coco_eval['bbox'],stats,identity,paths['val_ann'],out/'endpoint_epoch12',epoch=11,environment={'torch':str(torch.__version__),'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0)})
    old=plan['historical_baseline_stats'];current=stats['coco_eval_bbox'];error=max(abs(float(x)-float(y)) for x,y in zip(old,current))
    comparison={'historical_stats':old,'current_stats':current,'max_absolute_fraction_difference':error,'frozen_tolerance':1e-8,'historical_log_reproduction_pass':error<=1e-8,'current_same_evaluator_reference_preserved':True}
    write(out/'historical_comparison.json',comparison)
    write(out/'completion.json',{'status':'BASE12_REEVALUATION_COMPLETE','historical_log_reproduction_pass':comparison['historical_log_reproduction_pass'],'checkpoint_sha256':receipt['identity']['baseline_checkpoint_sha256'],'endpoint_manifest_sha256':sha(out/'endpoint_epoch12/manifest.json'),'stage_manifest_sha256':sha(out/'stages_epoch12/manifest.json'),'loaded_evaluator':loaded_fingerprint(receipt['identity']['evaluator_binding'],paths['evaluator_root']),'images':548,'model_forwards':137,'optimizer_steps':0,'epochs_added':0,'elapsed_seconds':time.monotonic()-start,'completed_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    assert comparison['historical_log_reproduction_pass'],'Historical BASE12 statistics differ; do not auto-launch candidate'
    print(json.dumps({'status':'PASS_BASE12_REPRODUCTION','seed':plan['seed'],'metrics':summary['overall_percent']}),flush=True)
if __name__=='__main__':main()
