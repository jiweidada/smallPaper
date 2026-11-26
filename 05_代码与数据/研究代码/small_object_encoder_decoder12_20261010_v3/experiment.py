"""One bounded candidate training command: native132 recipe prefix0..11."""
from pathlib import Path
import argparse,copy,datetime,json,os
from runtime import activate,register_model,read,write,sha,verify_bundle,load_common,optimizer_coverage
def train(bundle):
    receipt,plan,paths=verify_bundle(bundle)
    assert plan['seed']==2 and plan['variant']=='fine','This deployment permits only fresh fine seed2; seed1 budget decision pending'
    if paths['output'].exists():raise FileExistsError('Existing training run cannot be resumed or replaced')
    if plan['variant']=='degraded':
        promotion=read(paths['screen_root']/'dual_seed_fine_promotion.json')
        assert promotion['both_same_seed_native_gates_passed'] is True and promotion['protocol_sha256']==receipt['identity']['protocol_sha256']
    baseline=read(paths['screen_root']/'baseline_eval/completion.json')
    assert baseline['historical_log_reproduction_pass'] and baseline['checkpoint_sha256']==receipt['identity']['baseline_checkpoint_sha256']
    assert int(os.environ.get('WORLD_SIZE','1'))==1
    activate(paths['native_root'])
    from evaluator_binding import activate as bind_evaluator,loaded_fingerprint
    evaluator_runtime=bind_evaluator(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    register_model()
    import torch
    from engine.core import YAMLConfig
    from engine.solver.det_solver import DetSolver
    from engine.misc import dist_utils
    dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
    cfg=YAMLConfig(str(Path(bundle)/'resolved.yml'))
    class ScreenSolver(DetSolver):
        def load_resume_state(self,path):raise RuntimeError('Resume or AMP rollback/retraining forbidden')
        def load_tuning_state(self,path):self.initialization_receipt=load_common(self.model,path)
        def train(self):
            super().train()
            assert self.output_dir.resolve()==paths['output'].resolve()
            coverage=optimizer_coverage(dist_utils.de_parallel(self.model),self.optimizer,cfg.yaml_cfg)
            write(self.output_dir/'runtime_start.json',{'identity':receipt['identity'],'optimizer_coverage':coverage,'initialization':self.initialization_receipt,'evaluator_runtime':evaluator_runtime,'torch':str(torch.__version__),'cuda':torch.version.cuda,'resume':None,'automatic_retry':False})
        def _train_one_epoch_with_amp_rollback(self,epoch,args):
            assert 0<=epoch<=11 and not args.use_amp_fallback_fp32
            if not hasattr(self,'entered_epochs'):self.entered_epochs=set()
            assert epoch not in self.entered_epochs;self.entered_epochs.add(epoch)
            if epoch==0:
                from training_audit import TrainingAudit
                self.training_audit=TrainingAudit(dist_utils.de_parallel(self.model),self.optimizer,self.output_dir,paths['baseline_first100'],receipt['identity']['baseline_first100_sha256'])
            result=super()._train_one_epoch_with_amp_rollback(epoch,args)
            if epoch==11:
                try:
                    assert self.ema is not None
                    from stage_trace import StageTrace,TracedLoader
                    self.endpoint_trace=StageTrace(self.ema.module,receipt['identity'],paths['val_ann']);self.original_val_loader=self.val_dataloader;self.val_dataloader=TracedLoader(self.original_val_loader,self.endpoint_trace)
                except Exception:
                    import traceback
                    self.endpoint_trace_error=traceback.format_exc()
            return result
        def _check_baseline_log_monitor(self,log_stats,epoch):
            assert self.baseline_log_monitor is None and 0<=epoch<=11
            self.training_audit.epoch_complete(epoch,log_stats,self.ema)
            if epoch==11:
                from types import SimpleNamespace
                return SimpleNamespace(should_stop=True)
            return None
        def save_best_model(self,test_stats,best_stat,epoch):
            if epoch==11:
                if hasattr(self,'endpoint_trace'):self.endpoint_trace.close()
                if hasattr(self,'original_val_loader'):self.val_dataloader=self.original_val_loader
            result=super().save_best_model(test_stats,best_stat,epoch)
            if epoch==11:
                try:
                    assert hasattr(self,'endpoint_trace') and not hasattr(self,'endpoint_trace_error')
                    self.endpoint_trace.finish(self.output_dir/'stages_epoch12')
                    from results import capture
                    capture(self.evaluator.coco_eval['bbox'],test_stats,receipt['identity'],paths['val_ann'],self.output_dir/'endpoint_epoch12',epoch=11,environment={'torch':str(torch.__version__),'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0)})
                    self.endpoint_complete=True
                except Exception:
                    import traceback
                    self.endpoint_error=traceback.format_exc()
            return result
        def state_dict(self):
            state=super().state_dict();state['small_object_encoder_decoder12_identity']=copy.deepcopy(receipt['identity']);return state
        def load_state_dict(self,state):
            assert state.get('small_object_encoder_decoder12_identity')==receipt['identity']
            return super().load_state_dict(state)
    solver=ScreenSolver(cfg)
    try:
        solver.fit(json.dumps(cfg.__dict__,ensure_ascii=False,indent=2,default=str))
        assert getattr(solver,'endpoint_complete',False),getattr(solver,'endpoint_error','Fixed12 endpoint absent')
        assert solver.entered_epochs==set(range(12)) and (paths['output']/'last.pth').is_file()
        from results import load_endpoint,pair
        from evaluator_binding import same_result_evaluator
        base,bh,_=load_endpoint(paths['screen_root']/'baseline_eval/endpoint_epoch12');candidate,ch,_=load_endpoint(paths['output']/'endpoint_epoch12');same_result_evaluator(base,candidate)
        comparison=pair(base,candidate,bh,ch);delta=comparison['delta_AP_metrics_pp'];passed=delta['APs']>0 and delta['AP']>=0
        write(paths['output']/'fixed12_comparison.json',dict(comparison,native_gate_passed=passed,seed=plan['seed'],scope='Exploratory fixed12 screen; no best epoch,significance or novelty claim'))
        write(paths['output']/'training_completion.json',{'status':'NATIVE_FIT_FINISHED_EXACT12','identity':receipt['identity'],'completed_epochs':12,'entered_epochs':sorted(solver.entered_epochs),'native_gate_passed':passed,'last_checkpoint_sha256':sha(paths['output']/'last.pth'),'endpoint_manifest_sha256':sha(paths['output']/'endpoint_epoch12/manifest.json'),'stage_manifest_sha256':sha(paths['output']/'stages_epoch12/manifest.json'),'automatic_retry':False,'automatic_control_launch':False,'completed_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        print(json.dumps({'status':'COMPLETE_FIXED12','seed':plan['seed'],'delta_pp':delta,'native_gate_passed':passed}),flush=True)
    finally:
        if hasattr(solver,'training_audit'):solver.training_audit.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args();train(a.bundle)
if __name__=='__main__':main()
