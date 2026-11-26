"""One native training command, unchanged132 recipe stopped after epoch12."""
from pathlib import Path
import argparse
import copy
import datetime
import json
import os
from runtime import verify_bundle, activate, register_model, load_common, optimizer_coverage, read, write, sha


def train(bundle):
    receipt, plan, paths = verify_bundle(bundle)
    assert not paths['output'].exists(), 'No restart, overwrite or resume'
    assert int(os.environ.get('WORLD_SIZE','1')) == 1
    baseline = read(paths['screen_root']/'baseline_reuse.json')
    assert baseline['new_model_forwards'] == baseline['new_optimizer_steps'] == baseline['new_epochs'] == 0
    activate(paths['native_root'])
    from evaluator_binding import activate as bind_evaluator
    evaluator_runtime = bind_evaluator(paths['evaluator_root'], receipt['identity']['evaluator_binding'])
    register_model()
    from tail_grid import TailGridBinding
    binding = TailGridBinding(plan['endpoint_factor']).install()
    import torch
    from engine.core import YAMLConfig
    from engine.solver.det_solver import DetSolver
    from engine.misc import dist_utils
    dist_utils.setup_distributed(print_rank=0, print_method='builtin', seed=plan['seed'])
    cfg = YAMLConfig(str(Path(bundle)/'resolved.yml'))

    class FDRTailSolver(DetSolver):
        def load_resume_state(self, path):
            raise RuntimeError('No resume, rollback or retraining is authorized')

        def load_tuning_state(self, path):
            self.initialization_receipt = load_common(self.model, path)

        def train(self):
            super().train()
            assert self.output_dir.resolve() == paths['output'].resolve()
            assert self.initialization_receipt['extra_parameters'] == 0
            coverage = optimizer_coverage(dist_utils.de_parallel(self.model), self.optimizer, cfg.yaml_cfg)
            write(self.output_dir/'runtime_start.json', dict(identity=receipt['identity'],
                initialization=self.initialization_receipt, optimizer_coverage=coverage,
                tail_grid=binding.metadata(), evaluator_runtime=evaluator_runtime,
                torch=str(torch.__version__), cuda=torch.version.cuda,
                resume=None, automatic_retry=False, added_parameters=0,
                native_augmentation_and_supervision_weights_unchanged=True,
                caution='All ordinary and DN FGL labels use the same expanded prediction grid; no efficacy claim.'))

        def _train_one_epoch_with_amp_rollback(self, epoch, args):
            assert 0 <= epoch <= 11 and not args.use_amp_fallback_fp32
            if not hasattr(self,'entered_epochs'):
                self.entered_epochs = set()
            assert epoch not in self.entered_epochs
            self.entered_epochs.add(epoch)
            if epoch == 0:
                from training_audit import TrainingAudit
                self.training_audit = TrainingAudit(dist_utils.de_parallel(self.model), self.optimizer,
                    self.output_dir, paths['baseline_first100'], receipt['identity']['baseline_first100_sha256'])
            self.training_audit.enter_epoch(epoch)
            result = super()._train_one_epoch_with_amp_rollback(epoch, args)
            if epoch == 11:
                try:
                    assert self.ema is not None
                    from stage_trace import StageTrace, TracedLoader
                    self.endpoint_trace = StageTrace(self.ema.module, receipt['identity'], paths['val_ann'])
                    self.original_val_loader = self.val_dataloader
                    self.val_dataloader = TracedLoader(self.original_val_loader, self.endpoint_trace)
                except Exception:
                    import traceback
                    self.endpoint_trace_error = traceback.format_exc()
            return result

        def _check_baseline_log_monitor(self, stats, epoch):
            assert self.baseline_log_monitor is None and 0 <= epoch <= 11
            self.training_audit.epoch_complete(epoch, stats, self.ema)
            if epoch == 11:
                from types import SimpleNamespace
                return SimpleNamespace(should_stop=True)
            return None

        def save_best_model(self, test_stats, best_stat, epoch):
            if epoch == 11:
                if hasattr(self,'endpoint_trace'):
                    self.endpoint_trace.close()
                if hasattr(self,'original_val_loader'):
                    self.val_dataloader = self.original_val_loader
            result = super().save_best_model(test_stats, best_stat, epoch)
            if epoch == 11:
                try:
                    assert hasattr(self,'endpoint_trace') and not hasattr(self,'endpoint_trace_error')
                    self.endpoint_trace.finish(self.output_dir/'stages_epoch12')
                    from results import capture
                    capture(self.evaluator.coco_eval['bbox'], test_stats, receipt['identity'], paths['val_ann'],
                        self.output_dir/'endpoint_epoch12', epoch=11,
                        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, GPU=torch.cuda.get_device_name(0)))
                    self.endpoint_complete = True
                except Exception:
                    import traceback
                    self.endpoint_error = traceback.format_exc()
            return result

        def state_dict(self):
            state = super().state_dict()
            state['fdr_tail12_identity'] = copy.deepcopy(receipt['identity'])
            return state

        def load_state_dict(self, state):
            assert state.get('fdr_tail12_identity') == receipt['identity']
            return super().load_state_dict(state)

    solver = FDRTailSolver(cfg)
    try:
        solver.fit(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2, default=str))
        assert getattr(solver,'endpoint_complete',False), getattr(solver,'endpoint_error','Fixed12 endpoint missing')
        assert solver.entered_epochs == set(range(12)) and (paths['output']/'last.pth').is_file()
        from results import load_endpoint, pair
        from evaluator_binding import same_result_evaluator
        base, bh, _ = load_endpoint(paths['screen_root']/'baseline_eval/endpoint_epoch12')
        candidate, ch, _ = load_endpoint(paths['output']/'endpoint_epoch12')
        same_result_evaluator(base, candidate)
        comparison = pair(base, candidate, bh, ch)
        delta = comparison['delta_AP_metrics_pp']; passed = delta['APs'] > 0 and delta['AP'] >= 0
        write(paths['output']/'fixed12_comparison.json', dict(comparison, native_gate_passed=passed,
            seed=plan['seed'], arm=plan['arm'], scope='Fixed12 screen, no best-epoch, significance, novelty or unique-mediator claim'))
        write(paths['output']/'training_completion.json', dict(status='NATIVE_FIT_FINISHED_EXACT12',
            identity=receipt['identity'], completed_epochs=12, entered_epochs=sorted(solver.entered_epochs),
            native_gate_passed=passed, last_checkpoint_sha256=sha(paths['output']/'last.pth'),
            endpoint_manifest_sha256=sha(paths['output']/'endpoint_epoch12/manifest.json'),
            stage_manifest_sha256=sha(paths['output']/'stages_epoch12/manifest.json'), automatic_retry=False,
            automatic_control_launch=False, completed_remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(json.dumps(dict(status='COMPLETE_FIXED12', seed=plan['seed'], arm=plan['arm'], delta_pp=delta,
                              native_gate_passed=passed)), flush=True)
    finally:
        if hasattr(solver,'training_audit'):
            solver.training_audit.close()
        if hasattr(solver,'endpoint_trace'):
            solver.endpoint_trace.close()
        binding.close()
        dist_utils.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle', required=True); a = p.parse_args()
    train(a.bundle)
