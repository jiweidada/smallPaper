"""Prepare a reviewable native training run; training is an explicit subcommand.

`check` and `prepare` use only the Python standard library. They never import
torch, construct an optimizer, load a checkpoint, or apply data augmentation.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from runtime import (HERE, COMMON_INIT_SHA, sha, implementation_hashes,
                     verify_training_snapshot, activate_training_snapshot)

ARMS = ('wavelet', 'spatial')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def validate_plan(plan):
    if plan['schema'] != 'p3_band_attention12_20261010_v1' or plan['arm'] not in ARMS or plan['seed'] not in (1, 2):
        raise ValueError('Unreviewed plan schema, arm, or seed')
    from runtime import attention_contract
    contract = attention_contract()
    if plan['attention_contract'] != contract:
        raise ValueError('Attention contract changed after plan binding')
    cfg = plan['native_baseline_config']
    if plan['screen_protocol'] != {'recipe_epochs':132,'stop_after_epoch_one_based':12,
        'endpoint':'native_epoch12_EMA','automatic_132_promotion':False,
        'scope':'wavelet screen first; spatial control only after explicit screen acceptance; seeds 1 and 2'}:
        raise ValueError('Invalid short-screen boundary')
    checks = [cfg['epoches'] == 132, cfg['seed'] == plan['seed'], cfg['num_classes'] == 10,
              cfg['eval_spatial_size'] == [640, 640], cfg['use_focal_loss'] is True,
              cfg['train_dataloader']['total_batch_size'] == 4,
              cfg['DFINETransformer']['num_queries'] == 300,
              cfg['DFINETransformer']['query_select_method'] == 'default',
              cfg['PostProcessor']['num_top_queries'] == 300,
              cfg['DEIMCriterion']['losses'] == ['mal', 'boxes', 'local'],
              cfg['HGNetv2']['pretrained'] is False,
              cfg.get('use_amp_fallback_fp32', False) is False,
              plan['model_recipe'] == {k: cfg[k] for k in ('HGNetv2', 'HybridEncoder', 'DFINETransformer')},
              plan['criterion_recipe'] == cfg['DEIMCriterion']]
    if not all(checks):
        raise ValueError('The plan differs from the reviewed 132-epoch protocol')
    # Prevent silent drift of the prepared baseline recipe or new hyperparameters.
    expected = read_json(HERE/'plan_binding.json')[f"{plan['arm']}_seed{plan['seed']}"]
    canonical = json.dumps(plan, sort_keys=True, ensure_ascii=False).encode('utf-8')
    if hashlib.sha256(canonical).hexdigest() != expected:
        raise ValueError('Plan content differs from its reviewed binding')


def resolved_config(plan, paths):
    validate_plan(plan)
    cfg = deepcopy(plan['native_baseline_config'])
    # The archived args contain a fully merged configuration, so do not reload includes.
    cfg.pop('__include__', None)
    cfg.pop('config', None)
    cfg.update(model='P3BandAttentionDEIM', criterion='DEIMCriterion', resume=None,
               tuning=str(paths['common_init']), output_dir=str(paths['output']),
               device='cuda:0', use_swanlab=False)
    cfg['P3BandAttentionDEIM'] = dict(recipe=deepcopy(plan['model_recipe']), arm=plan['arm'])
    for split in ('train', 'val'):
        dataset = cfg[f'{split}_dataloader']['dataset']
        dataset['img_folder'] = str(paths[f'{split}_images'])
        dataset['ann_file'] = str(paths[f'{split}_ann'])
    return cfg


def check_code(native_root):
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    plans = sorted((HERE/'configs').glob('*.json'))
    if len(plans) != 4:
        raise RuntimeError('Expected wavelet and spatial plans with both seeds')
    for path in plans:
        validate_plan(read_json(path))
    bound = verify_training_snapshot(native_root)
    return {'status': 'PASS', 'plans': len(plans), 'training_source_files': len(bound['files']),
            'mode': 'static-only', 'optimizer_constructions': 0, 'detector_forwards': 0}


def prepare_run(args):
    from evaluator_binding import fingerprint
    paths = {k: Path(getattr(args, k)).resolve() for k in
             ('native_root', 'train_images', 'train_ann', 'val_images', 'val_ann', 'common_init', 'output', 'evaluator_root')}
    check_code(paths['native_root'])
    plan = read_json(args.plan)
    validate_plan(plan)
    for k in ('train_images', 'val_images'):
        if not paths[k].is_dir():
            raise FileNotFoundError(paths[k])
    for k in ('train_ann', 'val_ann', 'common_init'):
        if not paths[k].is_file():
            raise FileNotFoundError(paths[k])
    if sha(paths['common_init']) != COMMON_INIT_SHA:
        raise RuntimeError('The checkpoint is not the bound common initialization')
    bundle = Path(args.bundle).resolve()
    if paths['output'].exists() or bundle.exists():
        raise FileExistsError('Use new, separate bundle and training-output directories')
    if bundle == paths['output'] or bundle in paths['output'].parents or paths['output'] in bundle.parents:
        raise ValueError('Bundle and training output must be separate directory trees')
    cfg = resolved_config(plan, paths)
    identity = {'schema': plan['schema'], 'arm': plan['arm'], 'seed': plan['seed'],
                'plan_sha256': sha(args.plan), 'common_init_sha256': COMMON_INIT_SHA,
                'implementation': implementation_hashes(),
                'training_binding_sha256': sha(HERE/'training_source_binding.json'),
                'train_annotation_sha256': sha(paths['train_ann']),
                'val_annotation_sha256': sha(paths['val_ann']),
                'native_profiling': 'unchanged',
                'attention_contract_sha256': sha(HERE/'attention_contract.json'),
                'evaluator_binding': fingerprint(paths['evaluator_root'])}
    # JSON is a YAML subset; native YAMLConfig can read the .yml without includes.
    bundle.mkdir(parents=True, exist_ok=False)
    write_json(bundle/'resolved.yml', cfg)
    (bundle/'plan.json').write_bytes(Path(args.plan).read_bytes())
    receipt = {'status': 'PREPARED_NOT_TRAINED', 'identity': identity,
               'paths': {k: str(v) for k,v in paths.items()},
               'resolved_sha256': sha(bundle/'resolved.yml')}
    write_json(bundle/'receipt.json', receipt)
    return receipt


def verify_bundle(bundle):
    from evaluator_binding import verify_files
    bundle = Path(bundle).resolve()
    receipt = read_json(bundle/'receipt.json')
    plan = read_json(bundle/'plan.json')
    validate_plan(plan)
    paths = {k: Path(v) for k, v in receipt['paths'].items()}
    verify_training_snapshot(paths['native_root'])
    identity = receipt['identity']
    verify_files(paths['evaluator_root'], identity['evaluator_binding'])
    if identity['plan_sha256'] != sha(bundle/'plan.json') or identity['common_init_sha256'] != COMMON_INIT_SHA:
        raise RuntimeError('Plan or initialization identity changed')
    expected_plan = read_json(HERE/'plan_binding.json')[f"{plan['arm']}_seed{plan['seed']}"]
    canonical = json.dumps(plan, sort_keys=True, ensure_ascii=False).encode('utf-8')
    if hashlib.sha256(canonical).hexdigest() != expected_plan:
        raise RuntimeError('Plan mismatch')
    if identity['arm'] != plan['arm'] or identity['seed'] != plan['seed']:
        raise RuntimeError('Experiment identity does not match the plan')
    if identity['attention_contract_sha256'] != sha(HERE/'attention_contract.json'):
        raise RuntimeError('Attention contract changed after preparation')
    if identity['implementation'] != implementation_hashes():
        raise RuntimeError('Implementation changed after preparation; prepare a new bundle')
    if identity['training_binding_sha256'] != sha(HERE/'training_source_binding.json'):
        raise RuntimeError('Training source binding changed')
    if sha(bundle/'resolved.yml') != receipt['resolved_sha256']:
        raise RuntimeError('Resolved configuration changed after preparation')
    if read_json(bundle/'resolved.yml') != resolved_config(plan, paths):
        raise RuntimeError('Resolved configuration differs from the plan')
    for split in ('train', 'val'):
        if sha(paths[f'{split}_ann']) != identity[f'{split}_annotation_sha256']:
            raise RuntimeError(f'{split} annotations changed after preparation')
        if not paths[f'{split}_images'].is_dir():
            raise FileNotFoundError(paths[f'{split}_images'])
    if sha(paths['common_init']) != COMMON_INIT_SHA:
        raise RuntimeError('Common initialization changed')
    return receipt, plan, paths


def assert_optimizer_coverage(model, optimizer):
    # Called only by the explicit training path, after native optimizer creation.
    expected = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise RuntimeError('Optimizer has missing, duplicated, or extra parameters')
    return {'trainable_tensors': len(actual),
            'attention_arm': model.attention_arm}


def train_run(bundle):
    # Only this explicit command imports the full training environment or starts it.
    bundle = Path(bundle).resolve()
    receipt, plan, paths = verify_bundle(bundle)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('RANK', '-1')) not in (-1, 0):
        raise RuntimeError('This version supports the paired single-GPU protocol only')
    if paths['output'].exists():
        raise FileExistsError('Refusing to replace or increment an existing experiment directory')
    activate_training_snapshot(paths['native_root'])
    from evaluator_binding import activate, loaded_fingerprint
    evaluator_runtime = activate(paths['evaluator_root'], receipt['identity']['evaluator_binding'])
    import torch
    from engine.core import YAMLConfig
    from engine.solver.det_solver import DetSolver
    from engine.misc import dist_utils
    loaded_fingerprint(receipt['identity']['evaluator_binding'], paths['evaluator_root'])
    from runtime import register_components, load_common_initialization
    register_components()
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires the existing supported CUDA environment')
    dist_utils.setup_distributed(print_rank=0, print_method='builtin', seed=plan['seed'])
    cfg = YAMLConfig(str(bundle/'resolved.yml'))
    if cfg.yaml_cfg.get('use_amp_fallback_fp32', False):
        raise RuntimeError('Whole-epoch AMP rollback/retraining is forbidden')

    class ExperimentSolver(DetSolver):
        def load_resume_state(self, path):
            # Native AMP recovery reaches this before resetting/retraining an
            # epoch. This experiment never permits that continuation path.
            raise RuntimeError('Resume and whole-epoch rollback/retraining are forbidden')

        def _check_baseline_log_monitor(self, log_stats, epoch):
            # Native fit invokes this AFTER evaluation, checkpoint saving and log append.
            # Keep 132 in YAML so the LR/augmentation recipe remains its exact prefix.
            if self.baseline_log_monitor is not None:
                raise RuntimeError('Unexpected metric-based early stopper')
            if epoch > 11:
                raise RuntimeError('Screen exceeded its authorized epoch boundary')
            if hasattr(self, 'training_audit'):
                self.training_audit.epoch_complete(epoch, log_stats, self.ema)
            if epoch == 11:
                from types import SimpleNamespace
                return SimpleNamespace(should_stop=True)
            return None

        def _train_one_epoch_with_amp_rollback(self, epoch, args):
            if not 0 <= epoch <= 11:
                raise RuntimeError('Training outside the authorized 12-epoch prefix')
            if args.use_amp_fallback_fp32:
                raise RuntimeError('Native whole-epoch AMP fallback is forbidden')
            if not hasattr(self, '_entered_epochs'):
                self._entered_epochs = set()
            if epoch in self._entered_epochs:
                raise RuntimeError('An epoch was entered twice; no retraining is allowed')
            self._entered_epochs.add(epoch)
            if epoch == 0:
                from training_audit import TrainingAudit
                self.training_audit = TrainingAudit(self.model, self.optimizer, self.output_dir)
            result = super()._train_one_epoch_with_amp_rollback(epoch, args)
            if epoch == 11:
                # Install only after final-epoch training has returned. Observe
                # the EMA evaluation already called next by bound native fit.
                try:
                    if self.ema is None:
                        raise RuntimeError('Final stage trace requires EMA')
                    from stage_trace import StageTrace, TracedLoader
                    self.endpoint_trace = StageTrace(self.ema.module, receipt['identity'], paths['val_ann'])
                    self.original_val_loader = self.val_dataloader
                    self.val_dataloader = TracedLoader(self.original_val_loader, self.endpoint_trace)
                except Exception:
                    import traceback
                    self.endpoint_trace_error = traceback.format_exc()
            return result

        def load_tuning_state(self, path):
            self.initialization_receipt = load_common_initialization(self.model, path)

        def train(self):
            super().train()
            if self.output_dir.resolve() != paths['output'].resolve():
                raise RuntimeError('Native solver unexpectedly changed the output directory')
            coverage = assert_optimizer_coverage(dist_utils.de_parallel(self.model), self.optimizer)
            target = self.output_dir / 'runtime_start.json'
            write_json(target, {'identity': receipt['identity'], 'optimizer_coverage': coverage,
                               'initialization': getattr(self, 'initialization_receipt', None),
                               'resume': None, 'automatic_retry': False,
                               'evaluator_runtime': evaluator_runtime,
                               'torch': torch.__version__, 'cuda': torch.version.cuda})

        def state_dict(self):
            state = super().state_dict()
            state['p3_band_attention12_identity'] = deepcopy(receipt['identity'])
            return state

        def load_state_dict(self, state):
            if state.get('p3_band_attention12_identity') != receipt['identity']:
                raise RuntimeError('Checkpoint experiment identity mismatch')
            return super().load_state_dict(state)

        def save_best_model(self, test_stats, best_stat, epoch):
            if epoch == 11:
                if hasattr(self, 'endpoint_trace'):
                    self.endpoint_trace.close()
                if hasattr(self, 'original_val_loader'):
                    self.val_dataloader = self.original_val_loader
            result = super().save_best_model(test_stats, best_stat, epoch)
            if epoch == 11:
                # Observe the evaluation already performed by native fit. Do not
                # rerun evaluation or let export failures prevent last.pth saving.
                try:
                    if self.ema is None:
                        raise RuntimeError('The reviewed endpoint requires EMA evaluation')
                    if hasattr(self, 'endpoint_trace_error') or not hasattr(self, 'endpoint_trace'):
                        raise RuntimeError(getattr(self, 'endpoint_trace_error', 'Stage observer was not installed'))
                    self.endpoint_trace_manifest = self.endpoint_trace.finish(self.output_dir/'stages_epoch12')
                    from results import capture
                    capture(self.evaluator.coco_eval['bbox'], test_stats, receipt['identity'],
                            paths['val_ann'], self.output_dir/'endpoint_epoch12', epoch=epoch,
                            environment={'python': __import__('sys').version,
                                'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                                'numpy': __import__('numpy').__version__,
                                'torchvision': __import__('torchvision').__version__,
                                'device_name': torch.cuda.get_device_name(self.device)})
                    self.endpoint_capture_complete = True
                except Exception:
                    import traceback
                    self.endpoint_capture_error = traceback.format_exc()
            return result

    solver = ExperimentSolver(cfg)
    try:
        solver.fit(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2, default=str))
        if not getattr(solver, 'endpoint_capture_complete', False):
            error = getattr(solver, 'endpoint_capture_error', 'Fixed epoch 12 endpoint was not reached')
            raise RuntimeError('Native fit returned without a verified endpoint export: ' + error)
        endpoint = paths['output']/'endpoint_epoch12'
        last = paths['output']/'last.pth'
        if not last.is_file():
            raise RuntimeError('Native fit returned without last.pth; endpoint is not complete')
        if solver._entered_epochs != set(range(12)):
            raise RuntimeError('The completed run is not exactly the native 12-epoch prefix')
        write_json(endpoint/'training_completion.json', {
            'status': 'NATIVE_FIT_RETURNED_AND_LAST_CHECKPOINT_PRESENT',
            'identity': receipt['identity'], 'capture_manifest_sha256': sha(endpoint/'manifest.json'),
            'last_checkpoint': str(last), 'last_checkpoint_sha256': sha(last),
            'stage_trace_manifest': str(solver.endpoint_trace_manifest),
            'stage_trace_manifest_sha256': sha(solver.endpoint_trace_manifest),
            'completed_epochs': 12, 'epoch_indices': sorted(solver._entered_epochs),
            'forward_attempts': solver.training_audit.attempts,
            'actual_optimizer_steps': solver.training_audit.steps,
            'automatic_retry': False, 'whole_epoch_rollback': False,
            'independent_reproduction_verified': False})
    except BaseException:
        import traceback
        if paths['output'].is_dir():
            write_json(paths['output']/'failure.json', {'status':'FAILED_NO_AUTOMATIC_RETRY',
                'identity':receipt['identity'], 'error':traceback.format_exc()})
        raise
    finally:
        if hasattr(solver, 'training_audit'):
            solver.training_audit.close()
        if hasattr(solver, 'endpoint_trace'):
            solver.endpoint_trace.close()
        if hasattr(solver, 'original_val_loader'):
            solver.val_dataloader = solver.original_val_loader
        dist_utils.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='Static checks only; no torch import')
    check.add_argument('--native-root', required=True)
    prep = sub.add_parser('prepare', help='Create a reviewable configuration, without training')
    for name in ('plan', 'native-root', 'train-images', 'train-ann', 'val-images', 'val-ann', 'common-init', 'output', 'bundle', 'evaluator-root'):
        prep.add_argument('--' + name, required=True)
    run = sub.add_parser('train', help='Explicitly start native training; not executed during code delivery')
    run.add_argument('--bundle', required=True)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check_code(args.native_root), ensure_ascii=False, indent=2))
    elif args.command == 'prepare':
        print(json.dumps(prepare_run(args), ensure_ascii=False, indent=2))
    else:
        train_run(args.bundle)


if __name__ == '__main__':
    main()
