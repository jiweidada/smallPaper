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

ARMS = ('baseline', 'fine_only', 'grid_only', 'fine_grid')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def validate_plan(plan):
    if plan['schema'] != 'fine_geometry_v1' or plan['arm'] not in ARMS or plan['seed'] not in (1, 2):
        raise ValueError('Unreviewed plan schema, arm, or seed')
    cfg = plan['native_baseline_config']
    checks = [cfg['epoches'] == 132, cfg['seed'] == plan['seed'], cfg['num_classes'] == 10,
              cfg['eval_spatial_size'] == [640, 640], cfg['use_focal_loss'] is True,
              cfg['train_dataloader']['total_batch_size'] == 4,
              cfg['DFINETransformer']['num_queries'] == 300,
              cfg['DFINETransformer']['query_select_method'] == 'default',
              cfg['PostProcessor']['num_top_queries'] == 300,
              cfg['DEIMCriterion']['losses'] == ['mal', 'boxes', 'local'],
              cfg['HGNetv2']['pretrained'] is False,
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
    cfg.update(model='GeometryDEIM', criterion='DEIMCriterion', resume=None,
               tuning=str(paths['common_init']), output_dir=str(paths['output']),
               device='cuda:0', use_swanlab=False)
    cfg['GeometryDEIM'] = dict(recipe=deepcopy(plan['model_recipe']), arm=plan['arm'])
    for split in ('train', 'val'):
        dataset = cfg[f'{split}_dataloader']['dataset']
        dataset['img_folder'] = str(paths[f'{split}_images'])
        dataset['ann_file'] = str(paths[f'{split}_ann'])
    return cfg


def check_code(native_root):
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    plans = sorted((HERE/'configs').glob('*.json'))
    if len(plans) != 8:
        raise RuntimeError('Expected all four arms and both seeds')
    for path in plans:
        validate_plan(read_json(path))
    bound = verify_training_snapshot(native_root)
    return {'status': 'PASS', 'plans': len(plans), 'training_source_files': len(bound['files']),
            'mode': 'static-only', 'optimizer_constructions': 0, 'detector_forwards': 0}


def prepare_run(args):
    paths = {k: Path(getattr(args, k)).resolve() for k in
             ('native_root', 'train_images', 'train_ann', 'val_images', 'val_ann', 'common_init', 'output')}
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
                'native_profiling': 'unchanged'}
    # JSON is a YAML subset; native YAMLConfig can read the .yml without includes.
    bundle.mkdir(parents=True, exist_ok=False)
    write_json(bundle/'resolved.yml', cfg)
    write_json(bundle/'plan.json', plan)
    receipt = {'status': 'PREPARED_NOT_TRAINED', 'identity': identity,
               'paths': {k: str(v) for k,v in paths.items()},
               'resolved_sha256': sha(bundle/'resolved.yml')}
    write_json(bundle/'receipt.json', receipt)
    return receipt


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    receipt = read_json(bundle/'receipt.json')
    plan = read_json(bundle/'plan.json')
    validate_plan(plan)
    paths = {k: Path(v) for k, v in receipt['paths'].items()}
    verify_training_snapshot(paths['native_root'])
    identity = receipt['identity']
    if identity['plan_sha256'] != sha(bundle/'plan.json') or identity['common_init_sha256'] != COMMON_INIT_SHA:
        raise RuntimeError('Plan or initialization identity changed')
    expected_plan = read_json(HERE/'plan_binding.json')[f"{plan['arm']}_seed{plan['seed']}"]
    canonical = json.dumps(plan, sort_keys=True, ensure_ascii=False).encode('utf-8')
    if hashlib.sha256(canonical).hexdigest() != expected_plan:
        raise RuntimeError('Plan mismatch')
    if identity['arm'] != plan['arm'] or identity['seed'] != plan['seed']:
        raise RuntimeError('Experiment identity does not match the plan')
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
            'geometry_arm': model.geometry_arm}


def train_run(bundle, resume=None):
    # Only this explicit command imports the full training environment or starts it.
    bundle = Path(bundle).resolve()
    receipt, plan, paths = verify_bundle(bundle)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('RANK', '-1')) not in (-1, 0):
        raise RuntimeError('This version supports the paired single-GPU protocol only')
    if resume is None and paths['output'].exists():
        raise FileExistsError('Refusing to replace or increment an existing experiment directory')
    if resume is not None:
        resume = Path(resume).resolve()
        if resume.parent != paths['output'].resolve() or not resume.is_file():
            raise ValueError('Resume must belong to this experiment output directory')
    activate_training_snapshot(paths['native_root'])
    import torch
    from engine.core import YAMLConfig
    from engine.solver.det_solver import DetSolver
    from engine.misc import dist_utils
    from runtime import register_components, load_common_initialization
    register_components()
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires the existing supported CUDA environment')
    dist_utils.setup_distributed(print_rank=0, print_method='builtin', seed=plan['seed'])
    cfg = YAMLConfig(str(bundle/'resolved.yml'))
    if resume is not None:
        cfg.resume = str(resume)
        cfg.yaml_cfg['resume'] = str(resume)
        cfg.tuning = None
        cfg.yaml_cfg['tuning'] = None

    class ExperimentSolver(DetSolver):
        def load_tuning_state(self, path):
            self.initialization_receipt = load_common_initialization(self.model, path)

        def train(self):
            super().train()
            if self.output_dir.resolve() != paths['output'].resolve():
                raise RuntimeError('Native solver unexpectedly changed the output directory')
            coverage = assert_optimizer_coverage(dist_utils.de_parallel(self.model), self.optimizer)
            name = 'runtime_resume.json' if resume is not None else 'runtime_start.json'
            # Preserve any earlier resume receipt instead of overwriting it.
            import time
            target = self.output_dir / (name[:-5] + '_' + time.strftime('%Y%m%d_%H%M%S') + '.json')
            write_json(target, {'identity': receipt['identity'], 'optimizer_coverage': coverage,
                               'initialization': getattr(self, 'initialization_receipt', None),
                               'resume': str(resume) if resume is not None else None,
                               'torch': torch.__version__, 'cuda': torch.version.cuda})

        def state_dict(self):
            state = super().state_dict()
            state['fine_geometry_identity'] = deepcopy(receipt['identity'])
            return state

        def load_state_dict(self, state):
            if state.get('fine_geometry_identity') != receipt['identity']:
                raise RuntimeError('Checkpoint experiment identity mismatch')
            return super().load_state_dict(state)

    solver = ExperimentSolver(cfg)
    try:
        solver.fit(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2, default=str))
    finally:
        dist_utils.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='Static checks only; no torch import')
    check.add_argument('--native-root', required=True)
    prep = sub.add_parser('prepare', help='Create a reviewable configuration, without training')
    for name in ('plan', 'native-root', 'train-images', 'train-ann', 'val-images', 'val-ann', 'common-init', 'output', 'bundle'):
        prep.add_argument('--' + name, required=True)
    run = sub.add_parser('train', help='Explicitly start native training; not executed during code delivery')
    run.add_argument('--bundle', required=True)
    run.add_argument('--resume')
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check_code(args.native_root), ensure_ascii=False, indent=2))
    elif args.command == 'prepare':
        print(json.dumps(prepare_run(args), ensure_ascii=False, indent=2))
    else:
        train_run(args.bundle, args.resume)


if __name__ == '__main__':
    main()
