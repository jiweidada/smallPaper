"""Prepare or explicitly execute one bounded original-EMA GPU replay.

Importing this module and the prepare subcommand use only the standard library.
Execution requires separate human authorization; CLI switches are not permission.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
MAX_IMAGES = 548
MAX_BATCHES = 137
TIMEOUT_SECONDS = 3600


def sha(path, *, lf=False):
    path = Path(path)
    if lf:
        return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_support(experiment_root):
    root = Path(experiment_root).resolve()
    binding = read(HERE/'source_binding.json')
    for name, digest in binding['experiment_files'].items():
        if sha(root/name) != digest:
            raise ValueError('Bound v5 support changed: '+name)
    modules = []
    for name in ('runtime', 'evaluator_binding'):
        spec = importlib.util.spec_from_file_location('replay_bound_'+name, root/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def bound_config(args_data, paths):
    cfg = deepcopy(args_data['yaml_cfg'])
    if (cfg['model'] != 'DEIM' or cfg['HGNetv2']['pretrained'] is not False
            or cfg['val_dataloader']['total_batch_size'] != 4
            or cfg['val_dataloader']['shuffle'] or cfg['val_dataloader']['drop_last']
            or cfg['ram_cache'] or cfg['eval_spatial_size'] != [640, 640]
            or cfg['num_classes'] != 10 or cfg['remap_mscoco_category']):
        raise ValueError('Original validation recipe differs')
    cfg.pop('__include__', None)
    cfg.pop('config', None)
    cfg.update(resume=None, tuning=None, device='cuda:0', output_dir=str(paths['output']), use_swanlab=False)
    cfg['val_dataloader']['dataset']['img_folder'] = str(paths['val_images'])
    cfg['val_dataloader']['dataset']['ann_file'] = str(paths['val_ann'])
    return cfg


def validate_inputs(seed, paths):
    target = read(HERE/'source_binding.json')['targets'][str(seed)]
    for key in ('original_args', 'original_log'):
        if sha(paths[key], lf=True) != target[key+'_LF_sha256']:
            raise ValueError('Original run identity differs: '+key)
    for key in ('checkpoint', 'val_ann'):
        if sha(paths[key]) != target[key+'_sha256']:
            raise ValueError('Input identity differs: '+key)
    rows = [json.loads(line) for line in paths['original_log'].read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if [r['epoch'] for r in rows] != list(range(132)) or rows[-1]['test_coco_eval_bbox'] != target['original_stats']:
        raise ValueError('Original complete 132-epoch log differs')
    if not paths['val_images'].is_dir():
        raise ValueError('Validation image directory is missing')
    ann = read(paths['val_ann'])
    if sorted(int(im['id']) for im in ann['images']) != list(range(MAX_IMAGES)):
        raise ValueError('Expected fixed 548 validation images')
    return target


def prepare(args):
    keys = ('experiment_root', 'native_root', 'evaluator_root', 'original_args', 'original_log',
            'checkpoint', 'val_images', 'val_ann', 'output')
    paths = {key: Path(getattr(args, key)).resolve() for key in keys}
    bundle = Path(args.bundle).resolve()
    output = paths['output']
    if bundle.exists() or output.exists() or bundle == output or bundle in output.parents or output in bundle.parents:
        raise ValueError('Bundle and output must be new, separate directories')
    protected = [paths['original_args'].parent, paths['original_log'].parent, paths['checkpoint'].parent,
                 paths['native_root'], paths['evaluator_root'], paths['val_images'], paths['val_ann'].parent]
    for path in protected:
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('Replay output overlaps a protected source/input directory')
        if bundle == path or bundle in path.parents or path in bundle.parents:
            raise ValueError('Replay bundle overlaps a protected source/input directory')
    runtime, evaluator = load_support(paths['experiment_root'])
    runtime.verify_training_snapshot(paths['native_root'])
    target = validate_inputs(args.seed, paths)
    package = evaluator.fingerprint(paths['evaluator_root'])
    if (not package['backend'].endswith('.so')
            or package['files'][package['backend']] != read(HERE/'source_binding.json')['original_Linux_mask_sha256']):
        raise ValueError('Original Linux evaluator backend identity differs')
    config = bound_config(read(paths['original_args']), paths)
    bundle.mkdir(parents=True, exist_ok=False)
    write(bundle/'resolved.yml', config)
    write(bundle/'receipt.json', {'status': 'PREPARED_NOT_EXECUTED', 'seed': args.seed,
          'paths': {key: str(value) for key, value in paths.items()}, 'target': target,
          'evaluator_binding': package, 'resolved_sha256': sha(bundle/'resolved.yml'),
          'script_sha256': sha(__file__), 'source_binding_sha256': sha(HERE/'source_binding.json'),
          'maximum_image_forwards': MAX_IMAGES, 'maximum_batch_forwards': MAX_BATCHES,
          'timeout_seconds': TIMEOUT_SECONDS})
    return {'status': 'PREPARED_NOT_EXECUTED', 'bundle': str(bundle)}


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    receipt = read(bundle/'receipt.json')
    if (receipt['script_sha256'] != sha(__file__) or receipt['source_binding_sha256'] != sha(HERE/'source_binding.json')
            or receipt['resolved_sha256'] != sha(bundle/'resolved.yml')):
        raise ValueError('Replay code, binding or prepared config changed')
    paths = {key: Path(value) for key, value in receipt['paths'].items()}
    target = validate_inputs(receipt['seed'], paths)
    runtime, evaluator = load_support(paths['experiment_root'])
    runtime.verify_training_snapshot(paths['native_root'])
    evaluator.verify_files(paths['evaluator_root'], receipt['evaluator_binding'])
    if target != receipt['target'] or bound_config(read(paths['original_args']), paths) != read(bundle/'resolved.yml'):
        raise ValueError('Prepared recipe differs from bound inputs')
    return receipt, paths, runtime, evaluator


class ForwardBudget:
    def __init__(self):
        self.batches = 0
        self.images = 0

    def before(self, module, inputs):
        shape = tuple(inputs[0].shape)
        if module.training or shape != (4, 3, 640, 640):
            raise ValueError('Expected eval-mode four-image native validation batch')
        if self.images + shape[0] > MAX_IMAGES or self.batches + 1 > MAX_BATCHES:
            raise RuntimeError('Fixed replay forward budget exhausted BEFORE next forward')
        self.images += shape[0]
        self.batches += 1


def reproduction_gate(stats, original):
    import math
    if len(stats) != 12 or len(original) != 12 or not all(math.isfinite(float(x)) for x in (*stats, *original)):
        raise ValueError('Expected two finite native twelve-metric vectors')
    delta = {'AP': 100*(float(stats[0])-float(original[0])),
             'APs': 100*(float(stats[3])-float(original[3]))}
    return {'delta_from_original_log_pp': delta, 'gate_0_02pp_passed': all(abs(x) <= .02 for x in delta.values())}


def state_digest(model, torch):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def worker(bundle):
    """Only the explicitly invoked, supervised GPU worker imports ML packages."""
    launch_token = os.environ.get('ORIGINAL_EMA_REPLAY_WORKER_TOKEN')
    if not launch_token:
        raise RuntimeError('GPU worker requires the bounded supervisor')
    receipt, paths, runtime, evaluator_api = verify_bundle(bundle)
    output = paths['output']
    launch = read(output/'worker_launch.json')
    if launch['token'] != launch_token or launch['supervisor_pid'] != os.getppid():
        raise RuntimeError('GPU worker does not belong to its current supervisor')
    os.chdir(output)
    runtime.activate_training_snapshot(paths['native_root'])
    package = evaluator_api.activate(paths['evaluator_root'], receipt['evaluator_binding'])
    import numpy as np
    import torch
    import torchvision
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver.det_engine import evaluate
    if (sys.platform != 'linux' or sys.version_info[:2] != (3, 10) or str(torch.__version__) != '2.5.1+cu124'
            or torch.version.cuda != '12.4' or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3090'):
        raise RuntimeError('Requires the original Python3.10/PyTorch2.5.1+cu124/single RTX3090 environment')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('RANK', '-1')) not in (-1, 0):
        raise RuntimeError('Distributed execution is outside this replay')
    dist_utils.setup_seed(receipt['seed'])
    cfg = YAMLConfig(str(Path(bundle).resolve()/'resolved.yml'))
    model = cfg.model
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint['last_epoch'] != 131 or checkpoint['ema']['updates'] != 213444:
        raise ValueError('Expected exact final original EMA checkpoint')
    model.load_state_dict(checkpoint['ema']['module'], strict=True)
    del checkpoint
    model.to('cuda:0').eval()
    criterion, postprocessor = cfg.criterion.to('cuda:0').eval(), cfg.postprocessor.to('cuda:0').eval()
    loader, coco_evaluator = cfg.val_dataloader, cfg.evaluator
    if len(loader.dataset) != MAX_IMAGES or len(loader) != MAX_BATCHES or loader.batch_size != 4:
        raise ValueError('Validation loader differs from original batch4 population')
    if any(module.training for module in model.modules()):
        raise RuntimeError('Model is not entirely in eval mode')
    before = state_digest(model, torch)
    budget, predictions, image_ids = ForwardBudget(), [], set()
    native_prepare = coco_evaluator.prepare_for_coco_detection

    def observed_prepare(batch_predictions):
        ids = set(map(int, batch_predictions))
        if len(ids) != 4 or image_ids.intersection(ids) or not ids.issubset(set(range(MAX_IMAGES))):
            raise ValueError('Duplicate or foreign image in native evaluation')
        result = native_prepare(batch_predictions)
        predictions.extend(deepcopy(result))
        image_ids.update(ids)
        return result

    def before_forward(module, inputs):
        if torch.is_grad_enabled() or torch.is_autocast_enabled():
            raise RuntimeError('Original validation uses no gradient and no autocast')
        budget.before(module, inputs)

    coco_evaluator.prepare_for_coco_detection = observed_prepare
    handle = model.register_forward_pre_hook(before_forward)
    environment = {'python': sys.version, 'torch': str(torch.__version__), 'torchvision': torchvision.__version__,
                   'numpy': np.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'device_name': torch.cuda.get_device_name(0), 'device_capability': torch.cuda.get_device_capability(0),
                   'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
                   'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}
    write(output/'execution_identity.json', {'receipt': receipt, 'environment': environment, 'loaded_evaluator': package,
          'state_before_sha256': before, 'path': 'direct native evaluate, undeployed original EMA, test_only=False',
          'not_original_historical_prediction_cache': True})
    started = time.monotonic()
    try:
        # Do not call DetSolver.val(): it deploys/fuses and profiles the model.
        # The function below is exactly the one called by native fit after an epoch.
        stats, returned_evaluator = evaluate(model, criterion, postprocessor, loader, coco_evaluator,
                                              torch.device('cuda:0'), test_only=False,
                                              yolo_metrice=cfg.yolo_metrice)
    finally:
        handle.remove()
        coco_evaluator.prepare_for_coco_detection = native_prepare
        write(output/'forward_counts.json', {'attempted_batch_forwards': budget.batches, 'attempted_image_forwards': budget.images})
    if (budget.images != MAX_IMAGES or budget.batches != MAX_BATCHES or image_ids != set(range(MAX_IMAGES))
            or returned_evaluator is not coco_evaluator or state_digest(model, torch) != before):
        raise RuntimeError('Incomplete validation or model state changed')
    native = coco_evaluator.coco_eval['bbox']
    values = list(map(float, stats['coco_eval_bbox']))
    if not np.array_equal(np.asarray(values), np.asarray(native.stats)):
        raise ValueError('Returned statistics differ from native evaluator')
    gate = reproduction_gate(values, receipt['target']['original_stats'])
    write(output/'predictions.json', predictions)
    np.savez_compressed(output/'precision_recall.npz', precision=native.eval['precision'], recall=native.eval['recall'])
    write(output/'native_metrics.json', {'stats': values, **gate})
    validate_inputs(receipt['seed'], paths)
    runtime.verify_training_snapshot(paths['native_root'])
    evaluator_api.loaded_fingerprint(receipt['evaluator_binding'], paths['evaluator_root'])
    write(output/'completion.json', {'status': 'ORIGINAL_EMA_REPLAY_COMPLETED', 'mode': 'native_gpu_replay', 'seed': receipt['seed'],
          'images': budget.images, 'batch_forwards': budget.batches, 'prediction_rows': len(predictions),
          'elapsed_evaluation_seconds': time.monotonic()-started, 'native_evaluate_calls': 1,
          'native_yolo_metrics_enabled': cfg.yolo_metrice, 'state_unchanged': True,
          'checkpoint_loads': 1, 'optimizer_constructions': 0, 'parameter_or_EMA_updates': 0,
          'model_deploy_calls': 0, 'model_profiling_calls': 0, **gate,
          'claim_boundary': 'New replay, not original historical cached output; pass does not establish candidate effectiveness or a unique cause.'})
    write(output/'manifest.json', {'files': {p.name: sha(p) for p in output.iterdir()
                                           if p.is_file() and p.name not in ('worker_stdout.txt', 'worker_stderr.txt')},
                                   'script_sha256': sha(__file__), 'source_binding_sha256': sha(HERE/'source_binding.json')})


def execute_authorized(bundle):
    if sys.platform != 'linux':
        raise RuntimeError('Authorized GPU replay must run on the existing Linux host')
    receipt, paths, _, _ = verify_bundle(bundle)
    output = paths['output']
    output.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    write(output/'worker_launch.json', {'token': token, 'supervisor_pid': os.getpid()})
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', ORIGINAL_EMA_REPLAY_WORKER_TOKEN=token)
    with (output/'worker_stdout.txt').open('xb') as stdout, (output/'worker_stderr.txt').open('xb') as stderr:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '_worker', '--bundle', str(Path(bundle).resolve())],
                                   cwd=output, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            exit_code = process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = process.wait()
    result = {'seed': receipt['seed'], 'worker_pid': process.pid, 'exit_code': exit_code,
              'timeout_seconds': TIMEOUT_SECONDS, 'timed_out': timed_out,
              'complete': exit_code == 0 and (output/'completion.json').is_file() and (output/'manifest.json').is_file()}
    result['reproduction_gate_passed'] = bool(result['complete'] and read(output/'completion.json')['gate_0_02pp_passed'])
    write(output/'supervisor.json', result)
    return result


def check_replays(outputs):
    """Pure file gate for BOTH seeds; never deploys or starts training."""
    if len(outputs) != 2:
        raise ValueError('Both original seeds are required')
    binding = read(HERE/'source_binding.json')
    seen = set()
    for value in outputs:
        output = Path(value)
        manifest = read(output/'manifest.json')
        if manifest['script_sha256'] != sha(__file__) or manifest['source_binding_sha256'] != sha(HERE/'source_binding.json'):
            raise ValueError('Replay implementation differs from the reviewed one')
        for name, digest in manifest['files'].items():
            if Path(name).name != name or sha(output/name) != digest:
                raise ValueError('Replay artifact changed or path invalid')
        required = {'execution_identity.json', 'native_metrics.json', 'completion.json', 'predictions.json', 'precision_recall.npz', 'forward_counts.json'}
        if not required.issubset(manifest['files']):
            raise ValueError('Replay archive is incomplete')
        done, supervisor = read(output/'completion.json'), read(output/'supervisor.json')
        identity = read(output/'execution_identity.json')['receipt']
        if (done['status'] != 'ORIGINAL_EMA_REPLAY_COMPLETED' or done.get('mode') != 'native_gpu_replay'
                or not supervisor['complete'] or supervisor['exit_code'] != 0 or supervisor['timed_out']
                or not done['state_unchanged'] or done['images'] != 548 or done['batch_forwards'] != 137
                or done['optimizer_constructions'] != 0 or done['parameter_or_EMA_updates'] != 0):
            raise ValueError('Replay has not completed in the required frozen native mode')
        seed = done['seed']
        if seed not in (1, 2) or seed in seen or identity['seed'] != seed or supervisor['seed'] != seed:
            raise ValueError('Replay seed is duplicated or inconsistent')
        seen.add(seed)
        target = binding['targets'][str(seed)]
        if identity['target'] != target:
            raise ValueError('Replay uses different original inputs')
        metrics = read(output/'native_metrics.json')
        gate = reproduction_gate(metrics['stats'], target['original_stats'])
        if (not gate['gate_0_02pp_passed'] or metrics['delta_from_original_log_pp'] != gate['delta_from_original_log_pp']
                or not done['gate_0_02pp_passed'] or not supervisor['reproduction_gate_passed']):
            raise ValueError('At least one original replay fails the unchanged 0.02pp gate')
    return {'status': 'BOTH_ORIGINAL_REPLAYS_PASS_0_02PP', 'seeds': sorted(seen),
            'training_started': False, 'training_authorization_inferred': False,
            'claim_boundary': 'Numeric prerequisite only; separate human scope and six-run runtime checks still apply.'}


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Standard-library preparation only; no model/evaluator imports')
    p.add_argument('--seed', type=int, choices=(1, 2), required=True)
    for key in ('experiment-root', 'native-root', 'evaluator-root', 'original-args', 'original-log',
                'checkpoint', 'val-images', 'val-ann', 'bundle', 'output'):
        p.add_argument('--'+key, required=True)
    for name in ('run-authorized', '_worker'):
        p = sub.add_parser(name)
        p.add_argument('--bundle', required=True)
    p = sub.add_parser('check-replays', help='Check both completed replay archives; never starts training')
    p.add_argument('--outputs', nargs=2, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args), ensure_ascii=False))
    elif args.command == 'run-authorized':
        print(json.dumps(execute_authorized(args.bundle), ensure_ascii=False))
    elif args.command == 'check-replays':
        print(json.dumps(check_replays(args.outputs), ensure_ascii=False))
    else:
        try:
            worker(args.bundle)
        except Exception:
            output = Path(read(Path(args.bundle)/'receipt.json')['paths']['output'])
            if output.is_dir() and not (output/'failure.json').exists():
                write(output/'failure.json', {'status': 'FAILED', 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
