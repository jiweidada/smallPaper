"""Prepare six source-bound configuration bundles. No torch or training import."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from integration import HERE, sha, verify_sources


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)


def prepare(destination):
    verify_sources()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a fresh bundle directory; prior receipts are preserved')
    binding = read(HERE/'plan_binding.json')
    paths = read(HERE/'local_paths.json')
    for name in ('train_ann', 'val_ann', 'common_init'):
        if sha(paths[name]) != paths[name+'_sha256']:
            raise RuntimeError('Input identity changed: ' + name)
    for name in ('train_images', 'val_images'):
        if not Path(paths[name]).is_dir():
            raise FileNotFoundError(paths[name])
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    implementation = {p.name: sha(p) for p in HERE.glob('*.py')}
    configs = sorted((HERE/'configs').glob('*.json'))
    if len(configs) != 6:
        raise RuntimeError('Expected exactly three arms by two seeds')
    for path in configs:
        if sha(path) != binding[path.name]:
            raise RuntimeError('Frozen plan changed: ' + path.name)
    destination.mkdir(parents=True, exist_ok=False)
    for path in configs:
        plan = read(path)
        cfg = deepcopy(plan['native_baseline_config'])
        cfg.pop('__include__', None)
        cfg.pop('config', None)
        cfg.update(model='P3ReadoutDEIM', resume=None,
                   tuning=paths['common_init'], use_swanlab=False,
                   output_dir=str(HERE/'future_runs'/path.stem))
        cfg['P3ReadoutDEIM'] = {'recipe': plan['model_recipe'], 'arm': plan['arm']}
        for split in ('train', 'val'):
            cfg[split+'_dataloader']['dataset'].update(
                img_folder=paths[split+'_images'], ann_file=paths[split+'_ann'])
        out = destination/path.stem
        out.mkdir()
        write(out/'resolved.yml', cfg)  # JSON is also valid YAML.
        write(out/'plan.json', plan)
        write(out/'receipt.json', {
            'status': 'PREPARED_NOT_TRAINED', 'arm': plan['arm'], 'seed': plan['seed'],
            'resolved_sha256': sha(out/'resolved.yml'), 'plan_sha256': sha(out/'plan.json'),
            'source_binding_sha256': sha(HERE/'source_binding.json'),
            'implementation': implementation, 'paths': paths,
            'pending_runtime_checks': ['GPU AMP', 'GPU memory/latency',
                                       'native training entry integration and evaluator identity',
                                       'paired data/augmentation stream audit'],
            'optimizer_constructions': 0, 'training_started': False})
    return {'bundles': 6, 'destination': str(destination), 'training_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False))
